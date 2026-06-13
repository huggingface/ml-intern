"""FastAPI application for HF Agent web interface."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load .env before importing routes/session_manager so persistence and model
# modules see local settings during startup.
load_dotenv(Path(__file__).parent.parent / ".env")

from routes.agent import router as agent_router  # noqa: E402
from routes.auth import router as auth_router  # noqa: E402
from session_manager import session_manager  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
SHUTDOWN_USAGE_REFRESH_CONCURRENCY = 32


async def _flush_session_on_shutdown(sid: str, agent_session, semaphore) -> None:
    sess = agent_session.session
    if not sess.config.save_sessions:
        return
    try:
        async with semaphore:
            await session_manager.refresh_session_usage_metrics(
                agent_session,
                error_code="lifespan_billing_snapshot_error",
            )
            sess.save_and_upload_detached(sess.config.session_dataset_repo)
            logger.info("Flushed session %s on shutdown", sid)
    except Exception as e:
        logger.warning("Failed to flush session %s: %s", sid, e)


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    configured_origins = _split_env_list(os.environ.get("ML_INTERN_CORS_ORIGINS"))
    return [*default_origins, *configured_origins]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting HF Agent backend...")
    await session_manager.start()
    # Start in-process hourly KPI rollup. Replaces an external cron so the
    # rollup lives next to the data and reuses the Space's HF token.
    try:
        import kpis_scheduler

        kpis_scheduler.start()
    except Exception as e:
        logger.warning("KPI scheduler failed to start: %s", e)
    yield

    logger.info("Shutting down HF Agent backend...")
    try:
        import kpis_scheduler

        await kpis_scheduler.shutdown()
    except Exception as e:
        logger.warning("KPI scheduler shutdown failed: %s", e)

    # Final-flush: save every still-active session so we don't lose traces on
    # server restart. Billing refreshes are timeboxed and bounded; uploads are
    # detached subprocesses.
    try:
        semaphore = asyncio.Semaphore(SHUTDOWN_USAGE_REFRESH_CONCURRENCY)
        await asyncio.gather(
            *(
                _flush_session_on_shutdown(sid, agent_session, semaphore)
                for sid, agent_session in list(session_manager.sessions.items())
            )
        )
    except Exception as e:
        logger.warning("Lifespan final-flush skipped: %s", e)
    await session_manager.close()


# Disable FastAPI auto-docs when running on HF Spaces (SPACE_ID is set by the
# platform) to avoid exposing the full API surface to anonymous visitors. Local
# dev keeps /docs and /redoc available.
_DOCS_DISABLED = os.environ.get("SPACE_ID") is not None

app = FastAPI(
    title="HF Agent",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _DOCS_DISABLED else "/docs",
    redoc_url=None if _DOCS_DISABLED else "/redoc",
    openapi_url=None if _DOCS_DISABLED else "/openapi.json",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(auth_router)

# Serve static files (frontend build) in production
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Serving static files from {static_path}")
else:
    logger.info("No static directory found, running in API-only mode")


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "HF Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
