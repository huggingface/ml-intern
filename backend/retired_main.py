"""Minimal server for the retired ML Intern web application."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

HUGGINGCHAT_URL = "https://huggingface.co/chat/"
DEFAULT_STATIC_PATH = Path(__file__).parent.parent / "static"


def create_app(static_path: Path = DEFAULT_STATIC_PATH) -> FastAPI:
    """Create the retirement app, optionally serving a supplied static build."""
    retired_app = FastAPI(
        title="ML Intern",
        description="Retired ML Intern web application",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @retired_app.get("/api")
    async def retirement_status() -> dict[str, str]:
        """Return a lightweight status response for Space health checks."""
        return {
            "name": "ML Intern",
            "status": "retired",
            "moved_to": HUGGINGCHAT_URL,
        }

    # Register the catch-all static mount after /api so the health endpoint
    # remains reachable. No agent or authentication routers are imported by
    # this entrypoint.
    if static_path.exists():
        retired_app.mount(
            "/", StaticFiles(directory=str(static_path), html=True), name="static"
        )

    return retired_app


app = create_app()
