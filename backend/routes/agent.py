"""Agent API routes - WebSocket and REST endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

import logging
import os
from typing import Any

from dependencies import get_current_user, get_ws_user
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from litellm import acompletion

from agent.core.agent_loop import _resolve_hf_router_params
from models import (
    ApprovalRequest,
    HealthResponse,
    LLMHealthResponse,
    SessionInfo,
    SessionResponse,
    SubmitRequest,
)
from session_manager import MAX_SESSIONS, SessionCapacityError, session_manager
from websocket import manager as ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])

AVAILABLE_MODELS = [
    {
        "id": "huggingface/novita/minimax/minimax-m2.1",
        "label": "MiniMax M2.1",
        "provider": "huggingface",
        "recommended": True,
    },
    {
        "id": "anthropic/claude-opus-4-5-20251101",
        "label": "Claude Opus 4.5",
        "provider": "anthropic",
        "recommended": True,
    },
    {
        "id": "huggingface/novita/moonshotai/kimi-k2.5",
        "label": "Kimi K2.5",
        "provider": "huggingface",
    },
    {
        "id": "huggingface/novita/zai-org/glm-5",
        "label": "GLM 5",
        "provider": "huggingface",
    },
]


def _check_session_access(session_id: str, user: dict[str, Any]) -> None:
    """Verify the user has access to the given session. Raises 403 or 404."""
    info = session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session_manager.verify_session_access(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="Access denied to this session")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.active_session_count,
        max_sessions=MAX_SESSIONS,
    )


@router.get("/health/llm", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """Check if the LLM provider is reachable and the API key is valid.

    Makes a minimal 1-token completion call.  Catches common errors:
    - 401 → invalid API key
    - 402/insufficient_quota → out of credits
    - 429 → rate limited
    - timeout / network → provider unreachable
    """
    model = session_manager.config.model_name
    try:
        llm_params = _resolve_hf_router_params(model)
        await acompletion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=10,
            **llm_params,
        )
        return LLMHealthResponse(status="ok", model=model)
    except Exception as e:
        err_str = str(e).lower()
        error_type = "unknown"

        if (
            "401" in err_str
            or "auth" in err_str
            or "invalid" in err_str
            or "api key" in err_str
        ):
            error_type = "auth"
        elif (
            "402" in err_str
            or "credit" in err_str
            or "quota" in err_str
            or "insufficient" in err_str
            or "billing" in err_str
        ):
            error_type = "credits"
        elif "429" in err_str or "rate" in err_str:
            error_type = "rate_limit"
        elif "timeout" in err_str or "connect" in err_str or "network" in err_str:
            error_type = "network"

        logger.warning(f"LLM health check failed ({error_type}): {e}")
        return LLMHealthResponse(
            status="error",
            model=model,
            error=str(e)[:500],
            error_type=error_type,
        )


@router.get("/config/model")
async def get_model() -> dict:
    """Get current model and available models. No auth required."""
    return {
        "current": session_manager.config.model_name,
        "available": AVAILABLE_MODELS,
    }


@router.post("/config/model")
async def set_model(body: dict, user: dict = Depends(get_current_user)) -> dict:
    """Set the LLM model. Applies to new conversations."""
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    session_manager.config.model_name = model_id
    logger.info(f"Model changed to {model_id} by {user.get('username', 'unknown')}")
    return {"model": model_id}


@router.post("/title")
async def generate_title(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Generate a short title for a chat session based on the first user message."""
    model = session_manager.config.model_name
    llm_params = _resolve_hf_router_params(model)
    try:
        response = await acompletion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (max 6 words) for a chat conversation "
                        "that starts with the following user message. "
                        "Reply with ONLY the title, no quotes, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": request.text[:500]},
            ],
            max_tokens=20,
            temperature=0.3,
            timeout=8,
            **llm_params,
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        # Safety: cap at 50 chars
        if len(title) > 50:
            title = title[:50].rstrip() + "…"
        return {"title": title}
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        # Fallback: truncate the message
        fallback = request.text.strip()
        title = fallback[:40].rstrip() + "…" if len(fallback) > 40 else fallback
        return {"title": title}


@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: Request, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new agent session bound to the authenticated user.

    The user's HF access token is extracted from the Authorization header
    and stored in the session so that tools (e.g. hf_jobs) can act on
    behalf of the user.

    Returns 503 if the server or user has reached the session limit.
    """
    # Extract the user's HF token (Bearer header or HttpOnly cookie)
    hf_token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        hf_token = auth_header[7:]
    if not hf_token:
        hf_token = request.cookies.get("hf_access_token")

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"], hf_token=hf_token
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SessionResponse(session_id=session_id, ready=True)


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> SessionInfo:
    """Get session information. Only accessible by the session owner."""
    _check_session_access(session_id, user)
    info = session_manager.get_session_info(session_id)
    return SessionInfo(**info)


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: dict = Depends(get_current_user)) -> list[SessionInfo]:
    """List sessions belonging to the authenticated user."""
    sessions = session_manager.list_sessions(user_id=user["user_id"])
    return [SessionInfo(**s) for s in sessions]


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Delete a session. Only accessible by the session owner."""
    _check_session_access(session_id, user)
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit user input to a session. Only accessible by the session owner."""
    _check_session_access(request.session_id, user)
    success = await session_manager.submit_user_input(request.session_id, request.text)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit tool approvals to a session. Only accessible by the session owner."""
    _check_session_access(request.session_id, user)
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "feedback": a.feedback,
            "edited_script": a.edited_script,
        }
        for a in request.approvals
    ]
    success = await session_manager.submit_approval(request.session_id, approvals)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/interrupt/{session_id}")
async def interrupt_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Interrupt the current operation in a session."""
    _check_session_access(session_id, user)
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.post("/undo/{session_id}")
async def undo_session(session_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Undo the last turn in a session."""
    _check_session_access(session_id, user)
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Compact the context in a session."""
    _check_session_access(session_id, user)
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Shutdown a session."""
    _check_session_access(session_id, user)
    success = await session_manager.shutdown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "shutdown_requested", "session_id": session_id}


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time events.

    Authentication is done via:
    - ?token= query parameter (for browsers that can't send WS headers)
    - Cookie (automatic for same-origin connections)
    - Dev mode bypass (when OAUTH_CLIENT_ID is not set)

    NOTE: We must accept() before close() so the browser receives our custom
    close codes (4001, 4003, 4004).  If we close() before accept(), Starlette
    sends HTTP 403 and the browser only sees code 1006 (abnormal closure).
    """
    logger.info(f"WebSocket connection request for session {session_id}")

    # Authenticate the WebSocket connection
    user = await get_ws_user(websocket)
    if not user:
        logger.warning(
            f"WebSocket rejected: authentication failed for session {session_id}"
        )
        await websocket.accept()
        await websocket.close(code=4001, reason="Authentication required")
        return

    # Verify session exists
    info = session_manager.get_session_info(session_id)
    if not info:
        logger.warning(f"WebSocket rejected: session {session_id} not found")
        await websocket.accept()
        await websocket.close(code=4004, reason="Session not found")
        return

    # Verify user owns the session
    if not session_manager.verify_session_access(session_id, user["user_id"]):
        logger.warning(
            f"WebSocket rejected: user {user['user_id']} denied access to session {session_id}"
        )
        await websocket.accept()
        await websocket.close(code=4003, reason="Access denied")
        return

    await ws_manager.connect(websocket, session_id)

    # Send "ready" immediately on WebSocket connection so the frontend
    # knows the session is alive.  The original ready event from _run_session
    # fires before the WS is connected and is always lost.
    try:
        await websocket.send_json(
            {
                "event_type": "ready",
                "data": {"message": "Agent initialized"},
            }
        )
    except Exception as e:
        logger.error(f"Failed to send ready event for session {session_id}: {e}")

    try:
        while True:
            # Keep connection alive, handle ping/pong
            data = await websocket.receive_json()

            # Handle client messages (e.g., ping)
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        ws_manager.disconnect(session_id)
