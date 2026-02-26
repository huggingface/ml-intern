"""Authentication dependencies for FastAPI routes.

Provides auth validation for both REST and WebSocket endpoints.
- In dev mode (OAUTH_CLIENT_ID not set): auth is bypassed, returns a default "dev" user.
- In production: validates Bearer tokens or cookies against HF OAuth.
"""

import logging
import os
import time
from typing import Any

import httpx
from fastapi import HTTPException, Request, WebSocket, status

logger = logging.getLogger(__name__)

OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")
AUTH_ENABLED = bool(os.environ.get("OAUTH_CLIENT_ID", ""))

# Simple in-memory token cache: token -> (user_info, expiry_time)
_token_cache: dict[str, tuple[dict[str, Any], float]] = {}
TOKEN_CACHE_TTL = 300  # 5 minutes

DEV_USER: dict[str, Any] = {
    "user_id": "dev",
    "username": "dev",
    "authenticated": True,
}


async def _validate_token(token: str) -> dict[str, Any] | None:
    """Validate a token against HF OAuth userinfo endpoint.

    Results are cached for TOKEN_CACHE_TTL seconds to avoid excessive API calls.
    """
    now = time.time()

    # Check cache
    if token in _token_cache:
        user_info, expiry = _token_cache[token]
        if now < expiry:
            return user_info
        del _token_cache[token]

    # Validate against HF
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{OPENID_PROVIDER_URL}/oauth/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code != 200:
                logger.debug("Token validation failed: status %d", response.status_code)
                return None
            user_info = response.json()
            _token_cache[token] = (user_info, now + TOKEN_CACHE_TTL)
            return user_info
        except httpx.HTTPError as e:
            logger.warning("Token validation error: %s", e)
            return None


def _user_from_info(user_info: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized user dict from HF userinfo response."""
    return {
        "user_id": user_info.get("sub", user_info.get("preferred_username", "unknown")),
        "username": user_info.get("preferred_username", "unknown"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "authenticated": True,
    }


async def _extract_user_from_token(token: str) -> dict[str, Any] | None:
    """Validate a token and return a user dict, or None."""
    user_info = await _validate_token(token)
    if user_info:
        return _user_from_info(user_info)
    return None


async def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: extract and validate the current user.

    Checks (in order):
    1. Authorization: Bearer <token> header
    2. hf_access_token cookie

    In dev mode (AUTH_ENABLED=False), returns a default dev user.
    """
    if not AUTH_ENABLED:
        return DEV_USER

    # Try Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        user = await _extract_user_from_token(token)
        if user:
            return user

    # Try cookie
    token = request.cookies.get("hf_access_token")
    if token:
        user = await _extract_user_from_token(token)
        if user:
            return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Please log in via /auth/login.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_ws_user(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract and validate user from WebSocket connection.

    WebSocket doesn't support custom headers from browser, so we check:
    1. ?token= query parameter
    2. hf_access_token cookie (sent automatically for same-origin)

    Returns user dict or None if not authenticated.
    In dev mode, returns the default dev user.
    """
    if not AUTH_ENABLED:
        return DEV_USER

    # Try query param
    token = websocket.query_params.get("token")
    if token:
        user = await _extract_user_from_token(token)
        if user:
            return user

    # Try cookie (works for same-origin WebSocket)
    token = websocket.cookies.get("hf_access_token")
    if token:
        user = await _extract_user_from_token(token)
        if user:
            return user

    return None
