"""WebSocket connection manager for real-time communication."""

import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for multiple sessions."""

    def __init__(self) -> None:
        # session_id -> WebSocket
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a WebSocket connection and register it."""
        logger.info(f"Attempting to accept WebSocket for session {session_id}")
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected and registered for session {session_id}")

    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_event(
        self, session_id: str, event_type: str, data: dict[str, Any] | None = None
    ) -> None:
        """Send an event to a specific session's WebSocket."""
        if session_id not in self.active_connections:
            logger.warning(f"No active connection for session {session_id}")
            return

        message = {"event_type": event_type}
        if data is not None:
            message["data"] = data

        try:
            await self.active_connections[session_id].send_json(message)
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")
            self.disconnect(session_id)

    async def broadcast(
        self, event_type: str, data: dict[str, Any] | None = None
    ) -> None:
        """Broadcast an event to all connected sessions."""
        for session_id in list(self.active_connections.keys()):
            await self.send_event(session_id, event_type, data)

    def is_connected(self, session_id: str) -> bool:
        """Check if a session has an active WebSocket connection."""
        return session_id in self.active_connections


# Global connection manager instance
manager = ConnectionManager()
