import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import types

# Prevent importing heavy third-party modules when importing the backend module.
for _mod in ("litellm", "fastmcp", "thefuzz", "huggingface_hub"):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        # fastmcp is imported as `from fastmcp import Client` in some codepaths
        if _mod == "fastmcp":
            class _DummyClient:
                pass

            setattr(m, "Client", _DummyClient)
        sys.modules[_mod] = m

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from session_manager import SessionManager, AgentSession, MAX_SESSIONS


@pytest.mark.asyncio
async def test_restore_denied_when_at_capacity(caplog):
    manager = SessionManager()
    # Fill in-memory sessions up to MAX_SESSIONS
    for i in range(MAX_SESSIONS):
        manager.sessions[str(i)] = AgentSession(
            session_id=str(i),
            session=object(),
            tool_router=None,
            submission_queue=asyncio.Queue(),
            is_active=True,
        )

    class DummyStore:
        enabled = True

        async def load_session(self, sid):
            return {"metadata": {"user_id": "test", "model": "gpt"}, "messages": []}

    manager.persistence_store = DummyStore()

    caplog.set_level("WARNING")
    res = await manager.ensure_session_loaded("restored", user_id="test")
    assert res is None
    assert any("Cannot restore session" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_restore_allowed_under_capacity(monkeypatch):
    manager = SessionManager()
    manager.sessions.clear()

    class DummyStore:
        enabled = True

        async def load_session(self, sid):
            return {"metadata": {"user_id": "test", "model": "gpt"}, "messages": []}

    manager.persistence_store = DummyStore()

    # Replace the heavy sync constructor with a lightweight fake
    def fake_create_session_sync(*, session_id, user_id, hf_username, hf_token, model, event_queue, notification_destinations):
        class SimpleSession:
            def __init__(self):
                self.context_manager = SimpleNamespace(items=[object()])
                self.pending_approval = None
                self.turn_count = 0

        return None, SimpleSession()

    monkeypatch.setattr(manager, "_create_session_sync", fake_create_session_sync)

    # Fake _start_agent_session to register the session without starting tasks
    async def fake_start_agent_session(*, agent_session, event_queue, tool_router):
        async with manager._lock:
            manager.sessions[agent_session.session_id] = agent_session
        return agent_session

    monkeypatch.setattr(manager, "_start_agent_session", fake_start_agent_session)

    res = await manager.ensure_session_loaded("restored", user_id="test")
    assert res is not None
    assert res.session is not None