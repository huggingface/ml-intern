import asyncio
import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import types

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"


@pytest.fixture
def session_manager_module(monkeypatch):
    """Import backend.session_manager with temporary dependency stubs.

    The stubs are inserted with monkeypatch so they are restored after the test,
    and the imported session_manager module is removed from sys.modules to avoid
    leaking the stubbed import state into other tests.
    """
    with monkeypatch.context() as m:
        m.syspath_prepend(str(_BACKEND_DIR))

        litellm_stub = types.ModuleType("litellm")

        class _DummyMessage:
            @staticmethod
            def model_validate(raw):
                return SimpleNamespace(**raw)

        setattr(litellm_stub, "Message", _DummyMessage)
        m.setitem(sys.modules, "litellm", litellm_stub)

        fastmcp_stub = types.ModuleType("fastmcp")

        class _DummyClient:
            pass

        setattr(fastmcp_stub, "Client", _DummyClient)
        m.setitem(sys.modules, "fastmcp", fastmcp_stub)

        m.setitem(sys.modules, "thefuzz", types.ModuleType("thefuzz"))
        m.setitem(
            sys.modules,
            "huggingface_hub",
            types.ModuleType("huggingface_hub"),
        )

        module = importlib.import_module("session_manager")
        yield module

    sys.modules.pop("session_manager", None)


@pytest.mark.asyncio
async def test_restore_denied_when_at_capacity(session_manager_module, caplog):
    manager = session_manager_module.SessionManager()
    for i in range(session_manager_module.MAX_SESSIONS):
        manager.sessions[str(i)] = session_manager_module.AgentSession(
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
async def test_restore_allowed_under_capacity(session_manager_module, monkeypatch):
    manager = session_manager_module.SessionManager()
    manager.sessions.clear()

    class DummyStore:
        enabled = True

        async def load_session(self, sid):
            return {"metadata": {"user_id": "test", "model": "gpt"}, "messages": []}

    manager.persistence_store = DummyStore()

    def fake_create_session_sync(
        *, session_id, user_id, hf_username, hf_token, model, event_queue, notification_destinations
    ):
        class SimpleSession:
            def __init__(self):
                self.context_manager = SimpleNamespace(items=[object()])
                self.pending_approval = None
                self.turn_count = 0
                self.notification_destinations = []
                self.config = SimpleNamespace(model_name=model)

        return None, SimpleSession()

    monkeypatch.setattr(manager, "_create_session_sync", fake_create_session_sync)

    async def fake_start_agent_session(*, agent_session, event_queue, tool_router):
        async with manager._lock:
            manager.sessions[agent_session.session_id] = agent_session
        return agent_session

    monkeypatch.setattr(manager, "_start_agent_session", fake_start_agent_session)

    res = await manager.ensure_session_loaded("restored", user_id="test")
    assert res is not None
    assert res.session is not None


@pytest.mark.asyncio
async def test_restore_rolls_back_placeholder_on_load_failure(session_manager_module):
    manager = session_manager_module.SessionManager()

    class FailingStore:
        enabled = True

        async def load_session(self, sid):
            raise RuntimeError("load failed")

    manager.persistence_store = FailingStore()

    with pytest.raises(RuntimeError, match="load failed"):
        await manager.ensure_session_loaded("restored", user_id="test")

    assert manager.sessions == {}


@pytest.mark.asyncio
async def test_create_session_rolls_back_placeholder_on_failure(
    session_manager_module, monkeypatch
):
    manager = session_manager_module.SessionManager()

    def fake_create_session_sync(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_create_session_sync", fake_create_session_sync)

    with pytest.raises(RuntimeError, match="boom"):
        await manager.create_session(user_id="u1")

    assert manager.sessions == {}


@pytest.mark.asyncio
async def test_unload_inactive_session_persists_and_removes(session_manager_module):
    manager = session_manager_module.SessionManager()
    persisted = []

    async def fake_persist_session_snapshot(agent_session, **kwargs):
        persisted.append((agent_session.session_id, kwargs))

    manager.persist_session_snapshot = fake_persist_session_snapshot

    stale_session = session_manager_module.AgentSession(
        session_id="stale",
        session=SimpleNamespace(
            pending_approval=None,
            notification_destinations=[],
            turn_count=0,
            config=SimpleNamespace(model_name="gpt"),
        ),
        tool_router=None,
        submission_queue=asyncio.Queue(),
        is_active=True,
        last_access=datetime.utcnow()
        - session_manager_module.INACTIVE_SESSION_IDLE_THRESHOLD
        - timedelta(seconds=1),
    )
    manager.sessions["stale"] = stale_session

    await manager._unload_inactive_sessions_once()

    assert "stale" not in manager.sessions
    assert persisted == [
        (
            "stale",
            {"runtime_state": "idle", "status": "inactive"},
        )
    ]


@pytest.mark.asyncio
async def test_unload_inactive_sessions_skips_active_and_processing(
    session_manager_module,
):
    manager = session_manager_module.SessionManager()
    persisted = []

    async def fake_persist_session_snapshot(agent_session, **kwargs):
        persisted.append(agent_session.session_id)

    manager.persist_session_snapshot = fake_persist_session_snapshot

    active_session = session_manager_module.AgentSession(
        session_id="active",
        session=SimpleNamespace(
            pending_approval=None,
            notification_destinations=[],
            turn_count=0,
            config=SimpleNamespace(model_name="gpt"),
        ),
        tool_router=None,
        submission_queue=asyncio.Queue(),
        is_active=True,
        last_access=datetime.utcnow()
        - session_manager_module.INACTIVE_SESSION_IDLE_THRESHOLD
        - timedelta(seconds=1),
    )
    processing_session = session_manager_module.AgentSession(
        session_id="processing",
        session=SimpleNamespace(
            pending_approval=None,
            notification_destinations=[],
            turn_count=0,
            config=SimpleNamespace(model_name="gpt"),
        ),
        tool_router=None,
        submission_queue=asyncio.Queue(),
        is_active=True,
        is_processing=True,
        last_access=datetime.utcnow()
        - session_manager_module.INACTIVE_SESSION_IDLE_THRESHOLD
        - timedelta(seconds=1),
    )
    manager.sessions["active"] = active_session
    manager.sessions["processing"] = processing_session

    await manager._unload_inactive_sessions_once()

    assert "active" in manager.sessions
    assert "processing" in manager.sessions
    assert persisted == []
