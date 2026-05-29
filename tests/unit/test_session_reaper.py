"""Tests for the idle-session reaper and the global create-slot reservation.

Covers Parts B (hard global cap), C (idle reaper + activity stamps + safe
teardown + submit/reap race), and E (per-user concurrent cap interacts with
reaping) of the session-limit fix.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import session_manager as sm  # noqa: E402
from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from session_manager import (  # noqa: E402
    AgentSession,
    Operation,
    SessionCapacityError,
    SessionManager,
)
from agent.core.session import OpType  # noqa: E402


class RecordingStore(NoopSessionStore):
    """Captures every save_snapshot call so tests can assert persistence."""

    enabled = True

    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []

    async def save_snapshot(self, **kwargs: Any) -> None:
        self.snapshots.append(kwargs)

    def snapshots_for(self, session_id: str) -> list[dict[str, Any]]:
        return [s for s in self.snapshots if s.get("session_id") == session_id]


class FakeSession:
    """Minimal Session stand-in supporting both persistence and _run_session."""

    def __init__(self, *, hf_token: str | None = "token") -> None:
        self.hf_token = hf_token
        self.context_manager = SimpleNamespace(items=[])
        self.pending_approval: Any = None
        self.turn_count = 0
        self.config = SimpleNamespace(model_name="test-model", save_sessions=False)
        self.notification_destinations: list[str] = []
        self.auto_approval_enabled = False
        self.auto_approval_cost_cap_usd = None
        self.auto_approval_estimated_spend_usd = 0.0
        self.is_running = True

    async def send_event(self, event: Any) -> None:
        return None


class FakeToolRouter:
    async def __aenter__(self) -> "FakeToolRouter":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _manager() -> SessionManager:
    manager = object.__new__(SessionManager)
    manager.config = SimpleNamespace(model_name="test-model")
    manager.sessions = {}
    manager._lock = asyncio.Lock()
    manager.persistence_store = RecordingStore()
    manager.messaging_gateway = SimpleNamespace()
    manager._pending_creates = 0
    manager._reaper_task = None
    return manager


def _make_agent_session(
    session_id: str,
    *,
    user_id: str = "owner",
    last_active_at: datetime | None = None,
    is_processing: bool = False,
    pending_approval: Any = None,
) -> AgentSession:
    session = FakeSession()
    session.pending_approval = pending_approval
    agent_session = AgentSession(
        session_id=session_id,
        session=session,  # type: ignore[arg-type]
        tool_router=FakeToolRouter(),  # type: ignore[arg-type]
        submission_queue=asyncio.Queue(),
        user_id=user_id,
        hf_token="token",
    )
    agent_session.is_processing = is_processing
    if last_active_at is not None:
        agent_session.last_active_at = last_active_at
    return agent_session


async def _start_real_run_session(
    manager: SessionManager,
    session_id: str,
    *,
    user_id: str = "owner",
    last_active_at: datetime | None = None,
) -> AgentSession:
    """Insert a session and start the REAL _run_session task for it."""
    agent_session = _make_agent_session(
        session_id, user_id=user_id, last_active_at=last_active_at
    )
    event_queue: asyncio.Queue = asyncio.Queue()
    await manager._start_agent_session(
        agent_session=agent_session,
        event_queue=event_queue,
        tool_router=agent_session.tool_router,
    )
    await asyncio.sleep(0)  # let the run loop reach its queue wait
    return agent_session


def _install_fake_create(manager: SessionManager) -> asyncio.Event:
    """Replace blocking constructors + run loop with fakes for create tests."""
    stop = asyncio.Event()

    def fake_create_session_sync(**kwargs: Any):
        return object(), FakeSession(hf_token=kwargs.get("hf_token"))

    async def fake_run_session(*_: Any) -> None:
        await stop.wait()

    manager._create_session_sync = fake_create_session_sync  # type: ignore[method-assign]
    manager._run_session = fake_run_session  # type: ignore[method-assign]
    manager._start_cpu_sandbox_preload = lambda _agent_session: None  # type: ignore[method-assign]
    return stop


async def _cancel_tasks(manager: SessionManager) -> None:
    tasks = [
        agent_session.task
        for agent_session in manager.sessions.values()
        if agent_session.task and not agent_session.task.done()
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ── Reaper happy path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaper_evicts_idle_session_as_resumable():
    manager = _manager()
    cleaned: list[Any] = []

    async def fake_cleanup(session: Any) -> None:
        cleaned.append(session)

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]

    agent_session = await _start_real_run_session(
        manager,
        "stale",
        last_active_at=datetime.utcnow() - timedelta(hours=3),
    )
    session = agent_session.session

    await manager._reap_idle_sessions()

    # Evicted from the live pool, sandbox torn down by the task's finally.
    assert "stale" not in manager.sessions
    assert cleaned == [session]

    # Persisted resumable (status="active", runtime_state="idle"), never "ended".
    store = manager.persistence_store
    snapshots = store.snapshots_for("stale")
    assert snapshots, "reaper should persist a snapshot"
    assert all(s["status"] == "active" for s in snapshots)
    assert snapshots[-1]["runtime_state"] == "idle"
    assert not any(s["status"] == "ended" for s in snapshots)


# ── Spared sessions ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        # Fresh: touched just now.
        {"last_active_at": None},
        # Currently processing a turn.
        {
            "last_active_at": datetime.utcnow() - timedelta(hours=5),
            "is_processing": True,
        },
        # Awaiting tool approval ("approve later", not idle).
        {
            "last_active_at": datetime.utcnow() - timedelta(hours=5),
            "pending_approval": {"tool_calls": [object()]},
        },
        # Dev sessions are never reaped.
        {"last_active_at": datetime.utcnow() - timedelta(hours=5), "user_id": "dev"},
    ],
    ids=["fresh", "processing", "pending_approval", "dev"],
)
async def test_reaper_spares(kwargs):
    manager = _manager()
    agent_session = _make_agent_session("spared", **kwargs)
    manager.sessions["spared"] = agent_session

    await manager._reap_idle_sessions()

    assert "spared" in manager.sessions
    assert agent_session.is_reaping is False


# ── Submit / reap race ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reap_aborts_when_message_enqueued_first():
    """A message enqueued before teardown makes the queue non-empty, so the
    reaper's re-check aborts — the message is never lost."""
    manager = _manager()
    agent_session = _make_agent_session(
        "racing", last_active_at=datetime.utcnow() - timedelta(hours=3)
    )
    manager.sessions["racing"] = agent_session
    agent_session.submission_queue.put_nowait(object())

    cutoff = datetime.utcnow() - sm.REAPER_IDLE
    reaped = await manager._reap_one("racing", cutoff)

    assert reaped is False
    assert "racing" in manager.sessions
    assert agent_session.is_reaping is False
    assert agent_session.submission_queue.qsize() == 1


@pytest.mark.asyncio
async def test_submit_rejected_while_reaping():
    """submit() refuses a session being reaped instead of silently enqueuing
    onto a dying runtime (the caller then reloads a fresh one)."""
    manager = _manager()
    agent_session = _make_agent_session("reaping")
    agent_session.is_reaping = True
    manager.sessions["reaping"] = agent_session

    ok = await manager.submit("reaping", Operation(op_type=OpType.USER_INPUT, data={}))

    assert ok is False
    assert agent_session.submission_queue.empty()


# ── Turn-finish timestamp ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_finish_restamps_activity(monkeypatch):
    """A turn that runs longer than the idle window must not be reaped the
    instant it finishes — the turn-finish stamp resets the idle clock."""
    manager = _manager()

    async def fake_cleanup(session: Any) -> None:
        return None

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]

    agent_session = await _start_real_run_session(
        manager, "longturn", last_active_at=datetime.utcnow()
    )

    async def fake_process(session: Any, submission: Any) -> bool:
        # Simulate a turn that has been running far longer than the idle
        # window before it completes.
        agent_session.last_active_at = datetime.utcnow() - timedelta(hours=3)
        return True

    monkeypatch.setattr(sm, "process_submission", fake_process)

    agent_session.submission_queue.put_nowait(
        sm.Submission(id="s1", operation=Operation(op_type=OpType.USER_INPUT, data={}))
    )

    # Wait for the turn-finish stamp to land.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if datetime.utcnow() - agent_session.last_active_at < timedelta(minutes=1):
            break

    assert datetime.utcnow() - agent_session.last_active_at < timedelta(minutes=1)

    await _cancel_tasks(manager)


# ── Global reservation race (Part B) ────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_creates_cannot_exceed_global_cap(monkeypatch):
    manager = _manager()
    stop = _install_fake_create(manager)
    monkeypatch.setattr(sm, "MAX_SESSIONS", 3)

    try:
        results = await asyncio.gather(
            *[manager.create_session(user_id="owner") for _ in range(10)],
            return_exceptions=True,
        )
        created = [r for r in results if isinstance(r, str)]
        errors = [r for r in results if isinstance(r, SessionCapacityError)]

        assert len(created) == 3
        assert len(errors) == 7
        assert all(e.error_type == "global" for e in errors)
        assert len(manager.sessions) == 3
        assert manager._pending_creates == 0
    finally:
        stop.set()
        await _cancel_tasks(manager)


@pytest.mark.asyncio
async def test_failed_build_releases_reservation(monkeypatch):
    manager = _manager()
    _install_fake_create(manager)
    monkeypatch.setattr(sm, "MAX_SESSIONS", 5)

    def boom(**_: Any):
        raise RuntimeError("build failed")

    manager._create_session_sync = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await manager.create_session(user_id="owner")

    # The reservation must be released so a failed create can't shrink the pool.
    assert manager._pending_creates == 0
    assert manager.sessions == {}


# ── Per-user concurrent cap interacts with reclaimed slots (Part E) ──────


@pytest.mark.asyncio
async def test_per_user_cap_frees_up_after_slot_reclaimed():
    manager = _manager()
    stop = _install_fake_create(manager)

    try:
        for i in range(sm.MAX_SESSIONS_PER_USER):
            manager.sessions[f"owner-{i}"] = _make_agent_session(
                f"owner-{i}", user_id="owner"
            )

        # At the concurrent cap → rejected.
        with pytest.raises(SessionCapacityError) as exc:
            await manager.create_session(user_id="owner")
        assert exc.value.error_type == "per_user"

        # Reclaiming a slot (the reaper evicts an idle session) frees capacity.
        manager.sessions.pop("owner-0")
        new_id = await manager.create_session(user_id="owner")

        assert isinstance(new_id, str)
        assert manager._count_user_sessions("owner") == sm.MAX_SESSIONS_PER_USER
    finally:
        stop.set()
        await _cancel_tasks(manager)


# ── Persistence safety (resumability invariant) ──────────────────────────


@pytest.mark.asyncio
async def test_reaper_skips_when_persistence_disabled():
    """With no usable store a reaped session couldn't be restored, so eviction
    would destroy non-dev chats outright. The sweep must be a no-op."""
    manager = _manager()
    manager.persistence_store = NoopSessionStore()  # enabled = False
    agent_session = _make_agent_session(
        "idle", last_active_at=datetime.utcnow() - timedelta(hours=5)
    )
    manager.sessions["idle"] = agent_session

    await manager._reap_idle_sessions()

    assert "idle" in manager.sessions
    assert agent_session.is_reaping is False


@pytest.mark.asyncio
async def test_reap_aborts_when_snapshot_write_fails():
    """If the resumable snapshot can't be written (e.g. a transient Mongo
    error), abort rather than evict unrecoverable state — leave it live."""
    manager = _manager()

    class FailingStore(RecordingStore):
        async def save_snapshot(self, **kwargs: Any) -> None:
            raise RuntimeError("mongo write failed")

    manager.persistence_store = FailingStore()
    agent_session = _make_agent_session(
        "idle", last_active_at=datetime.utcnow() - timedelta(hours=5)
    )
    manager.sessions["idle"] = agent_session

    cutoff = datetime.utcnow() - sm.REAPER_IDLE
    reaped = await manager._reap_one("idle", cutoff)

    assert reaped is False
    assert "idle" in manager.sessions
    assert agent_session.is_reaping is False


@pytest.mark.asyncio
async def test_reap_aborts_when_message_write_fails_silently():
    """The real store swallows message bulk_write errors for best-effort callers
    and only surfaces them when asked to be strict. The reaper must request
    strict mode, so a silent message-write failure still aborts the reap (not
    only metadata/connection failures that already make save_snapshot raise)."""
    manager = _manager()

    class StrictModeStore(RecordingStore):
        # Mirrors MongoSessionStore: message-write failure is swallowed unless
        # the caller passes raise_on_error (which the reaper does).
        async def save_snapshot(
            self, *, raise_on_error: bool = False, **kwargs: Any
        ) -> None:
            if raise_on_error:
                raise RuntimeError("message bulk_write failed")

    manager.persistence_store = StrictModeStore()
    agent_session = _make_agent_session(
        "idle", last_active_at=datetime.utcnow() - timedelta(hours=5)
    )
    manager.sessions["idle"] = agent_session

    cutoff = datetime.utcnow() - sm.REAPER_IDLE
    reaped = await manager._reap_one("idle", cutoff)

    assert reaped is False
    assert "idle" in manager.sessions
    assert agent_session.is_reaping is False


# ── Shutdown safety (cancellation must propagate) ────────────────────────


@pytest.mark.asyncio
async def test_reaper_teardown_propagates_outer_cancellation():
    """Cancelling the reaper while it awaits a slow teardown must propagate, so
    close() can't hang. Regression for the CancelledError-conflation bug: the
    old wait_for + bare-except swallowed the reaper's own cancellation."""
    manager = _manager()

    release = asyncio.Event()

    async def slow_cleanup(session: Any) -> None:
        await release.wait()  # block teardown so _reap_one parks in the wait

    manager._cleanup_sandbox = slow_cleanup  # type: ignore[method-assign]

    agent_session = await _start_real_run_session(
        manager, "slow", last_active_at=datetime.utcnow() - timedelta(hours=3)
    )
    cutoff = datetime.utcnow() - sm.REAPER_IDLE

    reap_task = asyncio.create_task(manager._reap_one("slow", cutoff))
    # Let _reap_one persist + pop, then enter the teardown wait (the session
    # task is stuck in slow_cleanup, so the wait won't complete on its own).
    for _ in range(100):
        await asyncio.sleep(0.01)
        if "slow" not in manager.sessions:
            break
    await asyncio.sleep(0.02)

    # Simulate close() cancelling the reaper mid-teardown.
    reap_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reap_task

    # Unblock the stuck teardown and reap the orphaned session task.
    release.set()
    if agent_session.task is not None:
        await asyncio.gather(agent_session.task, return_exceptions=True)
