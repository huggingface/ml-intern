"""Tests for the idle-session reaper and the global create-slot reservation.

Covers Parts B (hard global cap), C (idle reaper + activity stamps + safe
teardown + submit/reap race), and E (per-user concurrent cap interacts with
reaping) of the session-limit fix.
"""

from __future__ import annotations

import asyncio
import logging
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
from agent.core.session_persistence import (  # noqa: E402
    MongoSessionStore,
    NoopSessionStore,
)
from session_manager import (  # noqa: E402
    AgentSession,
    Operation,
    SessionCapacityError,
    SessionManager,
)
from agent.core.session import OpType  # noqa: E402


def test_reaper_idle_default_is_fifteen_minutes():
    assert sm.REAPER_IDLE_MINUTES == 15
    assert sm.REAPER_IDLE == timedelta(minutes=15)


def test_reaper_window_defaults():
    assert sm.REAPER_TOOL_APPROVAL_IDLE_MINUTES == 60
    assert sm.REAPER_TOOL_APPROVAL_IDLE == timedelta(minutes=60)


class RecordingStore(NoopSessionStore):
    """Captures every save_snapshot call so tests can assert persistence."""

    enabled = True

    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []

    async def save_snapshot(self, **kwargs: Any) -> None:
        self.snapshots.append(kwargs)

    def snapshots_for(self, session_id: str) -> list[dict[str, Any]]:
        return [s for s in self.snapshots if s.get("session_id") == session_id]


class FakeContextManager:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def add_message(self, message: Any, token_count: int | None = None) -> None:
        self.items.append(message)


class FakeSession:
    """Minimal Session stand-in supporting both persistence and _run_session."""

    def __init__(
        self,
        *,
        hf_token: str | None = "token",
        user_plan: str | None = None,
    ) -> None:
        self.hf_token = hf_token
        self.user_plan = user_plan
        self.context_manager = FakeContextManager()
        self.pending_approval: Any = None
        self.turn_count = 0
        self.config = SimpleNamespace(model_name="test-model", save_sessions=False)
        self.notification_destinations: list[str] = []
        self.auto_approval_enabled = False
        self.auto_approval_cost_cap_usd = None
        self.auto_approval_estimated_spend_usd = 0.0
        self.is_running = True
        self.cancel_called = False
        self.events: list[Any] = []

    async def send_event(self, event: Any) -> None:
        self.events.append(event)

    def cancel(self) -> None:
        self.cancel_called = True


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
        return object(), FakeSession(
            hf_token=kwargs.get("hf_token"),
            user_plan=kwargs.get("user_plan"),
        )

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
        # Processing work is never reaped while in flight.
        {
            "last_active_at": datetime.utcnow() - timedelta(hours=5),
            "is_processing": True,
        },
        # Awaiting tool approval, still inside the 60-min grace window.
        {
            "last_active_at": datetime.utcnow() - timedelta(minutes=30),
            "pending_approval": {"tool_calls": [{"id": "tc-1"}]},
        },
        # Acknowledgement prompt raised moments ago.
        {
            "last_active_at": datetime.utcnow() - timedelta(minutes=5),
            "pending_approval": {"kind": "usage_threshold", "tool_call_id": "u1"},
        },
        # Dev sessions are never reaped.
        {"last_active_at": datetime.utcnow() - timedelta(hours=5), "user_id": "dev"},
    ],
    ids=["fresh", "processing", "pending_tool_in_window", "pending_ack_fresh", "dev"],
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

    reaped = await manager._reap_one(
        "racing", verdict="evict_idle", now=datetime.utcnow()
    )

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
    process_started = asyncio.Event()

    async def fake_process(session: Any, submission: Any) -> bool:
        # Simulate a turn that has been running far longer than the idle
        # window before it completes.
        agent_session.last_active_at = datetime.utcnow() - timedelta(hours=3)
        process_started.set()
        return True

    monkeypatch.setattr(sm, "process_submission", fake_process)

    agent_session.submission_queue.put_nowait(
        sm.Submission(id="s1", operation=Operation(op_type=OpType.USER_INPUT, data={}))
    )

    await asyncio.wait_for(process_started.wait(), timeout=2.0)
    # Wait for the turn-finish stamp to land.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if (
            not agent_session.is_processing
            and datetime.utcnow() - agent_session.last_active_at < timedelta(minutes=1)
        ):
            break

    assert not agent_session.is_processing
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
        message = str(exc.value)
        assert f"maximum of {sm.MAX_SESSIONS_PER_USER} live sessions" in message
        assert "Close an existing session" in message
        assert (
            f"wait {sm.REAPER_IDLE_MINUTES:g} minutes for idle or usage/cost "
            "prompt sessions"
        ) in message
        assert (
            f"{sm.REAPER_TOOL_APPROVAL_IDLE_MINUTES:g} minutes for unanswered "
            "tool approvals"
        ) in message

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

    reaped = await manager._reap_one(
        "idle", verdict="evict_idle", now=datetime.utcnow()
    )

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

    reaped = await manager._reap_one(
        "idle", verdict="evict_idle", now=datetime.utcnow()
    )

    assert reaped is False
    assert "idle" in manager.sessions
    assert agent_session.is_reaping is False


# ── Reap / restore race ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_old_reaped_task_does_not_end_freshly_restored_session():
    """If a user reopens after the reaper pops the old wrapper but before the
    old task finishes sandbox cleanup, the old finally must not mark the fresh
    wrapper ended."""
    manager = _manager()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def slow_cleanup(session: Any) -> None:
        cleanup_started.set()
        await release_cleanup.wait()

    manager._cleanup_sandbox = slow_cleanup  # type: ignore[method-assign]

    old = await _start_real_run_session(
        manager,
        "restore-race",
        last_active_at=datetime.utcnow() - timedelta(hours=3),
    )
    reap_task = asyncio.create_task(
        manager._reap_one("restore-race", verdict="evict_idle", now=datetime.utcnow())
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if "restore-race" not in manager.sessions and cleanup_started.is_set():
            break

    assert "restore-race" not in manager.sessions
    assert cleanup_started.is_set()

    fresh = _make_agent_session("restore-race")
    async with manager._lock:
        manager.sessions["restore-race"] = fresh

    release_cleanup.set()
    assert await reap_task is True
    if old.task is not None:
        assert old.task.done()

    assert manager.sessions["restore-race"] is fresh
    assert fresh.is_active is True
    assert all(
        snapshot.get("status") != "ended"
        for snapshot in manager.persistence_store.snapshots_for("restore-race")
    )


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

    reap_task = asyncio.create_task(
        manager._reap_one("slow", verdict="evict_idle", now=datetime.utcnow())
    )
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


# ── Pending-approval eviction windows ────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["usage_threshold", "yolo_budget"])
async def test_reaper_evicts_pending_ack_after_idle_window(kind):
    """Acknowledgement prompts (usage-threshold / YOLO-cap) are auto-created
    at turn end with no user action; they must not pin a slot past the normal
    idle window, and must stay answerable after restore."""
    manager = _manager()

    async def fake_cleanup(session: Any) -> None:
        return None

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]

    agent_session = await _start_real_run_session(
        manager,
        "ack",
        last_active_at=datetime.utcnow() - timedelta(minutes=20),
    )
    pending = {"kind": kind, "tool_call_id": "p1", "continuation": "continue_agent"}
    agent_session.session.pending_approval = pending

    await manager._reap_idle_sessions()

    assert "ack" not in manager.sessions
    snapshots = manager.persistence_store.snapshots_for("ack")
    assert snapshots, "eviction must persist a resumable snapshot"
    assert all(s["status"] == "active" for s in snapshots)
    assert snapshots[-1]["runtime_state"] == "waiting_approval"
    assert snapshots[-1]["pending_approval"] == [pending]
    restored = FakeSession()
    manager._restore_pending_approval(restored, snapshots[-1]["pending_approval"])
    assert restored.pending_approval == pending


@pytest.mark.asyncio
async def test_reaper_evicts_tool_approval_after_long_window():
    """Real tool-permission prompts expire before reaping.

    They must not remain actionable after their sandbox is torn down.
    """
    manager = _manager()

    async def fake_cleanup(session: Any) -> None:
        return None

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]
    from litellm import ChatCompletionMessageToolCall as ToolCall

    agent_session = await _start_real_run_session(
        manager,
        "tool-approval",
        last_active_at=datetime.utcnow() - timedelta(hours=2),
    )
    tool_call = ToolCall(
        id="tc-1",
        type="function",
        function={"name": "create_file", "arguments": '{"path":"app.py"}'},
    )
    agent_session.session.pending_approval = {"tool_calls": [tool_call]}
    serialized = manager._serialize_pending_approval(agent_session.session)
    restored = FakeSession()
    manager._restore_pending_approval(restored, serialized)
    restored_tool_calls = restored.pending_approval["tool_calls"]
    assert restored_tool_calls[0].id == "tc-1"
    assert restored_tool_calls[0].function.name == "create_file"

    await manager._reap_idle_sessions()

    assert "tool-approval" not in manager.sessions
    assert agent_session.session.pending_approval is None
    snapshots = manager.persistence_store.snapshots_for("tool-approval")
    assert snapshots
    assert all(s["status"] == "active" for s in snapshots)
    assert snapshots[-1]["runtime_state"] == "idle"
    assert snapshots[-1]["pending_approval"] == []
    [expired_msg] = agent_session.session.context_manager.items
    assert expired_msg.role == "tool"
    assert expired_msg.tool_call_id == "tc-1"
    assert expired_msg.name == "create_file"
    assert "expired after inactivity" in expired_msg.content
    [event] = [
        event
        for event in agent_session.session.events
        if event.event_type == "tool_state_change"
    ]
    assert event.event_type == "tool_state_change"
    assert event.data["state"] == "abandoned"
    assert event.data["reason"] == "expired_inactive"


@pytest.mark.asyncio
async def test_reaper_spares_quiet_processing_session():
    """Processing work is preserved even if it has been quiet for a long time."""
    manager = _manager()
    agent_session = _make_agent_session(
        "busy",
        last_active_at=datetime.utcnow() - timedelta(hours=5),
        is_processing=True,
    )
    manager.sessions["busy"] = agent_session

    await manager._reap_idle_sessions()

    assert "busy" in manager.sessions
    assert agent_session.is_reaping is False


@pytest.mark.asyncio
async def test_reaper_spares_quiet_processing_session_with_queued_messages():
    """Accepted queued submissions are not dropped by processing-session reaps."""
    manager = _manager()

    async def fake_cleanup(session: Any) -> None:
        return None

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]

    agent_session = _make_agent_session(
        "busy-queued",
        last_active_at=datetime.utcnow() - timedelta(hours=5),
        is_processing=True,
    )
    manager.sessions["busy-queued"] = agent_session
    agent_session.submission_queue.put_nowait(object())

    await manager._reap_idle_sessions()

    assert "busy-queued" in manager.sessions
    assert agent_session.submission_queue.qsize() == 1
    assert agent_session.is_reaping is False
    assert agent_session.session.cancel_called is False
    assert manager.persistence_store.snapshots_for("busy-queued") == []


# ── Verdict re-check ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reap_one_aborts_on_verdict_mismatch():
    """A session whose state changes between selection and teardown (e.g. the
    prompt was answered) must abort this reap cycle."""
    manager = _manager()
    agent_session = _make_agent_session(
        "answered",
        last_active_at=datetime.utcnow() - timedelta(hours=2),
        pending_approval={"tool_calls": [{"id": "tc-1"}]},
    )
    manager.sessions["answered"] = agent_session

    now = datetime.utcnow()
    assert manager._reaper_verdict(agent_session, now) == "evict_pending_tool"

    # Approval answered in the gap between selection and teardown.
    agent_session.session.pending_approval = None
    reaped = await manager._reap_one("answered", verdict="evict_pending_tool", now=now)

    assert reaped is False
    assert "answered" in manager.sessions
    assert agent_session.is_reaping is False


# ── Mongo store retry ────────────────────────────────────────────────────


class FlippableMongoStore(MongoSessionStore):
    """MongoSessionStore whose init() flips enabled without touching the
    network, to exercise the sweep's maybe_reconnect path."""

    def __init__(self, *, recovers: bool) -> None:
        super().__init__("mongodb://unreachable.invalid", "testdb")
        self.recovers = recovers
        self.init_calls = 0
        self.snapshots: list[dict[str, Any]] = []

    async def init(self) -> None:
        self.init_calls += 1
        self.enabled = self.recovers

    async def save_snapshot(self, **kwargs: Any) -> None:
        self.snapshots.append(kwargs)


@pytest.mark.asyncio
async def test_reaper_retries_disabled_mongo_store():
    """A Mongo store that failed its boot-time init is retried at sweep time,
    so one boot blip no longer disables reaping until the next restart."""
    manager = _manager()

    async def fake_cleanup(session: Any) -> None:
        return None

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]
    store = FlippableMongoStore(recovers=True)
    manager.persistence_store = store
    manager.sessions["idle"] = _make_agent_session(
        "idle", last_active_at=datetime.utcnow() - timedelta(hours=1)
    )

    await manager._reap_idle_sessions()

    assert store.init_calls == 1
    assert "idle" not in manager.sessions
    assert store.snapshots


@pytest.mark.asyncio
async def test_reaper_stays_noop_while_mongo_store_down():
    manager = _manager()
    store = FlippableMongoStore(recovers=False)
    manager.persistence_store = store
    agent_session = _make_agent_session(
        "idle", last_active_at=datetime.utcnow() - timedelta(hours=1)
    )
    manager.sessions["idle"] = agent_session

    await manager._reap_idle_sessions()

    assert store.init_calls == 1
    assert "idle" in manager.sessions
    assert agent_session.is_reaping is False


# ── Capacity-error breakdown ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capacity_error_message_breaks_down_held_slots():
    manager = _manager()
    sessions = []
    for i in range(3):
        sessions.append(
            _make_agent_session(
                f"tool-{i}", pending_approval={"tool_calls": [{"id": f"t{i}"}]}
            )
        )
    for i in range(2):
        sessions.append(
            _make_agent_session(
                f"ack-{i}",
                pending_approval={"kind": "usage_threshold", "tool_call_id": f"a{i}"},
            )
        )
    sessions.append(_make_agent_session("busy", is_processing=True))
    for i in range(4):
        sessions.append(_make_agent_session(f"idle-{i}"))
    assert len(sessions) == sm.MAX_SESSIONS_PER_USER
    for agent_session in sessions:
        manager.sessions[agent_session.session_id] = agent_session

    with pytest.raises(SessionCapacityError) as exc:
        await manager.create_session(user_id="owner")

    message = str(exc.value)
    assert f"maximum of {sm.MAX_SESSIONS_PER_USER} live sessions" in message
    assert "Close an existing session" in message
    assert "Currently held: 3 awaiting tool approval, 2 usage/cost prompts" in message
    assert "1 still processing, 4 idle." in message


# ── Sweep observability ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_log_emitted_only_when_nonzero(caplog):
    manager = _manager()
    manager.sessions["busy-only"] = _make_agent_session("busy-only", is_processing=True)

    with caplog.at_level(logging.INFO):
        await manager._reap_idle_sessions()
    assert "Reaper sweep:" not in caplog.text
    manager.sessions.clear()

    async def fake_cleanup(session: Any) -> None:
        return None

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]
    manager.sessions["idle"] = _make_agent_session(
        "idle", last_active_at=datetime.utcnow() - timedelta(hours=1)
    )
    manager.sessions["busy"] = _make_agent_session("busy", is_processing=True)

    with caplog.at_level(logging.INFO):
        await manager._reap_idle_sessions()

    assert "Reaper sweep:" in caplog.text
    assert "evicted_idle=1" in caplog.text
    assert "skipped_processing=1" in caplog.text
