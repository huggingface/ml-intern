"""Session manager for handling multiple concurrent agent sessions."""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from agent.config import load_config
from agent.core.agent_loop import process_submission
from agent.core.model_ids import (
    KIMI_K26_MODEL_ID,
    is_known_router_model_id,
    strip_huggingface_model_prefix,
)
from agent.core.session import Event, OpType, Session
from agent.core.session_persistence import get_session_store
from agent.core.tools import ToolRouter
from agent.core.usage_thresholds import (
    USAGE_THRESHOLD_TOOL_NAME,
    USAGE_WARNING_FIRST_THRESHOLD_USD,
    is_usage_threshold_pending,
    next_usage_warning_threshold,
    normalize_usage_threshold,
    usage_threshold_pending_to_tool,
)
from agent.messaging.gateway import NotificationGateway

# Get project root (parent of backend directory)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "frontend_agent_config.json")
USAGE_WARNING_SPEND_CACHE_TTL_SECONDS = 30.0


# These dataclasses match agent/main.py structure
@dataclass
class Operation:
    """Operation to be executed by the agent."""

    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    """Submission to the agent loop."""

    id: str
    operation: Operation


logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Reads from the agent's event queue and fans out to SSE subscribers.

    Events that arrive when no subscribers are listening are discarded by
    this in-memory fanout. Durable replay is handled by session_persistence.
    """

    def __init__(self, event_queue: asyncio.Queue):
        self._source = event_queue
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._counter = 0

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        """Create a new subscriber. Returns (id, queue)."""
        self._counter += 1
        sub_id = self._counter
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[sub_id] = q
        return sub_id, q

    def unsubscribe(self, sub_id: int) -> None:
        self._subscribers.pop(sub_id, None)

    async def run(self) -> None:
        """Main loop — reads from source queue and broadcasts."""
        while True:
            try:
                event: Event = await self._source.get()
                msg = {
                    "event_type": event.event_type,
                    "data": event.data,
                    "seq": event.seq,
                }
                for q in self._subscribers.values():
                    await q.put(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBroadcaster error: {e}")


@dataclass
class AgentSession:
    """Wrapper for an agent session with its associated resources."""

    session_id: str
    session: Session
    tool_router: ToolRouter
    submission_queue: asyncio.Queue
    user_id: str = "dev"  # Owner of this session
    hf_username: str | None = None  # HF namespace used for personal trace uploads
    hf_token: str | None = None  # User's HF OAuth token for tool execution
    user_plan: str | None = None  # Active HF account plan for plan-aware agent CTAs
    task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Last genuine activity (submit/turn-start/turn-finish/direct user write).
    # Drives the idle reaper. Defaults to load time so a freshly-restored but
    # untouched session isn't reaped for a full idle window.
    last_active_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_processing: bool = False  # True while a submission is being executed
    # Set under the lock by the reaper while tearing this session down. Blocks
    # submit() from enqueueing onto a session that's being evicted.
    is_reaping: bool = False
    broadcaster: Any = None
    title: str | None = None
    usage_window_started_at: datetime | None = None
    usage_window_baseline: dict[str, Any] | None = None
    usage_warning_next_threshold_usd: float = USAGE_WARNING_FIRST_THRESHOLD_USD
    usage_warning_spend_cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.usage_window_started_at is None:
            self.usage_window_started_at = self.created_at


class SessionCapacityError(Exception):
    """Raised when no more sessions can be created."""

    def __init__(self, message: str, error_type: str = "global") -> None:
        super().__init__(message)
        self.error_type = error_type  # "global" or "per_user"


# ── Capacity limits ─────────────────────────────────────────────────
# Sized for HF Spaces 8 vCPU / 32 GB RAM.
# Each session uses ~10-20 MB (context, tools, queues, task); 200 × 20 MB
# = 4 GB worst case, leaving plenty of headroom for the Python runtime
# and per-request overhead.
MAX_SESSIONS: int = 200
MAX_SESSIONS_PER_USER: int = 10
DEFAULT_YOLO_COST_CAP_USD: float = 5.0
SANDBOX_SHUTDOWN_CLEANUP_CONCURRENCY: int = 10
SANDBOX_SHUTDOWN_CLEANUP_TIMEOUT_S: float = 60.0

# ── Idle-session reaper ─────────────────────────────────────────────
# A live session idle ≥ REAPER_IDLE_MINUTES with no in-flight work has its
# sandbox + RAM released and is evicted from the live pool, while staying
# fully resumable from Mongo (it reappears as a normal chat, never "ended").
# This frees both the global pool and the user's concurrent slots.
REAPER_IDLE_MINUTES: float = float(os.environ.get("REAPER_IDLE_MINUTES", "15"))
REAPER_INTERVAL_S: float = float(os.environ.get("REAPER_INTERVAL_S", "300"))
REAP_TEARDOWN_TIMEOUT_S: float = float(os.environ.get("REAP_TEARDOWN_TIMEOUT_S", "30"))
REAPER_IDLE = timedelta(minutes=REAPER_IDLE_MINUTES)


class SessionManager:
    """Manages multiple concurrent agent sessions."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path or DEFAULT_CONFIG_PATH)
        normalized_default = strip_huggingface_model_prefix(self.config.model_name)
        if normalized_default:
            self.config.model_name = normalized_default
        self.messaging_gateway = NotificationGateway(self.config.messaging)
        self.sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()
        self.persistence_store = None
        # In-flight create_session calls that have passed the capacity check
        # but not yet inserted their session. Counted alongside
        # active_session_count to hard-cap the global pool against the
        # check-then-create race (see create_session).
        self._pending_creates: int = 0
        self._reaper_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start shared background resources."""
        self.persistence_store = get_session_store()
        await self.persistence_store.init()
        await self.messaging_gateway.start()
        self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def close(self) -> None:
        """Flush and close shared background resources."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        await self._cleanup_all_sandboxes_on_close()
        await self.messaging_gateway.close()
        if self.persistence_store is not None:
            await self.persistence_store.close()

    def _store(self):
        if self.persistence_store is None:
            self.persistence_store = get_session_store()
        return self.persistence_store

    def _count_user_sessions(self, user_id: str) -> int:
        """Count active sessions owned by a specific user."""
        return sum(
            1 for s in self.sessions.values() if s.user_id == user_id and s.is_active
        )

    @staticmethod
    def _touch(agent_session: "AgentSession") -> None:
        """Stamp genuine activity so the idle reaper's clock resets.

        Call on real user/agent activity (submit, turn start/finish, direct
        user-initiated writes) — never on passive reads or hydration, which
        would keep an otherwise-idle session alive forever.
        """
        agent_session.last_active_at = datetime.utcnow()

    @staticmethod
    def _model_from_saved_metadata(
        model: str | None,
    ) -> str:
        normalized = strip_huggingface_model_prefix(model)
        if normalized and is_known_router_model_id(normalized):
            return normalized

        fallback_model = KIMI_K26_MODEL_ID
        logger.warning(
            "Saved session model %r failed validation; using %r",
            model,
            fallback_model,
        )
        return fallback_model

    def _create_session_sync(
        self,
        *,
        session_id: str,
        user_id: str,
        hf_username: str | None,
        hf_token: str | None,
        user_plan: str | None,
        model: str | None,
        event_queue: asyncio.Queue,
        notification_destinations: list[str] | None = None,
    ) -> tuple[ToolRouter, Session]:
        """Build blocking per-session resources in a worker thread."""
        import time as _time

        t0 = _time.monotonic()
        tool_router = ToolRouter(self.config.mcpServers, hf_token=hf_token)
        # Deep-copy config so each session's model switches independently —
        # tab A picking GLM doesn't flip tab B off the default model.
        session_config = self.config.model_copy(deep=True)
        normalized_model = strip_huggingface_model_prefix(model)
        if normalized_model:
            session_config.model_name = normalized_model
        session = Session(
            event_queue=event_queue,
            config=session_config,
            tool_router=tool_router,
            hf_token=hf_token,
            user_id=user_id,
            hf_username=hf_username,
            user_plan=user_plan,
            notification_gateway=self.messaging_gateway,
            notification_destinations=notification_destinations or [],
            session_id=session_id,
            persistence_store=self._store(),
        )
        t1 = _time.monotonic()
        logger.info("Session initialized in %.2fs", t1 - t0)
        return tool_router, session

    def _serialize_messages(self, session: Session) -> list[dict[str, Any]]:
        return [msg.model_dump(mode="json") for msg in session.context_manager.items]

    def _serialize_pending_approval(self, session: Session) -> list[dict[str, Any]]:
        pending = session.pending_approval or {}
        if is_usage_threshold_pending(pending):
            return [dict(pending)]
        tool_calls = pending.get("tool_calls") or []
        serialized: list[dict[str, Any]] = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                serialized.append(tc.model_dump(mode="json"))
            elif isinstance(tc, dict):
                serialized.append(tc)
        return serialized

    @staticmethod
    def _pending_tools_for_api(session: Session) -> list[dict[str, Any]] | None:
        pending = session.pending_approval or {}
        if is_usage_threshold_pending(pending):
            return [usage_threshold_pending_to_tool(pending)]
        tool_calls = pending.get("tool_calls") or []
        if not tool_calls:
            return None
        result: list[dict[str, Any]] = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError, TypeError):
                args = {}
            result.append(
                {
                    "tool": getattr(tc.function, "name", None),
                    "tool_call_id": getattr(tc, "id", None),
                    "arguments": args,
                }
            )
        return result

    def _restore_pending_approval(
        self, session: Session, pending_approval: list[dict[str, Any]] | None
    ) -> None:
        if not pending_approval:
            session.pending_approval = None
            return
        first = pending_approval[0]
        if isinstance(first, dict) and first.get("kind") == USAGE_THRESHOLD_TOOL_NAME:
            session.pending_approval = dict(first)
            return
        from litellm import ChatCompletionMessageToolCall as ToolCall

        restored = []
        for raw in pending_approval:
            try:
                if "function" in raw:
                    restored.append(ToolCall(**raw))
                else:
                    restored.append(
                        ToolCall(
                            id=raw["tool_call_id"],
                            type="function",
                            function={
                                "name": raw["tool"],
                                "arguments": json.dumps(raw.get("arguments") or {}),
                            },
                        )
                    )
            except Exception as e:
                logger.warning("Dropping malformed pending approval: %s", e)
        session.pending_approval = {"tool_calls": restored} if restored else None

    @staticmethod
    def _pending_docs_for_api(
        pending_approval: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not pending_approval:
            return None
        first = pending_approval[0]
        if isinstance(first, dict) and first.get("kind") == USAGE_THRESHOLD_TOOL_NAME:
            return [usage_threshold_pending_to_tool(first)]
        result: list[dict[str, Any]] = []
        for raw in pending_approval:
            if "function" in raw:
                function = raw.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.append(
                    {
                        "tool": function.get("name"),
                        "tool_call_id": raw.get("id"),
                        "arguments": args,
                    }
                )
            elif {"tool", "tool_call_id"}.issubset(raw):
                result.append(
                    {
                        "tool": raw.get("tool"),
                        "tool_call_id": raw.get("tool_call_id"),
                        "arguments": raw.get("arguments") or {},
                    }
                )
        return result or None

    @staticmethod
    def _runtime_state(agent_session: AgentSession) -> str:
        if agent_session.session.pending_approval:
            return "waiting_approval"
        if agent_session.is_processing:
            return "processing"
        if not agent_session.is_active:
            return "ended"
        return "idle"

    @staticmethod
    def _auto_approval_summary(session: Session) -> dict[str, Any]:
        if hasattr(session, "auto_approval_policy_summary"):
            return session.auto_approval_policy_summary()
        cap = getattr(session, "auto_approval_cost_cap_usd", None)
        estimated = float(
            getattr(session, "auto_approval_estimated_spend_usd", 0.0) or 0.0
        )
        remaining = None if cap is None else round(max(0.0, float(cap) - estimated), 4)
        return {
            "enabled": bool(getattr(session, "auto_approval_enabled", False)),
            "cost_cap_usd": cap,
            "estimated_spend_usd": round(estimated, 4),
            "remaining_usd": remaining,
        }

    def _install_usage_threshold_checker(self, agent_session: AgentSession) -> None:
        threshold = normalize_usage_threshold(
            getattr(
                agent_session.session,
                "usage_warning_next_threshold_usd",
                agent_session.usage_warning_next_threshold_usd,
            )
        )
        agent_session.usage_warning_next_threshold_usd = threshold
        agent_session.session.usage_warning_next_threshold_usd = threshold

        async def _checker(payload: dict[str, Any]) -> bool:
            return await self._maybe_request_usage_threshold_approval(
                agent_session.session_id,
                payload,
            )

        agent_session.session.usage_threshold_checker = _checker

    @staticmethod
    def _usage_spend_from_response(response: dict[str, Any]) -> tuple[float, str]:
        def coerce_spend(value: Any) -> float | None:
            if isinstance(value, bool) or value is None:
                return None
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None

        hf_account = response.get("hf_account")
        if isinstance(hf_account, dict):
            current_session = hf_account.get("current_session")
            if isinstance(current_session, dict):
                spend = coerce_spend(current_session.get("total_usd"))
                if spend is not None:
                    return spend, "hf_billing_current_session"

        session_bucket = response.get("session")
        if isinstance(session_bucket, dict):
            spend = coerce_spend(session_bucket.get("total_usd"))
            if spend is not None:
                return spend, "app_telemetry_session"
        return 0.0, "app_telemetry_session"

    async def _current_session_usage_spend(
        self,
        agent_session: AgentSession,
        *,
        use_cache: bool = True,
    ) -> tuple[float, str]:
        now = datetime.now(UTC)
        cache = agent_session.usage_warning_spend_cache
        cache_expires_at = cache.get("expires_at")
        if (
            use_cache
            and isinstance(cache_expires_at, datetime)
            and cache_expires_at > now
        ):
            return (
                float(cache.get("spend_usd") or 0.0),
                str(cache.get("billing_source") or "app_telemetry_session"),
            )

        from usage import build_usage_response

        response = await build_usage_response(
            self,
            user_id=agent_session.user_id,
            hf_token=agent_session.hf_token,
            session_id=agent_session.session_id,
            timezone_name="UTC",
        )
        spend, billing_source = self._usage_spend_from_response(response)
        agent_session.usage_warning_spend_cache = {
            "spend_usd": spend,
            "billing_source": billing_source,
            "expires_at": now
            + timedelta(seconds=USAGE_WARNING_SPEND_CACHE_TTL_SECONDS),
        }
        return spend, billing_source

    @staticmethod
    def _runtime_session_usage_spend(agent_session: AgentSession) -> float:
        from usage import aggregate_usage_events, event_created_at

        window_start = agent_session.usage_window_started_at
        if isinstance(window_start, datetime):
            if window_start.tzinfo is None:
                window_start = window_start.replace(tzinfo=UTC)
            else:
                window_start = window_start.astimezone(UTC)
        events = []
        for raw_event in getattr(agent_session.session, "logged_events", []) or []:
            if raw_event.get("event_type") not in {"llm_call", "hf_job_complete"}:
                continue
            if isinstance(window_start, datetime):
                created_at = event_created_at(raw_event, timezone_name="UTC")
                if created_at is not None and created_at < window_start:
                    continue
            events.append(raw_event)
        bucket = aggregate_usage_events(
            events,
            session_id=agent_session.session_id,
        )
        return float(bucket.get("total_usd") or 0.0)

    async def _maybe_request_usage_threshold_approval(
        self,
        session_id: str,
        continuation_payload: dict[str, Any],
    ) -> bool:
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False

        session = agent_session.session
        if session.pending_approval:
            return False

        threshold = normalize_usage_threshold(
            getattr(
                session,
                "usage_warning_next_threshold_usd",
                agent_session.usage_warning_next_threshold_usd,
            )
        )
        force_check = bool(continuation_payload.get("force_check"))
        local_spend = self._runtime_session_usage_spend(agent_session)
        if not force_check and local_spend < threshold:
            return False

        current_spend, billing_source = await self._current_session_usage_spend(
            agent_session,
            use_cache=not force_check,
        )
        if current_spend < threshold:
            return False

        next_threshold = next_usage_warning_threshold(current_spend, threshold)
        tool_call_id = f"usage-threshold-{uuid.uuid4().hex[:10]}"
        pending: dict[str, Any] = {
            "kind": USAGE_THRESHOLD_TOOL_NAME,
            "tool_call_id": tool_call_id,
            "threshold_usd": round(threshold, 4),
            "current_spend_usd": round(current_spend, 6),
            "next_threshold_usd": next_threshold,
            "billing_source": billing_source,
            "continuation": continuation_payload.get("continuation")
            or "continue_agent",
            "history_size": int(
                continuation_payload.get("history_size")
                or len(session.context_manager.items)
            ),
        }
        final_response = continuation_payload.get("final_response")
        if isinstance(final_response, str):
            pending["final_response"] = final_response

        session.pending_approval = pending
        self._touch(agent_session)
        await session.send_event(
            Event(
                event_type="approval_required",
                data={
                    "tools": [usage_threshold_pending_to_tool(pending)],
                    "count": 1,
                    "usage_threshold": True,
                },
            )
        )
        return True

    async def _start_agent_session(
        self,
        *,
        agent_session: AgentSession,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> AgentSession:
        async with self._lock:
            existing = self.sessions.get(agent_session.session_id)
            if existing:
                return existing
            self.sessions[agent_session.session_id] = agent_session

        task = asyncio.create_task(
            self._run_session(
                agent_session.session_id,
                agent_session.submission_queue,
                event_queue,
                tool_router,
            )
        )
        agent_session.task = task
        return agent_session

    @staticmethod
    def _start_cpu_sandbox_preload(agent_session: AgentSession) -> None:
        """Kick off a best-effort cpu-basic sandbox for the session."""
        try:
            from agent.tools.sandbox_tool import start_cpu_sandbox_preload

            start_cpu_sandbox_preload(agent_session.session)
        except Exception as e:
            logger.warning(
                "Failed to start CPU sandbox preload for %s: %s",
                agent_session.session_id,
                e,
            )

    @staticmethod
    def _can_access_session(agent_session: AgentSession, user_id: str) -> bool:
        return (
            user_id == "dev"
            or agent_session.user_id == "dev"
            or agent_session.user_id == user_id
        )

    @staticmethod
    def _update_hf_identity(
        agent_session: AgentSession,
        *,
        hf_token: str | None,
        hf_username: str | None,
        user_plan: str | None = None,
    ) -> None:
        if hf_token:
            agent_session.hf_token = hf_token
            agent_session.session.hf_token = hf_token
        if hf_username:
            agent_session.hf_username = hf_username
            agent_session.session.hf_username = hf_username
        if user_plan is not None:
            agent_session.user_plan = user_plan
            agent_session.session.user_plan = user_plan

    @staticmethod
    def _has_active_sandbox_preload(agent_session: AgentSession) -> bool:
        task = getattr(agent_session.session, "sandbox_preload_task", None)
        return bool(task and not task.done())

    @staticmethod
    def _preload_failed_for_missing_hf_token(agent_session: AgentSession) -> bool:
        error = getattr(agent_session.session, "sandbox_preload_error", None)
        return isinstance(error, str) and error.startswith("No HF token available")

    def _restart_cpu_preload_if_token_recovered(
        self,
        agent_session: AgentSession,
        *,
        preload_sandbox: bool,
    ) -> None:
        if not preload_sandbox:
            return
        session = agent_session.session
        if getattr(session, "sandbox", None):
            return
        if self._has_active_sandbox_preload(agent_session):
            return
        if not (agent_session.hf_token or getattr(session, "hf_token", None)):
            return

        if not self._preload_failed_for_missing_hf_token(agent_session):
            return

        session.sandbox_preload_error = None
        session.sandbox_preload_task = None
        session.sandbox_preload_cancel_event = None
        self._start_cpu_sandbox_preload(agent_session)

    async def _clear_persisted_sandbox_metadata(self, session_id: str) -> None:
        try:
            await self._store().update_session_fields(
                session_id,
                sandbox_space_id=None,
                sandbox_hardware=None,
                sandbox_owner=None,
                sandbox_created_at=None,
                sandbox_status="destroyed",
            )
        except Exception as e:
            logger.warning("Failed to clear sandbox metadata for %s: %s", session_id, e)

    async def _cleanup_persisted_sandbox(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        hf_token: str | None,
    ) -> None:
        """Delete a sandbox recorded by a previous backend process, if any."""
        space_id = metadata.get("sandbox_space_id")
        if not isinstance(space_id, str) or not space_id:
            return
        if metadata.get("sandbox_status") == "destroyed":
            return

        tokens: list[tuple[str, str]] = []
        seen: set[str] = set()
        for label, token in (
            ("user", hf_token),
            ("admin", os.environ.get("HF_ADMIN_TOKEN")),
        ):
            if token and token not in seen:
                tokens.append((label, token))
                seen.add(token)

        if not tokens:
            logger.warning(
                "Cannot clean persisted sandbox %s for session %s: no HF token available",
                space_id,
                session_id,
            )
            return

        last_err: Exception | None = None
        for label, token in tokens:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=token)
                await asyncio.to_thread(
                    api.delete_repo,
                    repo_id=space_id,
                    repo_type="space",
                )
                logger.info(
                    "Deleted persisted sandbox %s for session %s with %s token",
                    space_id,
                    session_id,
                    label,
                )
                await self._clear_persisted_sandbox_metadata(session_id)
                return
            except Exception as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 404:
                    logger.info(
                        "Persisted sandbox %s for session %s is already gone",
                        space_id,
                        session_id,
                    )
                    await self._clear_persisted_sandbox_metadata(session_id)
                    return
                last_err = e

        logger.warning(
            "Failed to delete persisted sandbox %s for session %s: %s",
            space_id,
            session_id,
            last_err,
        )

    async def persist_session_snapshot(
        self,
        agent_session: AgentSession,
        *,
        runtime_state: str | None = None,
        status: str = "active",
        raise_on_error: bool = False,
    ) -> None:
        """Persist the current runtime context snapshot.

        Best-effort by default: a disabled store is a no-op and write failures
        are swallowed. Pass ``raise_on_error=True`` when the caller must know
        the snapshot was durably written (e.g. the reaper, which only evicts a
        session after confirming it stayed resumable) — then a disabled store
        or a write failure raises instead of silently dropping the snapshot.
        """
        store = self._store()
        if not getattr(store, "enabled", False):
            if raise_on_error:
                raise RuntimeError("persistence store is disabled")
            return
        try:
            await store.save_snapshot(
                session_id=agent_session.session_id,
                user_id=agent_session.user_id,
                model=agent_session.session.config.model_name,
                title=agent_session.title,
                messages=self._serialize_messages(agent_session.session),
                runtime_state=runtime_state or self._runtime_state(agent_session),
                status=status,
                turn_count=agent_session.session.turn_count,
                pending_approval=self._serialize_pending_approval(
                    agent_session.session
                ),
                created_at=agent_session.created_at,
                usage_window_started_at=agent_session.usage_window_started_at,
                usage_window_baseline=agent_session.usage_window_baseline,
                notification_destinations=list(
                    agent_session.session.notification_destinations
                ),
                auto_approval_enabled=bool(
                    getattr(agent_session.session, "auto_approval_enabled", False)
                ),
                auto_approval_cost_cap_usd=getattr(
                    agent_session.session, "auto_approval_cost_cap_usd", None
                ),
                auto_approval_estimated_spend_usd=float(
                    getattr(
                        agent_session.session,
                        "auto_approval_estimated_spend_usd",
                        0.0,
                    )
                    or 0.0
                ),
                usage_warning_next_threshold_usd=normalize_usage_threshold(
                    getattr(
                        agent_session.session,
                        "usage_warning_next_threshold_usd",
                        agent_session.usage_warning_next_threshold_usd,
                    )
                ),
                raise_on_error=raise_on_error,
            )
        except Exception as e:
            if raise_on_error:
                raise
            logger.warning(
                "Failed to persist snapshot for %s: %s",
                agent_session.session_id,
                e,
            )

    async def ensure_session_loaded(
        self,
        session_id: str,
        user_id: str,
        hf_token: str | None = None,
        hf_username: str | None = None,
        user_plan: str | None = None,
        preload_sandbox: bool = True,
    ) -> AgentSession | None:
        """Return a live runtime session, lazily restoring it from Mongo."""
        async with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            if self._can_access_session(existing, user_id):
                self._update_hf_identity(
                    existing,
                    hf_token=hf_token,
                    hf_username=hf_username,
                    user_plan=user_plan,
                )
                self._install_usage_threshold_checker(existing)
                self._restart_cpu_preload_if_token_recovered(
                    existing,
                    preload_sandbox=preload_sandbox,
                )
                return existing
            return None

        store = self._store()
        loaded = await store.load_session(session_id)
        if not loaded:
            return None

        async with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            if self._can_access_session(existing, user_id):
                self._update_hf_identity(
                    existing,
                    hf_token=hf_token,
                    hf_username=hf_username,
                    user_plan=user_plan,
                )
                self._install_usage_threshold_checker(existing)
                self._restart_cpu_preload_if_token_recovered(
                    existing,
                    preload_sandbox=preload_sandbox,
                )
                return existing
            return None

        meta = loaded.get("metadata") or {}
        owner = str(meta.get("user_id") or "")
        if user_id != "dev" and owner != "dev" and owner != user_id:
            return None

        await self._cleanup_persisted_sandbox(
            session_id,
            meta,
            hf_token=hf_token,
        )

        from litellm import Message

        model = self._model_from_saved_metadata(
            meta.get("model") or self.config.model_name,
        )
        event_queue: asyncio.Queue = asyncio.Queue()
        submission_queue: asyncio.Queue = asyncio.Queue()
        tool_router, session = await asyncio.to_thread(
            self._create_session_sync,
            session_id=session_id,
            user_id=owner or user_id,
            hf_username=hf_username,
            hf_token=hf_token,
            user_plan=user_plan,
            model=model,
            event_queue=event_queue,
            notification_destinations=meta.get("notification_destinations") or [],
        )

        restored_messages: list[Message] = []
        for raw in loaded.get("messages") or []:
            if not isinstance(raw, dict) or raw.get("role") == "system":
                continue
            try:
                restored_messages.append(Message.model_validate(raw))
            except Exception as e:
                logger.warning("Dropping malformed restored message: %s", e)
        if restored_messages:
            # Keep the freshly-rendered system prompt, then attach the durable
            # non-system context so tools/date/user context stay current.
            session.context_manager.items = [
                session.context_manager.items[0],
                *restored_messages,
            ]

        # If this session ever had a sandbox, its container did not survive the
        # resume (a fresh, empty one is lazily recreated). Tell the agent so it
        # recreates files/packages instead of assuming /app/train.py et al. still
        # exist. Gated on sandbox_status so pure Q&A chats get no note. Mirrors
        # the seed_from_summary note convention.
        #
        # Skip it when an approval is pending: the restored context ends with an
        # assistant tool-call message awaiting results, so injecting a user
        # message here would sit between the tool_calls and their results. On
        # approval the real results get appended after the note, leaving them
        # orphaned (the context manager stubs the "missing" result right after
        # the assistant message) — which the provider rejects. The agent still
        # learns the sandbox is empty when the approved tool runs against it.
        if meta.get("sandbox_status") and not meta.get("pending_approval"):
            session.context_manager.items.append(
                Message(
                    role="user",
                    content=(
                        "[SYSTEM: This session was resumed and its sandbox was "
                        "reset. Any files, installed packages, or running "
                        "processes from earlier are gone — recreate what you "
                        "need before using the sandbox.]"
                    ),
                )
            )

        self._restore_pending_approval(session, meta.get("pending_approval") or [])
        session.turn_count = int(meta.get("turn_count") or 0)
        session.auto_approval_enabled = bool(meta.get("auto_approval_enabled", False))
        raw_cap = meta.get("auto_approval_cost_cap_usd")
        session.auto_approval_cost_cap_usd = (
            float(raw_cap) if isinstance(raw_cap, int | float) else None
        )
        session.auto_approval_estimated_spend_usd = float(
            meta.get("auto_approval_estimated_spend_usd") or 0.0
        )
        session.usage_warning_next_threshold_usd = normalize_usage_threshold(
            meta.get("usage_warning_next_threshold_usd")
        )

        created_at = meta.get("created_at")
        if not isinstance(created_at, datetime):
            created_at = datetime.utcnow()
        usage_window_started_at = meta.get("usage_window_started_at")
        if not isinstance(usage_window_started_at, datetime):
            usage_window_started_at = created_at
        usage_window_baseline = meta.get("usage_window_baseline")
        if not isinstance(usage_window_baseline, dict):
            usage_window_baseline = None

        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            submission_queue=submission_queue,
            user_id=owner or user_id,
            hf_username=hf_username,
            hf_token=hf_token,
            user_plan=user_plan,
            created_at=created_at,
            usage_window_started_at=usage_window_started_at,
            usage_window_baseline=usage_window_baseline,
            usage_warning_next_threshold_usd=session.usage_warning_next_threshold_usd,
            is_active=True,
            is_processing=False,
            title=meta.get("title"),
        )
        self._install_usage_threshold_checker(agent_session)
        started = await self._start_agent_session(
            agent_session=agent_session,
            event_queue=event_queue,
            tool_router=tool_router,
        )
        if started is not agent_session:
            self._update_hf_identity(
                started,
                hf_token=hf_token,
                hf_username=hf_username,
                user_plan=user_plan,
            )
            return started
        if preload_sandbox:
            self._start_cpu_sandbox_preload(agent_session)
        logger.info("Restored session %s for user %s", session_id, owner or user_id)
        return agent_session

    async def create_session(
        self,
        user_id: str = "dev",
        hf_username: str | None = None,
        hf_token: str | None = None,
        user_plan: str | None = None,
        model: str | None = None,
        is_pro: bool | None = None,
    ) -> str:
        """Create a new agent session and return its ID.

        Session() and ToolRouter() constructors contain blocking I/O
        (e.g. HfApi().whoami(), litellm.get_max_tokens()) so they are
        executed in a thread pool to avoid freezing the async event loop.

        Args:
            user_id: The ID of the user who owns this session.
            hf_username: The HF username/namespace used for personal trace uploads.
            hf_token: The user's HF OAuth token, stored for tool execution.
            user_plan: The active HF account plan used for plan-aware agent CTAs.
            model: Optional model override. When set, replaces ``model_name``
                on the per-session config clone. None falls back to the
                config default.

        Raises:
            SessionCapacityError: If the server or user has reached the
                maximum number of live sessions.
        """
        # ── Capacity checks ──────────────────────────────────────────
        # Reserve a global slot under the lock so concurrent creates can't all
        # pass the check then over-admit past MAX_SESSIONS (the build + insert
        # happen later, outside the lock). active_session_count won't reflect
        # this session until _start_agent_session inserts it, so we count
        # _pending_creates alongside it to close that gap.
        async with self._lock:
            active_count = self.active_session_count
            projected = active_count + self._pending_creates
            if projected >= MAX_SESSIONS:
                raise SessionCapacityError(
                    f"Server is at capacity ({projected}/{MAX_SESSIONS} sessions). "
                    "Please try again later.",
                    error_type="global",
                )
            if user_id != "dev":
                user_count = self._count_user_sessions(user_id)
                if user_count >= MAX_SESSIONS_PER_USER:
                    raise SessionCapacityError(
                        f"You have reached the maximum of {MAX_SESSIONS_PER_USER} "
                        f"live sessions. Close an existing session, or wait "
                        f"{REAPER_IDLE_MINUTES:g} minutes after your last activity "
                        "for an idle session to be released.",
                        error_type="per_user",
                    )
            self._pending_creates += 1

        session_id = str(uuid.uuid4())

        # Create queues for this session
        submission_queue: asyncio.Queue = asyncio.Queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        reserved = True
        try:
            # Run blocking constructors in a thread to keep the event loop responsive.
            tool_router, session = await asyncio.to_thread(
                self._create_session_sync,
                session_id=session_id,
                user_id=user_id,
                hf_username=hf_username,
                hf_token=hf_token,
                user_plan=user_plan,
                model=model,
                event_queue=event_queue,
            )

            # Create wrapper
            agent_session = AgentSession(
                session_id=session_id,
                session=session,
                tool_router=tool_router,
                submission_queue=submission_queue,
                user_id=user_id,
                hf_username=hf_username,
                hf_token=hf_token,
                user_plan=user_plan,
            )
            self._install_usage_threshold_checker(agent_session)

            await self._start_agent_session(
                agent_session=agent_session,
                event_queue=event_queue,
                tool_router=tool_router,
            )
            # The session is now in self.sessions, so active_session_count
            # reflects it — release the reservation before the slower (and
            # non-capacity) persistence + preload work.
            async with self._lock:
                self._pending_creates -= 1
                reserved = False

            await self.persist_session_snapshot(agent_session, runtime_state="idle")
            self._start_cpu_sandbox_preload(agent_session)

            if is_pro is not None and user_id and user_id != "dev":
                await self._track_pro_status(agent_session, is_pro=is_pro)

            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
        finally:
            # Build/start failed before the session was inserted — always
            # release the reservation so a failed create can't permanently
            # shrink the pool.
            if reserved:
                async with self._lock:
                    self._pending_creates -= 1

    async def _track_pro_status(
        self, agent_session: AgentSession, *, is_pro: bool
    ) -> None:
        """Update Mongo per-user Pro state and emit a one-shot conversion
        event if the store reports a free→Pro transition. Best-effort: any
        Mongo failure is swallowed so we never fail session creation on
        telemetry."""
        store = self._store()
        if not getattr(store, "enabled", False):
            return
        try:
            result = await store.mark_pro_seen(agent_session.user_id, is_pro=is_pro)
        except Exception as e:
            logger.debug("mark_pro_seen failed: %s", e)
            return
        if not result or not result.get("converted"):
            return
        try:
            from agent.core import telemetry

            await telemetry.record_pro_conversion(
                agent_session.session,
                first_seen_at=result.get("first_seen_at"),
            )
        except Exception as e:
            logger.debug("record_pro_conversion failed: %s", e)

    async def seed_from_summary(self, session_id: str, messages: list[dict]) -> int:
        """Rehydrate a session from cached prior messages via summarization.

        Runs the standard summarization prompt (same one compaction uses)
        over the provided messages, then seeds the new session's context
        with that summary. Tool-call pairing concerns disappear because the
        output is plain text. Returns the number of messages summarized.
        """
        from litellm import Message

        from agent.context_manager.manager import _RESTORE_PROMPT, summarize_messages

        agent_session = self.sessions.get(session_id)
        if not agent_session:
            raise ValueError(f"Session {session_id} not found")

        # Parse into Message objects, tolerating malformed entries.
        parsed: list[Message] = []
        for raw in messages:
            if raw.get("role") == "system":
                continue  # the new session has its own system prompt
            try:
                parsed.append(Message.model_validate(raw))
            except Exception as e:
                logger.warning("Dropping malformed message during seed: %s", e)

        if not parsed:
            return 0

        session = agent_session.session
        # Pass the real tool specs so the summarizer sees what the agent
        # actually has. Without them, the summarizer can editorialize that
        # original tool calls were fabricated.
        tool_specs = None
        try:
            tool_specs = agent_session.tool_router.get_tool_specs_for_llm()
        except Exception:
            pass
        try:
            summary, _ = await summarize_messages(
                parsed,
                model_name=session.config.model_name,
                hf_token=session.hf_token,
                max_tokens=4000,
                prompt=_RESTORE_PROMPT,
                tool_specs=tool_specs,
                session=session,
                kind="restore",
            )
        except Exception as e:
            logger.error("Summary call failed during seed: %s", e)
            raise

        seed = Message(
            role="user",
            content=(
                "[SYSTEM: Your prior memory of this conversation — written "
                "in your own voice right before restart. Continue from here.]\n\n"
                + (summary or "(no summary returned)")
            ),
        )
        session.context_manager.items.append(seed)
        self._touch(agent_session)
        await self.persist_session_snapshot(agent_session, runtime_state="idle")
        return len(parsed)

    @staticmethod
    async def _cleanup_sandbox(session: Session) -> None:
        """Delete the sandbox Space if one was created for this session.

        Retries on transient failures (HF API 5xx, rate-limit, network blips)
        with exponential backoff. A single missed delete = a permanently
        orphaned Space, so the cost of an extra retry beats the alternative.
        """
        from agent.tools.sandbox_tool import teardown_session_sandbox

        await teardown_session_sandbox(session)

    async def _cleanup_all_sandboxes_on_close(self) -> None:
        """Best-effort sandbox cleanup for graceful backend shutdown."""
        async with self._lock:
            agent_sessions = list(self.sessions.values())
        if not agent_sessions:
            return

        semaphore = asyncio.Semaphore(SANDBOX_SHUTDOWN_CLEANUP_CONCURRENCY)

        async def _cleanup_one(agent_session: AgentSession) -> None:
            async with semaphore:
                try:
                    await self._cleanup_sandbox(agent_session.session)
                except Exception as e:
                    logger.warning(
                        "Shutdown sandbox cleanup failed for %s: %s",
                        agent_session.session_id,
                        e,
                    )

        tasks = [
            asyncio.create_task(_cleanup_one(agent_session))
            for agent_session in agent_sessions
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=SANDBOX_SHUTDOWN_CLEANUP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out after %.0fs cleaning up sandboxes on shutdown; "
                "orphan sweeper will handle any stragglers",
                SANDBOX_SHUTDOWN_CLEANUP_TIMEOUT_S,
            )

    async def _reaper_loop(self) -> None:
        """Periodically release resources held by idle sessions.

        Modeled on EventBroadcaster.run: a long-lived task started in start()
        and cancelled in close(). Per-sweep exceptions are swallowed so one bad
        sweep never kills the loop.
        """
        while True:
            try:
                await asyncio.sleep(REAPER_INTERVAL_S)
                await self._reap_idle_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Idle-session reaper sweep failed: %s", e)

    async def _reap_idle_sessions(self) -> None:
        """Select idle candidates under the lock, then tear each down.

        Candidates are non-dev sessions that are live, not processing, not
        awaiting tool approval (those are "approve later", not idle — reaping
        would destroy the sandbox the approved tool needs), and untouched for
        the idle window. We only snapshot IDs under the lock; the actual
        teardown in _reap_one re-acquires it, because tearing a session down
        while holding the lock would deadlock (the lock is non-reentrant).
        """
        # Reaping is only safe when sessions stay resumable from Mongo. With no
        # store, eviction would destroy non-dev chats outright, so don't reap.
        if not getattr(self._store(), "enabled", False):
            return

        cutoff = datetime.utcnow() - REAPER_IDLE
        async with self._lock:
            candidates = [
                agent_session.session_id
                for agent_session in self.sessions.values()
                if agent_session.is_active
                and not agent_session.is_processing
                and not agent_session.is_reaping
                and agent_session.user_id != "dev"
                and not agent_session.session.pending_approval
                and agent_session.last_active_at <= cutoff
            ]
        if not candidates:
            return

        reaped = 0
        for session_id in candidates:
            try:
                if await self._reap_one(session_id, cutoff):
                    reaped += 1
            except Exception as e:
                logger.warning("Failed to reap idle session %s: %s", session_id, e)
        if reaped:
            logger.info("Reaped %d idle session(s)", reaped)

    async def _reap_one(self, session_id: str, cutoff: datetime) -> bool:
        """Tear down one idle session, leaving it resumable from Mongo.

        Re-checks every idle condition under the lock (a user may have become
        active in the gap since selection), marks the session reaping, persists
        a resumable snapshot outside the lock, then does one final locked
        re-check before eviction. The runtime task is cancelled *outside* the
        lock: its own ``finally`` frees the sandbox, and its identity-gated
        persist no-ops because the session is already popped — so it can't
        overwrite our resumable snapshot with ``"ended"`` and there's no
        deadlock. Returns True if the session was reaped.
        """
        async with self._lock:
            agent_session = self.sessions.get(session_id)
            if (
                agent_session is None
                or not agent_session.is_active
                or agent_session.is_processing
                or agent_session.is_reaping
                or agent_session.session.pending_approval
                or agent_session.last_active_at > cutoff
                or not agent_session.submission_queue.empty()
            ):
                return False
            agent_session.is_reaping = True

        # Persist a resumable snapshot *before* eviction so a concurrent reopen
        # reloads clean state. status="active" (never "ended") keeps it a normal
        # chat in the sidebar. Do this outside the manager lock: Mongo writes can
        # take network round trips, and is_reaping=True is enough to block submit
        # from enqueueing while the snapshot is in flight.
        try:
            await self.persist_session_snapshot(
                agent_session,
                runtime_state="idle",
                status="active",
                raise_on_error=True,
            )
        except Exception as e:
            async with self._lock:
                if self.sessions.get(session_id) is agent_session:
                    agent_session.is_reaping = False
            logger.warning(
                "Skipping reap of %s: could not persist resumable snapshot: %s",
                session_id,
                e,
            )
            return False

        async with self._lock:
            current = self.sessions.get(session_id)
            if current is not agent_session:
                return False
            if (
                not agent_session.is_active
                or agent_session.is_processing
                or agent_session.session.pending_approval
                or agent_session.last_active_at > cutoff
                or not agent_session.submission_queue.empty()
            ):
                agent_session.is_reaping = False
                return False
            self.sessions.pop(session_id, None)
            task = agent_session.task
            session = agent_session.session

        if task is not None and not task.done():
            task.cancel()
            # Use asyncio.wait, not wait_for: wait_for re-raises the cancelled
            # task's CancelledError, which we'd have to swallow — and that same
            # bare except would also eat an *outer* cancel aimed at the reaper
            # itself (close() cancelling _reaper_task), hanging shutdown.
            # asyncio.wait returns the cancelled task in `done` and lets an
            # outer cancel propagate cleanly.
            done, _pending = await asyncio.wait({task}, timeout=REAP_TEARDOWN_TIMEOUT_S)
            if not done:
                logger.warning(
                    "Reaper teardown timed out after %.0fs for %s; orphan "
                    "sweeper will handle any sandbox straggler",
                    REAP_TEARDOWN_TIMEOUT_S,
                    session_id,
                )
            elif not task.cancelled():
                # Surface (and retrieve, to avoid "exception never retrieved")
                # any non-cancellation teardown error.
                exc = task.exception()
                if exc is not None:
                    logger.warning("Reaper teardown error for %s: %s", session_id, exc)
        else:
            # No live task to run the cleanup finally — free the sandbox here so
            # a reaped session never leaves an orphaned Space behind.
            await self._cleanup_sandbox(session)
        return True

    async def _run_session(
        self,
        session_id: str,
        submission_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> None:
        """Run the agent loop for a session and broadcast events via EventBroadcaster."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            logger.error(f"Session {session_id} not found")
            return

        session = agent_session.session

        # Start event broadcaster task
        broadcaster = EventBroadcaster(event_queue)
        agent_session.broadcaster = broadcaster
        broadcast_task = asyncio.create_task(broadcaster.run())

        try:
            async with tool_router:
                # Send ready event
                await session.send_event(
                    Event(event_type="ready", data={"message": "Agent initialized"})
                )

                while session.is_running:
                    try:
                        # Wait for submission with timeout to allow checking is_running
                        submission = await asyncio.wait_for(
                            submission_queue.get(), timeout=1.0
                        )
                        agent_session.is_processing = True
                        self._touch(agent_session)
                        try:
                            should_continue = await process_submission(
                                session, submission
                            )
                        finally:
                            agent_session.is_processing = False
                            # Stamp on turn finish too: a turn that ran longer
                            # than the idle window would otherwise be reaped the
                            # instant it completes.
                            self._touch(agent_session)
                            await self.persist_session_snapshot(agent_session)
                        if not should_continue:
                            break
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        logger.info(f"Session {session_id} cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error in session {session_id}: {e}")
                        await session.send_event(
                            Event(event_type="error", data={"error": str(e)})
                        )

        finally:
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

            await self._cleanup_sandbox(session)

            # Final-flush: always save on session death so we capture ended
            # sessions even if the client disconnects without /shutdown.
            # Idempotent via session_id key; detached subprocess.
            if session.config.save_sessions:
                try:
                    session.save_and_upload_detached(
                        session.config.session_dataset_repo
                    )
                except Exception as e:
                    logger.warning(f"Final-flush failed for {session_id}: {e}")

            async with self._lock:
                if self.sessions.get(session_id) is agent_session:
                    agent_session.is_active = False
                    await self.persist_session_snapshot(
                        agent_session,
                        runtime_state="ended",
                        status="ended",
                    )

            logger.info(f"Session {session_id} ended")

    async def submit(self, session_id: str, operation: Operation) -> bool:
        """Submit an operation to a session.

        Enqueues under the lock and rejects sessions being reaped, so submit
        and reap can't interleave: either the message is enqueued before the
        reaper's empty() re-check (which then aborts the reap), or the session
        is already popped (we return False and the caller reloads a fresh
        runtime from Mongo). The queue is unbounded, so put_nowait never blocks.
        """
        submission = Submission(id=f"sub_{uuid.uuid4().hex[:8]}", operation=operation)
        async with self._lock:
            agent_session = self.sessions.get(session_id)
            if (
                not agent_session
                or not agent_session.is_active
                or agent_session.is_reaping
            ):
                logger.warning(f"Session {session_id} not found or inactive")
                return False
            agent_session.submission_queue.put_nowait(submission)
            self._touch(agent_session)
        return True

    async def submit_user_input(self, session_id: str, text: str) -> bool:
        """Submit user input to a session."""
        operation = Operation(op_type=OpType.USER_INPUT, data={"text": text})
        return await self.submit(session_id, operation)

    async def submit_approval(
        self, session_id: str, approvals: list[dict[str, Any]]
    ) -> bool:
        """Submit tool approvals to a session."""
        operation = Operation(
            op_type=OpType.EXEC_APPROVAL, data={"approvals": approvals}
        )
        return await self.submit(session_id, operation)

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt a session by signalling cancellation directly (bypasses queue)."""
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        agent_session.session.cancel()
        return True

    async def undo(self, session_id: str) -> bool:
        """Undo last turn in a session."""
        operation = Operation(op_type=OpType.UNDO)
        return await self.submit(session_id, operation)

    async def truncate(self, session_id: str, user_message_index: int) -> bool:
        """Truncate conversation to before a specific user message (direct, no queue)."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        success = agent_session.session.context_manager.truncate_to_user_message(
            user_message_index
        )
        if success:
            self._touch(agent_session)
            await self.persist_session_snapshot(agent_session, runtime_state="idle")
        return success

    async def compact(self, session_id: str) -> bool:
        """Compact context in a session."""
        operation = Operation(op_type=OpType.COMPACT)
        return await self.submit(session_id, operation)

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a specific session."""
        operation = Operation(op_type=OpType.SHUTDOWN)
        success = await self.submit(session_id, operation)

        if success:
            async with self._lock:
                agent_session = self.sessions.get(session_id)
                if agent_session and agent_session.task:
                    # Wait for task to complete
                    try:
                        await asyncio.wait_for(agent_session.task, timeout=5.0)
                    except asyncio.TimeoutError:
                        agent_session.task.cancel()

        return success

    async def delete_session(self, session_id: str) -> bool:
        """Soft-delete a session and stop its runtime resources."""
        async with self._lock:
            agent_session = self.sessions.pop(session_id, None)

        if not agent_session:
            await self._store().soft_delete_session(session_id)
            return True

        await self._store().soft_delete_session(session_id)

        # Clean up sandbox Space before cancelling the task
        await self._cleanup_sandbox(agent_session.session)

        # Cancel the task if running
        if agent_session.task and not agent_session.task.done():
            agent_session.task.cancel()
            try:
                await agent_session.task
            except asyncio.CancelledError:
                pass

        return True

    async def teardown_sandbox(self, session_id: str) -> bool:
        """Delete only this session's sandbox runtime, preserving chat state."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)

        if not agent_session or not agent_session.is_active:
            return False

        await self._cleanup_sandbox(agent_session.session)
        await self.persist_session_snapshot(agent_session, runtime_state="idle")
        return True

    async def update_session_title(self, session_id: str, title: str | None) -> None:
        """Persist a user-visible title for sidebar rehydration."""
        agent_session = self.sessions.get(session_id)
        if agent_session:
            agent_session.title = title
        await self._store().update_session_fields(session_id, title=title)

    async def update_session_model(self, session_id: str, model_id: str) -> bool:
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        agent_session.session.update_model(model_id)
        self._touch(agent_session)
        await self.persist_session_snapshot(agent_session, runtime_state="idle")
        return True

    async def update_session_auto_approval(
        self,
        session_id: str,
        *,
        enabled: bool,
        cost_cap_usd: float | None,
        cap_provided: bool = False,
    ) -> dict[str, Any]:
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            raise ValueError("Session not found or inactive")

        session = agent_session.session
        if enabled:
            if not cap_provided and cost_cap_usd is None:
                cost_cap_usd = getattr(session, "auto_approval_cost_cap_usd", None)
                if cost_cap_usd is None:
                    cost_cap_usd = DEFAULT_YOLO_COST_CAP_USD
            elif cost_cap_usd is None:
                cost_cap_usd = DEFAULT_YOLO_COST_CAP_USD
        else:
            if not cap_provided:
                cost_cap_usd = getattr(session, "auto_approval_cost_cap_usd", None)

        if hasattr(session, "set_auto_approval_policy"):
            session.set_auto_approval_policy(
                enabled=enabled,
                cost_cap_usd=cost_cap_usd,
            )
        else:
            session.auto_approval_enabled = bool(enabled)
            session.auto_approval_cost_cap_usd = cost_cap_usd
        self._touch(agent_session)
        await self.persist_session_snapshot(agent_session)
        return self._auto_approval_summary(session)

    def get_session_owner(self, session_id: str) -> str | None:
        """Get the user_id that owns a session, or None if session doesn't exist."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None
        return agent_session.user_id

    def verify_session_access(self, session_id: str, user_id: str) -> bool:
        """Check if a user has access to a session.

        Returns True if:
        - The session exists AND the user owns it
        - The user_id is "dev" (dev mode bypass)
        """
        owner = self.get_session_owner(session_id)
        if owner is None:
            return False
        if user_id == "dev" or owner == "dev":
            return True
        return owner == user_id

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a session."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None

        pending_approval = self._pending_tools_for_api(agent_session.session)

        return {
            "session_id": session_id,
            "created_at": agent_session.created_at.isoformat(),
            "usage_window_started_at": (
                agent_session.usage_window_started_at or agent_session.created_at
            ).isoformat(),
            "is_active": agent_session.is_active,
            "is_processing": agent_session.is_processing,
            "message_count": len(agent_session.session.context_manager.items),
            "user_id": agent_session.user_id,
            "pending_approval": pending_approval,
            "model": agent_session.session.config.model_name,
            "title": agent_session.title,
            "notification_destinations": list(
                agent_session.session.notification_destinations
            ),
            "auto_approval": self._auto_approval_summary(agent_session.session),
        }

    async def reset_session_usage_window(
        self,
        session_id: str,
        *,
        started_at: datetime | None = None,
        baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Reset the account-billing window used for the visible usage meter."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None

        window_start = started_at or datetime.utcnow()
        agent_session.usage_window_started_at = window_start
        agent_session.usage_window_baseline = baseline
        self._touch(agent_session)

        store = self._store()
        if getattr(store, "enabled", False):
            await store.update_session_fields(
                session_id,
                usage_window_started_at=window_start,
                usage_window_baseline=baseline,
                last_active_at=agent_session.last_active_at,
            )
        return self.get_session_info(session_id)

    def set_notification_destinations(
        self, session_id: str, destinations: list[str]
    ) -> list[str]:
        """Replace the session's opted-in auto-notification destinations."""
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            raise ValueError("Session not found or inactive")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw_name in destinations:
            name = raw_name.strip()
            if not name:
                raise ValueError("Destination names must not be empty")
            destination = self.config.messaging.get_destination(name)
            if destination is None:
                raise ValueError(f"Unknown destination '{name}'")
            if not destination.allow_auto_events:
                raise ValueError(f"Destination '{name}' is not enabled for auto events")
            if name not in seen:
                normalized.append(name)
                seen.add(name)

        agent_session.session.set_notification_destinations(normalized)
        self._touch(agent_session)
        return normalized

    async def list_sessions(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user.

        Args:
            user_id: If provided, only return sessions owned by this user.
                     If "dev", return all sessions (dev mode).
        """
        results: list[dict[str, Any]] = []
        store = self._store()
        if getattr(store, "enabled", False):
            for row in await store.list_sessions(user_id or "dev"):
                sid = row.get("session_id") or row.get("_id")
                if not sid:
                    continue
                runtime_info = self.get_session_info(str(sid))
                if runtime_info:
                    results.append(runtime_info)
                    continue
                created_at = row.get("created_at")
                if isinstance(created_at, datetime):
                    created_at_str = created_at.isoformat()
                else:
                    created_at_str = str(created_at or datetime.utcnow().isoformat())
                usage_window_started_at = (
                    row.get("usage_window_started_at") or created_at
                )
                if isinstance(usage_window_started_at, datetime):
                    usage_window_started_at_str = usage_window_started_at.isoformat()
                else:
                    usage_window_started_at_str = str(
                        usage_window_started_at or created_at_str
                    )
                pending = self._pending_docs_for_api(row.get("pending_approval") or [])
                results.append(
                    {
                        "session_id": str(sid),
                        "created_at": created_at_str,
                        "usage_window_started_at": usage_window_started_at_str,
                        "is_active": row.get("status") != "ended",
                        "is_processing": row.get("runtime_state") == "processing",
                        "message_count": int(row.get("message_count") or 0),
                        "user_id": row.get("user_id") or "dev",
                        "pending_approval": pending or None,
                        "model": row.get("model"),
                        "title": row.get("title"),
                        "notification_destinations": row.get(
                            "notification_destinations"
                        )
                        or [],
                        "auto_approval": {
                            "enabled": bool(row.get("auto_approval_enabled", False)),
                            "cost_cap_usd": row.get("auto_approval_cost_cap_usd"),
                            "estimated_spend_usd": float(
                                row.get("auto_approval_estimated_spend_usd") or 0.0
                            ),
                            "remaining_usd": (
                                None
                                if row.get("auto_approval_cost_cap_usd") is None
                                else round(
                                    max(
                                        0.0,
                                        float(
                                            row.get("auto_approval_cost_cap_usd") or 0.0
                                        )
                                        - float(
                                            row.get("auto_approval_estimated_spend_usd")
                                            or 0.0
                                        ),
                                    ),
                                    4,
                                )
                            ),
                        },
                    }
                )
            return results

        for sid in self.sessions:
            info = self.get_session_info(sid)
            if not info:
                continue
            if user_id and user_id != "dev" and info.get("user_id") != user_id:
                continue
            results.append(info)
        return results

    @property
    def active_session_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self.sessions.values() if s.is_active)


# Global session manager instance
session_manager = SessionManager()
