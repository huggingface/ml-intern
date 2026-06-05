"""Usage aggregation for app-attributed ML Intern spend."""

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

USAGE_EVENT_TYPES = ("llm_call", "hf_job_complete")

HF_BILLING_URL = "https://huggingface.co/settings/billing"
HF_INFERENCE_PROVIDERS_USAGE_URL = "https://huggingface.co/settings/billing/usage"
HF_INFERENCE_PROVIDERS_PRICING_URL = (
    "https://huggingface.co/docs/inference-providers/en/pricing"
)
HF_JOBS_PRICING_URL = "https://huggingface.co/docs/hub/jobs-pricing"


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return _utc(dt).isoformat().replace("+00:00", "Z")


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _utc(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def event_created_at(event: dict[str, Any]) -> datetime | None:
    return _parse_timestamp(event.get("created_at") or event.get("timestamp"))


def resolve_usage_windows(
    timezone_name: str | None,
    *,
    now: datetime | None = None,
) -> dict[str, datetime | str]:
    """Return UTC day/month windows for a browser timezone."""
    try:
        tz = ZoneInfo(timezone_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")

    now_utc = _utc(now or datetime.now(UTC))
    local_now = now_utc.astimezone(tz)
    today_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_local = today_local.replace(day=1)
    return {
        "timezone": tz.key,
        "now_utc": now_utc,
        "today_start_utc": today_local.astimezone(UTC),
        "month_start_utc": month_local.astimezone(UTC),
    }


def _empty_bucket(
    *,
    session_id: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "window_start": _iso(window_start),
        "window_end": _iso(window_end),
        "timezone": timezone,
        "total_usd": 0.0,
        "inference_usd": 0.0,
        "hf_jobs_estimated_usd": 0.0,
        "llm_calls": 0,
        "hf_jobs_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_tokens": 0,
        "hf_jobs_billable_seconds_estimate": 0,
    }


def aggregate_usage_events(
    events: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    bucket = _empty_bucket(
        session_id=session_id,
        window_start=window_start,
        window_end=window_end,
        timezone=timezone,
    )
    for event in events:
        event_type = event.get("event_type")
        data = event.get("data") or {}
        if event_type == "llm_call":
            bucket["llm_calls"] += 1
            bucket["inference_usd"] += _coerce_float(data.get("cost_usd"))
            prompt_tokens = _coerce_int(data.get("prompt_tokens"))
            completion_tokens = _coerce_int(data.get("completion_tokens"))
            cache_read_tokens = _coerce_int(data.get("cache_read_tokens"))
            cache_creation_tokens = _coerce_int(data.get("cache_creation_tokens"))
            total_tokens = _coerce_int(data.get("total_tokens")) or (
                prompt_tokens
                + completion_tokens
                + cache_read_tokens
                + cache_creation_tokens
            )
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["cache_read_tokens"] += cache_read_tokens
            bucket["cache_creation_tokens"] += cache_creation_tokens
            bucket["total_tokens"] += total_tokens
        elif event_type == "hf_job_complete":
            bucket["hf_jobs_count"] += 1
            bucket["hf_jobs_estimated_usd"] += _coerce_float(
                data.get("estimated_cost_usd")
            )
            bucket["hf_jobs_billable_seconds_estimate"] += _coerce_int(
                data.get("billable_seconds_estimate") or data.get("wall_time_s")
            )

    bucket["inference_usd"] = round(bucket["inference_usd"], 6)
    bucket["hf_jobs_estimated_usd"] = round(bucket["hf_jobs_estimated_usd"], 6)
    bucket["total_usd"] = round(
        bucket["inference_usd"] + bucket["hf_jobs_estimated_usd"],
        6,
    )
    return bucket


def _event_in_window(
    event: dict[str, Any],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> bool:
    if start is None and end is None:
        return True
    created_at = event_created_at(event)
    if created_at is None:
        return False
    if start is not None and created_at < _utc(start):
        return False
    if end is not None and created_at >= _utc(end):
        return False
    return True


def _events_from_runtime_session(agent_session: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in getattr(agent_session.session, "logged_events", []) or []:
        if raw.get("event_type") not in USAGE_EVENT_TYPES:
            continue
        events.append(
            {
                "session_id": agent_session.session_id,
                "event_type": raw.get("event_type"),
                "data": raw.get("data") or {},
                "timestamp": raw.get("timestamp"),
            }
        )
    return events


def _runtime_sessions_for_user(manager: Any, user_id: str) -> list[Any]:
    sessions = list(getattr(manager, "sessions", {}).values())
    if user_id == "dev":
        return sessions
    return [session for session in sessions if session.user_id == user_id]


async def _load_usage_events(
    manager: Any,
    *,
    user_id: str,
    session_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict[str, Any]]:
    store = manager._store()
    if getattr(store, "enabled", False):
        return await store.load_usage_events(
            user_id,
            session_id=session_id,
            start=start,
            end=end,
        )

    events: list[dict[str, Any]] = []
    for agent_session in _runtime_sessions_for_user(manager, user_id):
        if session_id is not None and agent_session.session_id != session_id:
            continue
        for event in _events_from_runtime_session(agent_session):
            if _event_in_window(event, start=start, end=end):
                events.append(event)
    return events


async def build_usage_response(
    manager: Any,
    *,
    user_id: str,
    session_id: str | None = None,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    windows = resolve_usage_windows(timezone_name, now=now)
    timezone = str(windows["timezone"])
    now_utc = windows["now_utc"]
    today_start = windows["today_start_utc"]
    month_start = windows["month_start_utc"]

    session_events: list[dict[str, Any]] = []
    if session_id:
        session_events = await _load_usage_events(
            manager,
            user_id=user_id,
            session_id=session_id,
        )

    today_events = await _load_usage_events(
        manager,
        user_id=user_id,
        start=today_start,
        end=now_utc,
    )
    month_events = await _load_usage_events(
        manager,
        user_id=user_id,
        start=month_start,
        end=now_utc,
    )

    return {
        "source": "app_telemetry",
        "currency": "USD",
        "generated_at": _iso(now_utc),
        "timezone": timezone,
        "session": (
            aggregate_usage_events(session_events, session_id=session_id)
            if session_id
            else None
        ),
        "today": aggregate_usage_events(
            today_events,
            window_start=today_start,
            window_end=now_utc,
            timezone=timezone,
        ),
        "month": aggregate_usage_events(
            month_events,
            window_start=month_start,
            window_end=now_utc,
            timezone=timezone,
        ),
        "links": {
            "hf_billing": HF_BILLING_URL,
            "inference_providers_usage": HF_INFERENCE_PROVIDERS_USAGE_URL,
            "inference_providers_pricing": HF_INFERENCE_PROVIDERS_PRICING_URL,
            "jobs_pricing": HF_JOBS_PRICING_URL,
        },
    }
