"""Usage aggregation for app-attributed ML Intern spend."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from agent.core.cost_estimation import SPACE_PRICE_USD_PER_HOUR

USAGE_EVENT_TYPES = (
    "llm_call",
    "hf_job_complete",
    "sandbox_create",
    "sandbox_destroy",
)

logger = logging.getLogger(__name__)

HF_BILLING_USAGE_V2_URL = "https://huggingface.co/api/settings/billing/usage-v2"
HF_BILLING_URL = "https://huggingface.co/settings/billing"
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


def _nano_usd_to_usd(value: Any) -> float:
    return _coerce_float(value) / 1_000_000_000


def _micro_usd_to_usd(value: Any) -> float:
    return _coerce_float(value) / 1_000_000


def _coerce_timezone(timezone_name: str | None) -> ZoneInfo | None:
    if not timezone_name:
        return None
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _normalize_event_timestamp(
    dt: datetime,
    *,
    timezone_name: str | None = None,
) -> datetime:
    if dt.tzinfo is not None:
        return _utc(dt)
    timezone = _coerce_timezone(timezone_name)
    if timezone is not None:
        return dt.replace(tzinfo=timezone).astimezone(UTC)
    return dt.astimezone(UTC)


def _parse_timestamp(
    value: Any, *, timezone_name: str | None = None
) -> datetime | None:
    if isinstance(value, datetime):
        return _normalize_event_timestamp(value, timezone_name=timezone_name)
    if not isinstance(value, str) or not value:
        return None
    try:
        return _normalize_event_timestamp(
            datetime.fromisoformat(value.replace("Z", "+00:00")),
            timezone_name=timezone_name,
        )
    except ValueError:
        return None


def event_created_at(
    event: dict[str, Any],
    *,
    timezone_name: str | None = None,
) -> datetime | None:
    return _parse_timestamp(
        event.get("created_at") or event.get("timestamp"),
        timezone_name=timezone_name,
    )


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
        "sandbox_estimated_usd": 0.0,
        "llm_calls": 0,
        "hf_jobs_count": 0,
        "sandbox_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_tokens": 0,
        "hf_jobs_billable_seconds_estimate": 0,
        "sandbox_billable_seconds_estimate": 0,
    }


def _empty_hf_account_bucket(
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    return {
        "window_start": _iso(window_start),
        "window_end": _iso(window_end),
        "timezone": timezone,
        "total_usd": 0.0,
        "inference_providers_usd": 0.0,
        "hf_jobs_usd": 0.0,
        "inference_provider_requests": 0,
        "hf_jobs_minutes": 0.0,
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
        elif event_type == "sandbox_destroy":
            # Sandbox costs are paired and added after the main pass so the
            # create event can provide hardware pricing metadata.
            continue

    _aggregate_sandbox_usage(
        events,
        bucket,
        window_start=window_start,
        window_end=window_end,
        timezone=timezone,
    )

    bucket["inference_usd"] = round(bucket["inference_usd"], 6)
    bucket["hf_jobs_estimated_usd"] = round(bucket["hf_jobs_estimated_usd"], 6)
    bucket["sandbox_estimated_usd"] = round(bucket["sandbox_estimated_usd"], 6)
    bucket["total_usd"] = round(
        (
            bucket["inference_usd"]
            + bucket["hf_jobs_estimated_usd"]
            + bucket["sandbox_estimated_usd"]
        ),
        6,
    )
    return bucket


def _event_sort_key(
    indexed_event: tuple[int, dict[str, Any]],
    *,
    timezone_name: str | None,
) -> tuple[bool, datetime, int]:
    index, event = indexed_event
    created_at = event_created_at(event, timezone_name=timezone_name)
    return (
        created_at is None,
        created_at or datetime.min.replace(tzinfo=UTC),
        index,
    )


def _sandbox_id(event: dict[str, Any]) -> str | None:
    data = event.get("data") or {}
    sandbox_id = data.get("sandbox_id")
    return sandbox_id if isinstance(sandbox_id, str) and sandbox_id else None


def _sandbox_duration_seconds(
    create_event: dict[str, Any],
    destroy_event: dict[str, Any],
    *,
    window_start: datetime | None,
    window_end: datetime | None,
    timezone_name: str | None,
) -> int:
    create_data = create_event.get("data") or {}
    destroy_data = destroy_event.get("data") or {}
    lifetime_s = _coerce_int(destroy_data.get("lifetime_s"))
    create_at = event_created_at(create_event, timezone_name=timezone_name)
    destroy_at = event_created_at(destroy_event, timezone_name=timezone_name)

    if lifetime_s > 0:
        if destroy_at is None or (window_start is None and window_end is None):
            return lifetime_s
        interval_start = destroy_at - timedelta(seconds=lifetime_s)
        interval_end = destroy_at
    else:
        if create_at is None or destroy_at is None:
            return 0
        create_latency_s = max(0, _coerce_int(create_data.get("create_latency_s")))
        interval_start = create_at - timedelta(seconds=create_latency_s)
        interval_end = destroy_at

    if interval_end <= interval_start:
        return 0

    clamp_start = _utc(window_start) if window_start is not None else interval_start
    clamp_end = _utc(window_end) if window_end is not None else interval_end
    clamped_start = max(interval_start, clamp_start)
    clamped_end = min(interval_end, clamp_end)
    if clamped_end <= clamped_start:
        return 0
    return int((clamped_end - clamped_start).total_seconds())


def _aggregate_sandbox_usage(
    events: list[dict[str, Any]],
    bucket: dict[str, Any],
    *,
    window_start: datetime | None,
    window_end: datetime | None,
    timezone: str | None,
) -> None:
    lifecycle_events = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event_type") in {"sandbox_create", "sandbox_destroy"}
    ]
    ordered_events = [
        event
        for _, event in sorted(
            lifecycle_events,
            key=lambda item: _event_sort_key(item, timezone_name=timezone),
        )
    ]
    active_creates: dict[str, dict[str, Any]] = {}

    for event in ordered_events:
        event_type = event.get("event_type")
        sandbox_id = _sandbox_id(event)
        if sandbox_id is None:
            continue

        if event_type == "sandbox_create":
            active_creates[sandbox_id] = event
            continue

        if event_type != "sandbox_destroy":
            continue

        create_event = active_creates.pop(sandbox_id, None)
        if create_event is None:
            continue

        create_data = create_event.get("data") or {}
        hardware = str(create_data.get("hardware") or "cpu-basic")
        price_usd_per_hour = SPACE_PRICE_USD_PER_HOUR.get(hardware, 0.0)
        seconds = _sandbox_duration_seconds(
            create_event,
            event,
            window_start=window_start,
            window_end=window_end,
            timezone_name=timezone,
        )

        bucket["sandbox_count"] += 1
        if price_usd_per_hour > 0:
            bucket["sandbox_billable_seconds_estimate"] += seconds
        bucket["sandbox_estimated_usd"] += price_usd_per_hour * (seconds / 3600)


def _account_bucket_from_billing_usage(
    payload: dict[str, Any] | None,
    *,
    window_start: datetime,
    window_end: datetime,
    timezone: str,
) -> dict[str, Any]:
    bucket = _empty_hf_account_bucket(
        window_start=window_start,
        window_end=window_end,
        timezone=timezone,
    )
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        return bucket

    inference = usage.get("inferenceProviders")
    if not isinstance(inference, dict):
        inference = {}
    jobs = usage.get("jobs")
    if not isinstance(jobs, dict):
        jobs = {}

    bucket["inference_providers_usd"] = round(
        _nano_usd_to_usd(inference.get("usedNanoUsd")),
        6,
    )
    bucket["hf_jobs_usd"] = round(_micro_usd_to_usd(jobs.get("usedMicroUsd")), 6)
    bucket["inference_provider_requests"] = _coerce_int(inference.get("numRequests"))
    bucket["hf_jobs_minutes"] = round(_coerce_float(jobs.get("totalMinutes")), 3)
    bucket["total_usd"] = round(
        bucket["inference_providers_usd"] + bucket["hf_jobs_usd"],
        6,
    )
    return bucket


def _baseline_from_account_bucket(
    bucket: dict[str, Any],
    *,
    captured_at: datetime,
    month_start: datetime,
) -> dict[str, Any]:
    return {
        "captured_at": _utc(captured_at),
        "month_start": _utc(month_start),
        "total_usd": bucket.get("total_usd", 0.0),
        "inference_providers_usd": bucket.get("inference_providers_usd", 0.0),
        "hf_jobs_usd": bucket.get("hf_jobs_usd", 0.0),
        "inference_provider_requests": bucket.get("inference_provider_requests", 0),
        "hf_jobs_minutes": bucket.get("hf_jobs_minutes", 0.0),
    }


def _baseline_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _utc(value)
    parsed = _parse_timestamp(value)
    return _utc(parsed) if parsed is not None else None


def _baseline_delta_bucket(
    bucket: dict[str, Any],
    baseline: dict[str, Any],
    *,
    window_start: datetime,
    window_end: datetime,
    timezone: str,
) -> dict[str, Any]:
    delta = _empty_hf_account_bucket(
        window_start=window_start,
        window_end=window_end,
        timezone=timezone,
    )
    for key in ("inference_providers_usd", "hf_jobs_usd", "hf_jobs_minutes"):
        delta[key] = round(
            max(0.0, _coerce_float(bucket.get(key)) - _coerce_float(baseline.get(key))),
            6 if key != "hf_jobs_minutes" else 3,
        )
    delta["inference_provider_requests"] = max(
        0,
        _coerce_int(bucket.get("inference_provider_requests"))
        - _coerce_int(baseline.get("inference_provider_requests")),
    )
    delta["total_usd"] = round(
        delta["inference_providers_usd"] + delta["hf_jobs_usd"],
        6,
    )
    return delta


async def capture_hf_account_usage_baseline(
    hf_token: str | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if not hf_token:
        return None
    windows = resolve_usage_windows("UTC", now=now)
    now_utc = windows["now_utc"]
    month_start = windows["month_start_utc"]
    payload = await _fetch_hf_billing_usage_v2(
        hf_token,
        start=month_start,
        end=now_utc,
    )
    if payload is None:
        return None
    bucket = _account_bucket_from_billing_usage(
        payload,
        window_start=month_start,
        window_end=now_utc,
        timezone="UTC",
    )
    return _baseline_from_account_bucket(
        bucket,
        captured_at=now_utc,
        month_start=month_start,
    )


def _inference_credits_from_billing_usage(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        return None
    inference = usage.get("inferenceProviders")
    if not isinstance(inference, dict):
        return None

    included_usd = _nano_usd_to_usd(inference.get("includedNanoUsd"))
    used_usd = _nano_usd_to_usd(inference.get("usedNanoUsd"))
    limit_usd = _nano_usd_to_usd(inference.get("limitNanoUsd"))
    return {
        "included_usd": round(included_usd, 6),
        "used_usd": round(used_usd, 6),
        "remaining_included_usd": round(max(0.0, included_usd - used_usd), 6),
        "limit_usd": round(limit_usd, 6),
        "remaining_limit_usd": round(max(0.0, limit_usd - used_usd), 6),
        "num_requests": _coerce_int(inference.get("numRequests")),
        "period_start": inference.get("periodStart"),
        "period_end": inference.get("periodEnd"),
    }


async def _fetch_hf_billing_usage_v2(
    hf_token: str,
    *,
    start: datetime,
    end: datetime,
) -> dict[str, Any] | None:
    start_ts = max(1, int(_utc(start).timestamp()))
    end_ts = max(start_ts + 1, int(_utc(end).timestamp()))
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                HF_BILLING_USAGE_V2_URL,
                params={"startDate": start_ts, "endDate": end_ts},
                headers={"Authorization": f"Bearer {hf_token}"},
            )
            if response.status_code != 200:
                logger.debug(
                    "HF billing usage-v2 failed: status=%s body=%s",
                    response.status_code,
                    response.text[:200],
                )
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("HF billing usage-v2 failed: %s", e)
        return None


def _session_usage_window_started_at(
    manager: Any, session_id: str | None
) -> datetime | None:
    if not session_id:
        return None
    agent_session = getattr(manager, "sessions", {}).get(session_id)
    usage_window_started_at = getattr(agent_session, "usage_window_started_at", None)
    if isinstance(usage_window_started_at, datetime):
        return _utc(usage_window_started_at)
    created_at = getattr(agent_session, "created_at", None)
    if isinstance(created_at, datetime):
        return _utc(created_at)
    return None


def _session_usage_window_baseline(
    manager: Any,
    session_id: str | None,
) -> dict[str, Any] | None:
    if not session_id:
        return None
    agent_session = getattr(manager, "sessions", {}).get(session_id)
    baseline = getattr(agent_session, "usage_window_baseline", None)
    return baseline if isinstance(baseline, dict) else None


async def _load_persisted_session_usage_window_started_at(
    manager: Any,
    session_id: str | None,
) -> datetime | None:
    if not session_id:
        return None
    store = manager._store()
    if not getattr(store, "enabled", False) or not hasattr(store, "load_session"):
        return None
    loaded = await store.load_session(session_id)
    metadata = loaded.get("metadata") if isinstance(loaded, dict) else None
    started_at = None
    if isinstance(metadata, dict):
        started_at = metadata.get("usage_window_started_at") or metadata.get(
            "created_at"
        )
    if isinstance(started_at, datetime):
        return _utc(started_at)
    parsed = _parse_timestamp(started_at)
    return _utc(parsed) if parsed is not None else None


async def _load_persisted_session_usage_window_baseline(
    manager: Any,
    session_id: str | None,
) -> dict[str, Any] | None:
    if not session_id:
        return None
    store = manager._store()
    if not getattr(store, "enabled", False) or not hasattr(store, "load_session"):
        return None
    loaded = await store.load_session(session_id)
    metadata = loaded.get("metadata") if isinstance(loaded, dict) else None
    if not isinstance(metadata, dict):
        return None
    baseline = metadata.get("usage_window_baseline")
    return baseline if isinstance(baseline, dict) else None


async def _build_hf_account_usage(
    manager: Any,
    *,
    hf_token: str | None,
    session_id: str | None,
    timezone: str,
    now_utc: datetime,
    today_start: datetime,
    month_start: datetime,
    include_rollups: bool = True,
) -> dict[str, Any]:
    account_usage: dict[str, Any] = {
        "source": "hf_billing_usage_v2",
        "available": False,
        "current_session": None,
        "today": None,
        "month": None,
        "inference_providers_credits": None,
    }
    if not hf_token:
        account_usage["error"] = "missing_hf_token"
        return account_usage

    session_start = _session_usage_window_started_at(manager, session_id)
    if session_start is None:
        session_start = await _load_persisted_session_usage_window_started_at(
            manager, session_id
        )
    session_baseline = _session_usage_window_baseline(manager, session_id)
    if session_baseline is None:
        session_baseline = await _load_persisted_session_usage_window_baseline(
            manager,
            session_id,
        )
    baseline_month_start = None
    if session_baseline is not None:
        baseline_month_start = _baseline_datetime(session_baseline.get("month_start"))

    window_tasks: dict[str, tuple[datetime, asyncio.Task[dict[str, Any] | None]]] = {
        "month": (
            month_start,
            asyncio.create_task(
                _fetch_hf_billing_usage_v2(hf_token, start=month_start, end=now_utc)
            ),
        ),
    }
    if include_rollups:
        window_tasks["today"] = (
            today_start,
            asyncio.create_task(
                _fetch_hf_billing_usage_v2(hf_token, start=today_start, end=now_utc)
            ),
        )
    if session_start is not None:
        if baseline_month_start is not None:
            window_tasks["current_session_baseline"] = (
                baseline_month_start,
                asyncio.create_task(
                    _fetch_hf_billing_usage_v2(
                        hf_token,
                        start=baseline_month_start,
                        end=now_utc,
                    )
                ),
            )
        else:
            window_tasks["current_session"] = (
                session_start,
                asyncio.create_task(
                    _fetch_hf_billing_usage_v2(
                        hf_token,
                        start=session_start,
                        end=now_utc,
                    )
                ),
            )

    payloads: dict[str, dict[str, Any] | None] = {}
    for name, (_, task) in window_tasks.items():
        payloads[name] = await task

    any_payload = any(isinstance(payload, dict) for payload in payloads.values())
    account_usage["available"] = any_payload
    if not any_payload:
        account_usage["error"] = "billing_usage_unavailable"
        return account_usage

    for name, (start, _) in window_tasks.items():
        if name == "current_session_baseline":
            continue
        payload = payloads.get(name)
        if payload is None:
            continue
        account_usage[name] = _account_bucket_from_billing_usage(
            payload,
            window_start=start,
            window_end=now_utc,
            timezone=timezone,
        )

    if (
        session_start is not None
        and session_baseline is not None
        and baseline_month_start is not None
        and payloads.get("current_session_baseline") is not None
    ):
        baseline_bucket = _account_bucket_from_billing_usage(
            payloads.get("current_session_baseline"),
            window_start=baseline_month_start,
            window_end=now_utc,
            timezone=timezone,
        )
        account_usage["current_session"] = _baseline_delta_bucket(
            baseline_bucket,
            session_baseline,
            window_start=session_start,
            window_end=now_utc,
            timezone=timezone,
        )

    account_usage["inference_providers_credits"] = (
        _inference_credits_from_billing_usage(payloads.get("month"))
    )
    return account_usage


def _event_in_window(
    event: dict[str, Any],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    timezone_name: str | None = None,
) -> bool:
    if start is None and end is None:
        return True
    created_at = event_created_at(event, timezone_name=timezone_name)
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
    timezone_name: str | None = None,
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
            if _event_in_window(
                event,
                start=start,
                end=end,
                timezone_name=timezone_name,
            ):
                events.append(event)
    return events


async def build_usage_response(
    manager: Any,
    *,
    user_id: str,
    hf_token: str | None = None,
    session_id: str | None = None,
    timezone_name: str | None = None,
    now: datetime | None = None,
    include_rollups: bool = True,
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

    today_events: list[dict[str, Any]] = []
    month_events: list[dict[str, Any]] = []
    if include_rollups:
        today_events = await _load_usage_events(
            manager,
            user_id=user_id,
            start=today_start,
            end=now_utc,
            timezone_name=timezone,
        )
        month_events = await _load_usage_events(
            manager,
            user_id=user_id,
            start=month_start,
            end=now_utc,
            timezone_name=timezone,
        )
    hf_account = await _build_hf_account_usage(
        manager,
        hf_token=hf_token,
        session_id=session_id,
        timezone=timezone,
        now_utc=now_utc,
        today_start=today_start,
        month_start=month_start,
        include_rollups=include_rollups,
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
        "hf_account": hf_account,
        "links": {
            "hf_billing": HF_BILLING_URL,
            "inference_providers_pricing": HF_INFERENCE_PROVIDERS_PRICING_URL,
            "jobs_pricing": HF_JOBS_PRICING_URL,
        },
    }
