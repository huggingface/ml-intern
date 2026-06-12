"""Pure usage/billing summaries for session trajectory analytics."""

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from agent.core.cost_estimation import SPACE_PRICE_USD_PER_HOUR

USAGE_METRICS_VERSION = 1
BILLING_SCOPE_ACCOUNT_WINDOW_DELTA = "account_window_delta"

_USAGE_SCALAR_KEYS = (
    "usage_total_usd",
    "usage_total_usd_source",
    "usage_app_total_usd",
    "usage_hf_billing_total_usd",
    "usage_llm_calls",
    "usage_total_tokens",
    "usage_hf_job_submits",
    "usage_hf_job_status_snapshots",
    "usage_sandbox_creates",
    "usage_sandbox_pairs",
)


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if isfinite(parsed) else 0.0


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _round_usd(value: Any) -> float:
    return round(_coerce_float(value), 6)


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def event_created_at(event: dict[str, Any]) -> datetime | None:
    return _parse_timestamp(event.get("created_at") or event.get("timestamp"))


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") or {}
    return data if isinstance(data, dict) else {}


def _has_number(value: Any) -> bool:
    return _coerce_optional_float(value) is not None


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _empty_app_bucket(session_id: str | None) -> dict[str, Any]:
    return {
        "session_id": session_id,
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


def _sandbox_id(event: dict[str, Any]) -> str | None:
    sandbox_id = _event_data(event).get("sandbox_id")
    return sandbox_id if isinstance(sandbox_id, str) and sandbox_id else None


def _sandbox_duration_seconds(
    create_event: dict[str, Any],
    destroy_event: dict[str, Any],
) -> int:
    create_data = _event_data(create_event)
    destroy_data = _event_data(destroy_event)
    lifetime_s = _coerce_int(destroy_data.get("lifetime_s"))
    if lifetime_s > 0:
        return lifetime_s

    create_at = event_created_at(create_event)
    destroy_at = event_created_at(destroy_event)
    if create_at is None or destroy_at is None:
        return 0
    create_latency_s = max(0, _coerce_int(create_data.get("create_latency_s")))
    interval_start = create_at - timedelta(seconds=create_latency_s)
    if destroy_at <= interval_start:
        return 0
    return int((destroy_at - interval_start).total_seconds())


def summarize_sandbox_lifecycle(
    lifecycle_events: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Pair sandbox lifecycle events and estimate billed usage.

    Shared by dataset usage metrics and backend usage responses so sandbox
    pricing and create/destroy pairing semantics cannot drift.
    """
    ordered_events = [
        event
        for _, event in sorted(
            lifecycle_events,
            key=lambda indexed: (
                event_created_at(indexed[1]) is None,
                event_created_at(indexed[1]) or datetime.min.replace(tzinfo=UTC),
                indexed[0],
            ),
        )
    ]
    active_creates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_pairs = 0
    unpaired_destroys = 0
    estimated_usd = 0.0
    billable_seconds = 0

    for event in ordered_events:
        event_type = event.get("event_type")
        sandbox_id = _sandbox_id(event)
        if sandbox_id is None:
            continue
        if event_type == "sandbox_create":
            active_creates[sandbox_id].append(event)
            continue
        if event_type != "sandbox_destroy":
            continue

        creates = active_creates.get(sandbox_id)
        if not creates:
            unpaired_destroys += 1
            continue

        create_event = creates.pop()
        if not creates:
            active_creates.pop(sandbox_id, None)

        hardware = str(_event_data(create_event).get("hardware") or "cpu-basic")
        seconds = _sandbox_duration_seconds(create_event, event)
        price_usd_per_hour = _coerce_float(SPACE_PRICE_USD_PER_HOUR.get(hardware))
        matched_pairs += 1
        if price_usd_per_hour > 0:
            billable_seconds += seconds
        estimated_usd += price_usd_per_hour * (seconds / 3600)

    return {
        "matched_pairs": matched_pairs,
        "unpaired_creates": sum(len(events) for events in active_creates.values()),
        "unpaired_destroys": unpaired_destroys,
        "estimated_usd": _round_usd(estimated_usd),
        "billable_seconds_estimate": billable_seconds,
    }


def normalize_hf_billing_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Return a dataset-safe HF billing snapshot.

    Only current-session window rollups are retained. Monthly account totals,
    credit limits, and any caller-provided extra fields are intentionally
    dropped before the snapshot can be serialized into session artifacts.
    """
    hf_billing = snapshot.get("hf_billing") if isinstance(snapshot, dict) else None
    hf_billing = hf_billing if isinstance(hf_billing, dict) else {}
    current_session = hf_billing.get("current_session")
    current_session = current_session if isinstance(current_session, dict) else None

    sanitized_current = None
    if current_session is not None:
        sanitized_current = {
            "window_start": current_session.get("window_start"),
            "window_end": current_session.get("window_end"),
            "timezone": current_session.get("timezone"),
            "total_usd": _round_usd(current_session.get("total_usd")),
            "inference_providers_usd": _round_usd(
                current_session.get("inference_providers_usd")
            ),
            "hf_jobs_usd": _round_usd(current_session.get("hf_jobs_usd")),
            "inference_provider_requests": _coerce_int(
                current_session.get("inference_provider_requests")
            ),
            "hf_jobs_minutes": round(
                _coerce_float(current_session.get("hf_jobs_minutes")), 3
            ),
        }

    available = bool(hf_billing.get("available") and sanitized_current is not None)
    return {
        "billing_scope": BILLING_SCOPE_ACCOUNT_WINDOW_DELTA,
        "hf_billing": {
            "source": str(hf_billing.get("source") or "hf_billing_usage_v2"),
            "available": available,
            "error": None if available else hf_billing.get("error"),
            "current_session": sanitized_current if available else None,
        },
    }


def summarize_usage_events(
    events: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    hf_billing_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    app = _empty_app_bucket(session_id)
    llm_by_kind: Counter[str] = Counter()
    llm_by_model: Counter[str] = Counter()
    job_statuses: Counter[str] = Counter()
    job_submit_flavors: Counter[str] = Counter()
    job_status_flavors: Counter[str] = Counter()
    sandbox_hardware: Counter[str] = Counter()
    lifecycle_events: list[tuple[int, dict[str, Any]]] = []

    event_count = 0
    events_without_timestamp = 0
    llm_calls_with_cost_usd = 0
    llm_calls_with_nonzero_cost_usd = 0
    job_submits = 0
    job_status_snapshots = 0
    job_snapshots_with_estimated_cost = 0
    job_snapshots_with_nonzero_estimated_cost = 0
    sandbox_creates = 0
    sandbox_destroys = 0
    turn_complete_count = 0
    assistant_stream_end_count = 0

    for index, event in enumerate(events or []):
        if not isinstance(event, dict):
            continue
        event_count += 1
        if event_created_at(event) is None:
            events_without_timestamp += 1

        event_type = event.get("event_type")
        data = _event_data(event)
        if event_type == "llm_call":
            app["llm_calls"] += 1
            if "cost_usd" in data:
                llm_calls_with_cost_usd += 1
            cost_usd = _coerce_float(data.get("cost_usd"))
            if cost_usd > 0:
                llm_calls_with_nonzero_cost_usd += 1
            app["inference_usd"] += cost_usd

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
            app["prompt_tokens"] += prompt_tokens
            app["completion_tokens"] += completion_tokens
            app["cache_read_tokens"] += cache_read_tokens
            app["cache_creation_tokens"] += cache_creation_tokens
            app["total_tokens"] += total_tokens
            llm_by_kind[str(data.get("kind") or "unknown")] += 1
            llm_by_model[str(data.get("model") or "unknown")] += 1
        elif event_type == "hf_job_submit":
            job_submits += 1
            job_submit_flavors[str(data.get("flavor") or "unknown")] += 1
        elif event_type == "hf_job_complete":
            job_status_snapshots += 1
            app["hf_jobs_count"] += 1
            estimated_cost = _coerce_float(data.get("estimated_cost_usd"))
            app["hf_jobs_estimated_usd"] += estimated_cost
            app["hf_jobs_billable_seconds_estimate"] += _coerce_int(
                data.get("billable_seconds_estimate") or data.get("wall_time_s")
            )
            if _has_number(data.get("estimated_cost_usd")):
                job_snapshots_with_estimated_cost += 1
            if estimated_cost > 0:
                job_snapshots_with_nonzero_estimated_cost += 1
            job_statuses[str(data.get("final_status") or "unknown")] += 1
            job_status_flavors[str(data.get("flavor") or "unknown")] += 1
        elif event_type == "sandbox_create":
            sandbox_creates += 1
            sandbox_hardware[str(data.get("hardware") or "cpu-basic")] += 1
            lifecycle_events.append((index, event))
        elif event_type == "sandbox_destroy":
            sandbox_destroys += 1
            lifecycle_events.append((index, event))
        elif event_type == "turn_complete":
            turn_complete_count += 1
        elif event_type == "assistant_stream_end":
            assistant_stream_end_count += 1

    sandbox = summarize_sandbox_lifecycle(lifecycle_events)
    app["sandbox_count"] = sandbox["matched_pairs"]
    app["sandbox_estimated_usd"] = sandbox["estimated_usd"]
    app["sandbox_billable_seconds_estimate"] = sandbox["billable_seconds_estimate"]
    app["inference_usd"] = _round_usd(app["inference_usd"])
    app["hf_jobs_estimated_usd"] = _round_usd(app["hf_jobs_estimated_usd"])
    app["total_usd"] = _round_usd(
        app["inference_usd"]
        + app["hf_jobs_estimated_usd"]
        + app["sandbox_estimated_usd"]
    )

    billing = normalize_hf_billing_snapshot(hf_billing_snapshot)
    current_billing = billing["hf_billing"]["current_session"]
    hf_billing_total = None
    if billing["hf_billing"]["available"] and current_billing is not None:
        hf_billing_total = _round_usd(current_billing.get("total_usd"))
        usage_total = _round_usd(hf_billing_total + app["sandbox_estimated_usd"])
        usage_total_source = "hf_billing_plus_sandbox_estimate"
    else:
        usage_total = app["total_usd"]
        usage_total_source = "app_telemetry_fallback"

    job_flavors = job_submit_flavors + job_status_flavors

    return {
        "version": USAGE_METRICS_VERSION,
        "session_id": session_id,
        "billing_scope": BILLING_SCOPE_ACCOUNT_WINDOW_DELTA,
        "total_usd": usage_total,
        "total_usd_source": usage_total_source,
        "app_total_usd": app["total_usd"],
        "hf_billing_total_usd": hf_billing_total,
        "app_telemetry": app,
        "hf_billing": billing["hf_billing"],
        "llm": {
            "calls": app["llm_calls"],
            "calls_by_kind": _counter_dict(llm_by_kind),
            "calls_by_model": _counter_dict(llm_by_model),
            "prompt_tokens": app["prompt_tokens"],
            "completion_tokens": app["completion_tokens"],
            "cache_read_tokens": app["cache_read_tokens"],
            "cache_creation_tokens": app["cache_creation_tokens"],
            "total_tokens": app["total_tokens"],
        },
        "turns": {
            "turn_complete_count": turn_complete_count,
            "assistant_stream_end_count": assistant_stream_end_count,
        },
        "hf_jobs": {
            "submits": job_submits,
            "status_snapshots": job_status_snapshots,
            "statuses": _counter_dict(job_statuses),
            "flavors": _counter_dict(job_flavors),
            "submit_flavors": _counter_dict(job_submit_flavors),
            "status_snapshot_flavors": _counter_dict(job_status_flavors),
            "estimated_usd": app["hf_jobs_estimated_usd"],
            "billable_seconds_estimate": app["hf_jobs_billable_seconds_estimate"],
            "snapshots_with_estimated_cost": job_snapshots_with_estimated_cost,
            "snapshots_with_nonzero_estimated_cost": (
                job_snapshots_with_nonzero_estimated_cost
            ),
        },
        "sandboxes": {
            "creates": sandbox_creates,
            "destroys": sandbox_destroys,
            "matched_pairs": sandbox["matched_pairs"],
            "unpaired_creates": sandbox["unpaired_creates"],
            "unpaired_destroys": sandbox["unpaired_destroys"],
            "hardware": _counter_dict(sandbox_hardware),
            "estimated_usd": app["sandbox_estimated_usd"],
            "billable_seconds_estimate": app["sandbox_billable_seconds_estimate"],
        },
        "data_quality": {
            "event_count": event_count,
            "events_without_timestamp": events_without_timestamp,
            "llm_calls_with_cost_usd": llm_calls_with_cost_usd,
            "llm_calls_with_nonzero_cost_usd": llm_calls_with_nonzero_cost_usd,
            "job_snapshots_with_estimated_cost": job_snapshots_with_estimated_cost,
            "job_snapshots_missing_estimated_cost": (
                job_status_snapshots - job_snapshots_with_estimated_cost
            ),
        },
    }


def usage_metric_scalar_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    app = metrics.get("app_telemetry") if isinstance(metrics, dict) else {}
    llm = metrics.get("llm") if isinstance(metrics, dict) else {}
    jobs = metrics.get("hf_jobs") if isinstance(metrics, dict) else {}
    sandboxes = metrics.get("sandboxes") if isinstance(metrics, dict) else {}
    values = {
        "usage_total_usd": metrics.get("total_usd"),
        "usage_total_usd_source": metrics.get("total_usd_source"),
        "usage_app_total_usd": metrics.get("app_total_usd"),
        "usage_hf_billing_total_usd": metrics.get("hf_billing_total_usd"),
        "usage_llm_calls": app.get("llm_calls") if isinstance(app, dict) else None,
        "usage_total_tokens": llm.get("total_tokens")
        if isinstance(llm, dict)
        else None,
        "usage_hf_job_submits": (
            jobs.get("submits") if isinstance(jobs, dict) else None
        ),
        "usage_hf_job_status_snapshots": (
            jobs.get("status_snapshots") if isinstance(jobs, dict) else None
        ),
        "usage_sandbox_creates": (
            sandboxes.get("creates") if isinstance(sandboxes, dict) else None
        ),
        "usage_sandbox_pairs": (
            sandboxes.get("matched_pairs") if isinstance(sandboxes, dict) else None
        ),
    }
    return {key: values.get(key) for key in _USAGE_SCALAR_KEYS}
