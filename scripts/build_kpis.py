#!/usr/bin/env python3
"""Build KPI dataset v2 facts and rollups.

The v2 pipeline deliberately stops producing the legacy hourly/daily schema.
It emits exact per-session facts plus hourly, daily, and monthly rollups under
``v2/``. Raw user ids, session ids, and Hub artifact repo ids are never written;
all identifiers are salted HMAC-SHA256 hashes.
"""

import argparse
import calendar
import csv
import hashlib
import hmac
import io
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger("build_kpis")

SCHEMA_VERSION = 2
V2_PREFIX = "v2"
VALID_PLANS = {"free", "pro", "unknown"}
VALID_REPO_TYPES = {"model", "dataset", "space"}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_hash_salt(salt: str | None = None) -> str:
    resolved = salt if salt is not None else os.environ.get("KPI_USER_HASH_SALT")
    if not resolved:
        raise RuntimeError("KPI_USER_HASH_SALT is required for KPI v2 exports")
    return resolved


def _hash_identifier(value: Any, salt: str) -> str:
    raw = "" if value is None else str(value)
    return hmac.new(
        salt.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _integer(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_ts(value: Any) -> datetime | None:
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
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _date_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _hour_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")


def _month_key_from_date_key(day: str) -> str:
    return day[:7]


def _normalize_plan(value: Any) -> str:
    plan = str(value or "unknown").strip().lower()
    return plan if plan in VALID_PLANS else "unknown"


def _normalize_job_status(value: Any) -> str:
    status = str(value or "unknown").strip().lower()
    if status in {"completed", "complete", "succeeded", "success", "done"}:
        return "completed"
    if status in {"cancelled", "canceled", "cancelled_by_user", "canceled_by_user"}:
        return "cancelled"
    if "cancel" in status or status in {"stopped", "killed"}:
        return "cancelled"
    if status in {"failed", "failure", "error", "errored", "timeout", "timed_out"}:
        return "failed"
    if "fail" in status or "error" in status or "timeout" in status:
        return "failed"
    return "unknown"


def _session_events(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event for event in _as_list(session.get("events")) if isinstance(event, dict)
    ]


def _session_messages(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [msg for msg in _as_list(session.get("messages")) if isinstance(msg, dict)]


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(event.get("data"))


def _session_usage_metrics(session: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(session.get("usage_metrics"))
    if metrics:
        return metrics
    # Historical rows can lack usage_metrics. Keep the fallback minimal and do
    # not use llm_call.cost_usd for the billing fields.
    prompt = completion = cache_read = cache_creation = total = calls = 0
    calls_by_model: Counter[str] = Counter()
    for event in _session_events(session):
        if event.get("event_type") != "llm_call":
            continue
        data = _event_data(event)
        calls += 1
        prompt += _integer(data.get("prompt_tokens"))
        completion += _integer(data.get("completion_tokens"))
        cache_read += _integer(data.get("cache_read_tokens"))
        cache_creation += _integer(data.get("cache_creation_tokens"))
        total += _integer(data.get("total_tokens")) or (
            _integer(data.get("prompt_tokens"))
            + _integer(data.get("completion_tokens"))
            + _integer(data.get("cache_read_tokens"))
            + _integer(data.get("cache_creation_tokens"))
        )
        calls_by_model[
            str(data.get("model") or session.get("model_name") or "unknown")
        ] += 1
    return {
        "total_usd": 0.0,
        "total_usd_source": "missing_usage_metrics",
        "app_total_usd": 0.0,
        "hf_billing_total_usd": None,
        "app_telemetry": {
            "inference_usd": 0.0,
            "hf_jobs_estimated_usd": 0.0,
            "sandbox_estimated_usd": 0.0,
        },
        "hf_billing": {"available": False, "current_session": None},
        "llm": {
            "calls": calls,
            "calls_by_model": dict(calls_by_model),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "total_tokens": total,
        },
        "hf_jobs": {"submits": 0, "estimated_usd": 0.0},
        "sandboxes": {"creates": 0, "estimated_usd": 0.0},
    }


def _active_times(
    session: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    event_timestamps = [
        ts
        for ts in (
            _parse_ts(event.get("created_at") or event.get("timestamp"))
            for event in events
        )
        if ts is not None
    ]
    timestamps = list(event_timestamps)
    for key in ("session_start_time", "session_end_time"):
        ts = _parse_ts(session.get(key))
        if ts is not None:
            timestamps.append(ts)
    if not timestamps:
        now = datetime.now(timezone.utc)
        timestamps.append(now)

    active_hours = {_hour_key(ts) for ts in timestamps}
    start_ts = _parse_ts(session.get("session_start_time"))
    end_ts = _parse_ts(session.get("session_end_time"))
    if start_ts is not None and end_ts is not None and end_ts >= start_ts:
        current = start_ts.replace(minute=0, second=0, microsecond=0)
        final = end_ts.replace(minute=0, second=0, microsecond=0)
        while current <= final:
            active_hours.add(_hour_key(current))
            current += timedelta(hours=1)

    active_dates = {_date_key(ts) for ts in timestamps}
    active_dates.update(hour[:10] for hour in active_hours)
    return (sorted(active_dates), sorted(active_hours))


def _session_status(events: list[dict[str, Any]]) -> str:
    event_types = {event.get("event_type") for event in events}
    if "error" in event_types:
        return "failed"
    if (
        "interrupted" in event_types
        or "cancelled" in event_types
        or "canceled" in event_types
    ):
        return "cancelled"
    if "turn_complete" in event_types:
        return "completed"
    return "unknown"


def _usage_components(metrics: dict[str, Any]) -> dict[str, float | str]:
    app = _as_dict(metrics.get("app_telemetry"))
    hf_billing = _as_dict(metrics.get("hf_billing"))
    current_session = _as_dict(hf_billing.get("current_session"))
    source = str(metrics.get("total_usd_source") or "usage_metrics")

    if hf_billing.get("available") and current_session:
        inference = _number(current_session.get("inference_providers_usd"))
        jobs = _number(current_session.get("hf_jobs_usd"))
    else:
        inference = _number(app.get("inference_usd"))
        jobs = _number(app.get("hf_jobs_estimated_usd"))

    sandboxes = _number(app.get("sandbox_estimated_usd"))
    return {
        "usage_total_usd": round(_number(metrics.get("total_usd")), 6),
        "usage_inference_providers_usd": round(inference, 6),
        "usage_hf_jobs_usd": round(jobs, 6),
        "usage_sandboxes_usd": round(sandboxes, 6),
        "usage_cost_source": source,
    }


def _model_usage_from_events(
    session: dict[str, Any],
    events: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, dict[str, int]]:
    usage: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "sessions": 0,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    for event in events:
        if event.get("event_type") != "llm_call":
            continue
        data = _event_data(event)
        model = str(data.get("model") or session.get("model_name") or "unknown")
        input_tokens = (
            _integer(data.get("prompt_tokens"))
            + _integer(data.get("cache_read_tokens"))
            + _integer(data.get("cache_creation_tokens"))
        )
        output_tokens = _integer(data.get("completion_tokens"))
        total_tokens = _integer(data.get("total_tokens")) or (
            input_tokens + output_tokens
        )
        usage[model]["calls"] += 1
        usage[model]["input_tokens"] += input_tokens
        usage[model]["output_tokens"] += output_tokens
        usage[model]["total_tokens"] += total_tokens

    if not usage:
        llm = _as_dict(metrics.get("llm"))
        calls_by_model = _as_dict(llm.get("calls_by_model"))
        if calls_by_model:
            for model, calls in calls_by_model.items():
                usage[str(model)]["calls"] = _integer(calls)
            if len(calls_by_model) == 1:
                model = next(iter(calls_by_model))
                usage[str(model)]["input_tokens"] = (
                    _integer(llm.get("prompt_tokens"))
                    + _integer(llm.get("cache_read_tokens"))
                    + _integer(llm.get("cache_creation_tokens"))
                )
                usage[str(model)]["output_tokens"] = _integer(
                    llm.get("completion_tokens")
                )
                usage[str(model)]["total_tokens"] = _integer(llm.get("total_tokens"))
        else:
            model = str(session.get("model_name") or "unknown")
            usage[model]["calls"] = _integer(llm.get("calls"))
            usage[model]["input_tokens"] = (
                _integer(llm.get("prompt_tokens"))
                + _integer(llm.get("cache_read_tokens"))
                + _integer(llm.get("cache_creation_tokens"))
            )
            usage[model]["output_tokens"] = _integer(llm.get("completion_tokens"))
            usage[model]["total_tokens"] = _integer(llm.get("total_tokens"))

    for model_usage in usage.values():
        if model_usage["calls"] or model_usage["total_tokens"]:
            model_usage["sessions"] = 1
    return dict(sorted((model, dict(values)) for model, values in usage.items()))


def _job_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    submitted = 0
    terminal_by_id: dict[str, str] = {}
    anonymous_terminal_statuses: list[str] = []
    for idx, event in enumerate(events):
        event_type = event.get("event_type")
        data = _event_data(event)
        if event_type == "hf_job_submit":
            submitted += 1
        elif event_type in {"hf_job_complete", "hf_job_cancel"}:
            status = _normalize_job_status(
                data.get("final_status") or data.get("status")
            )
            job_id = data.get("job_id")
            if job_id:
                terminal_by_id[str(job_id)] = status
            else:
                anonymous_terminal_statuses.append(status or f"unknown-{idx}")

    statuses = Counter(terminal_by_id.values())
    statuses.update(anonymous_terminal_statuses)
    return {
        "hf_jobs_submitted": submitted,
        "hf_jobs_completed": int(statuses.get("completed", 0)),
        "hf_jobs_failed": int(statuses.get("failed", 0)),
        "hf_jobs_cancelled": int(statuses.get("cancelled", 0)),
    }


def _sandbox_counts(
    events: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, int]:
    created = cpu = gpu = 0
    for event in events:
        if event.get("event_type") != "sandbox_create":
            continue
        data = _event_data(event)
        created += 1
        hardware = str(data.get("hardware") or "cpu-basic").lower()
        if hardware.startswith("cpu-"):
            cpu += 1
        else:
            gpu += 1
    if created:
        return {
            "sandboxes_created": created,
            "sandboxes_cpu": cpu,
            "sandboxes_gpu": gpu,
        }

    sandboxes = _as_dict(metrics.get("sandboxes"))
    hardware = _as_dict(sandboxes.get("hardware"))
    created = _integer(sandboxes.get("creates"))
    cpu = sum(
        _integer(count)
        for flavor, count in hardware.items()
        if str(flavor).lower().startswith("cpu-")
    )
    gpu = max(0, created - cpu)
    return {
        "sandboxes_created": created,
        "sandboxes_cpu": int(cpu),
        "sandboxes_gpu": int(gpu),
    }


def _artifact_counts(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "hub_artifact":
            continue
        data = _event_data(event)
        if data.get("success") is False:
            continue
        artifact_hash = str(data.get("artifact_hash") or "").strip()
        if not artifact_hash:
            continue
        repo_type = str(data.get("repo_type") or "").strip().lower()
        if repo_type not in VALID_REPO_TYPES:
            continue
        by_hash[artifact_hash] = data

    type_counts: Counter[str] = Counter()
    non_sandbox_spaces = 0
    for data in by_hash.values():
        repo_type = str(data.get("repo_type") or "").lower()
        type_counts[repo_type] += 1
        if repo_type == "space" and not bool(data.get("is_sandbox")):
            non_sandbox_spaces += 1

    by_type = {
        "model": int(type_counts.get("model", 0)),
        "dataset": int(type_counts.get("dataset", 0)),
        "space": int(type_counts.get("space", 0)),
        "non_sandbox_space": int(non_sandbox_spaces),
    }
    return {
        "hub_models_created": by_type["model"],
        "hub_datasets_created": by_type["dataset"],
        "hub_spaces_created": by_type["space"],
        "hub_non_sandbox_spaces_created": by_type["non_sandbox_space"],
        "hub_artifacts_by_type_json": _json_dumps(by_type),
    }


def _session_fact(session: dict[str, Any], salt: str | None = None) -> dict[str, Any]:
    salt = _require_hash_salt(salt)
    events = _session_events(session)
    messages = _session_messages(session)
    metrics = _session_usage_metrics(session)
    llm = _as_dict(metrics.get("llm"))
    active_dates, active_hours = _active_times(session, events)
    usage = _usage_components(metrics)
    jobs = _job_counts(events)
    sandboxes = _sandbox_counts(events, metrics)
    artifacts = _artifact_counts(events)
    model_usage = _model_usage_from_events(session, events, metrics)

    session_id = str(session.get("session_id") or "")
    user_id = session.get("user_id") or f"session:{session_id}"
    prompt_tokens = _integer(llm.get("prompt_tokens"))
    completion_tokens = _integer(llm.get("completion_tokens"))
    cache_read_tokens = _integer(llm.get("cache_read_tokens"))
    cache_creation_tokens = _integer(llm.get("cache_creation_tokens"))
    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens
    output_tokens = completion_tokens
    total_tokens = _integer(llm.get("total_tokens")) or (input_tokens + output_tokens)

    models_used = sorted(model_usage.keys())
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id_hash": _hash_identifier(session_id, salt),
        "user_id_hash": _hash_identifier(user_id, salt),
        "user_plan": _normalize_plan(session.get("user_plan")),
        "session_start_time": session.get("session_start_time"),
        "session_end_time": session.get("session_end_time"),
        "active_dates": active_dates,
        "active_hours": active_hours,
        "turns": sum(1 for msg in messages if msg.get("role") == "user"),
        "status": _session_status(events),
        "models_used_json": _json_dumps(models_used),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        **jobs,
        **sandboxes,
        **artifacts,
        **usage,
        "model_usage_json": _json_dumps(model_usage),
    }


def _fact_active_in_hour(fact: dict[str, Any], hour_key: str) -> bool:
    return hour_key in (fact.get("active_hours") or [])


def _fact_active_in_day(fact: dict[str, Any], day_key: str) -> bool:
    return day_key in (fact.get("active_dates") or [])


def _fact_active_in_month(fact: dict[str, Any], month_key: str) -> bool:
    return any(
        str(day).startswith(f"{month_key}-") for day in fact.get("active_dates") or []
    )


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_avg": 0.0,
        }
    return {
        f"{prefix}_min": round(min(values), 6),
        f"{prefix}_max": round(max(values), 6),
        f"{prefix}_avg": _avg(values),
    }


def _latest_plan_counts(facts: list[dict[str, Any]]) -> dict[str, int]:
    latest: dict[str, tuple[datetime, str]] = {}
    for fact in facts:
        user_hash = fact.get("user_id_hash")
        if not user_hash:
            continue
        ts = _parse_ts(fact.get("session_end_time")) or _parse_ts(
            fact.get("session_start_time")
        )
        if ts is None:
            ts = datetime.min.replace(tzinfo=timezone.utc)
        plan = _normalize_plan(fact.get("user_plan"))
        previous = latest.get(user_hash)
        if previous is None or ts >= previous[0]:
            latest[user_hash] = (ts, plan)
    counter = Counter(plan for _, plan in latest.values())
    return {
        "free_users": int(counter.get("free", 0)),
        "pro_users": int(counter.get("pro", 0)),
        "unknown_plan_users": int(counter.get("unknown", 0)),
    }


def _merge_model_usage(facts: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "sessions": 0,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    sessions_by_model: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        session_hash = str(fact.get("session_id_hash") or "")
        usage = _as_dict(fact.get("model_usage_json"))
        for model, raw_values in usage.items():
            values = _as_dict(raw_values)
            if session_hash:
                sessions_by_model[str(model)].add(session_hash)
            merged[str(model)]["calls"] += _integer(values.get("calls"))
            merged[str(model)]["input_tokens"] += _integer(values.get("input_tokens"))
            merged[str(model)]["output_tokens"] += _integer(values.get("output_tokens"))
            merged[str(model)]["total_tokens"] += _integer(values.get("total_tokens"))
    for model, sessions in sessions_by_model.items():
        merged[model]["sessions"] = len(sessions)
    return dict(sorted((model, dict(values)) for model, values in merged.items()))


def _rollup(facts: list[dict[str, Any]], bucket: str) -> dict[str, Any]:
    facts = list(facts)
    unique_users = {
        fact.get("user_id_hash") for fact in facts if fact.get("user_id_hash")
    }
    unique_sessions = {
        fact.get("session_id_hash") for fact in facts if fact.get("session_id_hash")
    }
    status_counts = Counter(str(fact.get("status") or "unknown") for fact in facts)
    plan_counts = _latest_plan_counts(facts)
    model_usage = _merge_model_usage(facts)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bucket": bucket,
        "active_users": len(unique_users),
        "active_sessions": len(unique_sessions),
        **plan_counts,
        "completed_sessions": int(status_counts.get("completed", 0)),
        "failed_sessions": int(status_counts.get("failed", 0)),
        "cancelled_sessions": int(status_counts.get("cancelled", 0)),
        "unknown_status_sessions": int(status_counts.get("unknown", 0)),
    }

    stat_fields = [
        "turns",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "usage_total_usd",
        "usage_inference_providers_usd",
        "usage_hf_jobs_usd",
        "usage_sandboxes_usd",
    ]
    for field in stat_fields:
        row.update(_stats(field, [_number(fact.get(field)) for fact in facts]))

    sum_fields = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "hf_jobs_submitted",
        "hf_jobs_completed",
        "hf_jobs_failed",
        "hf_jobs_cancelled",
        "sandboxes_created",
        "sandboxes_cpu",
        "sandboxes_gpu",
        "hub_models_created",
        "hub_datasets_created",
        "hub_spaces_created",
        "hub_non_sandbox_spaces_created",
        "usage_total_usd",
        "usage_inference_providers_usd",
        "usage_hf_jobs_usd",
        "usage_sandboxes_usd",
    ]
    for field in sum_fields:
        total = sum(_number(fact.get(field)) for fact in facts)
        if field.endswith("_usd"):
            row[f"total_{field}"] = round(total, 6)
        else:
            row[f"total_{field}"] = int(total)

    row["model_usage_json"] = _json_dumps(model_usage)
    return row


def _hourly_rollup(facts: list[dict[str, Any]], hour_key: str) -> dict[str, Any]:
    return _rollup(
        [fact for fact in facts if _fact_active_in_hour(fact, hour_key)], hour_key
    )


def _daily_rollup(facts: list[dict[str, Any]], day_key: str) -> dict[str, Any]:
    return _rollup(
        [fact for fact in facts if _fact_active_in_day(fact, day_key)], day_key
    )


def _monthly_rollup(facts: list[dict[str, Any]], month_key: str) -> dict[str, Any]:
    return _rollup(
        [fact for fact in facts if _fact_active_in_month(fact, month_key)],
        month_key,
    )


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
        for row in sorted(rows, key=lambda item: str(item.get("session_id_hash") or ""))
    )


def _upload_bytes(
    api: Any,
    *,
    repo_id: str,
    token: str,
    path_in_repo: str,
    content: bytes,
    commit_message: str,
) -> None:
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            private=False,
            exist_ok=True,
        )
    except Exception as e:
        logger.debug("create_repo(%s) skipped: %s", repo_id, e)

    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _write_csv(
    api: Any,
    *,
    repo_id: str,
    token: str,
    path_in_repo: str,
    row: dict[str, Any],
) -> None:
    _upload_bytes(
        api,
        repo_id=repo_id,
        token=token,
        path_in_repo=path_in_repo,
        content=_csv_bytes([row]),
        commit_message=f"Update KPI v2 {path_in_repo}",
    )


def _fact_start_day(fact: dict[str, Any]) -> str:
    start_ts = _parse_ts(fact.get("session_start_time"))
    if start_ts:
        return _date_key(start_ts)
    return str((fact.get("active_dates") or ["unknown"])[0])


def _write_session_facts(
    api: Any,
    *,
    repo_id: str,
    token: str,
    facts: list[dict[str, Any]],
) -> None:
    by_start_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        by_start_day[_fact_start_day(fact)].append(fact)

    for day_key, day_facts in by_start_day.items():
        _upload_bytes(
            api,
            repo_id=repo_id,
            token=token,
            path_in_repo=f"{V2_PREFIX}/session_facts/{day_key}.jsonl",
            content=_jsonl_bytes(day_facts),
            commit_message=f"Update KPI v2 session facts {day_key}",
        )


def _iter_session_files(api: Any, repo_id: str, day: date, token: str) -> Iterable[str]:
    prefix = f"sessions/{day.isoformat()}/"
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception as e:
        logger.warning("list_repo_files(%s) failed: %s", repo_id, e)
        return []
    return [
        path for path in files if path.startswith(prefix) and path.endswith(".jsonl")
    ]


def _download_session(repo_id: str, path: str, token: str) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type="dataset",
            token=token,
        )
    except Exception as e:
        logger.warning("hf_hub_download(%s) failed: %s", path, e)
        return None

    try:
        with open(local, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        row = json.loads(line)
        for key in ("messages", "events", "tools", "usage_metrics"):
            value = row.get(key)
            if isinstance(value, str):
                try:
                    row[key] = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    row[key] = [] if key in {"messages", "events", "tools"} else {}
        return row
    except Exception as e:
        logger.warning("parse(%s) failed: %s", path, e)
        return None


def _sessions_for_start_dates(
    api: Any,
    source_repo: str,
    dates: Iterable[date],
    token: str,
) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for day in sorted(set(dates)):
        for path in _iter_session_files(api, source_repo, day, token):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            session = _download_session(source_repo, path, token)
            if session:
                sessions.append(session)
    return sessions


def _facts_for_start_dates(
    api: Any,
    source_repo: str,
    dates: Iterable[date],
    token: str,
    salt: str,
) -> list[dict[str, Any]]:
    return [
        _session_fact(session, salt=salt)
        for session in _sessions_for_start_dates(api, source_repo, dates, token)
    ]


def _daily_start_dates(day: date) -> list[date]:
    # Rollups read the target day and the previous start-date partition to
    # include normal midnight-spanning sessions. Sessions active for more than
    # one day before this window are not included; ML Intern sessions are
    # expected to be short-lived.
    return [day - timedelta(days=1), day]


def _month_start_dates(month_key: str, through_day: date | None = None) -> list[date]:
    year, month = (int(part) for part in month_key.split("-", 1))
    last_day = calendar.monthrange(year, month)[1]
    start = date(year, month, 1)
    end = date(year, month, last_day)
    if through_day is not None and through_day.strftime("%Y-%m") == month_key:
        end = min(end, through_day)
    dates = []
    # Include the previous day to catch ordinary sessions that started before
    # midnight on the first of the month and remained active after it.
    current = start - timedelta(days=1)
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _write_daily_rollup(
    api: Any,
    *,
    source_repo: str,
    target_repo: str,
    day: date,
    token: str,
    salt: str,
) -> dict[str, Any]:
    facts = _facts_for_start_dates(
        api, source_repo, _daily_start_dates(day), token, salt
    )
    row = _daily_rollup(facts, day.isoformat())
    _write_csv(
        api,
        repo_id=target_repo,
        token=token,
        path_in_repo=f"{V2_PREFIX}/daily/{day.isoformat()}.csv",
        row=row,
    )
    return row


def _write_monthly_rollup(
    api: Any,
    *,
    source_repo: str,
    target_repo: str,
    month_key: str,
    token: str,
    salt: str,
    through_day: date | None = None,
) -> dict[str, Any]:
    facts = _facts_for_start_dates(
        api,
        source_repo,
        _month_start_dates(month_key, through_day=through_day),
        token,
        salt,
    )
    row = _monthly_rollup(facts, month_key)
    _write_csv(
        api,
        repo_id=target_repo,
        token=token,
        path_in_repo=f"{V2_PREFIX}/monthly/{month_key}.csv",
        row=row,
    )
    return row


def run_for_hour(
    api: Any,
    *,
    source_repo: str,
    target_repo: str,
    hour_dt: datetime,
    token: str,
    salt: str | None = None,
) -> dict[str, Any]:
    salt = _require_hash_salt(salt)
    hour_dt = hour_dt.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    day = hour_dt.date()
    facts = _facts_for_start_dates(
        api,
        source_repo,
        _daily_start_dates(day),
        token,
        salt,
    )
    current_day = day.isoformat()
    _write_session_facts(
        api,
        repo_id=target_repo,
        token=token,
        facts=[fact for fact in facts if _fact_start_day(fact) == current_day],
    )

    hour = _hour_key(hour_dt)
    row = _hourly_rollup(facts, hour)
    _write_csv(
        api,
        repo_id=target_repo,
        token=token,
        path_in_repo=f"{V2_PREFIX}/hourly/{day.isoformat()}/{hour_dt:%H}.csv",
        row=row,
    )

    _write_daily_rollup(
        api,
        source_repo=source_repo,
        target_repo=target_repo,
        day=day,
        token=token,
        salt=salt,
    )
    _write_monthly_rollup(
        api,
        source_repo=source_repo,
        target_repo=target_repo,
        month_key=day.strftime("%Y-%m"),
        token=token,
        salt=salt,
        through_day=day,
    )
    return row


def run_for_day(
    api: Any,
    *,
    source_repo: str,
    target_repo: str,
    day: date,
    token: str,
    salt: str | None = None,
) -> dict[str, Any]:
    salt = _require_hash_salt(salt)
    row = _write_daily_rollup(
        api,
        source_repo=source_repo,
        target_repo=target_repo,
        day=day,
        token=token,
        salt=salt,
    )
    _write_monthly_rollup(
        api,
        source_repo=source_repo,
        target_repo=target_repo,
        month_key=day.strftime("%Y-%m"),
        token=token,
        salt=salt,
        through_day=day,
    )
    return row


def _parse_hour(value: str) -> datetime:
    if len(value) == 13:
        value = value + ":00:00"
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KPI dataset v2 rollups")
    parser.add_argument("--source-repo", default="smolagents/ml-intern-sessions")
    parser.add_argument("--target-repo", default="smolagents/ml-intern-kpis")
    parser.add_argument(
        "--hours", type=int, default=1, help="Number of completed hours to build"
    )
    parser.add_argument("--datetime", help="Explicit UTC hour, e.g. 2026-04-24T14")
    parser.add_argument(
        "--daily-backfill",
        type=int,
        default=0,
        help="Backfill N UTC days ending yesterday",
    )
    parser.add_argument("--month", help="Build one monthly rollup, e.g. 2026-06")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token = os.environ.get("HF_KPI_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_KPI_WRITE_TOKEN or HF_TOKEN is required")
        return 2
    try:
        salt = _require_hash_salt()
    except RuntimeError as e:
        logger.error("%s", e)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()
    if args.month:
        _write_monthly_rollup(
            api,
            source_repo=args.source_repo,
            target_repo=args.target_repo,
            month_key=args.month,
            token=token,
            salt=salt,
        )
        return 0

    if args.daily_backfill:
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        for offset in range(args.daily_backfill):
            day = yesterday - timedelta(days=offset)
            run_for_day(
                api,
                source_repo=args.source_repo,
                target_repo=args.target_repo,
                day=day,
                token=token,
                salt=salt,
            )
        return 0

    if args.datetime:
        hours = [_parse_hour(args.datetime)]
    else:
        last_completed = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ) - timedelta(hours=1)
        hours = [
            last_completed - timedelta(hours=offset) for offset in range(args.hours)
        ]

    for hour_dt in hours:
        run_for_hour(
            api,
            source_repo=args.source_repo,
            target_repo=args.target_repo,
            hour_dt=hour_dt,
            token=token,
            salt=salt,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
