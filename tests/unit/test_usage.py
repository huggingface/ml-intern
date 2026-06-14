from datetime import UTC, datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from usage import (  # noqa: E402
    USAGE_EVENT_TYPES,
    _account_bucket_from_billing_usage,
    _session_bucket_from_inference_session_usage,
    aggregate_usage_events,
    build_hf_billing_snapshot,
    build_usage_response,
    resolve_usage_windows,
)
from agent.core import session_persistence  # noqa: E402
from agent.core.usage_metrics import (  # noqa: E402
    summarize_usage_events,
    usage_metric_scalar_fields,
)

BILLING_SESSION_ID = "00000000-0000-4000-8000-000000000001"


def _event(event_type, data=None, created_at="2026-06-01T12:00:00+00:00"):
    return {
        "event_type": event_type,
        "data": data or {},
        "timestamp": created_at,
    }


def test_aggregate_usage_events_sums_inference_jobs_and_sandboxes():
    events = [
        _event(
            "llm_call",
            {
                "cost_usd": 0.125,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cache_read_tokens": 25,
                "cache_creation_tokens": 5,
                "total_tokens": 180,
            },
        ),
        _event("llm_call", {"cost_usd": 0.25, "prompt_tokens": 10}),
        _event(
            "hf_job_complete",
            {
                "estimated_cost_usd": 1.5,
                "billable_seconds_estimate": 1800,
            },
        ),
        _event(
            "sandbox_create",
            {
                "sandbox_id": "alice/sandbox-12345678",
                "hardware": "cpu-upgrade",
            },
            created_at="2026-06-01T12:30:00+00:00",
        ),
        _event(
            "sandbox_destroy",
            {
                "sandbox_id": "alice/sandbox-12345678",
                "lifetime_s": 3600,
            },
            created_at="2026-06-01T13:30:00+00:00",
        ),
    ]

    usage = aggregate_usage_events(events, session_id="s1")

    assert usage["session_id"] == "s1"
    assert usage["llm_calls"] == 2
    assert usage["hf_jobs_count"] == 1
    assert usage["sandbox_count"] == 1
    assert usage["prompt_tokens"] == 110
    assert usage["completion_tokens"] == 50
    assert usage["cache_read_tokens"] == 25
    assert usage["cache_creation_tokens"] == 5
    assert usage["total_tokens"] == 190
    assert usage["hf_jobs_billable_seconds_estimate"] == 1800
    assert usage["sandbox_billable_seconds_estimate"] == 3600
    assert usage["inference_usd"] == 0.375
    assert usage["hf_jobs_estimated_usd"] == 1.5
    assert usage["sandbox_estimated_usd"] == 0.05
    assert usage["total_usd"] == 1.925


def test_aggregate_usage_events_treats_missing_costs_as_zero():
    usage = aggregate_usage_events(
        [
            _event("llm_call", {"prompt_tokens": 7}),
            _event("hf_job_complete", {"wall_time_s": 60}),
        ]
    )

    assert usage["llm_calls"] == 1
    assert usage["hf_jobs_count"] == 1
    assert usage["prompt_tokens"] == 7
    assert usage["hf_jobs_billable_seconds_estimate"] == 60
    assert usage["total_usd"] == 0.0


def test_aggregate_usage_events_ignores_active_sandbox_before_destroy():
    usage = aggregate_usage_events(
        [
            _event(
                "sandbox_create",
                {
                    "sandbox_id": "alice/sandbox-12345678",
                    "hardware": "a100-large",
                },
            )
        ]
    )

    assert usage["sandbox_count"] == 0
    assert usage["sandbox_estimated_usd"] == 0.0
    assert usage["sandbox_billable_seconds_estimate"] == 0
    assert usage["total_usd"] == 0.0


def test_aggregate_usage_events_counts_cpu_basic_sandbox_as_free():
    usage = aggregate_usage_events(
        [
            _event(
                "sandbox_create",
                {
                    "sandbox_id": "alice/sandbox-12345678",
                    "hardware": "cpu-basic",
                },
            ),
            _event(
                "sandbox_destroy",
                {
                    "sandbox_id": "alice/sandbox-12345678",
                    "lifetime_s": 3600,
                },
            ),
        ]
    )

    assert usage["sandbox_count"] == 1
    assert usage["sandbox_estimated_usd"] == 0.0
    assert usage["sandbox_billable_seconds_estimate"] == 0
    assert usage["total_usd"] == 0.0


def test_aggregate_usage_events_falls_back_to_sandbox_timestamps():
    usage = aggregate_usage_events(
        [
            _event(
                "sandbox_create",
                {
                    "sandbox_id": "alice/sandbox-12345678",
                    "hardware": "t4-small",
                },
                created_at="2026-06-01T12:00:00+00:00",
            ),
            _event(
                "sandbox_destroy",
                {"sandbox_id": "alice/sandbox-12345678"},
                created_at="2026-06-01T12:30:00+00:00",
            ),
        ]
    )

    assert usage["sandbox_count"] == 1
    assert usage["sandbox_billable_seconds_estimate"] == 1800
    assert usage["sandbox_estimated_usd"] == 0.3
    assert usage["total_usd"] == 0.3


def test_sandbox_lifecycle_pairing_is_shared_for_duplicate_creates():
    events = [
        _event(
            "sandbox_create",
            {"sandbox_id": "alice/sandbox-reused", "hardware": "t4-small"},
            created_at="2026-06-01T12:00:00+00:00",
        ),
        _event(
            "sandbox_create",
            {"sandbox_id": "alice/sandbox-reused", "hardware": "cpu-basic"},
            created_at="2026-06-01T12:05:00+00:00",
        ),
        _event(
            "sandbox_destroy",
            {"sandbox_id": "alice/sandbox-reused", "lifetime_s": 300},
            created_at="2026-06-01T12:10:00+00:00",
        ),
        _event(
            "sandbox_destroy",
            {"sandbox_id": "alice/sandbox-reused", "lifetime_s": 1200},
            created_at="2026-06-01T12:20:00+00:00",
        ),
    ]

    usage = aggregate_usage_events(events, session_id="s1")
    metrics = summarize_usage_events(events, session_id="s1")

    assert usage["sandbox_count"] == 2
    assert usage["sandbox_billable_seconds_estimate"] == 1200
    assert usage["sandbox_estimated_usd"] == 0.2
    assert metrics["sandboxes"]["matched_pairs"] == usage["sandbox_count"]
    assert metrics["sandboxes"]["unpaired_creates"] == 0
    assert metrics["sandboxes"]["unpaired_destroys"] == 0
    assert metrics["sandboxes"]["estimated_usd"] == usage["sandbox_estimated_usd"]


def test_usage_event_type_allowlists_include_sandbox_lifecycle():
    assert set(USAGE_EVENT_TYPES) >= {"sandbox_create", "sandbox_destroy"}
    assert set(session_persistence.USAGE_EVENT_TYPES) >= {
        "sandbox_create",
        "sandbox_destroy",
    }


def test_summarize_usage_events_aggregates_dataset_analytics():
    events = [
        _event(
            "llm_call",
            {
                "model": "model-a",
                "kind": "main",
                "cost_usd": 0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_tokens": 2,
            },
        ),
        _event(
            "llm_call",
            {
                "model": "model-b",
                "kind": "research",
                "cost_usd": 0.125,
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "cache_creation_tokens": 3,
                "total_tokens": 40,
            },
        ),
        _event("hf_job_submit", {"flavor": "a10g-small"}),
        _event(
            "hf_job_complete",
            {
                "flavor": "a10g-small",
                "final_status": "succeeded",
                "estimated_cost_usd": 0.5,
                "billable_seconds_estimate": 600,
            },
        ),
        _event(
            "hf_job_complete",
            {
                "flavor": "cpu-basic",
                "final_status": "failed",
                "wall_time_s": 30,
            },
        ),
        _event(
            "sandbox_create",
            {"sandbox_id": "alice/sandbox-1", "hardware": "t4-small"},
            created_at="2026-06-01T12:00:00+00:00",
        ),
        _event(
            "sandbox_destroy",
            {"sandbox_id": "alice/sandbox-1", "lifetime_s": 1800},
            created_at="2026-06-01T12:30:00+00:00",
        ),
        _event(
            "sandbox_create",
            {"sandbox_id": "alice/sandbox-2", "hardware": "a100-large"},
            created_at="2026-06-01T13:00:00+00:00",
        ),
        _event(
            "sandbox_destroy",
            {"sandbox_id": "alice/sandbox-missing", "lifetime_s": 60},
            created_at="2026-06-01T13:05:00+00:00",
        ),
        _event("turn_complete"),
        _event("assistant_stream_end"),
        {"event_type": "debug", "data": {}},
    ]

    metrics = summarize_usage_events(events, session_id="s1")

    assert metrics["version"] == 1
    assert metrics["total_usd_source"] == "app_telemetry_fallback"
    assert metrics["total_usd"] == 0.925
    assert metrics["llm"] == {
        "calls": 2,
        "calls_by_kind": {"main": 1, "research": 1},
        "calls_by_model": {"model-a": 1, "model-b": 1},
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "cache_read_tokens": 2,
        "cache_creation_tokens": 3,
        "total_tokens": 57,
    }
    assert metrics["turns"] == {
        "turn_complete_count": 1,
        "assistant_stream_end_count": 1,
    }
    assert metrics["hf_jobs"]["submits"] == 1
    assert metrics["hf_jobs"]["status_snapshots"] == 2
    assert metrics["hf_jobs"]["statuses"] == {"failed": 1, "succeeded": 1}
    assert metrics["hf_jobs"]["flavors"] == {
        "a10g-small": 2,
        "cpu-basic": 1,
    }
    assert metrics["hf_jobs"]["estimated_usd"] == 0.5
    assert metrics["hf_jobs"]["billable_seconds_estimate"] == 630
    assert metrics["sandboxes"]["creates"] == 2
    assert metrics["sandboxes"]["destroys"] == 2
    assert metrics["sandboxes"]["matched_pairs"] == 1
    assert metrics["sandboxes"]["unpaired_creates"] == 1
    assert metrics["sandboxes"]["unpaired_destroys"] == 1
    assert metrics["sandboxes"]["hardware"] == {"a100-large": 1, "t4-small": 1}
    assert metrics["sandboxes"]["estimated_usd"] == 0.3
    assert metrics["sandboxes"]["billable_seconds_estimate"] == 1800
    assert metrics["data_quality"] == {
        "event_count": 12,
        "events_without_timestamp": 1,
        "llm_calls_with_cost_usd": 2,
        "llm_calls_with_nonzero_cost_usd": 1,
        "job_snapshots_with_estimated_cost": 1,
        "job_snapshots_missing_estimated_cost": 1,
    }

    assert usage_metric_scalar_fields(metrics) == {
        "usage_total_usd": 0.925,
        "usage_total_usd_source": "app_telemetry_fallback",
        "usage_app_total_usd": 0.925,
        "usage_hf_billing_total_usd": None,
        "usage_llm_calls": 2,
        "usage_total_tokens": 57,
        "usage_hf_job_submits": 1,
        "usage_hf_job_status_snapshots": 2,
        "usage_sandbox_creates": 2,
        "usage_sandbox_pairs": 1,
    }


def test_summarize_usage_events_uses_hf_billing_plus_sandbox_when_available():
    events = [
        _event("llm_call", {"cost_usd": 99.0, "total_tokens": 10}),
        _event("hf_job_complete", {"estimated_cost_usd": 99.0}),
        _event(
            "sandbox_create",
            {"sandbox_id": "alice/sandbox-1", "hardware": "t4-small"},
            created_at="2026-06-01T12:00:00+00:00",
        ),
        _event(
            "sandbox_destroy",
            {"sandbox_id": "alice/sandbox-1", "lifetime_s": 1800},
            created_at="2026-06-01T12:30:00+00:00",
        ),
    ]

    metrics = summarize_usage_events(
        events,
        session_id="s1",
        hf_billing_snapshot={
            "hf_billing": {
                "source": "hf_billing_usage_v2",
                "available": True,
                "current_session": {
                    "window_start": "2026-06-01T12:00:00Z",
                    "window_end": "2026-06-01T12:30:00Z",
                    "timezone": "UTC",
                    "total_usd": 1.25,
                    "inference_providers_usd": 1.0,
                    "hf_jobs_usd": 0.25,
                    "inference_provider_requests": 3,
                    "hf_jobs_minutes": 1.5,
                    "unexpected": "dropped",
                },
            },
            "month": {"total_usd": 999},
            "inference_providers_credits": {"limit_usd": 999},
        },
    )

    assert metrics["total_usd"] == 1.55
    assert metrics["total_usd_source"] == "hf_billing_plus_sandbox_estimate"
    assert metrics["app_total_usd"] == 198.3
    assert metrics["hf_billing_total_usd"] == 1.25
    assert metrics["hf_billing"] == {
        "source": "hf_billing_usage_v2",
        "available": True,
        "error": None,
        "current_session": {
            "window_start": "2026-06-01T12:00:00Z",
            "window_end": "2026-06-01T12:30:00Z",
            "timezone": "UTC",
            "total_usd": 1.25,
            "inference_providers_usd": 1.0,
            "hf_jobs_usd": 0.25,
            "inference_provider_requests": 3,
            "hf_jobs_minutes": 1.5,
        },
    }


def test_account_bucket_from_hf_billing_usage_v2():
    usage = _account_bucket_from_billing_usage(
        {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 1_500_000_000,
                    "numRequests": 12,
                },
                "jobs": {
                    "usedMicroUsd": 250_000,
                    "totalMinutes": 3.5,
                },
            }
        },
        window_start=datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
        window_end=datetime(2026, 6, 1, 1, 0, tzinfo=UTC),
        timezone="UTC",
    )

    assert usage["inference_providers_usd"] == 1.5
    assert usage["hf_jobs_usd"] == 0.25
    assert usage["total_usd"] == 1.75
    assert usage["inference_provider_requests"] == 12
    assert usage["hf_jobs_minutes"] == 3.5


def test_session_bucket_from_inference_session_usage_sums_periods():
    usage = _session_bucket_from_inference_session_usage(
        {
            "currency": "USD",
            "periods": [
                {
                    "period": "2026-06-01T00:00:00.000Z",
                    "sessions": [
                        {"id": "s1", "requestCount": 2, "costCents": 123.4},
                        {"id": "s2", "requestCount": 20, "costCents": 999},
                    ],
                },
                {
                    "period": "2026-07-01T00:00:00.000Z",
                    "sessions": [
                        {"id": "s1", "requestCount": 3, "costCents": 76.6},
                    ],
                },
            ],
        },
        session_id="s1",
        window_start=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
        window_end=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
        timezone="UTC",
    )

    assert usage["inference_providers_usd"] == 2.0
    assert usage["inference_provider_requests"] == 5
    assert usage["hf_jobs_usd"] == 0.0
    assert usage["total_usd"] == 2.0


def test_session_bucket_from_inference_session_usage_handles_no_matching_session():
    usage = _session_bucket_from_inference_session_usage(
        {
            "currency": "USD",
            "periods": [
                {
                    "period": "2026-06-01T00:00:00.000Z",
                    "sessions": [{"id": "other", "requestCount": 1, "costCents": 50}],
                },
            ],
        },
        session_id="s1",
        window_start=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
        window_end=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
        timezone="UTC",
    )

    assert usage["inference_providers_usd"] == 0.0
    assert usage["inference_provider_requests"] == 0
    assert usage["total_usd"] == 0.0


def test_usage_windows_respect_browser_timezone():
    windows = resolve_usage_windows(
        "America/Los_Angeles",
        now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
    )

    assert windows["timezone"] == "America/Los_Angeles"
    assert windows["month_start_utc"] == datetime(2026, 6, 1, 7, 0, tzinfo=UTC)


class _NoopStore:
    enabled = False


class _RecordingStore:
    enabled = True

    def __init__(self):
        self.calls = []

    async def load_usage_events(self, user_id, **kwargs):
        self.calls.append((user_id, kwargs))
        return []


class _Manager:
    def __init__(self, sessions, store=None):
        self.sessions = sessions
        self.store = store or _NoopStore()

    def _store(self):
        return self.store


class _MetadataStore(_NoopStore):
    enabled = True

    def __init__(self, metadata):
        self.metadata = metadata

    async def load_session(self, session_id):
        return {"metadata": {"session_id": session_id, **self.metadata}, "messages": []}

    async def load_usage_events(self, user_id, **kwargs):
        return []


def _agent_session(session_id, user_id, events):
    return SimpleNamespace(
        session_id=session_id,
        user_id=user_id,
        inference_billing_session_id=BILLING_SESSION_ID,
        session=SimpleNamespace(logged_events=events),
    )


@pytest.mark.asyncio
async def test_hf_billing_snapshot_uses_session_window_without_account_totals(
    monkeypatch,
):
    usage_window_started_at = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                created_at=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
                usage_window_started_at=usage_window_started_at,
                session=SimpleNamespace(logged_events=[]),
            )
        }
    )
    calls = []

    async def fake_usage_v2(_token, *, start, end):
        calls.append((start, end))
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 1_500_000_000,
                    "includedNanoUsd": 2_000_000_000,
                    "limitNanoUsd": 5_000_000_000,
                    "numRequests": 4,
                },
                "jobs": {"usedMicroUsd": 250_000, "totalMinutes": 3.5},
            }
        }

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_usage_v2)

    snapshot = await build_hf_billing_snapshot(
        manager,
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert calls == [(usage_window_started_at, datetime(2026, 6, 5, 13, 0, tzinfo=UTC))]
    assert snapshot == {
        "billing_scope": "account_window_delta",
        "hf_billing": {
            "source": "hf_billing_usage_v2",
            "available": True,
            "error": None,
            "current_session": {
                "window_start": "2026-06-05T12:30:00Z",
                "window_end": "2026-06-05T13:00:00Z",
                "timezone": "UTC",
                "total_usd": 1.75,
                "inference_providers_usd": 1.5,
                "hf_jobs_usd": 0.25,
                "inference_provider_requests": 4,
                "hf_jobs_minutes": 3.5,
            },
        },
    }


@pytest.mark.asyncio
async def test_usage_response_omits_app_rollups_without_session():
    manager = _Manager(
        {
            "owner-session": _agent_session(
                "owner-session",
                "owner",
                [_event("llm_call", {"cost_usd": 0.5})],
            ),
            "other-session": _agent_session(
                "other-session",
                "other",
                [_event("llm_call", {"cost_usd": 99.0})],
            ),
        }
    )

    usage = await build_usage_response(
        manager,
        user_id="owner",
        session_id=None,
        timezone_name="UTC",
        now=datetime(2026, 6, 1, 13, 0, tzinfo=UTC),
    )

    assert usage["session"] is None


@pytest.mark.asyncio
async def test_runtime_usage_includes_requested_session_total():
    manager = _Manager(
        {
            "s1": _agent_session(
                "s1",
                "owner",
                [
                    _event(
                        "llm_call",
                        {"cost_usd": 0.25},
                        created_at="2026-05-01T12:00:00+00:00",
                    )
                ],
            )
        }
    )

    usage = await build_usage_response(
        manager,
        user_id="owner",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 1, 13, 0, tzinfo=UTC),
    )

    assert usage["session"]["session_id"] == "s1"
    assert usage["session"]["inference_usd"] == 0.25


@pytest.mark.asyncio
async def test_runtime_usage_clips_requested_session_to_usage_window():
    usage_window_started_at = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                usage_window_started_at=usage_window_started_at,
                session=SimpleNamespace(
                    logged_events=[
                        _event(
                            "llm_call",
                            {"cost_usd": 99.0},
                            created_at="2026-06-05T12:00:00+00:00",
                        ),
                        _event(
                            "llm_call",
                            {"cost_usd": 0.25},
                            created_at="2026-06-05T12:45:00+00:00",
                        ),
                        _event(
                            "hf_job_complete",
                            {
                                "estimated_cost_usd": 1.5,
                                "billable_seconds_estimate": 1800,
                            },
                            created_at="2026-06-05T12:50:00+00:00",
                        ),
                    ]
                ),
            )
        }
    )

    usage = await build_usage_response(
        manager,
        user_id="owner",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["session"]["inference_usd"] == 0.25
    assert usage["session"]["hf_jobs_estimated_usd"] == 1.5
    assert usage["session"]["total_usd"] == 1.75


@pytest.mark.asyncio
async def test_runtime_usage_includes_requested_session_tokens():
    manager = _Manager(
        {
            "s1": _agent_session(
                "s1",
                "owner",
                [
                    _event(
                        "llm_call",
                        {"cost_usd": 0.25, "total_tokens": 42},
                        created_at="2026-06-05T15:00:00",
                    )
                ],
            )
        }
    )

    usage = await build_usage_response(
        manager,
        user_id="owner",
        session_id="s1",
        timezone_name="Europe/Zurich",
        now=datetime(2026, 6, 5, 13, 30, tzinfo=UTC),
    )

    assert usage["session"]["llm_calls"] == 1
    assert usage["session"]["total_tokens"] == 42


@pytest.mark.asyncio
async def test_hf_account_usage_reports_missing_token_error():
    manager = _Manager({})

    usage = await build_usage_response(
        manager,
        user_id="owner",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["available"] is False
    assert usage["hf_account"]["error"] == "missing_hf_token"


@pytest.mark.asyncio
async def test_hf_account_usage_reports_billing_unavailable_error(monkeypatch):
    manager = _Manager({})

    async def fake_fetch(_token, *, start, end):
        return None

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_fetch)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["available"] is False
    assert usage["hf_account"]["error"] == "billing_usage_unavailable"


@pytest.mark.asyncio
async def test_hf_account_usage_uses_session_endpoint_for_current_session(
    monkeypatch,
):
    session_created_at = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    usage_window_started_at = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                created_at=session_created_at,
                usage_window_started_at=usage_window_started_at,
                inference_billing_session_id=BILLING_SESSION_ID,
                session=SimpleNamespace(logged_events=[]),
            )
        }
    )
    usage_v2_calls = []
    session_usage_calls = []

    async def fake_usage_v2(_token, *, start, end):
        usage_v2_calls.append((start, end))
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 2_000_000_000,
                    "includedNanoUsd": 2_000_000_000,
                    "limitNanoUsd": 5_000_000_000,
                    "numRequests": 4,
                },
                "jobs": {"usedMicroUsd": 0, "totalMinutes": 0},
            }
        }

    async def fake_session_usage(_token, *, start, end):
        session_usage_calls.append((start, end))
        return {
            "currency": "USD",
            "periods": [
                {
                    "period": "2026-06-01T00:00:00.000Z",
                    "sessions": [
                        {
                            "id": BILLING_SESSION_ID,
                            "requestCount": 3,
                            "costCents": 125,
                        },
                        {"id": "s1", "requestCount": 11, "costCents": 500},
                        {"id": "other", "requestCount": 9, "costCents": 999},
                    ],
                }
            ],
        }

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_usage_v2)
    monkeypatch.setattr("usage._fetch_hf_inference_session_usage", fake_session_usage)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["available"] is True
    assert usage["hf_account"]["current_session"]["inference_providers_usd"] == 1.25
    assert usage["hf_account"]["current_session"]["inference_provider_requests"] == 3
    assert usage["hf_account"]["month"]["inference_providers_usd"] == 2.0
    assert usage["hf_account"]["inference_providers_credits"] == {
        "included_usd": 2.0,
        "used_usd": 2.0,
        "remaining_included_usd": 0.0,
        "limit_usd": 5.0,
        "remaining_limit_usd": 3.0,
        "num_requests": 4,
        "period_start": None,
        "period_end": None,
    }
    assert usage_v2_calls == [
        (
            datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
        )
    ]
    assert session_usage_calls == [
        (usage_window_started_at, datetime(2026, 6, 5, 13, 0, tzinfo=UTC))
    ]


@pytest.mark.asyncio
async def test_hf_account_usage_keeps_account_available_when_session_endpoint_fails(
    monkeypatch,
):
    usage_window_started_at = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                created_at=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
                usage_window_started_at=usage_window_started_at,
                inference_billing_session_id=BILLING_SESSION_ID,
                session=SimpleNamespace(logged_events=[]),
            )
        }
    )

    async def fake_usage_v2(_token, *, start, end):
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 2_000_000_000,
                    "numRequests": 10,
                },
                "jobs": {"usedMicroUsd": 500_000, "totalMinutes": 4.0},
            }
        }

    async def fake_session_usage(_token, *, start, end):
        return None

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_usage_v2)
    monkeypatch.setattr("usage._fetch_hf_inference_session_usage", fake_session_usage)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["available"] is True
    assert usage["hf_account"]["current_session"] is None
    assert usage["hf_account"]["month"]["inference_providers_usd"] == 2.0
    assert "error" not in usage["hf_account"]


@pytest.mark.asyncio
async def test_hf_account_usage_falls_back_to_persisted_created_at(monkeypatch):
    session_created_at = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    store = _MetadataStore(
        {
            "created_at": session_created_at,
            "inference_billing_session_id": BILLING_SESSION_ID,
        }
    )
    manager = _Manager({}, store=store)
    usage_v2_calls = []
    session_usage_calls = []

    async def fake_usage_v2(_token, *, start, end):
        usage_v2_calls.append((start, end))
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 0,
                    "includedNanoUsd": 0,
                    "limitNanoUsd": 0,
                },
                "jobs": {"usedMicroUsd": 0, "totalMinutes": 0},
            }
        }

    async def fake_session_usage(_token, *, start, end):
        session_usage_calls.append((start, end))
        return {"currency": "USD", "periods": []}

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_usage_v2)
    monkeypatch.setattr("usage._fetch_hf_inference_session_usage", fake_session_usage)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["current_session"]["window_start"] == (
        "2026-06-05T12:00:00Z"
    )
    assert usage_v2_calls == [
        (
            datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
        )
    ]
    assert session_usage_calls == [
        (session_created_at, datetime(2026, 6, 5, 13, 0, tzinfo=UTC))
    ]


@pytest.mark.asyncio
async def test_usage_response_loads_only_session_events(monkeypatch):
    session_created_at = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    store = _RecordingStore()
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                created_at=session_created_at,
                inference_billing_session_id=BILLING_SESSION_ID,
                session=SimpleNamespace(logged_events=[]),
            )
        },
        store=store,
    )
    usage_v2_starts = []
    session_usage_starts = []

    async def fake_usage_v2(_token, *, start, end):
        usage_v2_starts.append(start)
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 0,
                    "includedNanoUsd": 2_000_000_000,
                    "limitNanoUsd": 0,
                },
                "jobs": {"usedMicroUsd": 0, "totalMinutes": 0},
            }
        }

    async def fake_session_usage(_token, *, start, end):
        session_usage_starts.append(start)
        return {"currency": "USD", "periods": []}

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_usage_v2)
    monkeypatch.setattr("usage._fetch_hf_inference_session_usage", fake_session_usage)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert store.calls == [
        (
            "owner",
            {"session_id": "s1", "start": session_created_at, "end": None},
        )
    ]
    assert usage_v2_starts == [datetime(2026, 6, 1, 0, 0, tzinfo=UTC)]
    assert session_usage_starts == [session_created_at]
    assert datetime(2026, 6, 5, 0, 0, tzinfo=UTC) not in usage_v2_starts
    assert datetime(2026, 6, 5, 0, 0, tzinfo=UTC) not in session_usage_starts
    assert usage["hf_account"]["month"]["inference_providers_usd"] == 0.0
