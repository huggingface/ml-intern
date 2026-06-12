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
    aggregate_usage_events,
    build_usage_response,
    resolve_usage_windows,
)
from agent.core import session_persistence  # noqa: E402
from agent.core.usage_metrics import summarize_usage_events  # noqa: E402


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


def test_usage_event_type_allowlists_include_sandbox_lifecycle():
    assert set(USAGE_EVENT_TYPES) >= {
        "hf_job_submit",
        "sandbox_create",
        "sandbox_destroy",
    }
    assert set(session_persistence.USAGE_EVENT_TYPES) >= {
        "hf_job_submit",
        "sandbox_create",
        "sandbox_destroy",
    }


def test_summarize_usage_events_aggregates_llm_turns_and_quality():
    usage = summarize_usage_events(
        [
            _event(
                "llm_call",
                {
                    "kind": "main",
                    "model": "moonshotai/Kimi-K2.6",
                    "cost_usd": 0.125,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cache_read_tokens": 25,
                    "cache_creation_tokens": 5,
                    "total_tokens": 180,
                },
            ),
            _event(
                "llm_call",
                {
                    "kind": "research",
                    "model": "anthropic/claude-opus-4.8",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            ),
            {
                "event_type": "llm_call",
                "data": {
                    "kind": "main",
                    "model": "moonshotai/Kimi-K2.6",
                    "cost_usd": 0,
                },
            },
            _event("turn_complete"),
            _event("assistant_stream_end"),
        ],
        session_id="s1",
    )

    assert usage["version"] == 1
    assert usage["session_id"] == "s1"
    assert usage["llm"]["calls"] == 3
    assert usage["llm"]["calls_by_kind"] == {"main": 2, "research": 1}
    assert usage["llm"]["calls_by_model"] == {
        "anthropic/claude-opus-4.8": 1,
        "moonshotai/Kimi-K2.6": 2,
    }
    assert usage["llm"]["prompt_tokens"] == 110
    assert usage["llm"]["completion_tokens"] == 55
    assert usage["llm"]["cache_read_tokens"] == 25
    assert usage["llm"]["cache_creation_tokens"] == 5
    assert usage["llm"]["total_tokens"] == 195
    assert usage["turns"] == {
        "turn_complete_count": 1,
        "assistant_stream_end_count": 1,
    }
    assert usage["data_quality"]["events_without_timestamp"] == 1
    assert usage["data_quality"]["llm_calls_with_cost_field"] == 2
    assert usage["data_quality"]["llm_calls_with_nonzero_cost"] == 1


def test_summarize_usage_events_aggregates_jobs_and_sandbox_lifecycle():
    usage = summarize_usage_events(
        [
            _event("hf_job_submit", {"flavor": "t4-small"}),
            _event(
                "hf_job_complete",
                {
                    "flavor": "t4-small",
                    "final_status": "success",
                    "estimated_cost_usd": 0.4,
                    "billable_seconds_estimate": 120,
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
                {
                    "sandbox_id": "alice/sandbox-paid",
                    "hardware": "t4-small",
                },
                created_at="2026-06-01T12:00:00+00:00",
            ),
            _event(
                "sandbox_destroy",
                {
                    "sandbox_id": "alice/sandbox-paid",
                    "lifetime_s": 1800,
                },
                created_at="2026-06-01T12:30:00+00:00",
            ),
            _event(
                "sandbox_create",
                {
                    "sandbox_id": "alice/sandbox-active",
                    "hardware": "a100-large",
                },
            ),
            _event("sandbox_destroy", {"sandbox_id": "alice/sandbox-missing"}),
        ]
    )

    assert usage["hf_jobs"]["submits"] == 1
    assert usage["hf_jobs"]["status_snapshots"] == 2
    assert usage["hf_jobs"]["statuses"] == {"failed": 1, "success": 1}
    assert usage["hf_jobs"]["flavors"] == {"t4-small": 1}
    assert usage["hf_jobs"]["submit_flavors"] == {"t4-small": 1}
    assert usage["hf_jobs"]["status_snapshot_flavors"] == {
        "cpu-basic": 1,
        "t4-small": 1,
    }
    assert usage["hf_jobs"]["estimated_usd"] == 0.4
    assert usage["hf_jobs"]["billable_seconds_estimate"] == 150
    assert usage["hf_jobs"]["snapshots_with_estimated_cost"] == 1
    assert usage["hf_jobs"]["snapshots_with_nonzero_estimated_cost"] == 1
    assert usage["sandboxes"]["creates"] == 2
    assert usage["sandboxes"]["destroys"] == 2
    assert usage["sandboxes"]["matched_pairs"] == 1
    assert usage["sandboxes"]["unpaired_creates"] == 1
    assert usage["sandboxes"]["unpaired_destroys"] == 1
    assert usage["sandboxes"]["hardware"] == {"a100-large": 1, "t4-small": 1}
    assert usage["sandboxes"]["estimated_usd"] == 0.3
    assert usage["sandboxes"]["billable_seconds_estimate"] == 1800
    assert usage["data_quality"]["job_snapshots_with_estimated_cost"] == 1
    assert usage["data_quality"]["job_snapshots_missing_estimated_cost"] == 1


def test_summarize_usage_events_uses_hf_billing_plus_sandbox_when_available():
    usage = summarize_usage_events(
        [
            _event("llm_call", {"cost_usd": 0.25, "total_tokens": 42}),
            _event("hf_job_complete", {"estimated_cost_usd": 0.5}),
            _event(
                "sandbox_create",
                {
                    "sandbox_id": "alice/sandbox-paid",
                    "hardware": "t4-small",
                },
            ),
            _event(
                "sandbox_destroy",
                {
                    "sandbox_id": "alice/sandbox-paid",
                    "lifetime_s": 3600,
                },
            ),
        ],
        hf_billing_snapshot={
            "hf_billing": {
                "available": True,
                "source": "hf_billing_usage_v2",
                "current_session": {
                    "window_start": "2026-06-01T12:00:00Z",
                    "window_end": "2026-06-01T13:00:00Z",
                    "timezone": "UTC",
                    "total_usd": 2.5,
                    "inference_providers_usd": 2.0,
                    "hf_jobs_usd": 0.5,
                    "inference_provider_requests": 3,
                    "hf_jobs_minutes": 12.5,
                    "monthly_total_usd": 99,
                },
                "month": {"total_usd": 99},
                "inference_providers_credits": {"limit_usd": 100},
            }
        },
    )

    assert usage["app_total_usd"] == 1.35
    assert usage["hf_billing_total_usd"] == 2.5
    assert usage["total_usd"] == 3.1
    assert usage["total_usd_source"] == "hf_billing_plus_sandbox_estimate"
    assert usage["hf_billing"]["current_session"]["total_usd"] == 2.5
    assert "month" not in usage["hf_billing"]
    assert "inference_providers_credits" not in usage["hf_billing"]
    assert "monthly_total_usd" not in usage["hf_billing"]["current_session"]


def test_summarize_usage_events_falls_back_to_app_telemetry_without_hf_billing():
    usage = summarize_usage_events(
        [
            _event("llm_call", {"cost_usd": 0.25, "total_tokens": 42}),
            _event("hf_job_complete", {"estimated_cost_usd": 0.5}),
        ],
        hf_billing_snapshot={
            "hf_billing": {
                "available": False,
                "error": "missing_hf_token",
                "current_session": None,
            }
        },
    )

    assert usage["total_usd"] == 0.75
    assert usage["app_total_usd"] == 0.75
    assert usage["hf_billing_total_usd"] is None
    assert usage["total_usd_source"] == "app_telemetry_fallback"
    assert usage["hf_billing"]["available"] is False
    assert usage["hf_billing"]["error"] == "missing_hf_token"


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
        session=SimpleNamespace(logged_events=events),
    )


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
async def test_hf_account_usage_uses_usage_window_for_current_delta(monkeypatch):
    session_created_at = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    usage_window_started_at = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                created_at=session_created_at,
                usage_window_started_at=usage_window_started_at,
                session=SimpleNamespace(logged_events=[]),
            )
        }
    )
    calls = []

    async def fake_fetch(_token, *, start, end):
        calls.append((start, end))
        if start == usage_window_started_at:
            used_nano = 500_000_000
        else:
            used_nano = 2_000_000_000
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": used_nano,
                    "includedNanoUsd": 2_000_000_000,
                    "limitNanoUsd": 5_000_000_000,
                    "numRequests": 4,
                },
                "jobs": {"usedMicroUsd": 0, "totalMinutes": 0},
            }
        }

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_fetch)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["available"] is True
    assert usage["hf_account"]["current_session"]["inference_providers_usd"] == 0.5
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
    assert {start for start, _ in calls} == {
        datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
        usage_window_started_at,
    }


@pytest.mark.asyncio
async def test_hf_account_usage_uses_baseline_for_current_delta(monkeypatch):
    usage_window_started_at = datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    month_start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    manager = _Manager(
        {
            "s1": SimpleNamespace(
                session_id="s1",
                user_id="owner",
                created_at=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
                usage_window_started_at=usage_window_started_at,
                usage_window_baseline={
                    "captured_at": usage_window_started_at,
                    "month_start": month_start,
                    "total_usd": 2.5,
                    "inference_providers_usd": 2.0,
                    "hf_jobs_usd": 0.5,
                    "inference_provider_requests": 10,
                    "hf_jobs_minutes": 4.0,
                },
                session=SimpleNamespace(logged_events=[]),
            )
        }
    )
    calls = []

    async def fake_fetch(_token, *, start, end):
        calls.append((start, end))
        if start == month_start:
            return {
                "usage": {
                    "inferenceProviders": {
                        "usedNanoUsd": 2_000_000_000,
                        "numRequests": 10,
                    },
                    "jobs": {"usedMicroUsd": 500_000, "totalMinutes": 4.0},
                }
            }
        return {
            "usage": {
                "inferenceProviders": {
                    "usedNanoUsd": 5_000_000_000,
                    "numRequests": 99,
                },
                "jobs": {"usedMicroUsd": 0, "totalMinutes": 0},
            }
        }

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_fetch)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert usage["hf_account"]["current_session"]["total_usd"] == 0.0
    assert usage["hf_account"]["current_session"]["inference_provider_requests"] == 0
    assert usage_window_started_at not in {start for start, _ in calls}


@pytest.mark.asyncio
async def test_hf_account_usage_falls_back_to_persisted_created_at(monkeypatch):
    session_created_at = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    store = _MetadataStore({"created_at": session_created_at})
    manager = _Manager({}, store=store)
    calls = []

    async def fake_fetch(_token, *, start, end):
        calls.append((start, end))
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

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_fetch)

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
    assert {start for start, _ in calls} == {
        datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
        session_created_at,
    }


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
                session=SimpleNamespace(logged_events=[]),
            )
        },
        store=store,
    )
    billing_starts = []

    async def fake_fetch(_token, *, start, end):
        billing_starts.append(start)
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

    monkeypatch.setattr("usage._fetch_hf_billing_usage_v2", fake_fetch)

    usage = await build_usage_response(
        manager,
        user_id="owner",
        hf_token="hf_fake",
        session_id="s1",
        timezone_name="UTC",
        now=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
    )

    assert store.calls == [("owner", {"session_id": "s1", "start": None, "end": None})]
    assert set(billing_starts) == {
        datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
        session_created_at,
    }
    assert datetime(2026, 6, 5, 0, 0, tzinfo=UTC) not in billing_starts
    assert usage["hf_account"]["month"]["inference_providers_usd"] == 0.0
