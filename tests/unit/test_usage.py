from datetime import UTC, datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from usage import (  # noqa: E402
    aggregate_usage_events,
    build_usage_response,
    resolve_usage_windows,
)


def _event(event_type, data=None, created_at="2026-06-01T12:00:00+00:00"):
    return {
        "event_type": event_type,
        "data": data or {},
        "timestamp": created_at,
    }


def test_aggregate_usage_events_sums_inference_and_jobs():
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
    ]

    usage = aggregate_usage_events(events, session_id="s1")

    assert usage["session_id"] == "s1"
    assert usage["llm_calls"] == 2
    assert usage["hf_jobs_count"] == 1
    assert usage["prompt_tokens"] == 110
    assert usage["completion_tokens"] == 50
    assert usage["cache_read_tokens"] == 25
    assert usage["cache_creation_tokens"] == 5
    assert usage["total_tokens"] == 190
    assert usage["hf_jobs_billable_seconds_estimate"] == 1800
    assert usage["inference_usd"] == 0.375
    assert usage["hf_jobs_estimated_usd"] == 1.5
    assert usage["total_usd"] == 1.875


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


def test_usage_windows_respect_browser_timezone():
    windows = resolve_usage_windows(
        "America/Los_Angeles",
        now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
    )

    assert windows["timezone"] == "America/Los_Angeles"
    assert windows["today_start_utc"] == datetime(2026, 6, 1, 7, 0, tzinfo=UTC)
    assert windows["month_start_utc"] == datetime(2026, 6, 1, 7, 0, tzinfo=UTC)


class _NoopStore:
    enabled = False


class _Manager:
    def __init__(self, sessions):
        self.sessions = sessions

    def _store(self):
        return _NoopStore()


def _agent_session(session_id, user_id, events):
    return SimpleNamespace(
        session_id=session_id,
        user_id=user_id,
        session=SimpleNamespace(logged_events=events),
    )


@pytest.mark.asyncio
async def test_runtime_usage_excludes_other_users():
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

    assert usage["today"]["llm_calls"] == 1
    assert usage["today"]["inference_usd"] == 0.5
    assert usage["month"]["inference_usd"] == 0.5


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
    assert usage["today"]["inference_usd"] == 0.0


@pytest.mark.asyncio
async def test_runtime_usage_interprets_naive_timestamps_in_browser_timezone():
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
    assert usage["today"]["llm_calls"] == 1
    assert usage["month"]["llm_calls"] == 1
    assert usage["today"]["total_tokens"] == 42
