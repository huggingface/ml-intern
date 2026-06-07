from types import SimpleNamespace

import pytest

from agent.core import telemetry


class FakeSession:
    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


def test_extract_usage_reads_hf_router_cache_write_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_write_tokens=20,
            ),
        )
    )

    usage = telemetry.extract_usage(response)

    assert usage["cache_read_tokens"] == 80
    assert usage["cache_creation_tokens"] == 20


@pytest.mark.asyncio
async def test_record_hf_job_complete_emits_runtime_cost(monkeypatch):
    async def fake_catalog():
        return {"a100-large": 4.0}

    monkeypatch.setattr(telemetry, "hf_jobs_price_catalog", fake_catalog)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: 130.0)
    session = FakeSession()

    await telemetry.record_hf_job_complete(
        session,
        SimpleNamespace(id="job-1"),
        flavor="a100-large",
        final_status="COMPLETED",
        submit_ts=100.0,
    )

    event = session.events[0]
    assert event.event_type == "hf_job_complete"
    assert event.data["wall_time_s"] == 30
    assert event.data["billable_seconds_estimate"] == 30
    assert event.data["price_usd_per_hour"] == 4.0
    assert event.data["estimated_cost_usd"] == round(4.0 * 30 / 3600, 4)
    assert event.data["cost_estimate_source"] == "runtime_price_catalog"


@pytest.mark.asyncio
async def test_record_hf_job_complete_keeps_unknown_hardware_countable(monkeypatch):
    async def fake_catalog():
        return {}

    monkeypatch.setattr(telemetry, "hf_jobs_price_catalog", fake_catalog)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: 160.0)
    session = FakeSession()

    await telemetry.record_hf_job_complete(
        session,
        SimpleNamespace(id="job-2"),
        flavor="future-gpu",
        final_status="FAILED",
        submit_ts=100.0,
    )

    data = session.events[0].data
    assert data["wall_time_s"] == 60
    assert data["billable_seconds_estimate"] == 60
    assert data["price_usd_per_hour"] is None
    assert data["estimated_cost_usd"] is None
    assert data["cost_estimate_source"] == "unknown_price"
