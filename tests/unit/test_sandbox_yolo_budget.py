import asyncio
import time

import pytest

from agent.core.cost_estimation import CostEstimate
from agent.core.yolo_budget import reserve_session_budget
from agent.tools import sandbox_tool


class FakeSandbox:
    def __init__(self):
        self._owns_space = True
        self.space_id = "owner/sandbox-1234abcd"
        self.url = "https://example.test"
        self.deleted = False

    def delete(self, log=None):
        self.deleted = True
        if log:
            log("deleted")


class FakeSession:
    def __init__(self, *, cap: float = 5.0):
        self.auto_approval_enabled = True
        self.auto_approval_cost_cap_usd = cap
        self.auto_approval_estimated_spend_usd = 0.0
        self.sandbox = FakeSandbox()
        self.sandbox_hardware = "a10g-large"
        self.sandbox_preload_task = None
        self._sandbox_yolo_reservation_id = None
        self._sandbox_yolo_finalized_cost_usd = 0.0
        self._sandbox_created_at = time.monotonic()
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_sandbox_early_teardown_releases_unused_reservation():
    session = FakeSession(cap=5.0)
    reserve_session_budget(
        session,
        CostEstimate(estimated_cost_usd=2.0, billable=True),
        spend_kind="sandbox",
        reservation_id="sandbox-call",
    )
    session._sandbox_yolo_reservation_id = "sandbox-call"
    session._sandbox_created_at = time.monotonic() - 1800

    await sandbox_tool.teardown_session_sandbox(session)

    assert session.auto_approval_estimated_spend_usd == 1.0
    assert session.sandbox is None
    assert session.sandbox_hardware is None
    assert session._sandbox_yolo_reservation_id is None


@pytest.mark.asyncio
async def test_sandbox_renewal_tears_down_when_cap_expires(monkeypatch):
    session = FakeSession(cap=2.0)
    reserve_session_budget(
        session,
        CostEstimate(estimated_cost_usd=2.0, billable=True),
        spend_kind="sandbox",
        reservation_id="sandbox-call",
    )
    session._sandbox_yolo_reservation_id = "sandbox-call"
    session._sandbox_created_at = time.monotonic() - 3600
    monkeypatch.setattr(sandbox_tool, "_sandbox_yolo_renewal_delay_s", lambda: 0.01)

    sandbox_tool._start_sandbox_yolo_renewal(
        session,
        hardware="a10g-large",
        reservation_id="sandbox-call",
    )

    await asyncio.wait_for(session._sandbox_yolo_renewal_task, timeout=1)

    assert session.sandbox is None
    assert session.sandbox_hardware is None
    assert session._sandbox_yolo_reservation_id is None
    assert session.auto_approval_estimated_spend_usd == 2.0
    assert any(event.event_type == "tool_log" for event in session.events)
