from types import SimpleNamespace

import pytest

from agent.core.agent_loop import Handlers
from agent.core.session import Event
from agent.core.usage_thresholds import (
    USAGE_THRESHOLD_TOOL_NAME,
    next_usage_warning_threshold,
)


class FakeUsageApprovalSession:
    def __init__(self, *, continuation="continue_agent"):
        self.pending_approval = {
            "kind": USAGE_THRESHOLD_TOOL_NAME,
            "tool_call_id": "usage-threshold-1",
            "threshold_usd": 5.0,
            "current_spend_usd": 12.25,
            "next_threshold_usd": 10.0,
            "billing_source": "app_telemetry_session",
            "continuation": continuation,
            "history_size": 3,
            "final_response": "done",
        }
        self.context_manager = SimpleNamespace(items=[])
        self.usage_warning_next_threshold_usd = 5.0
        self.events: list[Event] = []
        self.turn_count = 0
        self.auto_saved = False

    async def send_event(self, event: Event):
        self.events.append(event)

    def increment_turn(self):
        self.turn_count += 1

    async def auto_save_if_needed(self):
        self.auto_saved = True


def test_next_usage_warning_threshold_advances_past_current_spend():
    assert next_usage_warning_threshold(4.99, 5.0) == 5.0
    assert next_usage_warning_threshold(5.0, 5.0) == 10.0
    assert next_usage_warning_threshold(12.25, 5.0) == 20.0
    assert next_usage_warning_threshold(40.0, 20.0) == 80.0


@pytest.mark.asyncio
async def test_usage_threshold_approval_resumes_agent(monkeypatch):
    session = FakeUsageApprovalSession(continuation="continue_agent")
    resumed = False

    async def fake_run_agent(run_session, text):
        nonlocal resumed
        assert run_session is session
        assert text == ""
        resumed = True

    monkeypatch.setattr(Handlers, "run_agent", fake_run_agent)

    await Handlers.exec_approval(
        session,
        [{"tool_call_id": "usage-threshold-1", "approved": True}],
    )

    assert session.pending_approval is None
    assert resumed is True
    assert session.usage_warning_next_threshold_usd == 20.0
    assert [event.event_type for event in session.events] == [
        "tool_state_change",
        "tool_output",
    ]
    assert session.events[-1].data["success"] is True


@pytest.mark.asyncio
async def test_usage_threshold_approval_completes_finished_turn(monkeypatch):
    session = FakeUsageApprovalSession(continuation="complete_turn")

    async def fail_run_agent(*args, **kwargs):
        raise AssertionError("complete_turn must not call run_agent")

    monkeypatch.setattr(Handlers, "run_agent", fail_run_agent)

    await Handlers.exec_approval(
        session,
        [{"tool_call_id": "usage-threshold-1", "approved": True}],
    )

    assert session.pending_approval is None
    assert [event.event_type for event in session.events] == [
        "tool_state_change",
        "tool_output",
        "turn_complete",
    ]
    assert session.events[-1].data == {
        "history_size": 3,
        "final_response": "done",
    }
    assert session.turn_count == 1
    assert session.auto_saved is True


@pytest.mark.asyncio
async def test_usage_threshold_rejection_stops_turn(monkeypatch):
    session = FakeUsageApprovalSession()

    async def fail_run_agent(*args, **kwargs):
        raise AssertionError("rejection must not call run_agent")

    monkeypatch.setattr(Handlers, "run_agent", fail_run_agent)

    await Handlers.exec_approval(
        session,
        [{"tool_call_id": "usage-threshold-1", "approved": False}],
    )

    assert session.pending_approval is None
    assert [event.event_type for event in session.events] == [
        "tool_state_change",
        "tool_output",
        "interrupted",
    ]
    assert session.events[1].data["success"] is False
    assert session.turn_count == 1
    assert session.auto_saved is True


@pytest.mark.asyncio
async def test_abandon_complete_turn_usage_threshold_completes_prior_turn():
    session = FakeUsageApprovalSession(continuation="complete_turn")

    await Handlers._abandon_pending_approval(session)

    assert session.pending_approval is None
    assert [event.event_type for event in session.events] == [
        "tool_state_change",
        "turn_complete",
    ]
    assert session.events[-1].data == {
        "history_size": 3,
        "final_response": "done",
    }
    assert session.turn_count == 1
    assert session.auto_saved is True
