from types import SimpleNamespace

import pytest
from litellm import Message

from agent.context_manager import manager as context_manager
from agent.core.cost_estimation import CostEstimate
from agent.core.session import Event
from agent.core import yolo_budget


def _session(*, enabled: bool = True, cap: float = 5.0, spent: float = 0.0):
    class FakeSession:
        def __init__(self):
            self.auto_approval_enabled = enabled
            self.auto_approval_cost_cap_usd = cap
            self.auto_approval_estimated_spend_usd = spent
            self.pending_approval = None
            self.context_manager = SimpleNamespace(items=[])
            self.events: list[Event] = []
            self.session_id = "budget-session"

        async def send_event(self, event: Event) -> None:
            self.events.append(event)

    return FakeSession()


def test_reservation_reconcile_replaces_estimate_with_actual_cost():
    session = _session(cap=3.0, spent=1.0)

    decision = yolo_budget.reserve_session_budget(
        session,
        CostEstimate(estimated_cost_usd=1.25, billable=True),
        spend_kind="sandbox",
        reservation_id="sandbox-1",
    )

    assert decision.allowed is True
    assert session.auto_approval_estimated_spend_usd == 2.25

    yolo_budget.reconcile_budget_reservation(session, "sandbox-1", 0.5)

    assert session.auto_approval_estimated_spend_usd == 1.5


def test_zero_cost_reconcile_retains_reserved_estimate_by_default():
    session = _session(cap=3.0, spent=1.0)

    yolo_budget.reserve_session_budget(
        session,
        CostEstimate(estimated_cost_usd=1.25, billable=True),
        spend_kind="llm_call",
        reservation_id="llm-1",
    )

    yolo_budget.reconcile_budget_reservation(session, "llm-1", 0.0)

    assert session.auto_approval_estimated_spend_usd == 2.25


def test_zero_cost_reconcile_can_release_measured_zero_runtime_cost():
    session = _session(cap=3.0, spent=1.0)

    yolo_budget.reserve_session_budget(
        session,
        CostEstimate(estimated_cost_usd=1.25, billable=True),
        spend_kind="sandbox",
        reservation_id="sandbox-1",
    )

    yolo_budget.reconcile_budget_reservation(
        session,
        "sandbox-1",
        0.0,
        allow_zero_actual=True,
    )

    assert session.auto_approval_estimated_spend_usd == 1.0


@pytest.mark.asyncio
async def test_summarization_call_runs_before_yolo_cap_check(monkeypatch):
    session = _session(cap=0.5, spent=0.0)
    calls: list[str] = []

    class FakeUsage:
        completion_tokens = 1

    class FakeChoice:
        finish_reason = "stop"
        message = SimpleNamespace(content="summary")

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    async def fake_acompletion(*args, **kwargs):
        calls.append("acompletion")
        return FakeResponse()

    async def fake_checker(payload):
        calls.append(str(payload["spend_kind"]))
        return False

    async def fake_record_llm_call(*args, **kwargs):
        return {}

    session.yolo_budget_checker = fake_checker
    monkeypatch.setattr(context_manager, "acompletion", fake_acompletion)
    monkeypatch.setattr("agent.core.telemetry.record_llm_call", fake_record_llm_call)

    summary, tokens = await context_manager.summarize_messages(
        [Message(role="user", content="summarize me")],
        model_name="org/unknown",
        max_tokens=100,
        session=session,
    )

    assert summary == "summary"
    assert tokens == 1
    assert calls == ["acompletion", "compaction"]


@pytest.mark.asyncio
async def test_post_call_yolo_pause_created_when_observed_spend_reaches_cap():
    session = _session(cap=1.0, spent=0.95)

    paused = await yolo_budget.maybe_pause_yolo_after_spend(
        session,
        spend_kind="llm_call",
        observed_cost_usd=0.05,
    )

    assert paused is True
    assert session.pending_approval["kind"] == yolo_budget.YOLO_BUDGET_TOOL_NAME
    assert session.pending_approval["estimated_next_usd"] is None
    assert session.events[0].event_type == "approval_required"
