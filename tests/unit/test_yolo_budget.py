from types import SimpleNamespace

import pytest
from litellm import Message

from agent.context_manager import manager as context_manager
from agent.core.cost_estimation import CostEstimate
from agent.core.hf_router_catalog import ModelInfo, ProviderInfo
from agent.core.session import Event
from agent.core import yolo_budget


def _provider(name: str, input_price: float, output_price: float) -> ProviderInfo:
    return ProviderInfo(
        provider=name,
        status="live",
        context_length=128_000,
        input_price=input_price,
        output_price=output_price,
        supports_tools=True,
        supports_structured_output=False,
    )


def _model_info() -> ModelInfo:
    return ModelInfo(
        id="org/model",
        providers=[
            _provider("provider-a", 1.0, 10.0),
            _provider("provider-b", 2.0, 3.0),
        ],
    )


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


@pytest.fixture
def priced_router(monkeypatch):
    monkeypatch.setattr(yolo_budget, "_count_prompt_tokens", lambda **_: 1_000_000)
    monkeypatch.setattr(
        "agent.core.hf_router_catalog.lookup",
        lambda _model: _model_info(),
    )


def test_llm_estimate_uses_provider_pinned_price(priced_router):
    estimate = yolo_budget.estimate_llm_call_cost(
        model_name="org/model:provider-b",
        litellm_model="openai/org/model:provider-b",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=1_000_000,
    )

    assert estimate.estimated_cost_usd == 5.0
    assert estimate.billable is True


def test_llm_estimate_uses_cheapest_route_price(priced_router):
    estimate = yolo_budget.estimate_llm_call_cost(
        model_name="org/model:cheapest",
        litellm_model="openai/org/model:cheapest",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=1_000_000,
    )

    assert estimate.estimated_cost_usd == 5.0


def test_llm_estimate_uses_highest_live_price_for_auto_route(priced_router):
    estimate = yolo_budget.estimate_llm_call_cost(
        model_name="org/model",
        litellm_model="openai/org/model",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=1_000_000,
    )

    assert estimate.estimated_cost_usd == 12.0


def test_llm_estimate_treats_local_models_as_free(monkeypatch):
    monkeypatch.setattr(
        "agent.core.hf_router_catalog.lookup",
        lambda _model: pytest.fail("local models must not hit the router catalog"),
    )

    estimate = yolo_budget.estimate_llm_call_cost(
        model_name="ollama/llama3",
        litellm_model="openai/llama3",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=1_000_000,
    )

    assert estimate.estimated_cost_usd == 0.0
    assert estimate.billable is False


def test_llm_estimate_fails_closed_when_price_is_unknown(monkeypatch):
    monkeypatch.setattr(yolo_budget, "_count_prompt_tokens", lambda **_: 10)
    monkeypatch.setattr("agent.core.hf_router_catalog.lookup", lambda _model: None)
    monkeypatch.setattr("agent.core.hf_router_catalog.last_fetch_error", lambda: None)

    estimate = yolo_budget.estimate_llm_call_cost(
        model_name="org/unknown",
        litellm_model="openai/org/unknown",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=100,
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "No HF Router price" in estimate.block_reason


def test_llm_estimate_reports_router_catalog_fetch_failure(monkeypatch):
    monkeypatch.setattr(yolo_budget, "_count_prompt_tokens", lambda **_: 10)
    monkeypatch.setattr("agent.core.hf_router_catalog.lookup", lambda _model: None)
    monkeypatch.setattr(
        "agent.core.hf_router_catalog.last_fetch_error",
        lambda: "timed out",
    )

    estimate = yolo_budget.estimate_llm_call_cost(
        model_name="org/model",
        litellm_model="openai/org/model",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=100,
    )

    assert estimate.estimated_cost_usd is None
    assert "Could not fetch HF Router price metadata" in estimate.block_reason


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


def test_yolo_output_bound_uses_remaining_router_context(monkeypatch):
    session = _session()
    monkeypatch.setattr(yolo_budget, "_count_prompt_tokens", lambda **_: 120_000)
    monkeypatch.setattr(yolo_budget, "_router_context_length", lambda _model: 128_000)

    params = yolo_budget.with_yolo_llm_output_bound(
        {"model": "openai/org/model"},
        session,
        model_name="org/model",
        litellm_model="openai/org/model",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert params["max_completion_tokens"] == 8000


@pytest.mark.asyncio
async def test_llm_budget_pause_happens_before_summarization_call(monkeypatch):
    session = _session(cap=0.5, spent=0.0)
    monkeypatch.setattr(yolo_budget, "_count_prompt_tokens", lambda **_: 1_000_000)
    monkeypatch.setattr(yolo_budget, "_router_prices", lambda _model: (1.0, 1.0))

    async def fail_acompletion(*args, **kwargs):
        raise AssertionError("acompletion must not be called when YOLO blocks")

    monkeypatch.setattr(context_manager, "acompletion", fail_acompletion)

    with pytest.raises(yolo_budget.YoloBudgetPaused):
        await context_manager.summarize_messages(
            [Message(role="user", content="summarize me")],
            model_name="org/model",
            max_tokens=100,
            session=session,
        )

    assert session.pending_approval["kind"] == yolo_budget.YOLO_BUDGET_TOOL_NAME
    assert session.events[0].event_type == "approval_required"
    assert session.events[0].data["yolo_budget"] is True
