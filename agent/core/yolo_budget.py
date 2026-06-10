"""Session-scoped YOLO budget guardrails."""

import os
import uuid
from dataclasses import dataclass
from typing import Any

from agent.core.cost_estimation import CostEstimate
from agent.core.local_models import local_model_provider
from agent.core.model_ids import strip_huggingface_model_prefix

YOLO_BUDGET_TOOL_NAME = "yolo_budget"
DEFAULT_LLM_OUTPUT_TOKEN_RESERVE = int(
    os.environ.get("ML_INTERN_YOLO_LLM_OUTPUT_TOKEN_RESERVE", "8192")
)
_ROUTING_POLICIES = {"fastest", "cheapest", "preferred"}


class YoloBudgetPaused(Exception):
    """Raised when a session LLM call is paused by the YOLO budget."""


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    amount_usd: float
    spend_kind: str


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    estimated_cost_usd: float | None = None
    remaining_cap_usd: float | None = None
    block_reason: str | None = None
    billable: bool = False
    reservation: BudgetReservation | None = None


@dataclass(frozen=True)
class LlmBudgetResult:
    llm_params: dict[str, Any]
    decision: BudgetDecision
    blocked: bool = False


def session_yolo_enabled(session: Any | None) -> bool:
    return bool(session and getattr(session, "auto_approval_enabled", False))


def session_spend_usd(session: Any | None) -> float:
    if not session:
        return 0.0
    return max(
        0.0,
        float(getattr(session, "auto_approval_estimated_spend_usd", 0.0) or 0.0),
    )


def session_remaining_usd(
    session: Any | None, reserved_spend_usd: float = 0.0
) -> float | None:
    if not session or getattr(session, "auto_approval_cost_cap_usd", None) is None:
        return None
    cap = float(getattr(session, "auto_approval_cost_cap_usd") or 0.0)
    return round(max(0.0, cap - session_spend_usd(session) - reserved_spend_usd), 4)


def _set_session_spend(session: Any, amount_usd: float) -> None:
    session.auto_approval_estimated_spend_usd = round(max(0.0, amount_usd), 4)


def add_session_spend(session: Any, amount_usd: float | None) -> None:
    if amount_usd is None or amount_usd <= 0:
        return
    if hasattr(session, "add_auto_approval_estimated_spend"):
        session.add_auto_approval_estimated_spend(amount_usd)
    else:
        _set_session_spend(session, session_spend_usd(session) + float(amount_usd))


def adjust_session_spend(session: Any, delta_usd: float | None) -> None:
    if delta_usd is None or delta_usd == 0:
        return
    _set_session_spend(session, session_spend_usd(session) + float(delta_usd))


def seed_session_spend(session: Any, amount_usd: float | None) -> None:
    if amount_usd is None:
        return
    _set_session_spend(session, max(session_spend_usd(session), float(amount_usd)))


def _reservation_store(session: Any) -> dict[str, BudgetReservation]:
    store = getattr(session, "_yolo_budget_reservations", None)
    if not isinstance(store, dict):
        store = {}
        setattr(session, "_yolo_budget_reservations", store)
    return store


def _coerce_cost(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def check_session_budget(
    session: Any | None,
    estimate: CostEstimate,
    *,
    reserved_spend_usd: float = 0.0,
) -> BudgetDecision:
    if not session_yolo_enabled(session) or not estimate.billable:
        return BudgetDecision(
            allowed=True,
            estimated_cost_usd=estimate.estimated_cost_usd,
            billable=estimate.billable,
        )

    remaining = session_remaining_usd(session, reserved_spend_usd=reserved_spend_usd)
    amount = _coerce_cost(estimate.estimated_cost_usd)
    if amount is None:
        return BudgetDecision(
            allowed=False,
            estimated_cost_usd=None,
            remaining_cap_usd=remaining,
            block_reason=estimate.block_reason
            or "Could not estimate this session spend safely.",
            billable=True,
        )
    if remaining is not None and amount > remaining:
        return BudgetDecision(
            allowed=False,
            estimated_cost_usd=round(amount, 4),
            remaining_cap_usd=remaining,
            block_reason=(
                f"Estimated cost ${amount:.2f} exceeds remaining YOLO cap "
                f"${remaining:.2f}."
            ),
            billable=True,
        )
    return BudgetDecision(
        allowed=True,
        estimated_cost_usd=round(amount, 4),
        remaining_cap_usd=remaining,
        billable=True,
    )


def reserve_session_budget(
    session: Any | None,
    estimate: CostEstimate,
    *,
    spend_kind: str,
    reservation_id: str | None = None,
) -> BudgetDecision:
    decision = check_session_budget(session, estimate)
    if not session or not session_yolo_enabled(session) or not decision.billable:
        return decision
    if not decision.allowed:
        return decision
    amount = _coerce_cost(decision.estimated_cost_usd)
    if amount is None or amount <= 0:
        return decision

    add_session_spend(session, amount)
    rid = reservation_id or f"{spend_kind}-{uuid.uuid4().hex[:10]}"
    reservation = BudgetReservation(
        reservation_id=rid,
        amount_usd=round(amount, 4),
        spend_kind=spend_kind,
    )
    _reservation_store(session)[rid] = reservation
    return BudgetDecision(
        allowed=True,
        estimated_cost_usd=round(amount, 4),
        remaining_cap_usd=session_remaining_usd(session),
        billable=True,
        reservation=reservation,
    )


def release_budget_reservation(session: Any | None, reservation_id: str | None) -> None:
    if not session or not reservation_id:
        return
    reservation = _reservation_store(session).pop(reservation_id, None)
    if reservation is None:
        return
    adjust_session_spend(session, -reservation.amount_usd)


def reconcile_budget_reservation(
    session: Any | None,
    reservation_id: str | None,
    actual_cost_usd: Any,
) -> None:
    if not session or not reservation_id:
        return
    reservation = _reservation_store(session).pop(reservation_id, None)
    if reservation is None:
        return
    actual = _coerce_cost(actual_cost_usd)
    if actual is None:
        return
    adjust_session_spend(session, actual - reservation.amount_usd)


def llm_output_token_bound(llm_params: dict[str, Any]) -> int | None:
    for key in ("max_completion_tokens", "max_tokens"):
        value = llm_params.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            continue
        if tokens > 0:
            return tokens
    return None


def with_yolo_llm_output_bound(
    llm_params: dict[str, Any],
    session: Any | None,
) -> dict[str, Any]:
    if (
        not session_yolo_enabled(session)
        or llm_output_token_bound(llm_params) is not None
    ):
        return llm_params
    return {**llm_params, "max_completion_tokens": DEFAULT_LLM_OUTPUT_TOKEN_RESERVE}


def _message_payload(message: Any) -> Any:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return message


def _count_prompt_tokens(
    *,
    litellm_model: str,
    messages: list[Any],
    tools: Any = None,
) -> int | None:
    try:
        from litellm import token_counter

        payload = [_message_payload(message) for message in messages]
        try:
            return int(
                token_counter(model=litellm_model, messages=payload, tools=tools)
            )
        except TypeError:
            return int(token_counter(model=litellm_model, messages=payload))
    except Exception:
        return None


def _router_prices(model_name: str) -> tuple[float, float] | None:
    from agent.core import hf_router_catalog

    normalized = strip_huggingface_model_prefix(model_name) or model_name
    bare, _, tag = normalized.partition(":")
    info = hf_router_catalog.lookup(normalized)
    if info is None:
        return None
    live = [
        p
        for p in info.live_providers
        if p.input_price is not None and p.output_price is not None
    ]
    if not live:
        return None

    if tag and tag not in _ROUTING_POLICIES:
        pinned = [p for p in live if p.provider == tag]
        if not pinned:
            return None
        p = pinned[0]
        return float(p.input_price), float(p.output_price)

    if tag == "cheapest":
        p = min(
            live, key=lambda item: float(item.input_price) + float(item.output_price)
        )
        return float(p.input_price), float(p.output_price)

    # Auto/fastest/preferred routing can choose any live provider. Use the
    # highest advertised price per side so the reservation is a safe upper bound.
    return (
        max(float(p.input_price) for p in live),
        max(float(p.output_price) for p in live),
    )


def estimate_llm_call_cost(
    *,
    model_name: str,
    litellm_model: str,
    messages: list[Any],
    tools: Any = None,
    max_output_tokens: int | None,
) -> CostEstimate:
    normalized = strip_huggingface_model_prefix(model_name) or model_name
    if local_model_provider(normalized) is not None:
        return CostEstimate(estimated_cost_usd=0.0, billable=False, label=normalized)
    if max_output_tokens is None or max_output_tokens <= 0:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="No safe output-token bound is available for this LLM call.",
            label=normalized,
        )
    prompt_tokens = _count_prompt_tokens(
        litellm_model=litellm_model,
        messages=messages,
        tools=tools,
    )
    if prompt_tokens is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="Could not count prompt tokens for this LLM call.",
            label=normalized,
        )
    prices = _router_prices(normalized)
    if prices is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"No HF Router price is available for model '{normalized}'.",
            label=normalized,
        )
    input_price, output_price = prices
    estimated = (
        (prompt_tokens * input_price) + (max_output_tokens * output_price)
    ) / 1_000_000
    return CostEstimate(
        estimated_cost_usd=round(estimated, 4),
        billable=estimated > 0,
        label=normalized,
    )


def is_yolo_budget_pending(pending_approval: Any) -> bool:
    return (
        isinstance(pending_approval, dict)
        and pending_approval.get("kind") == YOLO_BUDGET_TOOL_NAME
    )


def yolo_budget_pending_to_tool(pending_approval: dict[str, Any]) -> dict[str, Any]:
    tool_call_id = str(pending_approval.get("tool_call_id") or "")
    arguments = {
        "cap_usd": pending_approval.get("cap_usd"),
        "current_spend_usd": pending_approval.get("current_spend_usd"),
        "remaining_cap_usd": pending_approval.get("remaining_cap_usd"),
        "estimated_next_usd": pending_approval.get("estimated_next_usd"),
        "spend_kind": pending_approval.get("spend_kind"),
        "reason": pending_approval.get("reason"),
    }
    return {
        "tool": YOLO_BUDGET_TOOL_NAME,
        "tool_call_id": tool_call_id,
        "arguments": arguments,
        "auto_approval_blocked": True,
        "block_reason": pending_approval.get("reason"),
        "estimated_cost_usd": pending_approval.get("estimated_next_usd"),
        "remaining_cap_usd": pending_approval.get("remaining_cap_usd"),
    }


async def request_yolo_budget_approval(
    session: Any,
    decision: BudgetDecision,
    *,
    spend_kind: str,
    continuation: str = "continue_agent",
    final_response: str | None = None,
) -> bool:
    if session.pending_approval:
        return False
    from agent.core.session import Event

    pending = {
        "kind": YOLO_BUDGET_TOOL_NAME,
        "tool_call_id": f"yolo-budget-{uuid.uuid4().hex[:10]}",
        "cap_usd": getattr(session, "auto_approval_cost_cap_usd", None),
        "current_spend_usd": round(session_spend_usd(session), 6),
        "remaining_cap_usd": decision.remaining_cap_usd,
        "estimated_next_usd": decision.estimated_cost_usd,
        "spend_kind": spend_kind,
        "reason": decision.block_reason or "YOLO budget requires confirmation.",
        "continuation": continuation,
        "history_size": len(session.context_manager.items),
    }
    if final_response is not None:
        pending["final_response"] = final_response
    session.pending_approval = pending
    tool = yolo_budget_pending_to_tool(pending)
    await session.send_event(
        Event(
            event_type="approval_required",
            data={
                "tools": [tool],
                "count": 1,
                "yolo_budget": True,
                "auto_approval_blocked": True,
                "block_reason": pending["reason"],
                "estimated_cost_usd": pending["estimated_next_usd"],
                "remaining_cap_usd": pending["remaining_cap_usd"],
            },
        )
    )
    return True


async def reserve_llm_call_budget(
    session: Any | None,
    *,
    model_name: str,
    messages: list[Any],
    tools: Any,
    llm_params: dict[str, Any],
    spend_kind: str,
) -> LlmBudgetResult:
    bounded_params = with_yolo_llm_output_bound(llm_params, session)
    if not session_yolo_enabled(session):
        return LlmBudgetResult(
            llm_params=bounded_params,
            decision=BudgetDecision(allowed=True),
        )

    estimate = estimate_llm_call_cost(
        model_name=model_name,
        litellm_model=str(bounded_params.get("model") or model_name),
        messages=messages,
        tools=tools,
        max_output_tokens=llm_output_token_bound(bounded_params),
    )
    decision = reserve_session_budget(
        session,
        estimate,
        spend_kind=spend_kind,
    )
    if decision.allowed:
        return LlmBudgetResult(llm_params=bounded_params, decision=decision)

    assert session is not None
    await request_yolo_budget_approval(
        session,
        decision,
        spend_kind=spend_kind,
    )
    return LlmBudgetResult(
        llm_params=bounded_params,
        decision=decision,
        blocked=True,
    )


def yolo_budget_can_resume(
    session: Any, pending: dict[str, Any]
) -> tuple[bool, str | None]:
    if not session_yolo_enabled(session):
        return True, None
    estimated_next = _coerce_cost(pending.get("estimated_next_usd"))
    if estimated_next is None:
        return False, str(
            pending.get("reason") or "Unknown-cost spend requires disabling YOLO."
        )
    remaining = session_remaining_usd(session)
    if remaining is not None and estimated_next > remaining:
        return (
            False,
            f"Estimated cost ${estimated_next:.2f} exceeds remaining YOLO cap ${remaining:.2f}.",
        )
    return True, None
