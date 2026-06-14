"""Session-scoped YOLO budget guardrails."""

import uuid
from dataclasses import dataclass
from typing import Any

from agent.core.cost_estimation import CostEstimate

YOLO_BUDGET_TOOL_NAME = "yolo_budget"


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


def _cap_usd(session: Any | None) -> float | None:
    if not session or getattr(session, "auto_approval_cost_cap_usd", None) is None:
        return None
    return max(0.0, float(getattr(session, "auto_approval_cost_cap_usd") or 0.0))


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
    *,
    allow_zero_actual: bool = False,
) -> None:
    if not session or not reservation_id:
        return
    reservation = _reservation_store(session).pop(reservation_id, None)
    if reservation is None:
        return
    actual = _coerce_cost(actual_cost_usd)
    if actual is None or (actual == 0 and not allow_zero_actual):
        return
    adjust_session_spend(session, actual - reservation.amount_usd)


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
    current_spend_usd: float | None = None,
    cap_usd: float | None = None,
    billing_source: str | None = None,
    continuation: str | None = None,
    final_response: str | None = None,
    history_size: int | None = None,
) -> bool:
    if session.pending_approval:
        return False
    from agent.core.session import Event

    current_spend = (
        session_spend_usd(session)
        if current_spend_usd is None
        else max(0.0, float(current_spend_usd))
    )
    cap = getattr(session, "auto_approval_cost_cap_usd", None)
    if cap_usd is not None:
        cap = max(0.0, float(cap_usd))
    pending = {
        "kind": YOLO_BUDGET_TOOL_NAME,
        "tool_call_id": f"yolo-budget-{uuid.uuid4().hex[:10]}",
        "cap_usd": cap,
        "current_spend_usd": round(current_spend, 6),
        "remaining_cap_usd": decision.remaining_cap_usd,
        "estimated_next_usd": decision.estimated_cost_usd,
        "spend_kind": spend_kind,
        "reason": decision.block_reason or "YOLO budget requires confirmation.",
        "history_size": history_size
        if history_size is not None
        else len(session.context_manager.items),
    }
    if billing_source:
        pending["billing_source"] = billing_source
    if continuation:
        pending["continuation"] = continuation
    if isinstance(final_response, str):
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


async def request_yolo_budget_exceeded_approval(
    session: Any,
    *,
    spend_kind: str,
    current_spend_usd: float,
    cap_usd: float,
    billing_source: str | None = None,
    reason: str | None = None,
    continuation: str | None = None,
    final_response: str | None = None,
    history_size: int | None = None,
) -> bool:
    current_spend = max(0.0, float(current_spend_usd))
    cap = max(0.0, float(cap_usd))
    seed_session_spend(session, current_spend)
    if not session_yolo_enabled(session) or current_spend < cap:
        return False
    decision = BudgetDecision(
        allowed=False,
        estimated_cost_usd=None,
        remaining_cap_usd=round(max(0.0, cap - current_spend), 4),
        block_reason=reason
        or (
            "YOLO cap paused session usage after "
            f"{spend_kind}: current session spend ${current_spend:.2f} "
            f"has reached the ${cap:.2f} cap."
        ),
        billable=True,
    )
    return await request_yolo_budget_approval(
        session,
        decision,
        spend_kind=spend_kind,
        current_spend_usd=current_spend,
        cap_usd=cap,
        billing_source=billing_source,
        continuation=continuation,
        final_response=final_response,
        history_size=history_size,
    )


async def maybe_pause_yolo_after_spend(
    session: Any | None,
    *,
    spend_kind: str,
    observed_cost_usd: Any = None,
    continuation: str | None = None,
    final_response: str | None = None,
) -> bool:
    if not session or not session_yolo_enabled(session) or session.pending_approval:
        return False

    observed = _coerce_cost(observed_cost_usd)
    if observed is not None and observed > 0:
        add_session_spend(session, observed)

    checker = getattr(session, "yolo_budget_checker", None)
    if checker is not None:
        try:
            return bool(
                await checker(
                    {
                        "spend_kind": spend_kind,
                        "observed_cost_usd": observed,
                        "continuation": continuation,
                        "final_response": final_response,
                        "history_size": len(session.context_manager.items),
                    }
                )
            )
        except Exception:
            pass

    cap = _cap_usd(session)
    current_spend = session_spend_usd(session)
    if cap is None or current_spend < cap:
        return False
    return await request_yolo_budget_exceeded_approval(
        session,
        spend_kind=spend_kind,
        current_spend_usd=current_spend,
        cap_usd=cap,
        continuation=continuation,
        final_response=final_response,
        history_size=len(session.context_manager.items),
    )


def yolo_budget_can_resume(
    session: Any, pending: dict[str, Any]
) -> tuple[bool, str | None]:
    if not session_yolo_enabled(session):
        return True, None
    estimated_next = _coerce_cost(pending.get("estimated_next_usd"))
    remaining = session_remaining_usd(session)
    if estimated_next is None:
        if remaining is None or remaining > 0:
            return True, None
        return (
            False,
            str(
                pending.get("reason")
                or "YOLO cap is reached. Raise or disable the cap to continue."
            ),
        )
    if remaining is not None and estimated_next > remaining:
        return (
            False,
            f"Estimated cost ${estimated_next:.2f} exceeds remaining YOLO cap ${remaining:.2f}.",
        )
    return True, None
