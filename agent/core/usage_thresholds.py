"""Helpers for session usage-threshold approval warnings."""

from typing import Any

USAGE_THRESHOLD_TOOL_NAME = "usage_threshold"
USAGE_WARNING_FIRST_THRESHOLD_USD = 5.0
USAGE_WARNING_MULTIPLIER = 2.0


def normalize_usage_threshold(value: Any) -> float:
    """Return a usable positive threshold, defaulting to the first warning."""
    if isinstance(value, bool):
        return USAGE_WARNING_FIRST_THRESHOLD_USD
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return USAGE_WARNING_FIRST_THRESHOLD_USD
    if threshold <= 0:
        return USAGE_WARNING_FIRST_THRESHOLD_USD
    return threshold


def next_usage_warning_threshold(
    current_spend_usd: float,
    acknowledged_threshold_usd: float,
) -> float:
    """Advance the next threshold until it is above the current spend."""
    threshold = normalize_usage_threshold(acknowledged_threshold_usd)
    current = max(0.0, float(current_spend_usd or 0.0))
    while threshold <= current:
        threshold *= USAGE_WARNING_MULTIPLIER
    return round(threshold, 4)


def is_usage_threshold_pending(pending_approval: Any) -> bool:
    return (
        isinstance(pending_approval, dict)
        and pending_approval.get("kind") == USAGE_THRESHOLD_TOOL_NAME
    )


def usage_threshold_pending_to_tool(pending_approval: dict[str, Any]) -> dict[str, Any]:
    """Represent a synthetic usage approval as the existing pending-tool shape."""
    tool_call_id = str(pending_approval.get("tool_call_id") or "")
    arguments = {
        "threshold_usd": pending_approval.get("threshold_usd"),
        "current_spend_usd": pending_approval.get("current_spend_usd"),
        "next_threshold_usd": pending_approval.get("next_threshold_usd"),
        "billing_source": pending_approval.get("billing_source"),
    }
    return {
        "tool": USAGE_THRESHOLD_TOOL_NAME,
        "tool_call_id": tool_call_id,
        "arguments": arguments,
    }
