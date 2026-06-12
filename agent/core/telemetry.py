"""All agent observability in one module.

Every telemetry signal the agent emits — LLM-call usage / cost, hf_jobs
lifecycle, sandbox lifecycle, user feedback, mid-turn heartbeat saves — is
defined here so business-logic files stay free of instrumentation noise.

Callsites are one-liners::

    await telemetry.record_llm_call(session, model=..., response=r, ...)
    await telemetry.record_hf_job_submit(session, job, args, image=..., job_type="Python")
    HeartbeatSaver.maybe_fire(session)

All ``record_*`` functions emit a single ``Event`` via ``session.send_event``
and never raise — telemetry is best-effort and must not break the agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent.core.cost_estimation import hf_jobs_price_catalog

logger = logging.getLogger(__name__)


# ── usage extraction ────────────────────────────────────────────────────────


def extract_usage(response_or_chunk: Any) -> dict:
    """Flat usage dict from a litellm response or final-chunk usage object.

    Normalizes cache-token details across provider response shapes. Exposed
    under the stable keys ``cache_read_tokens`` / ``cache_creation_tokens``.
    """
    u = getattr(response_or_chunk, "usage", None)
    if u is None and isinstance(response_or_chunk, dict):
        u = response_or_chunk.get("usage")
    if u is None:
        return {}

    def _g(name, default=0):
        if isinstance(u, dict):
            return u.get(name, default) or default
        return getattr(u, name, default) or default

    prompt = _g("prompt_tokens")
    completion = _g("completion_tokens")
    total = _g("total_tokens") or (prompt + completion)

    cache_read = _g("cache_read_input_tokens")
    cache_creation = _g("cache_creation_input_tokens")
    details = _g("prompt_tokens_details", None)

    if not cache_read and details is not None:
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens", 0) or 0
        else:
            cache_read = getattr(details, "cached_tokens", 0) or 0
    if not cache_creation and details is not None:
        if isinstance(details, dict):
            cache_creation = details.get("cache_write_tokens", 0) or 0
        else:
            cache_creation = getattr(details, "cache_write_tokens", 0) or 0

    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "cache_read_tokens": int(cache_read),
        "cache_creation_tokens": int(cache_creation),
    }


# ── llm_call ────────────────────────────────────────────────────────────────


async def record_llm_call(
    session: Any,
    *,
    model: str,
    response: Any = None,
    latency_ms: int,
    finish_reason: str | None,
    kind: str = "main",
) -> dict:
    """Emit an ``llm_call`` event and return the extracted usage dict so
    callers can stash it on their result object if they want.

    ``kind`` tags the call site so downstream analytics can break spend
    down by category. Values currently emitted by the codebase:

    * ``main``        — agent loop turn (user-facing reply or tool follow-up)
    * ``research``    — research sub-agent inner loop (3 call sites)
    * ``compaction``  — context-window summary on overflow
    * ``effort_probe``— effort cascade walk on rejection / model switch
    * ``restore``     — session re-seed summary after a Space restart

    Pre-2026-04-29 only ``main`` calls were instrumented; observed gap on
    Cost Explorer was ~67%, with the other 5 call sites accounting for
    the rest. Tagging lets us split the dataset's ``total_cost_usd`` by
    category and validate against billing data.

    The ``/title`` and ``/health/llm`` diagnostic call sites are intentionally
    not instrumented because they have no session context and are tiny.
    """
    usage = extract_usage(response) if response is not None else {}
    cost_usd = 0.0
    if response is not None:
        try:
            from litellm import completion_cost

            cost_usd = float(completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost_usd = 0.0
    from agent.core.session import Event  # local import to avoid cycle

    try:
        payload = {
            "model": model,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "cost_usd": cost_usd,
            "kind": kind,
            **usage,
        }
        await session.send_event(
            Event(
                event_type="llm_call",
                data=payload,
            )
        )
    except Exception as e:
        logger.debug("record_llm_call failed (non-fatal): %s", e)
    return {"cost_usd": cost_usd, **usage}


# ── hf_jobs ────────────────────────────────────────────────────────────────


def _infer_push_to_hub(script_or_cmd: Any) -> bool:
    if not isinstance(script_or_cmd, str):
        return False
    return (
        "push_to_hub=True" in script_or_cmd
        or "push_to_hub=true" in script_or_cmd
        or "hub_model_id" in script_or_cmd
    )


async def record_hf_job_submit(
    session: Any,
    job: Any,
    args: dict,
    *,
    image: str,
    job_type: str,
) -> float:
    """Emit ``hf_job_submit``. Returns the monotonic start timestamp so the
    caller can pass it back into :func:`record_hf_job_complete`."""
    from agent.core.session import Event

    t_start = time.monotonic()
    try:
        script_text = args.get("script") or args.get("command") or ""
        await session.send_event(
            Event(
                event_type="hf_job_submit",
                data={
                    "job_id": getattr(job, "id", None),
                    "job_url": getattr(job, "url", None),
                    "flavor": args.get("hardware_flavor", "cpu-basic"),
                    "timeout": args.get("timeout", "30m"),
                    "job_type": job_type,
                    "image": image,
                    "namespace": args.get("namespace"),
                    "push_to_hub": _infer_push_to_hub(script_text),
                },
            )
        )
    except Exception as e:
        logger.debug("record_hf_job_submit failed (non-fatal): %s", e)
    return t_start


async def record_hf_job_complete(
    session: Any,
    job: Any,
    *,
    flavor: str,
    final_status: str,
    submit_ts: float,
) -> dict:
    from agent.core.session import Event

    try:
        wall_time_s = int(time.monotonic() - submit_ts)
        billable_seconds = max(0, wall_time_s)
        price_usd_per_hour = None
        estimated_cost_usd = None
        cost_estimate_source = "unknown_price"
        prices = await hf_jobs_price_catalog()
        if flavor in prices:
            price_usd_per_hour = float(prices[flavor])
            estimated_cost_usd = round(
                price_usd_per_hour * (billable_seconds / 3600),
                4,
            )
            cost_estimate_source = "runtime_price_catalog"
        payload = {
            "job_id": getattr(job, "id", None),
            "flavor": flavor,
            "final_status": final_status,
            "wall_time_s": wall_time_s,
            "billable_seconds_estimate": billable_seconds,
            "price_usd_per_hour": price_usd_per_hour,
            "estimated_cost_usd": estimated_cost_usd,
            "cost_estimate_source": cost_estimate_source,
        }
        await session.send_event(
            Event(
                event_type="hf_job_complete",
                data=payload,
            )
        )
        return payload
    except Exception as e:
        logger.debug("record_hf_job_complete failed (non-fatal): %s", e)
    return {}


# ── sandbox ─────────────────────────────────────────────────────────────────


async def record_sandbox_create(
    session: Any,
    sandbox: Any,
    *,
    hardware: str,
    create_latency_s: int,
) -> None:
    from agent.core.session import Event

    try:
        # Pin created-at on the session so record_sandbox_destroy can diff.
        session._sandbox_created_at = time.monotonic() - create_latency_s
        await session.send_event(
            Event(
                event_type="sandbox_create",
                data={
                    "sandbox_id": getattr(sandbox, "space_id", None),
                    "hardware": hardware,
                    "create_latency_s": int(create_latency_s),
                },
            )
        )
    except Exception as e:
        logger.debug("record_sandbox_create failed (non-fatal): %s", e)


async def record_sandbox_destroy(session: Any, sandbox: Any) -> dict:
    from agent.core.session import Event

    try:
        created = getattr(session, "_sandbox_created_at", None)
        lifetime_s = int(time.monotonic() - created) if created else None
        hardware = getattr(session, "sandbox_hardware", None) or "cpu-basic"
        estimated_cost_usd = None
        try:
            from agent.core.cost_estimation import SPACE_PRICE_USD_PER_HOUR

            price_usd_per_hour = SPACE_PRICE_USD_PER_HOUR.get(str(hardware))
            if price_usd_per_hour is not None and lifetime_s is not None:
                estimated_cost_usd = round(
                    float(price_usd_per_hour) * (max(0, lifetime_s) / 3600),
                    4,
                )
        except Exception:
            estimated_cost_usd = None
        payload = {
            "sandbox_id": getattr(sandbox, "space_id", None),
            "hardware": hardware,
            "lifetime_s": lifetime_s,
            "estimated_cost_usd": estimated_cost_usd,
        }
        await session.send_event(
            Event(
                event_type="sandbox_destroy",
                data=payload,
            )
        )
        return payload
    except Exception as e:
        logger.debug("record_sandbox_destroy failed (non-fatal): %s", e)
    return {}


# ── feedback ───────────────────────────────────────────────────────────────


async def record_feedback(
    session: Any,
    *,
    rating: str,
    turn_index: int | None = None,
    message_id: str | None = None,
    comment: str | None = None,
) -> None:
    from agent.core.session import Event

    try:
        await session.send_event(
            Event(
                event_type="feedback",
                data={
                    "rating": rating,
                    "turn_index": turn_index,
                    "message_id": message_id,
                    "comment": (comment or "")[:500],
                },
            )
        )
    except Exception as e:
        logger.debug("record_feedback failed (non-fatal): %s", e)


async def record_pro_cta_click(
    session: Any,
    *,
    source: str,
    target: str = "pro_pricing",
) -> None:
    from agent.core.session import Event

    try:
        await session.send_event(
            Event(
                event_type="pro_cta_click",
                data={"source": source, "target": target},
            )
        )
    except Exception as e:
        logger.debug("record_pro_cta_click failed (non-fatal): %s", e)


async def record_pro_conversion(
    session: Any,
    *,
    first_seen_at: str | None = None,
) -> None:
    """Emit a ``pro_conversion`` event for a user we've previously observed
    as non-Pro and now see as Pro for the first time. Detected upstream in
    ``MongoSessionStore.mark_pro_seen``; fired into the user's first Pro
    session so the rollup picks it up alongside other event-driven KPIs."""
    from agent.core.session import Event

    try:
        await session.send_event(
            Event(
                event_type="pro_conversion",
                data={"first_seen_at": first_seen_at},
            )
        )
    except Exception as e:
        logger.debug("record_pro_conversion failed (non-fatal): %s", e)


async def record_credits_topped_up(
    session: Any,
    *,
    namespace: str | None = None,
) -> None:
    """Emit a ``credits_topped_up`` event when an hf_job submits successfully
    in a session that previously hit ``jobs_access_blocked`` — i.e. the user
    came back from the HF billing top-up flow and unblocked themselves.
    Caller is responsible for firing this at most once per session."""
    from agent.core.session import Event

    try:
        await session.send_event(
            Event(
                event_type="credits_topped_up",
                data={"namespace": namespace},
            )
        )
    except Exception as e:
        logger.debug("record_credits_topped_up failed (non-fatal): %s", e)


# ── heartbeat ──────────────────────────────────────────────────────────────

# Module-level reference set for fire-and-forget heartbeat tasks. asyncio only
# keeps *weak* references to tasks, so the returned Task would otherwise be
# eligible for GC before running — the task gets discarded and the upload
# silently never happens. Hold strong refs until the task completes.
_heartbeat_tasks: set[asyncio.Task] = set()


class HeartbeatSaver:
    """Time-gated mid-turn flush.

    Called from ``Session.send_event`` after every event. Fires
    ``save_and_upload_detached`` in a worker thread at most once per
    ``heartbeat_interval_s`` (default 60s). Guards against losing trace data
    on long-running turns that crash before ``turn_complete``.
    """

    @staticmethod
    def maybe_fire(session: Any) -> None:
        if not getattr(session.config, "save_sessions", False):
            return
        interval = getattr(session.config, "heartbeat_interval_s", 0) or 0
        if interval <= 0:
            return
        now = time.monotonic()
        last = getattr(session, "_last_heartbeat_ts", None)
        if last is None:
            # Initialise on first event; no save yet.
            session._last_heartbeat_ts = now
            return
        if now - last < interval:
            return
        session._last_heartbeat_ts = now
        repo_id = session.config.session_dataset_repo
        try:
            task = asyncio.get_running_loop().create_task(
                asyncio.to_thread(session.save_and_upload_detached, repo_id)
            )
            # Hold a strong reference until the task finishes so asyncio can't
            # GC it. ``set.discard`` is a no-op on missing keys → safe callback.
            _heartbeat_tasks.add(task)
            task.add_done_callback(_heartbeat_tasks.discard)
        except RuntimeError:
            try:
                session.save_and_upload_detached(repo_id)
            except Exception as e:
                logger.debug("Heartbeat save failed (non-fatal): %s", e)
