"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

import os

from agent.core.hf_tokens import (
    get_hf_bill_to,
    resolve_hf_router_token,
    resolve_hf_token,
)
from agent.core.local_models import (
    LOCAL_MODEL_API_KEY_DEFAULT,
    LOCAL_MODEL_API_KEY_ENV,
    LOCAL_MODEL_BASE_URL_ENV,
    is_reserved_local_model_id,
    local_model_name,
    local_model_provider,
)


def _resolve_hf_router_token(session_hf_token: str | None = None) -> str | None:
    """Backward-compatible private wrapper used by tests and older imports."""
    return resolve_hf_router_token(session_hf_token)


def _patch_litellm_effort_validation() -> None:
    """Neuter LiteLLM 1.83's hardcoded effort-level validation.

    Context: at ``litellm/llms/anthropic/chat/transformation.py:~1443`` the
    Anthropic adapter validates ``output_config.effort ∈ {high, medium,
    low, max}`` and gates ``max`` behind an ``_is_opus_4_6_model`` check
    that only matches the substring ``opus-4-6`` / ``opus_4_6``. Result:

    * ``xhigh`` — valid on Anthropic's real API for Claude 4.7 — is
      rejected pre-flight with "Invalid effort value: xhigh".
    * ``max`` on Opus 4.7 is rejected with "effort='max' is only supported
      by Claude Opus 4.6", even though Opus 4.7 accepts it in practice.

    We don't want to maintain a parallel model table, so we let the
    Anthropic API itself be the validator: widen ``_is_opus_4_6_model``
    to also match ``opus-4-7``+ families, and drop the valid-effort-set
    check entirely. If Anthropic rejects an effort level, we see a 400
    and the cascade walks down — exactly the behavior we want for any
    future model family.

    Removable once litellm ships 1.83.8-stable (which merges PR #25867,
    "Litellm day 0 opus 4.7 support") — see commit 0868a82 on their main
    branch. Until then, this one-time patch is the escape hatch.
    """
    try:
        from litellm.llms.anthropic.chat import transformation as _t
    except Exception:
        return

    cfg = getattr(_t, "AnthropicConfig", None)
    if cfg is None:
        return

    original = getattr(cfg, "_is_opus_4_6_model", None)
    if original is None or getattr(original, "_hf_agent_patched", False):
        return

    def _widened(model: str) -> bool:
        m = model.lower()
        # Original 4.6 match plus any future Opus >= 4.6. We only need this
        # to return True for families where "max" / "xhigh" are acceptable
        # at the API; the cascade handles the case when they're not.
        return any(
            v in m
            for v in (
                "opus-4-6",
                "opus_4_6",
                "opus-4.6",
                "opus_4.6",
                "opus-4-7",
                "opus_4_7",
                "opus-4.7",
                "opus_4.7",
            )
        )

    _widened._hf_agent_patched = True  # type: ignore[attr-defined]
    cfg._is_opus_4_6_model = staticmethod(_widened)


_patch_litellm_effort_validation()


# Effort levels accepted on the wire.
#   Anthropic (4.6+):  low | medium | high | xhigh | max   (output_config.effort)
#   OpenAI direct:     minimal | low | medium | high | xhigh (reasoning_effort top-level)
#   HF router default: low | medium | high                 (extra_body.reasoning_effort)
#   HF router premium user-billed overflow keeps the subsidized endpoint's
#   provider-native effort set so overflow cannot silently lower or drop effort.
#
# We validate *shape* here and let the probe cascade walk down on rejection;
# we deliberately do NOT maintain a per-model capability table.
_ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_OPENAI_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
_HF_EFFORTS = {"low", "medium", "high"}

# Production still uses the historical subsidized endpoints while users are
# within their daily premium allowance. Once a session overflows to user
# billing, route the same logical model through the HF Inference Router/FAL
# with the user's OAuth token.
_USER_BILLED_HF_ROUTER_MODEL_BY_SUBSIDIZED_MODEL = {
    "bedrock/us.anthropic.claude-opus-4-6-v1": (
        "huggingface/anthropic/claude-opus-4.6:fal-ai"
    ),
    "bedrock/us.anthropic.claude-sonnet-4-6": (
        "huggingface/anthropic/claude-sonnet-4-6:fal-ai"
    ),
    "openai/gpt-5.5": "huggingface/openai/gpt-5.5:fal-ai",
}
_SUBSIDIZED_MODEL_BY_USER_BILLED_HF_ROUTER_MODEL = {
    user_billed: subsidized
    for subsidized, user_billed in _USER_BILLED_HF_ROUTER_MODEL_BY_SUBSIDIZED_MODEL.items()
}


def _hf_router_effort_spec(
    hf_model: str, bill_user: bool, reasoning_effort: str
) -> tuple[str, set[str], str]:
    """Return the effort value and accepted set for an HF-router call.

    Generic HF-router models use the router's low/medium/high contract. Premium
    overflow is different: the subsidized endpoint is the source of truth for
    the effective effort we cached on the session, so the user-billed FAL call
    must keep the same provider-native effort instead of silently dropping
    ``max``/``xhigh``.
    """
    if bill_user and hf_model.startswith("anthropic/"):
        level = "low" if reasoning_effort == "minimal" else reasoning_effort
        return level, _ANTHROPIC_EFFORTS, "Anthropic"
    if bill_user and hf_model.startswith("openai/"):
        return reasoning_effort, _OPENAI_EFFORTS, "OpenAI"
    level = "low" if reasoning_effort == "minimal" else reasoning_effort
    return level, _HF_EFFORTS, "HF router"


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


def _local_api_base(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _resolve_local_model_params(
    model_name: str,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict:
    if reasoning_effort and strict:
        raise UnsupportedEffortError(
            "Local OpenAI-compatible endpoints don't accept reasoning_effort"
        )

    local_name = local_model_name(model_name)
    if local_name is None:
        raise ValueError(f"Unsupported local model id: {model_name}")

    provider = local_model_provider(model_name)
    assert provider is not None
    raw_base = (
        os.environ.get(provider["base_url_env"])
        or os.environ.get(LOCAL_MODEL_BASE_URL_ENV)
        or provider["base_url_default"]
    )
    api_key = (
        os.environ.get(provider["api_key_env"])
        or os.environ.get(LOCAL_MODEL_API_KEY_ENV)
        or LOCAL_MODEL_API_KEY_DEFAULT
    )
    return {
        "model": f"openai/{local_name}",
        "api_base": _local_api_base(raw_base),
        "api_key": api_key,
    }


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
    strict: bool = False,
    bill_to_user: bool = False,
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``anthropic/<model>`` — native thinking config. We bypass LiteLLM's
      ``reasoning_effort`` → ``thinking`` mapping (which lags new Claude
      releases like 4.7 and sends the wrong API shape). Instead we pass
      both ``thinking={"type": "adaptive"}`` and ``output_config=
      {"effort": <level>}`` as top-level kwargs — LiteLLM's Anthropic
      adapter forwards unknown top-level kwargs into the request body
      verbatim (confirmed by live probe; ``extra_body`` does NOT work
      here because Anthropic's API rejects it as "Extra inputs are not
      permitted"). This is the stable API for 4.6 and 4.7. Older
      extended-thinking models that only accept ``thinking.type.enabled``
      will reject this; the probe's cascade catches that and falls back
      to no thinking.

    • ``openai/<model>`` — ``reasoning_effort`` forwarded as a top-level
      kwarg (GPT-5 / o-series). LiteLLM uses the user's ``OPENAI_API_KEY``.

    • ``ollama/<model>``, ``vllm/<model>``, ``lm_studio/<model>``, and
      ``llamacpp/<model>`` — local OpenAI-compatible endpoints. The id prefix
      selects a configurable localhost base URL, and the model suffix is sent
      to LiteLLM as ``openai/<model>``. These endpoints don't receive
      ``reasoning_effort``.

    • Anything else is treated as a HuggingFace router id. We hit the
      auto-routing OpenAI-compatible endpoint at
      ``https://router.huggingface.co/v1``. The id can be bare or carry an
      HF routing suffix (``:fastest`` / ``:cheapest`` / ``:<provider>``).
      A leading ``huggingface/`` is stripped. ``reasoning_effort`` is
      forwarded via ``extra_body`` (LiteLLM's OpenAI adapter refuses it as
      a top-level kwarg for non-OpenAI models). "minimal" normalizes to
      "low".

    ``strict=True`` raises ``UnsupportedEffortError`` when the requested
    effort isn't in the provider's accepted set, instead of silently
    dropping it. The probe cascade uses strict mode so it can walk down
    (``max`` → ``xhigh`` → ``high`` …) without making an API call. Regular
    runtime callers leave ``strict=False``, so a stale cached effort
    can't crash a turn — it just doesn't get sent.

    Token precedence for HF-router calls (first non-empty wins):
      1. INFERENCE_TOKEN env — shared key on the hosted Space (inference is
         free for users, billed to the Space owner via ``X-HF-Bill-To``).
      2. session.hf_token — the user's own token (CLI / OAuth / cache file).
      3. huggingface_hub cache — ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` /
         local ``hf auth login`` cache.

    The production premium ids intentionally remain the old subsidized
    endpoints (Bedrock Claude, direct OpenAI GPT-5.5). Pass
    ``bill_to_user=True`` only after the daily subsidized allowance is spent;
    those ids then map to the HF Router/FAL ids and use the caller's own token
    (skipping ``INFERENCE_TOKEN`` and omitting ``X-HF-Bill-To``).
    """
    if bill_to_user and (
        user_billed_model := _USER_BILLED_HF_ROUTER_MODEL_BY_SUBSIDIZED_MODEL.get(
            model_name
        )
    ):
        return _resolve_llm_params(
            user_billed_model,
            session_hf_token,
            reasoning_effort=reasoning_effort,
            strict=strict,
            bill_to_user=True,
        )

    if not bill_to_user and (
        subsidized_model := _SUBSIDIZED_MODEL_BY_USER_BILLED_HF_ROUTER_MODEL.get(
            model_name
        )
    ):
        return _resolve_llm_params(
            subsidized_model,
            session_hf_token,
            reasoning_effort=reasoning_effort,
            strict=strict,
            bill_to_user=False,
        )

    if model_name.startswith("anthropic/"):
        params: dict = {"model": model_name}
        if reasoning_effort:
            level = reasoning_effort
            if level == "minimal":
                level = "low"
            if level not in _ANTHROPIC_EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"Anthropic doesn't accept effort={level!r}"
                    )
            else:
                # Adaptive thinking + output_config.effort is the stable
                # Anthropic API for Claude 4.6 / 4.7. Both kwargs are
                # passed top-level: LiteLLM forwards unknown params into
                # the request body for Anthropic, so ``output_config``
                # reaches the API. ``extra_body`` does NOT work here —
                # Anthropic rejects it as "Extra inputs are not
                # permitted".
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": level}
        return params

    if model_name.startswith("bedrock/"):
        # LiteLLM routes ``bedrock/...`` through the Converse adapter, which
        # picks up AWS credentials from the standard env vars
        # (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_REGION``).
        # The Anthropic thinking/effort shape is not forwarded through Converse
        # the same way, so we leave it off for now.
        return {"model": model_name}

    if model_name.startswith("openai/"):
        params = {"model": model_name}
        if reasoning_effort:
            if reasoning_effort not in _OPENAI_EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"OpenAI doesn't accept effort={reasoning_effort!r}"
                    )
            else:
                params["reasoning_effort"] = reasoning_effort
        return params

    if is_reserved_local_model_id(model_name):
        raise ValueError(f"Unsupported local model id: {model_name}")

    if local_model_provider(model_name) is not None:
        return _resolve_local_model_params(model_name, reasoning_effort, strict)

    hf_model = model_name.removeprefix("huggingface/")
    # Premium models routed through the HF router (Anthropic Claude, OpenAI GPT)
    # can be billed to the *user's* own HF account instead of the Space: when
    # ``bill_to_user`` is set (the backend flips it on once a user is past their
    # subsidized daily allowance) use the caller's session token (never
    # INFERENCE_TOKEN) and omit X-HF-Bill-To so the spend lands on their wallet.
    # Otherwise — within the allowance, or for any free model — keep the
    # subsidized path.
    bill_user = bill_to_user and hf_model.startswith(("anthropic/", "openai/"))
    api_key = (
        resolve_hf_token(session_hf_token, include_cached=False)
        if bill_user
        else _resolve_hf_router_token(session_hf_token)
    )
    params = {
        "model": f"openai/{hf_model}",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": api_key,
    }
    if not bill_user and (bill_to := get_hf_bill_to()):
        params["extra_headers"] = {"X-HF-Bill-To": bill_to}
    if reasoning_effort:
        hf_level, accepted_efforts, effort_owner = _hf_router_effort_spec(
            hf_model, bill_user, reasoning_effort
        )
        if hf_level not in accepted_efforts:
            if strict:
                raise UnsupportedEffortError(
                    f"{effort_owner} doesn't accept effort={hf_level!r}"
                )
        else:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params
