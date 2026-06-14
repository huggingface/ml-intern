"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

import os

from agent.core.hf_tokens import resolve_hf_router_token
from agent.core.local_models import (
    LOCAL_MODEL_API_KEY_DEFAULT,
    LOCAL_MODEL_API_KEY_ENV,
    LOCAL_MODEL_BASE_URL_ENV,
    is_reserved_local_model_id,
    local_model_name,
    local_model_provider,
)
from agent.core.model_ids import (
    HF_ROUTER_BASE_URL,
    strip_huggingface_model_prefix,
)


def _resolve_hf_router_token(session_hf_token: str | None = None) -> str | None:
    """Backward-compatible private wrapper used by tests and older imports."""
    return resolve_hf_router_token(session_hf_token)


# Effort levels accepted on the wire.
# HF Router exposes reasoning controls through the OpenAI-compatible
# ``extra_body`` field. The probe cascade walks down when a provider rejects
# an accepted-looking value, so this stays intentionally small and generic.
_HF_EFFORTS = {"low", "medium", "high"}

# When ``HF_BILL_TO`` names an org you belong to, the router charges that org's
# credits instead of your personal allowance — same header huggingface_hub's
# ``bill_to=`` sets, applied here because we go through LiteLLM, not InferenceClient.
HF_BILL_TO_ENV = "HF_BILL_TO"
HF_BILL_TO_HEADER = "X-HF-Bill-To"


def _resolve_hf_bill_to() -> str | None:
    """Org to bill HF Inference Providers usage to, or None if unset."""
    value = os.environ.get(HF_BILL_TO_ENV)
    return value.strip() or None if value else None


def _hf_router_effort_level(reasoning_effort: str) -> str:
    level = "low" if reasoning_effort == "minimal" else reasoning_effort
    return level


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
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``ollama/<model>``, ``vllm/<model>``, ``lm_studio/<model>``, and
      ``llamacpp/<model>`` — local OpenAI-compatible endpoints. The id prefix
      selects a configurable localhost base URL, and the model suffix is sent
      to LiteLLM as ``openai/<model>``. These endpoints don't receive
      ``reasoning_effort``.

    • Anything else is treated as an HF Router id. We hit the auto-routing
      OpenAI-compatible endpoint at ``https://router.huggingface.co/v1``.
      The id can be bare or carry an HF routing suffix (``:fastest`` /
      ``:cheapest`` / ``:<provider>``). A leading ``huggingface/`` is
      stripped. ``reasoning_effort`` is forwarded via ``extra_body``.
      "minimal" normalizes to "low".

    ``strict=True`` raises ``UnsupportedEffortError`` when the requested
    effort isn't in the provider's accepted set, instead of silently
    dropping it. The probe cascade uses strict mode so it can walk down
    (``max`` → ``xhigh`` → ``high`` …) without making an API call. Regular
    runtime callers leave ``strict=False``, so a stale cached effort
    can't crash a turn — it just doesn't get sent.

    Token precedence for HF-router calls (first non-empty wins):
      1. session.hf_token — the user's own token (CLI / OAuth / cache file).
      2. huggingface_hub cache — ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` /
         local ``hf auth login`` cache.

    When ``HF_BILL_TO`` is set, an ``X-HF-Bill-To`` header is attached to
    HF-router calls so Inference Providers usage is charged to that org's
    credits instead of the token owner's personal monthly allowance.
    """
    normalized_model = strip_huggingface_model_prefix(model_name) or model_name

    if is_reserved_local_model_id(normalized_model):
        raise ValueError(f"Unsupported local model id: {normalized_model}")

    if local_model_provider(normalized_model) is not None:
        return _resolve_local_model_params(normalized_model, reasoning_effort, strict)

    hf_model = normalized_model
    api_key = _resolve_hf_router_token(session_hf_token)
    params = {
        "model": f"openai/{hf_model}",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": api_key,
    }
    bill_to = _resolve_hf_bill_to()
    if bill_to:
        params["extra_headers"] = {HF_BILL_TO_HEADER: bill_to}
    if reasoning_effort:
        hf_level = _hf_router_effort_level(reasoning_effort)
        if hf_level not in _HF_EFFORTS:
            if strict:
                raise UnsupportedEffortError(
                    f"HF Router doesn't accept effort={hf_level!r}"
                )
        else:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params
