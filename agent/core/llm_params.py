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
    endpoint_model_name,
    endpoint_provider,
    is_openai_compat_model_id,
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


def _hf_router_effort_level(reasoning_effort: str) -> str:
    level = "low" if reasoning_effort == "minimal" else reasoning_effort
    return level


class EndpointNotConfiguredError(ValueError):
    """A direct endpoint id was used without a base URL to send it to.

    Its own type so the CLI can surface the fix instead of a traceback.
    """


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


def endpoint_api_base(model_name: str) -> str | None:
    """The base URL a direct-endpoint id resolves to, for display.

    ``None`` for HF Router ids (always the same endpoint, so showing it is
    noise) and for a direct-endpoint id with nothing configured — that case
    is already reported by ``EndpointNotConfiguredError`` when it's used.
    """
    normalized = strip_huggingface_model_prefix(model_name) or model_name
    provider = endpoint_provider(normalized)
    if provider is None:
        return None
    raw_base = (
        os.environ.get(provider["base_url_env"])
        or os.environ.get(LOCAL_MODEL_BASE_URL_ENV)
        or provider["base_url_default"]
    )
    return _local_api_base(raw_base) if raw_base else None


def _resolve_endpoint_params(
    model_name: str,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict:
    is_gateway = is_openai_compat_model_id(model_name)

    # Localhost servers don't do thinking params. Gateways front the same
    # frontier models HF Router does, so effort goes through and the probe
    # cascade discovers what the far side accepts — no capability table.
    if reasoning_effort and strict and not is_gateway:
        raise UnsupportedEffortError(
            "Local OpenAI-compatible endpoints don't accept reasoning_effort"
        )

    local_name = endpoint_model_name(model_name)
    if local_name is None:
        raise ValueError(f"Unsupported local model id: {model_name}")

    provider = endpoint_provider(model_name)
    assert provider is not None
    raw_base = (
        os.environ.get(provider["base_url_env"])
        or os.environ.get(LOCAL_MODEL_BASE_URL_ENV)
        or provider["base_url_default"]
    )
    if not raw_base:
        raise EndpointNotConfiguredError(
            f"No base URL configured for '{model_name}'. Set "
            f"{provider['base_url_env']} (or {LOCAL_MODEL_BASE_URL_ENV}) to your "
            "OpenAI-compatible endpoint, e.g. https://gateway.example.com/v1"
        )
    api_key = (
        os.environ.get(provider["api_key_env"])
        or os.environ.get(LOCAL_MODEL_API_KEY_ENV)
        or LOCAL_MODEL_API_KEY_DEFAULT
    )
    params = {
        "model": f"openai/{local_name}",
        "api_base": _local_api_base(raw_base),
        "api_key": api_key,
    }
    if is_gateway and reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    return params


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

    • ``openai-compat/<model>`` — any other OpenAI-compatible endpoint, such
      as a self-hosted or corporate LLM gateway. ``OPENAI_COMPAT_BASE_URL``
      is required since there is no localhost default. The model suffix may
      itself contain slashes (``openai-compat/vendor/model-name``); only the
      prefix is stripped. Unlike the local prefixes these do forward
      ``reasoning_effort`` — a gateway usually fronts the same frontier
      models HF Router does — as a top-level OpenAI parameter, with the
      probe cascade walking down whatever the far side rejects.

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
    """
    normalized_model = strip_huggingface_model_prefix(model_name) or model_name

    if endpoint_provider(normalized_model) is not None:
        return _resolve_endpoint_params(normalized_model, reasoning_effort, strict)

    hf_model = normalized_model
    api_key = _resolve_hf_router_token(session_hf_token)
    params = {
        "model": f"openai/{hf_model}",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": api_key,
    }
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
