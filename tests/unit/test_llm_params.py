import pytest

from agent.core.hf_tokens import resolve_hf_request_token
from agent.core.llm_params import (
    EndpointNotConfiguredError,
    UnsupportedEffortError,
    _resolve_hf_router_token,
    _resolve_llm_params,
)
from agent.core.model_ids import HF_ROUTER_BASE_URL


def test_hf_router_params_for_default_model_uses_session_token():
    params = _resolve_llm_params(
        "anthropic/claude-opus-4.8:fal-ai",
        "session-token",
        reasoning_effort="high",
        strict=True,
    )

    assert params == {
        "model": "openai/anthropic/claude-opus-4.8:fal-ai",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": "session-token",
        "extra_body": {"reasoning_effort": "high"},
    }


def test_hf_router_rejects_max_effort_in_strict_mode():
    with pytest.raises(UnsupportedEffortError, match="HF Router"):
        _resolve_llm_params(
            "anthropic/claude-opus-4.8:fal-ai",
            reasoning_effort="max",
            strict=True,
        )


def test_hf_router_drops_unsupported_effort_in_non_strict_mode(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    params = _resolve_llm_params(
        "anthropic/claude-opus-4.8:fal-ai",
        reasoning_effort="max",
        strict=False,
    )

    assert params["api_base"] == HF_ROUTER_BASE_URL
    assert params["api_key"] == "hf-token"
    assert "extra_body" not in params


def test_router_params_fall_back_to_hf_cache_when_session_token_missing(monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("HF_TOKEN", "server-token")
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    params = _resolve_llm_params(
        "anthropic/claude-opus-4.8:fal-ai",
        None,
    )

    assert params["api_key"] == "cached-token"
    assert "extra_headers" not in params


def test_router_params_never_set_bill_to_headers():
    params = _resolve_llm_params("moonshotai/Kimi-K2.7-Code", "session-token")

    assert params["api_key"] == "session-token"
    assert "extra_headers" not in params


def test_huggingface_prefix_is_stripped_for_router_calls():
    params = _resolve_llm_params("huggingface/openai/gpt-5.5:fal-ai")

    assert params["model"] == "openai/openai/gpt-5.5:fal-ai"
    assert params["api_base"] == HF_ROUTER_BASE_URL


def test_resolve_ollama_params_adds_v1_and_uses_default_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    params = _resolve_llm_params("ollama/llama3.1:8b")

    assert params == {
        "model": "openai/llama3.1:8b",
        "api_base": "http://localhost:11434/v1",
        "api_key": "sk-local-no-key-required",
    }


def test_resolve_vllm_params_keeps_existing_v1_and_trims_slash(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1/")

    params = _resolve_llm_params("vllm/meta-llama/Llama-3.1-8B-Instruct")

    assert params["model"] == "openai/meta-llama/Llama-3.1-8B-Instruct"
    assert params["api_base"] == "http://localhost:8000/v1"
    assert params["api_key"] == "sk-local-no-key-required"


def test_resolve_lm_studio_params_uses_api_key_override(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "local-secret")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:9999")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "shared-secret")

    params = _resolve_llm_params("lm_studio/google/gemma-3-4b")

    assert params["model"] == "openai/google/gemma-3-4b"
    assert params["api_base"] == "http://127.0.0.1:1234/v1"
    assert params["api_key"] == "local-secret"


def test_resolve_local_params_uses_shared_fallback_env(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:9000/v1/")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "shared-local-secret")

    params = _resolve_llm_params("vllm/custom-model")

    assert params["model"] == "openai/custom-model"
    assert params["api_base"] == "http://localhost:9000/v1"
    assert params["api_key"] == "shared-local-secret"


def test_resolve_llamacpp_params_strips_provider_prefix(monkeypatch):
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://localhost:8080")

    params = _resolve_llm_params("llamacpp/unsloth/Qwen3.5-2B")

    assert params["model"] == "openai/unsloth/Qwen3.5-2B"
    assert params["api_base"] == "http://localhost:8080/v1"


def test_local_params_reject_reasoning_effort_in_strict_mode():
    with pytest.raises(UnsupportedEffortError, match="reasoning_effort"):
        _resolve_llm_params("ollama/llama3.1", reasoning_effort="high", strict=True)


def test_local_params_drop_reasoning_effort_in_non_strict_mode():
    params = _resolve_llm_params(
        "ollama/llama3.1",
        reasoning_effort="high",
        strict=False,
    )

    assert params["model"] == "openai/llama3.1"
    assert "reasoning_effort" not in params
    assert "extra_body" not in params


def test_openai_compat_forwards_reasoning_effort(monkeypatch):
    """A gateway fronts the same frontier models HF Router does, so effort
    has to reach it — the local prefixes deliberately drop it."""
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "gateway-secret")

    params = _resolve_llm_params(
        "openai-compat/vertex/claude-opus-5",
        reasoning_effort="high",
        strict=True,
    )

    assert params["reasoning_effort"] == "high"


def test_openai_compat_without_effort_sends_no_effort_key(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "gateway-secret")

    params = _resolve_llm_params("openai-compat/custom-model")

    assert "reasoning_effort" not in params


def test_local_prefixes_still_reject_effort_in_strict_mode(monkeypatch):
    """The gateway carve-out must not leak into the localhost servers."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

    with pytest.raises(UnsupportedEffortError, match="reasoning_effort"):
        _resolve_llm_params("vllm/custom-model", reasoning_effort="high", strict=True)


def test_openai_compat_params_target_configured_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway.example.com/openai")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "gateway-secret")

    params = _resolve_llm_params("openai-compat/custom-model")

    assert params == {
        "model": "openai/custom-model",
        "api_base": "https://gateway.example.com/openai/v1",
        "api_key": "gateway-secret",
    }


def test_openai_compat_keeps_multi_segment_model_names(monkeypatch):
    """Gateways namespace models (``vertex/claude-...``); only strip the prefix."""
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway.example.com/v1/")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "gateway-secret")

    params = _resolve_llm_params("openai-compat/vertex/claude-opus-4-7")

    assert params["model"] == "openai/vertex/claude-opus-4-7"
    assert params["api_base"] == "https://gateway.example.com/v1"


def test_openai_compat_falls_back_to_shared_local_env(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "shared-secret")

    params = _resolve_llm_params("openai-compat/custom-model")

    assert params["api_base"] == "https://gateway.example.com/v1"
    assert params["api_key"] == "shared-secret"


def test_openai_compat_without_base_url_raises_actionable_error(monkeypatch):
    """No localhost default to fall back on, so say what to set."""
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)

    with pytest.raises(EndpointNotConfiguredError, match="OPENAI_COMPAT_BASE_URL"):
        _resolve_llm_params("openai-compat/custom-model")


def test_unconfigured_endpoint_is_reported_without_a_traceback():
    """The message is already an instruction; don't bury it in stack frames."""
    from agent.core.agent_loop import _friendly_error_message

    error = EndpointNotConfiguredError("No base URL configured for 'openai-compat/x'.")

    assert _friendly_error_message(error) == str(error)


def test_empty_openai_compat_model_id_is_not_treated_as_hf_router():
    with pytest.raises(ValueError, match="Unsupported local model id"):
        _resolve_llm_params("openai-compat/")


def test_empty_local_model_id_is_not_treated_as_hf_router():
    with pytest.raises(ValueError, match="Unsupported local model id"):
        _resolve_llm_params("ollama/")


def test_hf_router_token_prefers_session_over_hf_cache(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token(" session-token ") == "session-token"


def test_hf_router_token_uses_hf_token_env_via_huggingface_hub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", " hf-token ")

    assert _resolve_hf_router_token(None) == "hf-token"


def test_hf_router_token_uses_huggingface_hub_cache(monkeypatch):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert _resolve_hf_router_token(None) == "cached-token"


def test_hf_router_token_swallows_huggingface_hub_errors(monkeypatch):
    import huggingface_hub

    def fail():
        raise RuntimeError("cache unavailable")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", fail)

    assert _resolve_hf_router_token(None) is None


def test_hf_router_params_allow_missing_token_without_headers(monkeypatch):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)

    params = _resolve_llm_params("moonshotai/Kimi-K2.7-Code")

    assert params["api_key"] is None
    assert "extra_headers" not in params


def test_hf_request_token_keeps_browser_user_precedence(monkeypatch):
    class Request:
        headers = {"Authorization": "Bearer browser-token"}
        cookies = {"hf_access_token": "cookie-token"}

    monkeypatch.setenv("HF_TOKEN", "server-token")

    assert resolve_hf_request_token(Request()) == "browser-token"


def test_hf_request_token_does_not_use_cached_login(monkeypatch):
    import huggingface_hub

    class Request:
        headers = {}
        cookies = {}

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert resolve_hf_request_token(Request()) is None


def test_endpoint_api_base_shows_where_a_gateway_id_resolves(monkeypatch):
    """Surfaced at startup so a .env the user forgot about is visible."""
    from agent.core.llm_params import endpoint_api_base

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway.example.com/openai")

    assert (
        endpoint_api_base("openai-compat/vertex/claude-opus-5")
        == "https://gateway.example.com/openai/v1"
    )


def test_endpoint_api_base_is_none_for_router_and_unconfigured_ids(monkeypatch):
    from agent.core.llm_params import endpoint_api_base

    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)

    # HF Router always goes to the same place — showing it would be noise.
    assert endpoint_api_base("zai-org/GLM-5.2:novita") is None
    # Unconfigured gateway: EndpointNotConfiguredError already covers this.
    assert endpoint_api_base("openai-compat/custom-model") is None


def test_endpoint_api_base_reports_the_mixed_source_combination(monkeypatch):
    """Precedence is per-variable, so a URL and key can come from different
    files. Showing the resolved URL is what makes that visible."""
    from agent.core.llm_params import endpoint_api_base

    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "https://gateway-a.example.com/v1")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway-b.example.com/v1")

    assert (
        endpoint_api_base("openai-compat/custom-model")
        == "https://gateway-b.example.com/v1"
    )
