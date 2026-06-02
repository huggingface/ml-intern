import pytest

from agent.core.hf_tokens import resolve_hf_request_token
from agent.core.llm_params import (
    UnsupportedEffortError,
    _resolve_hf_router_token,
    _resolve_llm_params,
)
from agent.core.model_ids import HF_ROUTER_BASE_URL


def test_hf_router_params_for_default_premium_model(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_BILL_TO", "smolagents")

    params = _resolve_llm_params(
        "anthropic/claude-opus-4.8:fal-ai",
        "session-token",
        reasoning_effort="high",
        strict=True,
    )

    assert params == {
        "model": "openai/anthropic/claude-opus-4.8:fal-ai",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": "inference-token",
        "extra_headers": {"X-HF-Bill-To": "smolagents"},
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


def test_bedrock_native_provider_id_is_rejected():
    with pytest.raises(ValueError, match="Native Anthropic"):
        _resolve_llm_params("bedrock/us.anthropic.unknown-model")


def test_direct_openai_provider_id_is_rejected():
    with pytest.raises(ValueError, match="Native Anthropic"):
        _resolve_llm_params("openai/gpt-5.5")


def test_user_billed_premium_uses_session_token_without_bill_to(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_BILL_TO", "smolagents")

    params = _resolve_llm_params(
        "anthropic/claude-opus-4.8:fal-ai",
        "session-token",
        reasoning_effort="high",
        strict=True,
        bill_to_user=True,
    )

    assert params["model"] == "openai/anthropic/claude-opus-4.8:fal-ai"
    assert params["api_base"] == HF_ROUTER_BASE_URL
    assert params["api_key"] == "session-token"
    assert "extra_headers" not in params
    assert params["extra_body"] == {"reasoning_effort": "high"}


def test_user_billed_premium_does_not_fall_back_to_cached_token(monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_TOKEN", "server-token")
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    params = _resolve_llm_params(
        "anthropic/claude-opus-4.8:fal-ai",
        None,
        bill_to_user=True,
    )

    assert params["api_key"] is None
    assert "extra_headers" not in params


def test_bill_to_user_ignored_for_free_models(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_BILL_TO", "smolagents")

    params = _resolve_llm_params(
        "moonshotai/Kimi-K2.6", "session-token", bill_to_user=True
    )

    assert params["api_key"] == "inference-token"
    assert params["extra_headers"] == {"X-HF-Bill-To": "smolagents"}


def test_huggingface_prefix_is_stripped_for_router_calls(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")

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


def test_openai_compat_prefix_is_not_a_local_escape_hatch():
    with pytest.raises(ValueError, match="Unsupported local model id"):
        _resolve_llm_params("openai-compat/custom-model")


def test_empty_local_model_id_is_not_treated_as_hf_router():
    with pytest.raises(ValueError, match="Unsupported local model id"):
        _resolve_llm_params("ollama/")


def test_hf_router_token_prefers_inference_token(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", " inference-token ")
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token("session-token") == "inference-token"


def test_hf_router_token_prefers_session_over_hf_cache(monkeypatch):
    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token(" session-token ") == "session-token"


def test_hf_router_token_uses_hf_token_env_via_huggingface_hub(monkeypatch):
    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", " hf-token ")

    assert _resolve_hf_router_token(None) == "hf-token"


def test_hf_router_token_uses_huggingface_hub_cache(monkeypatch):
    import huggingface_hub

    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert _resolve_hf_router_token(None) == "cached-token"


def test_hf_router_token_swallows_huggingface_hub_errors(monkeypatch):
    import huggingface_hub

    def fail():
        raise RuntimeError("cache unavailable")

    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", fail)

    assert _resolve_hf_router_token(None) is None


def test_hf_router_params_set_bill_to_only_for_inference_token(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_BILL_TO", "test-org")

    params = _resolve_llm_params("moonshotai/Kimi-K2.6")

    assert params["api_key"] == "inference-token"
    assert params["extra_headers"] == {"X-HF-Bill-To": "test-org"}


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
