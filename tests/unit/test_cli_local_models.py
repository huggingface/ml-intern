import pytest

from agent.config import load_config
from agent.core import model_switcher
from agent.core.local_models import is_local_model_id
from agent.main import CLI_CONFIG_PATH


def test_local_model_helper_accepts_supported_prefixes():
    assert is_local_model_id("ollama/llama3.1:8b")
    assert is_local_model_id("vllm/meta-llama/Llama-3.1-8B-Instruct")
    assert is_local_model_id("lm_studio/google/gemma-3-4b")
    assert is_local_model_id("llamacpp/unsloth/Qwen3.5-2B")
    assert is_local_model_id("custom-proxy/Azure AI/gpt-5.5")


def test_model_switcher_accepts_supported_local_prefixes():
    assert model_switcher.is_valid_model_id("ollama/llama3.1:8b")
    assert model_switcher.is_valid_model_id("vllm/meta-llama/Llama-3.1-8B")
    assert model_switcher.is_valid_model_id("lm_studio/google/gemma-3-4b")
    assert model_switcher.is_valid_model_id("llamacpp/llama-3.1-8b")
    assert model_switcher.is_valid_model_id("custom-proxy/Azure AI/gpt-5.5")


def test_model_switcher_rejects_empty_or_whitespace_local_ids():
    assert not model_switcher.is_valid_model_id("ollama/")
    assert not model_switcher.is_valid_model_id("vllm/")
    assert not model_switcher.is_valid_model_id("lm_studio/")
    assert not model_switcher.is_valid_model_id("llamacpp/")
    assert not model_switcher.is_valid_model_id("custom-proxy/")
    assert not model_switcher.is_valid_model_id("ollama/llama 3.1")


def test_openai_compat_prefix_is_not_supported():
    assert not model_switcher.is_valid_model_id("openai-compat/custom-model")


def test_suggested_models_include_router_claude_models_and_no_native_ids():
    ids = {m["id"] for m in model_switcher.SUGGESTED_MODELS}

    assert "anthropic/claude-opus-4.8:fal-ai" in ids
    assert all(model_id.count("/") >= 1 for model_id in ids)


def test_cli_default_model_is_opus():
    config = load_config(CLI_CONFIG_PATH)

    assert config.model_name == "anthropic/claude-opus-4.8:fal-ai"


def test_model_switcher_accepts_router_model_ids():
    assert model_switcher.is_valid_model_id("openai/gpt-5.5:fal-ai")
    assert model_switcher.is_valid_model_id("openai/gpt-oss-120b")
    assert model_switcher.is_valid_model_id("moonshotai/Kimi-K2.6")


def test_local_models_skip_hf_router_catalog_output():
    class NoPrintConsole:
        def print(self, *args, **kwargs):
            raise AssertionError("local models should not print HF catalog info")

    assert model_switcher._print_hf_routing_info(
        "ollama/llama3.1:8b",
        NoPrintConsole(),
    )


@pytest.mark.asyncio
async def test_probe_and_switch_local_model_uses_no_effort(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(model_switcher, "acompletion", fake_acompletion)

    class Config:
        model_name = "anthropic/claude-opus-4.8:fal-ai"
        reasoning_effort = "high"

    class Session:
        def __init__(self):
            self.model_id = None
            self.model_effective_effort = {}

        def update_model(self, model_id):
            self.model_id = model_id

    class Console:
        def print(self, *args, **kwargs):
            pass

    session = Session()
    await model_switcher.probe_and_switch_model(
        "ollama/llama3.1:8b",
        Config(),
        session,
        Console(),
        hf_token=None,
    )

    assert session.model_id == "ollama/llama3.1:8b"
    assert session.model_effective_effort["ollama/llama3.1:8b"] is None
    assert calls[0]["model"] == "openai/llama3.1:8b"
    assert "reasoning_effort" not in calls[0]
    assert "extra_body" not in calls[0]


@pytest.mark.asyncio
async def test_probe_and_switch_local_model_rejects_probe_errors(monkeypatch):
    async def failing_acompletion(**kwargs):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(model_switcher, "acompletion", failing_acompletion)

    class Config:
        model_name = "anthropic/claude-opus-4.8:fal-ai"
        reasoning_effort = None

    class Session:
        def __init__(self):
            self.model_id = None
            self.model_effective_effort = {}

        def update_model(self, model_id):
            self.model_id = model_id

    class Console:
        def print(self, *args, **kwargs):
            pass

    config = Config()
    session = Session()
    await model_switcher.probe_and_switch_model(
        "ollama/llama3.1:8b",
        config,
        session,
        Console(),
        hf_token=None,
    )

    assert config.model_name == "anthropic/claude-opus-4.8:fal-ai"
    assert session.model_id is None
    assert "ollama/llama3.1:8b" not in session.model_effective_effort
