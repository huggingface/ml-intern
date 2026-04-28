"""Tests for backend custom local model validation."""

from backend import model_catalog


def test_custom_local_model_ids_require_feature_flag(monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_MODELS", raising=False)

    assert not model_catalog.is_valid_model_id("ollama/qwen2.5-coder")
    assert not model_catalog.is_custom_local_model_id("ollama/qwen2.5-coder")

    monkeypatch.setenv("ENABLE_LOCAL_MODELS", "true")

    assert model_catalog.is_valid_model_id("ollama/qwen2.5-coder")
    assert model_catalog.is_valid_model_id("vllm/Qwen/Qwen3-4B")
    assert model_catalog.is_valid_model_id("llamacpp/models/qwen.gguf")
    assert model_catalog.is_valid_model_id("local://my-model")


def test_custom_local_model_ids_reject_empty_or_whitespace(monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_MODELS", "true")

    assert not model_catalog.is_valid_model_id("ollama/")
    assert not model_catalog.is_valid_model_id("local://")
    assert not model_catalog.is_valid_model_id(" ollama/qwen")
    assert not model_catalog.is_valid_model_id("ollama/qwen coder")
    assert not model_catalog.is_valid_model_id("some-org/model")


def test_anthropic_detection_is_anchored_to_cloud_prefixes():
    assert model_catalog.is_anthropic_model("anthropic/claude-opus-4-6")
    assert model_catalog.is_anthropic_model("bedrock/us.anthropic.claude-opus-4-6-v1")
    assert not model_catalog.is_anthropic_model("local://my-anthropic-wrapper")
    assert not model_catalog.is_anthropic_model("ollama/anthropic-clone")
