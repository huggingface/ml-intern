"""Tests for backend custom local model validation."""

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _ROOT_DIR / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import model_catalog


def test_custom_local_model_ids_require_feature_flag(monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_MODELS", raising=False)

    assert not model_catalog.is_valid_model_id("ollama/qwen2.5-coder")

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
