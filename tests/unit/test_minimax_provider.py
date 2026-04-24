"""Unit tests for MiniMax provider support in _resolve_llm_params."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Load agent.core.llm_params directly from its file to avoid triggering
# agent/__init__.py (which pulls in agent_loop -> tools -> whoosh and other
# heavy optional deps that aren't installed in all test environments).
_AGENT_DIR = Path(__file__).resolve().parent.parent.parent
_LLM_PARAMS_PATH = _AGENT_DIR / "agent" / "core" / "llm_params.py"

spec = importlib.util.spec_from_file_location("agent.core.llm_params", _LLM_PARAMS_PATH)
_llm_params_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(_llm_params_mod)  # type: ignore[union-attr]

_resolve_llm_params = _llm_params_mod._resolve_llm_params
UnsupportedEffortError = _llm_params_mod.UnsupportedEffortError
_MINIMAX_DEFAULT_BASE_URL = _llm_params_mod._MINIMAX_DEFAULT_BASE_URL


class TestMinimaxProvider:
    """Tests for the minimax/ prefix provider routing."""

    def test_minimax_model_uses_openai_adapter(self):
        """minimax/X is forwarded to litellm as openai/X with api_base set."""
        params = _resolve_llm_params("minimax/MiniMax-M2.7")
        assert params["model"] == "openai/MiniMax-M2.7"

    def test_minimax_highspeed_model(self):
        params = _resolve_llm_params("minimax/MiniMax-M2.7-highspeed")
        assert params["model"] == "openai/MiniMax-M2.7-highspeed"

    def test_minimax_default_base_url(self):
        """Default base URL points to api.minimax.io."""
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_BASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            params = _resolve_llm_params("minimax/MiniMax-M2.7")
        assert params["api_base"] == _MINIMAX_DEFAULT_BASE_URL
        assert params["api_base"].startswith("https://api.minimax.io")

    def test_minimax_custom_base_url(self):
        """MINIMAX_BASE_URL env var overrides the default endpoint."""
        custom_url = "https://api.minimaxi.com/v1"
        with patch.dict(os.environ, {"MINIMAX_BASE_URL": custom_url}):
            params = _resolve_llm_params("minimax/MiniMax-M2.7")
        assert params["api_base"] == custom_url

    def test_minimax_api_key_from_env(self):
        """MINIMAX_API_KEY is picked up from environment."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
            params = _resolve_llm_params("minimax/MiniMax-M2.7")
        assert params["api_key"] == "test-key-123"

    def test_minimax_no_api_key(self):
        """When MINIMAX_API_KEY is unset, api_key is None (litellm raises later)."""
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            params = _resolve_llm_params("minimax/MiniMax-M2.7")
        assert params["api_key"] is None

    def test_minimax_reasoning_effort_silently_ignored(self):
        """Reasoning effort is not forwarded — MiniMax doesn't support it."""
        params = _resolve_llm_params("minimax/MiniMax-M2.7", reasoning_effort="high")
        assert "reasoning_effort" not in params
        assert "thinking" not in params
        assert "output_config" not in params
        assert "extra_body" not in params

    def test_minimax_reasoning_effort_strict_mode_no_error(self):
        """strict=True doesn't raise for minimax — effort is silently dropped."""
        params = _resolve_llm_params(
            "minimax/MiniMax-M2.7", reasoning_effort="max", strict=True
        )
        assert params["model"] == "openai/MiniMax-M2.7"
        assert "reasoning_effort" not in params

    def test_minimax_does_not_include_hf_headers(self):
        """No X-HF-Bill-To or extra_headers for minimax."""
        params = _resolve_llm_params("minimax/MiniMax-M2.7")
        assert "extra_headers" not in params

    def test_other_prefixes_unaffected(self):
        """anthropic/ and openai/ prefixes are unaffected by the minimax branch."""
        anthropic_params = _resolve_llm_params("anthropic/claude-opus-4-6")
        assert anthropic_params["model"] == "anthropic/claude-opus-4-6"
        assert "api_base" not in anthropic_params

        openai_params = _resolve_llm_params("openai/gpt-4o")
        assert openai_params["model"] == "openai/gpt-4o"
        assert "api_base" not in openai_params


class TestMinimaxSuggestedModels:
    """Tests that SUGGESTED_MODELS includes direct MiniMax entries."""

    def test_minimax_direct_in_suggested_models(self):
        # Load model_switcher directly to avoid agent/__init__.py imports
        _SWITCHER_PATH = _AGENT_DIR / "agent" / "core" / "model_switcher.py"

        # We need to stub out the effort_probe import that model_switcher uses
        from unittest.mock import MagicMock
        mock_module = MagicMock()
        sys.modules.setdefault("agent.core.effort_probe", mock_module)
        sys.modules.setdefault("agent.core", MagicMock())

        spec_s = importlib.util.spec_from_file_location("model_switcher_test", _SWITCHER_PATH)
        mod = importlib.util.module_from_spec(spec_s)  # type: ignore[arg-type]
        spec_s.loader.exec_module(mod)  # type: ignore[union-attr]

        ids = [m["id"] for m in mod.SUGGESTED_MODELS]
        assert "minimax/MiniMax-M2.7" in ids
        assert "minimax/MiniMax-M2.7-highspeed" in ids

    def test_minimax_model_ids_are_valid(self):
        # is_valid_model_id only does string checks, load similarly
        _SWITCHER_PATH = _AGENT_DIR / "agent" / "core" / "model_switcher.py"
        from unittest.mock import MagicMock
        sys.modules.setdefault("agent.core.effort_probe", MagicMock())

        spec_s = importlib.util.spec_from_file_location("model_switcher_test2", _SWITCHER_PATH)
        mod = importlib.util.module_from_spec(spec_s)  # type: ignore[arg-type]
        spec_s.loader.exec_module(mod)  # type: ignore[union-attr]

        assert mod.is_valid_model_id("minimax/MiniMax-M2.7")
        assert mod.is_valid_model_id("minimax/MiniMax-M2.7-highspeed")
