"""Regression tests for interactive CLI rendering and research model routing."""

from io import StringIO
from types import SimpleNamespace

from agent.tools.research_tool import _get_research_model
from agent.core.model_switcher import SUGGESTED_MODELS, is_valid_model_id
from agent.utils import terminal_display


def test_direct_anthropic_research_model_stays_off_bedrock():
    assert _get_research_model("anthropic/claude-opus-4-6") == "anthropic/claude-sonnet-4-6"


def test_bedrock_anthropic_research_model_stays_on_bedrock():
    assert (
        _get_research_model("bedrock/us.anthropic.claude-opus-4-6-v1")
        == "bedrock/us.anthropic.claude-sonnet-4-6"
    )


def test_non_anthropic_research_model_is_unchanged():
    assert _get_research_model("openai/gpt-5.4") == "openai/gpt-5.4"


def test_gemini_and_vertex_ai_model_ids_are_valid():
    assert is_valid_model_id("google/gemini-3.1-pro-preview")
    assert is_valid_model_id("google/gemini-2.5-flash-lite-preview-09-2025")
    assert is_valid_model_id("google-geap/gemini-3-flash-preview")
    assert is_valid_model_id("google-geap/gemini-2.5-pro")


def test_google_preview_models_are_suggested():
    suggested = {m["id"] for m in SUGGESTED_MODELS}

    assert "google/gemini-3.1-pro-preview" in suggested
    assert "google/gemini-3-flash-preview" in suggested
    assert "google/gemini-3.1-flash-lite-preview" in suggested
    assert "google-geap/gemini-3.1-pro-preview" in suggested
    assert "google-geap/gemini-3-flash-preview" in suggested
    assert "google/deep-research-pro-preview-12-2025" not in suggested
    assert "google/deep-research-max-preview-04-2026" not in suggested


def test_subagent_display_does_not_spawn_background_redraw(monkeypatch):
    calls: list[object] = []

    def _unexpected_future(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("background redraw task should not be created")

    monkeypatch.setattr("asyncio.ensure_future", _unexpected_future)
    monkeypatch.setattr(
        terminal_display,
        "_console",
        SimpleNamespace(file=StringIO(), width=100),
    )

    mgr = terminal_display.SubAgentDisplayManager()
    mgr.start("agent-1", "research")
    mgr.add_call("agent-1", "▸ hf_papers  {\"operation\": \"search\"}")
    mgr.clear("agent-1")

    assert calls == []
