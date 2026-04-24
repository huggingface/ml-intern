"""Tests for agent/tools/web_search_tool.py — Exa-backed web search."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is importable (tests/unit is two levels below root).
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Stub exa_py so tests do not require the package to be installed.
if "exa_py" not in sys.modules:
    stub = types.ModuleType("exa_py")
    stub.Exa = MagicMock()  # type: ignore[attr-defined]
    sys.modules["exa_py"] = stub

from agent.tools.web_search_tool import (  # noqa: E402
    ENV_VAR,
    INTEGRATION_NAME,
    WEB_SEARCH_TOOL_SPEC,
    WebSearchResult,
    _coerce_result,
    _format_results,
    web_search_enabled,
    web_search_handler,
)


# ---------------------------------------------------------------------------
# Env gating
# ---------------------------------------------------------------------------


def test_enabled_only_when_env_var_set(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert web_search_enabled() is False

    monkeypatch.setenv(ENV_VAR, "sk-test")
    assert web_search_enabled() is True


@pytest.mark.asyncio
async def test_handler_refuses_without_api_key(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    output, ok = await web_search_handler({"query": "anything"})
    assert ok is False
    assert ENV_VAR in output


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_requires_query(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")
    output, ok = await web_search_handler({"query": "   "})
    assert ok is False
    assert "query" in output.lower()


@pytest.mark.asyncio
async def test_handler_rejects_invalid_type(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")
    output, ok = await web_search_handler({"query": "x", "type": "keyword"})
    assert ok is False
    assert "type" in output.lower()


@pytest.mark.asyncio
async def test_handler_rejects_invalid_category(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")
    output, ok = await web_search_handler({"query": "x", "category": "bogus"})
    assert ok is False
    assert "category" in output.lower()


@pytest.mark.asyncio
async def test_handler_rejects_non_list_domains(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")
    output, ok = await web_search_handler(
        {"query": "x", "include_domains": "arxiv.org"}
    )
    assert ok is False
    assert "include_domains" in output


# ---------------------------------------------------------------------------
# Result parsing and snippet cascade
# ---------------------------------------------------------------------------


def test_coerce_result_from_object():
    raw = types.SimpleNamespace(
        title="A paper",
        url="https://example.com/paper",
        published_date="2024-06-01",
        author="Author",
        score=0.83,
        summary="A short summary.",
        highlights=["h1", "h2"],
        text="Full text body",
    )
    r = _coerce_result(raw)
    assert r.title == "A paper"
    assert r.url == "https://example.com/paper"
    assert r.published_date == "2024-06-01"
    assert r.summary == "A short summary."
    assert r.highlights == ["h1", "h2"]


def test_coerce_result_from_dict_with_camel_case_date():
    raw = {
        "title": "T",
        "url": "https://x",
        "publishedDate": "2025-01-01",
    }
    r = _coerce_result(raw)
    assert r.published_date == "2025-01-01"


def test_snippet_prefers_summary():
    r = WebSearchResult(
        title="t",
        url="u",
        summary="summary wins",
        highlights=["h1", "h2"],
        text="text loses",
    )
    assert r.snippet() == "summary wins"


def test_snippet_falls_back_to_highlights_then_text():
    r1 = WebSearchResult(title="t", url="u", highlights=["h1", "h2"], text="text")
    assert "h1" in r1.snippet() and "h2" in r1.snippet()

    r2 = WebSearchResult(title="t", url="u", text="just text")
    assert r2.snippet() == "just text"

    r3 = WebSearchResult(title="t", url="u")
    assert r3.snippet() == ""


def test_snippet_truncates_long_content():
    long_summary = "x" * 1000
    r = WebSearchResult(title="t", url="u", summary=long_summary)
    snippet = r.snippet(max_characters=50)
    assert len(snippet) <= 50
    assert snippet.endswith("…")


def test_format_results_empty():
    out = _format_results("no hits", [], "auto", None)
    assert "No web results" in out


def test_format_results_shows_metadata_and_snippet():
    results = [
        WebSearchResult(
            title="Cool blog post",
            url="https://example.com/post",
            published_date="2025-02-01",
            author="Jane Doe",
            score=0.91,
            summary="TL;DR of the post.",
        )
    ]
    out = _format_results("cool post", results, "auto", "neural")
    assert "Cool blog post" in out
    assert "https://example.com/post" in out
    assert "2025-02-01" in out
    assert "Jane Doe" in out
    assert "0.91" in out
    assert "TL;DR" in out
    assert "auto → resolved=neural" in out


# ---------------------------------------------------------------------------
# Integration header + end-to-end handler path (with mocked Exa client)
# ---------------------------------------------------------------------------


class _FakeExa:
    """Minimal Exa stand-in that records its call and returns canned results."""

    last_instance: "_FakeExa | None" = None

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers: dict[str, str] = {}
        self.search_calls: list[dict] = []
        self.search_and_contents_calls: list[dict] = []
        _FakeExa.last_instance = self

    def _response(self):
        return types.SimpleNamespace(
            results=[
                types.SimpleNamespace(
                    title="Result 1",
                    url="https://example.com/1",
                    published_date="2025-03-01",
                    author=None,
                    score=0.7,
                    summary="Summary 1",
                    highlights=["hl1"],
                    text=None,
                )
            ],
            resolved_search_type="neural",
        )

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self._response()

    def search_and_contents(self, **kwargs):
        self.search_and_contents_calls.append(kwargs)
        return self._response()


@pytest.mark.asyncio
async def test_handler_happy_path_sets_integration_header(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")

    # Patch on the source module — the tool imports Exa locally via
    # `from exa_py import Exa`, so monkeypatching the consuming namespace
    # (agent.tools.web_search_tool.Exa) would miss it.
    with patch("exa_py.Exa", _FakeExa):
        output, ok = await web_search_handler(
            {
                "query": "flash attention",
                "num_results": 3,
                "type": "auto",
                "category": "research paper",
                "include_domains": ["arxiv.org"],
                "start_published_date": "2024-01-01",
            }
        )

    assert ok is True
    assert "Result 1" in output
    assert "https://example.com/1" in output

    inst = _FakeExa.last_instance
    assert inst is not None
    # Integration attribution header must be set on every client.
    assert inst.headers.get("x-exa-integration") == INTEGRATION_NAME
    # Defaults enable summary + highlights, so search_and_contents is used.
    assert len(inst.search_and_contents_calls) == 1
    call = inst.search_and_contents_calls[0]
    assert call["query"] == "flash attention"
    assert call["num_results"] == 3
    assert call["type"] == "auto"
    assert call["category"] == "research paper"
    assert call["include_domains"] == ["arxiv.org"]
    assert call["start_published_date"] == "2024-01-01"
    assert call.get("summary") is True
    assert call.get("highlights") is True


@pytest.mark.asyncio
async def test_handler_plain_search_when_no_contents_requested(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")

    with patch("exa_py.Exa", _FakeExa):
        output, ok = await web_search_handler(
            {
                "query": "no extras",
                "include_summary": False,
                "include_highlights": False,
                "include_text": False,
            }
        )

    assert ok is True
    inst = _FakeExa.last_instance
    assert inst is not None
    assert len(inst.search_calls) == 1
    assert inst.search_and_contents_calls == []


@pytest.mark.asyncio
async def test_handler_caps_num_results(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")

    with patch("exa_py.Exa", _FakeExa):
        _, ok = await web_search_handler({"query": "x", "num_results": 9999})

    assert ok is True
    inst = _FakeExa.last_instance
    assert inst is not None
    call = inst.search_and_contents_calls[0]
    assert 1 <= call["num_results"] <= 25


# ---------------------------------------------------------------------------
# Router gating
# ---------------------------------------------------------------------------


def test_tool_not_registered_when_api_key_unset(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)

    # Import lazily to avoid pulling in heavy agent modules during collection.
    from agent.core.tools import create_builtin_tools

    names = {t.name for t in create_builtin_tools(local_mode=True)}
    assert "web_search" not in names


def test_tool_registered_when_api_key_set(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "sk-test")

    from agent.core.tools import create_builtin_tools

    names = {t.name for t in create_builtin_tools(local_mode=True)}
    assert "web_search" in names


# ---------------------------------------------------------------------------
# Tool spec shape
# ---------------------------------------------------------------------------


def test_tool_spec_shape():
    assert WEB_SEARCH_TOOL_SPEC["name"] == "web_search"
    params = WEB_SEARCH_TOOL_SPEC["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["required"] == ["query"]
