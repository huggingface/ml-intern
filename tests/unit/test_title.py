from types import SimpleNamespace

import pytest

from agent.core.title import (
    extract_first_user_text,
    fallback_title,
    generate_conversation_title,
)


def test_fallback_title_collapses_and_caps_words():
    out = fallback_title("  fine-tune   a   model   on   squad   data   please   now  ")
    assert out == "fine-tune a model on squad data"  # capped at 6 words


def test_fallback_title_empty_input():
    assert fallback_title("") == ""
    assert fallback_title(None) == ""
    assert fallback_title("    ") == ""


def test_fallback_title_strips_long_secrets():
    secret = "hf_" + "a" * 40
    out = fallback_title(f"use token {secret} to login")
    assert secret not in out
    assert "use token" in out


def test_extract_first_user_text_from_dicts():
    items = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first task here"},
        {"role": "user", "content": "second"},
    ]
    assert extract_first_user_text(items) == "first task here"


def test_extract_first_user_text_from_blocks():
    # Block-list content appears in dict-shaped messages (e.g. from saved JSON);
    # litellm's Message only accepts string content.
    items = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "block one"}]},
    ]
    assert extract_first_user_text(items) == "block one"


def test_extract_first_user_text_none_when_no_user():
    items = [{"role": "system", "content": "sys"}]
    assert extract_first_user_text(items) == ""


def _fake_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.asyncio
async def test_generate_title_cleans_model_output(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _fake_response('"Fine-Tune Llama On SQuAD."')

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    out = await generate_conversation_title(
        "openai/gpt-5.5", None, "help me fine-tune llama", "sure"
    )
    assert out == "Fine-Tune Llama On SQuAD"  # quotes + trailing dot stripped


@pytest.mark.asyncio
async def test_generate_title_falls_back_on_empty(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _fake_response("")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    out = await generate_conversation_title(
        "openai/gpt-5.5", None, "process my dataset", None
    )
    assert out == fallback_title("process my dataset")


@pytest.mark.asyncio
async def test_generate_title_falls_back_on_error(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("litellm.acompletion", boom)
    out = await generate_conversation_title(
        "openai/gpt-5.5", None, "run inference on a model", None
    )
    assert out == fallback_title("run inference on a model")


@pytest.mark.asyncio
async def test_generate_title_empty_user_text_returns_fallback():
    # No LLM call should be needed when there's nothing to title from.
    out = await generate_conversation_title("openai/gpt-5.5", None, "", None)
    assert out == ""
