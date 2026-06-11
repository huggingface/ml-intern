from litellm import Message
from types import SimpleNamespace

from agent.core.model_ids import HF_ROUTER_BASE_URL
from agent.core.prompt_caching import (
    router_session_id_for,
    with_prompt_cache_params,
    with_prompt_caching,
)


def _anthropic_fal_params() -> dict:
    return {
        "model": "openai/anthropic/claude-opus-4.8:fal-ai",
        "api_base": HF_ROUTER_BASE_URL,
    }


def _gpt55_fal_params() -> dict:
    return {
        "model": "openai/openai/gpt-5.5:fal-ai",
        "api_base": HF_ROUTER_BASE_URL,
    }


def _kimi_novita_params() -> dict:
    return {
        "model": "openai/moonshotai/Kimi-K2.6:novita",
        "api_base": HF_ROUTER_BASE_URL,
    }


def _gpt_oss_cerebras_params() -> dict:
    return {
        "model": "openai/openai/gpt-oss-120b:cerebras",
        "api_base": HF_ROUTER_BASE_URL,
    }


def test_prompt_caching_marks_system_prefix_and_tools_for_fal_router_model():
    messages = [
        Message(role="system", content="stable system prompt"),
        Message(role="user", content="current question"),
    ]
    tools = [
        {"type": "function", "function": {"name": "read"}},
        {"type": "function", "function": {"name": "write"}},
    ]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _anthropic_fal_params()
    )

    assert cached_tools is not tools
    assert cached_tools == [
        {"type": "function", "function": {"name": "read"}},
        {
            "type": "function",
            "function": {"name": "write"},
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert "cache_control" not in tools[-1]
    assert cached_messages is not messages
    assert cached_messages[0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "stable system prompt",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    assert messages[0].content == "stable system prompt"


def test_prompt_caching_marks_last_stable_user_before_current_message():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "stable reference"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "current question"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _anthropic_fal_params())

    assert cached_messages[0]["content"] == "stable system"
    assert cached_messages[1]["content"] == [
        {
            "type": "text",
            "text": "stable reference",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert messages[1]["content"] == "stable reference"


def test_prompt_caching_marks_last_text_block_in_content_list():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "stable part one"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/i.png"},
                },
                {"type": "text", "text": "stable part two"},
            ],
        },
        {"role": "user", "content": "current question"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _anthropic_fal_params())

    assert cached_messages[0]["content"][0] == {
        "type": "text",
        "text": "stable part one",
    }
    assert cached_messages[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/i.png"},
    }
    assert cached_messages[0]["content"][2] == {
        "type": "text",
        "text": "stable part two",
        "cache_control": {"type": "ephemeral"},
    }
    assert "cache_control" not in messages[0]["content"][2]


def test_prompt_caching_marks_tools_without_message_prefix():
    messages = [{"role": "user", "content": "current question"}]
    tools = [{"type": "function", "function": {"name": "read"}}]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _anthropic_fal_params()
    )

    assert cached_messages is messages
    assert cached_tools == [
        {
            "type": "function",
            "function": {"name": "read"},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert "cache_control" not in tools[0]


def test_prompt_caching_is_noop_for_non_fal_router_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    llm_params = {
        "model": "openai/moonshotai/Kimi-K2.6",
        "api_base": HF_ROUTER_BASE_URL,
    }

    cached_messages, cached_tools = with_prompt_caching(messages, None, llm_params)

    assert cached_messages is messages
    assert cached_tools is None


def test_prompt_caching_is_noop_for_gpt55_fal_router_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    tools = [{"type": "function", "function": {"name": "read"}}]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _gpt55_fal_params()
    )

    assert cached_messages is messages
    assert cached_tools is tools


def test_prompt_caching_is_noop_for_non_router_fal_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    llm_params = {
        "model": "openai/anthropic/claude-opus-4.8:fal-ai",
        "api_base": "http://localhost:8000/v1",
    }

    cached_messages, _ = with_prompt_caching(messages, None, llm_params)

    assert cached_messages is messages


def test_prompt_cache_params_add_session_id_for_fal_router_model():
    llm_params = _anthropic_fal_params()

    cached_params = with_prompt_cache_params(llm_params, session_id="session-1")

    assert cached_params is not llm_params
    assert cached_params["extra_body"] == {
        "session_id": "session-1",
        "cache_control": {"type": "ephemeral"},
    }
    assert "extra_body" not in llm_params


def test_prompt_cache_params_adds_session_id_for_novita_router_model():
    llm_params = _kimi_novita_params()

    cached_params = with_prompt_cache_params(llm_params, session_id="session-1")

    assert cached_params is not llm_params
    assert cached_params["extra_body"] == {"session_id": "session-1"}
    assert "extra_body" not in llm_params


def test_prompt_cache_params_adds_session_id_for_cerebras_router_model():
    llm_params = _gpt_oss_cerebras_params()

    cached_params = with_prompt_cache_params(llm_params, session_id="session-1")

    assert cached_params is not llm_params
    assert cached_params["extra_body"] == {"session_id": "session-1"}
    assert "extra_body" not in llm_params


def test_router_session_id_prefers_billing_window_id():
    assert (
        router_session_id_for(
            SimpleNamespace(
                session_id="session-1",
                inference_billing_session_id="session-1:usage:window-1",
            )
        )
        == "session-1:usage:window-1"
    )
    assert router_session_id_for(SimpleNamespace(session_id="session-1")) == "session-1"


def test_prompt_cache_params_adds_anthropic_cache_control_without_session_id():
    cached_params = with_prompt_cache_params(_anthropic_fal_params())

    assert cached_params["extra_body"] == {"cache_control": {"type": "ephemeral"}}


def test_prompt_cache_params_merges_gpt55_cache_hints():
    llm_params = {
        **_gpt55_fal_params(),
        "extra_body": {"reasoning_effort": "high"},
    }

    cached_params = with_prompt_cache_params(llm_params, session_id="session-1")

    assert cached_params["extra_body"] == {
        "reasoning_effort": "high",
        "session_id": "session-1",
        "prompt_cache_key": "session-1",
        "prompt_cache_retention": "24h",
    }
    assert llm_params["extra_body"] == {"reasoning_effort": "high"}


def test_prompt_cache_params_adds_gpt55_retention_without_session():
    cached_params = with_prompt_cache_params(_gpt55_fal_params())

    assert cached_params["extra_body"] == {"prompt_cache_retention": "24h"}


def test_prompt_cache_params_is_noop_for_non_router_model():
    llm_params = {
        "model": "openai/openai/gpt-5.5:fal-ai",
        "api_base": "http://localhost:8000/v1",
    }

    cached_params = with_prompt_cache_params(llm_params, session_id="session-1")

    assert cached_params is llm_params
