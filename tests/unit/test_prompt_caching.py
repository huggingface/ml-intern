from litellm import Message

from agent.core.model_ids import HF_ROUTER_BASE_URL
from agent.core.prompt_caching import with_prompt_caching


def _fal_params() -> dict:
    return {
        "model": "openai/anthropic/claude-sonnet-4-6:fal-ai",
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

    cached_messages, cached_tools = with_prompt_caching(messages, tools, _fal_params())

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

    cached_messages, _ = with_prompt_caching(messages, None, _fal_params())

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

    cached_messages, _ = with_prompt_caching(messages, None, _fal_params())

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

    cached_messages, cached_tools = with_prompt_caching(messages, tools, _fal_params())

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


def test_prompt_caching_is_noop_for_non_router_fal_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    llm_params = {
        "model": "openai/anthropic/claude-sonnet-4-6:fal-ai",
        "api_base": "http://localhost:8000/v1",
    }

    cached_messages, _ = with_prompt_caching(messages, None, llm_params)

    assert cached_messages is messages
