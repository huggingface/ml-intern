from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.core.agent_loop import (
    LLMResult,
    _assistant_message_from_result,
    _call_llm_streaming,
    _strip_thinking_state_from_messages,
)


def test_assistant_message_from_result_keeps_content_and_tool_calls():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "date"}'},
    )
    result = LLMResult(
        content="working",
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
    )

    message = _assistant_message_from_result(result, tool_calls=[tool_call])

    assert message.content == "working"
    assert message.tool_calls == [tool_call]
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_strip_thinking_state_from_saved_messages():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "stale"},
                {"type": "text", "text": "done"},
            ],
            "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
            "reasoning_content": "stale",
            "provider_specific_fields": {
                "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
                "reasoning_content": "stale",
                "other": "kept",
            },
        }
    ]

    stripped = _strip_thinking_state_from_messages(messages)

    assert stripped == 5
    assert messages == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "provider_specific_fields": {"other": "kept"},
        }
    ]


@pytest.mark.asyncio
async def test_streaming_call_returns_wire_safe_result(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4.8:fal-ai"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/anthropic/claude-opus-4.8:fal-ai"},
    )

    assert result.content == "done"
    assert result.token_count == 3
