from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.core import agent_loop
from agent.core.agent_loop import (
    LLMResult,
    _call_llm_streaming,
    _assistant_message_from_result,
    _extract_thinking_state,
)


def test_extract_thinking_state_from_litellm_message():
    message = Message(
        role="assistant",
        content="working",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    assert thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert reasoning_content == "reasoned"


def test_extract_thinking_state_from_provider_fields():
    message = SimpleNamespace(
        provider_specific_fields={
            "thinking_blocks": [{"type": "thinking", "thinking": "reasoned"}],
            "reasoning_content": "reasoned",
        },
    )

    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    assert thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert reasoning_content == "reasoned"


def test_assistant_message_from_result_strips_router_thinking_with_tool_calls():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "date"}'},
    )
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="anthropic/claude-opus-4.8:fal-ai",
        tool_calls=[tool_call],
    )

    assert message.tool_calls == [tool_call]
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_assistant_message_from_result_strips_router_reasoning_content():
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="openai/gpt-5.5:fal-ai",
    )

    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_assistant_message_from_result_omits_absent_thinking_fields():
    result = LLMResult(
        content="done",
        tool_calls_acc={},
        token_count=12,
        finish_reason="stop",
    )

    message = _assistant_message_from_result(
        result,
        model_name="anthropic/claude-opus-4.8:fal-ai",
    )

    assert message.content == "done"
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


@pytest.mark.asyncio
async def test_streaming_call_skips_chunk_rebuild_for_router_models(monkeypatch):
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

    def fail_chunk_builder(*_args, **_kwargs):
        raise AssertionError("stream_chunk_builder should not run")

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4.8:fal-ai"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fail_chunk_builder)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/anthropic/claude-opus-4.8:fal-ai"},
    )

    assert result.content == "done"
    assert result.thinking_blocks is None
    assert result.reasoning_content is None
