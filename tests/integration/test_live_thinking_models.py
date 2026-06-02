"""Opt-in live HF Router checks.

These tests intentionally call paid model APIs and are skipped unless
``ML_INTERN_LIVE_LLM_TESTS=1`` plus ``HF_TOKEN`` are set. They verify the
router-only request path without requiring provider-specific API keys.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from litellm import Message

from agent.core.agent_loop import (
    _assistant_message_from_result,
    _call_llm_streaming,
)
from agent.core.llm_params import _resolve_llm_params
from agent.core.model_ids import DEFAULT_MODEL_ID


if env_file := os.environ.get("ML_INTERN_LIVE_ENV_FILE"):
    load_dotenv(Path(env_file))

LIVE_TESTS_ENABLED = os.environ.get("ML_INTERN_LIVE_LLM_TESTS") == "1"
REPORT_RESULT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_result",
            "description": "Report the final test result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The exact marker requested by the test.",
                    }
                },
                "required": ["answer"],
            },
        },
    }
]


def _skip_without_live_flag() -> None:
    if not LIVE_TESTS_ENABLED:
        pytest.skip("set ML_INTERN_LIVE_LLM_TESTS=1 to run paid live LLM tests")


def _skip_without_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("set HF_TOKEN to run this live router test")


def _session(model_name: str):
    events = []

    async def send_event(event):
        events.append(event)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        is_cancelled=False,
        send_event=send_event,
        events=events,
    )


@pytest.mark.asyncio
async def test_live_default_router_model_does_not_replay_reasoning_metadata():
    _skip_without_live_flag()
    _skip_without_hf_token()

    session = _session(DEFAULT_MODEL_ID)
    llm_params = _resolve_llm_params(
        DEFAULT_MODEL_ID,
        os.environ["HF_TOKEN"],
        reasoning_effort="low",
        bill_to_user=True,
    )

    result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content="Call report_result with answer ROUTER_OK.",
            )
        ],
        tools=REPORT_RESULT_TOOL,
        llm_params=llm_params,
    )

    replay = _assistant_message_from_result(result)

    assert result.content or result.tool_calls_acc
    assert getattr(replay, "thinking_blocks", None) is None
    assert getattr(replay, "reasoning_content", None) is None
