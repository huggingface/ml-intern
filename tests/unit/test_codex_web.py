import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from codex_web import (  # noqa: E402
    CodexWebRuntime,
    _catalog_to_web_models,
    codex_web_enabled,
)


class StubContextManager:
    def __init__(self):
        self.items = []

    def add_message(self, message):
        self.items.append(message)


class StubSession:
    def __init__(self):
        self.config = SimpleNamespace(model_name="codex/default")
        self.context_manager = StubContextManager()
        self.events = []
        self.is_cancelled = False
        self.turn_count = 0

    def reset_cancel(self):
        self.is_cancelled = False

    async def send_event(self, event):
        self.events.append(event)

    def increment_turn(self):
        self.turn_count += 1

    async def auto_save_if_needed(self):
        return None


def test_codex_web_requires_explicit_local_only_flag(monkeypatch):
    monkeypatch.setenv("ML_INTERN_ENABLE_CODEX_WEB", "1")
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setattr("codex_web.shutil.which", lambda _name: "/usr/bin/codex")

    assert codex_web_enabled() is True

    monkeypatch.setenv("SPACE_ID", "owner/space")
    assert codex_web_enabled() is False


def test_codex_catalog_maps_models_and_reasoning_options():
    models = _catalog_to_web_models(
        [
            {
                "id": "gpt-5.6-sol",
                "model": "gpt-5.6-sol",
                "displayName": "GPT-5.6-Sol",
                "description": "Latest frontier model",
                "isDefault": True,
                "defaultReasoningEffort": "low",
                "supportedReasoningEfforts": [
                    {
                        "reasoningEffort": "low",
                        "description": "Fast responses",
                    },
                    {
                        "reasoningEffort": "max",
                        "description": "Maximum reasoning",
                    },
                ],
            },
            {
                "id": "hidden-model",
                "model": "hidden-model",
                "hidden": True,
            },
        ]
    )

    assert [model["id"] for model in models] == [
        "codex/default",
        "codex/gpt-5.6-sol",
    ]
    assert models[0]["label"] == "Codex Auto (GPT-5.6-Sol)"
    assert models[0]["default_reasoning_effort"] == "low"
    assert models[1]["reasoning_efforts"] == [
        {"id": "low", "description": "Fast responses"},
        {"id": "max", "description": "Maximum reasoning"},
    ]


@pytest.mark.asyncio
async def test_web_runtime_emits_sse_contract_and_persists_messages(tmp_path):
    session = StubSession()
    agent_session = SimpleNamespace(
        session=session,
        tool_router=SimpleNamespace(),
        hf_token=None,
    )
    web_runtime = CodexWebRuntime(
        agent_session=agent_session,
        project_root=tmp_path,
    )

    class Runtime:
        async def run_turn(self, prompt, *, stream):
            assert prompt == "Find one LoRA paper"
            assert stream is True
            await web_runtime._on_delta("Found it")
            return "Found it"

    web_runtime.runtime = Runtime()

    result = await web_runtime.run_user_input("Find one LoRA paper")

    assert result == "Found it"
    assert [message.role for message in session.context_manager.items] == [
        "user",
        "assistant",
    ]
    assert [event.event_type for event in session.events] == [
        "processing",
        "assistant_chunk",
        "assistant_stream_end",
        "turn_complete",
    ]
    assert session.events[-1].data["final_response"] == "Found it"
    assert session.turn_count == 1


@pytest.mark.asyncio
async def test_web_runtime_surfaces_ml_tool_events(tmp_path):
    session = StubSession()
    web_runtime = CodexWebRuntime(
        agent_session=SimpleNamespace(
            session=session,
            tool_router=SimpleNamespace(),
            hf_token=None,
        ),
        project_root=tmp_path,
    )

    await web_runtime._on_tool(
        "hf_papers",
        {"query": "LoRA"},
        None,
        None,
        "call-1",
    )
    await web_runtime._on_tool(
        "hf_papers",
        {"query": "LoRA"},
        "paper result",
        True,
        "call-1",
    )

    assert [event.event_type for event in session.events] == [
        "tool_call",
        "tool_output",
    ]
    assert session.events[0].data["tool_call_id"] == "call-1"
    assert session.events[1].data["success"] is True
