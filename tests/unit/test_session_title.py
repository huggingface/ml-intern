import asyncio
from types import SimpleNamespace

import pytest
from litellm import Message

import agent.main as main_mod
from agent.core.agent_loop import _generate_and_set_title
from agent.core.session import Event, Session


class _FakeConfig:
    model_name = "openai/gpt-5.5:fal-ai"
    save_sessions = False
    session_dataset_repo = "fake/repo"
    auto_save_interval = 1
    heartbeat_interval_s = 60
    max_iterations = 10
    yolo_mode = False
    confirm_cpu_jobs = False
    auto_file_upload = False
    reasoning_effort = None
    share_traces = False
    personal_trace_repo_template = None
    mcpServers: dict = {}


class _FakeContext:
    def __init__(self) -> None:
        self.items = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="train a model"),
        ]
        self.model_max_tokens = 200_000
        self.running_context_usage = 0
        self.on_message_added = None


def _make_session() -> Session:
    return Session(
        event_queue=asyncio.Queue(),
        config=_FakeConfig(),
        tool_router=None,
        context_manager=_FakeContext(),
        hf_token=None,
        user_id="user-a",
        local_mode=True,
    )


def test_get_trajectory_includes_session_title_default_none():
    session = _make_session()
    traj = session.get_trajectory()
    assert "session_title" in traj
    assert traj["session_title"] is None


def test_get_trajectory_reflects_set_title():
    session = _make_session()
    session.session_title = "my experiment"
    assert session.get_trajectory()["session_title"] == "my experiment"


def test_start_new_conversation_resets_title():
    session = _make_session()
    session.session_title = "named run"
    session._title_user_set = True
    session.start_new_conversation()
    assert session.session_title is None
    assert session._title_user_set is False


@pytest.mark.asyncio
async def test_rename_command_sets_title():
    session = _make_session()
    result = await main_mod._handle_slash_command(
        "/rename my-experiment",
        config=session.config,
        session_holder=[session],
        submission_queue=asyncio.Queue(),
        submission_id=[0],
    )
    assert result is None
    assert session.session_title == "my-experiment"
    assert session._title_user_set is True


def _fake_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


async def _drain_events(queue: asyncio.Queue) -> list[Event]:
    events: list[Event] = []
    while not queue.empty():
        events.append(await queue.get())
    return events


@pytest.mark.asyncio
async def test_auto_title_sets_title_and_emits_event(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _fake_response("Train A Model")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    session = _make_session()  # context has a "train a model" user message

    await _generate_and_set_title(session, "here is how")

    assert session.session_title == "Train A Model"
    events = await _drain_events(session.event_queue)
    titled = [e for e in events if e.event_type == "conversation_title"]
    assert titled, "expected a conversation_title event"
    assert titled[0].data["title"] == "Train A Model"


@pytest.mark.asyncio
async def test_auto_title_does_not_override_user_rename(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _fake_response("Auto Generated Name")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    session = _make_session()
    session.session_title = "my-name"
    session._title_user_set = True

    await _generate_and_set_title(session, "here is how")

    assert session.session_title == "my-name"  # user's name preserved
    events = await _drain_events(session.event_queue)
    assert not [e for e in events if e.event_type == "conversation_title"]


@pytest.mark.asyncio
async def test_rename_command_empty_arg_is_noop():
    session = _make_session()
    result = await main_mod._handle_slash_command(
        "/rename",
        config=session.config,
        session_holder=[session],
        submission_queue=asyncio.Queue(),
        submission_id=[0],
    )
    assert result is None
    assert session.session_title is None
    assert session._title_user_set is False
