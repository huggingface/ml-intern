"""Regression tests for interactive CLI rendering and research model routing."""

import asyncio
import sys
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import agent.main as main_mod
from agent.core.session import Event
from agent.tools.research_tool import _get_research_model
from agent.utils import terminal_display


async def _fake_hf_identity(_token):
    return "tester", "pro"


def test_router_anthropic_research_model_is_unchanged():
    assert (
        _get_research_model("anthropic/claude-opus-4.8:fal-ai")
        == "anthropic/claude-opus-4.8:fal-ai"
    )


def test_non_anthropic_research_model_is_unchanged():
    assert _get_research_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"


def test_huggingface_prefix_research_model_strips_prefix():
    assert (
        _get_research_model("huggingface/anthropic/claude-opus-4.8:fal-ai")
        == "anthropic/claude-opus-4.8:fal-ai"
    )


def test_hf_router_openai_research_model_is_unchanged():
    assert (
        _get_research_model("huggingface/openai/gpt-5.5:fal-ai")
        == "openai/gpt-5.5:fal-ai"
    )


def test_help_output_keeps_descriptions_aligned(monkeypatch):
    output = StringIO()
    console = Console(
        file=output,
        color_system=None,
        theme=terminal_display._THEME,
        width=120,
    )
    monkeypatch.setattr(terminal_display, "_console", console)

    terminal_display.print_help()

    lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
    description_columns = []
    for command, args, description in terminal_display.HELP_ROWS:
        line = next(line for line in lines if command in line)
        if args:
            assert args in line
        description_columns.append(line.index(description))

    assert len(set(description_columns)) == 1


def test_help_output_recomputes_widths_from_rows():
    rows = terminal_display.HELP_ROWS + (
        ("/longer-command", "[longer-args]", "Synthetic help row"),
    )
    output = StringIO()
    Console(
        file=output,
        color_system=None,
        theme=terminal_display._THEME,
        width=140,
    ).print(terminal_display.format_help_text(rows))

    lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
    description_columns = [
        next(line for line in lines if command in line).index(description)
        for command, _args, description in rows
    ]

    assert len(set(description_columns)) == 1


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
    mgr.add_call("agent-1", '▸ hf_papers  {"operation": "search"}')
    mgr.clear("agent-1")

    assert calls == []


@pytest.mark.asyncio
async def test_hf_identity_uses_single_whoami_v2_payload(monkeypatch):
    calls: list[tuple[str, float]] = []

    async def fake_fetch_whoami_v2(token, timeout=5.0):
        calls.append((token, timeout))
        return {"name": "tester", "isPro": True}

    def fail_get_hf_user(_token):
        raise AssertionError("fallback whoami should not run when whoami-v2 succeeds")

    monkeypatch.setattr(main_mod, "fetch_whoami_v2", fake_fetch_whoami_v2)
    monkeypatch.setattr(main_mod, "_get_hf_user", fail_get_hf_user)

    assert await main_mod._get_hf_identity("hf-token") == ("tester", "pro")
    assert calls == [("hf-token", 5.0)]


def test_cli_forwards_model_flag_to_interactive_main(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_main(*, model=None, sandbox_tools=False):
        seen["model"] = model
        seen["sandbox_tools"] = sandbox_tools

    monkeypatch.setattr(sys, "argv", ["ml-intern", "--model", "openai/gpt-5.5:fal-ai"])
    monkeypatch.setattr(main_mod, "main", fake_main)

    main_mod.cli()

    assert seen["model"] == "openai/gpt-5.5:fal-ai"
    assert seen["sandbox_tools"] is False


def test_cli_forwards_sandbox_flag_to_interactive_main(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_main(*, model=None, sandbox_tools=False):
        seen["model"] = model
        seen["sandbox_tools"] = sandbox_tools

    monkeypatch.setattr(sys, "argv", ["ml-intern", "--sandbox-tools"])
    monkeypatch.setattr(main_mod, "main", fake_main)

    main_mod.cli()

    assert seen == {"model": None, "sandbox_tools": True}


def test_cli_forwards_sandbox_flag_to_headless_main(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_headless_main(
        prompt,
        *,
        model=None,
        max_iterations=None,
        stream=True,
        sandbox_tools=False,
    ):
        seen.update(
            {
                "prompt": prompt,
                "model": model,
                "max_iterations": max_iterations,
                "stream": stream,
                "sandbox_tools": sandbox_tools,
            }
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ml-intern", "--sandbox-tools", "--no-stream", "train a model"],
    )
    monkeypatch.setattr(main_mod, "headless_main", fake_headless_main)

    main_mod.cli()

    assert seen == {
        "prompt": "train a model",
        "model": None,
        "max_iterations": None,
        "stream": False,
        "sandbox_tools": True,
    }


@pytest.mark.asyncio
async def test_interactive_main_applies_model_override_before_banner(monkeypatch):
    class StopAfterBanner(Exception):
        pass

    def fake_banner(*, model=None, hf_user=None, tool_runtime=None):
        assert model == "openai/gpt-5.5:fal-ai"
        assert hf_user == "tester"
        assert tool_runtime == "local filesystem"
        raise StopAfterBanner

    monkeypatch.setattr(main_mod.os, "system", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: "hf-token")
    monkeypatch.setattr(main_mod, "_get_hf_identity", _fake_hf_identity)
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="moonshotai/Kimi-K2.6",
            mcpServers={},
            tool_runtime="local",
        ),
    )
    monkeypatch.setattr(main_mod, "print_banner", fake_banner)

    with pytest.raises(StopAfterBanner):
        await main_mod.main(model="openai/gpt-5.5:fal-ai")


@pytest.mark.asyncio
async def test_local_model_local_runtime_skips_hf_token_prompt(monkeypatch):
    class StopAfterBanner(Exception):
        pass

    async def fail_prompt(_prompt_session):
        raise AssertionError("local model with local tools should not prompt")

    def fake_banner(*, model=None, hf_user=None, tool_runtime=None):
        assert model == "llamacpp/model"
        assert hf_user is None
        assert tool_runtime == "local filesystem"
        raise StopAfterBanner

    monkeypatch.setattr(main_mod.os, "system", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: None)
    monkeypatch.setattr(main_mod, "_prompt_and_save_hf_token", fail_prompt)
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="llamacpp/model",
            mcpServers={},
            tool_runtime="local",
        ),
    )
    monkeypatch.setattr(main_mod, "print_banner", fake_banner)

    with pytest.raises(StopAfterBanner):
        await main_mod.main()


@pytest.mark.asyncio
async def test_local_model_sandbox_runtime_prompts_for_hf_token(monkeypatch):
    class StopAfterBanner(Exception):
        pass

    prompted = False

    async def fake_prompt(_prompt_session):
        nonlocal prompted
        prompted = True
        return "hf-token"

    def fake_banner(*, model=None, hf_user=None, tool_runtime=None):
        assert model == "llamacpp/model"
        assert hf_user == "tester"
        assert tool_runtime == "HF sandbox"
        raise StopAfterBanner

    monkeypatch.setattr(main_mod.os, "system", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: None)
    monkeypatch.setattr(main_mod, "_prompt_and_save_hf_token", fake_prompt)
    monkeypatch.setattr(main_mod, "_get_hf_identity", _fake_hf_identity)
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="llamacpp/model",
            mcpServers={},
            tool_runtime="local",
        ),
    )
    monkeypatch.setattr(main_mod, "print_banner", fake_banner)

    with pytest.raises(StopAfterBanner):
        await main_mod.main(sandbox_tools=True)

    assert prompted is True


@pytest.mark.asyncio
async def test_interactive_main_passes_sandbox_runtime_to_tool_router(monkeypatch):
    class StopAfterToolRouter(Exception):
        pass

    seen: dict[str, object] = {}

    class FakeGateway:
        def __init__(self, _config):
            pass

        async def start(self):
            pass

    class FakeToolRouter:
        def __init__(self, mcp_servers, *, hf_token=None, local_mode=True):
            seen["mcp_servers"] = mcp_servers
            seen["hf_token"] = hf_token
            seen["local_mode"] = local_mode
            raise StopAfterToolRouter

    from agent.core import hf_router_catalog

    monkeypatch.setattr(main_mod.os, "system", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: "hf-token")
    monkeypatch.setattr(main_mod, "_get_hf_identity", _fake_hf_identity)
    monkeypatch.setattr(main_mod, "print_banner", lambda **_kwargs: None)
    monkeypatch.setattr(hf_router_catalog, "prewarm", lambda: None)
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="llamacpp/model",
            mcpServers={"server": object()},
            messaging=SimpleNamespace(default_auto_destinations=lambda: []),
            tool_runtime="local",
        ),
    )
    monkeypatch.setattr(main_mod, "NotificationGateway", FakeGateway)
    monkeypatch.setattr(main_mod, "ToolRouter", FakeToolRouter)

    with pytest.raises(StopAfterToolRouter):
        await main_mod.main(sandbox_tools=True)

    assert seen["hf_token"] == "hf-token"
    assert seen["local_mode"] is False


@pytest.mark.asyncio
async def test_interactive_main_passes_user_plan_to_submission_loop(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_get_hf_identity(token):
        seen["plan_token"] = token
        return "tester", "free"

    class FakeGateway:
        def __init__(self, _config):
            pass

        async def start(self):
            pass

        async def close(self):
            pass

    class FakeToolRouter:
        def __init__(self, _mcp_servers, *, hf_token=None, local_mode=True):
            self.hf_token = hf_token
            self.local_mode = local_mode

        async def __aexit__(self, *_args):
            pass

    async def fake_submission_loop(submission_queue, _event_queue, **kwargs):
        seen["user_plan"] = kwargs.get("user_plan")
        seen["hf_token"] = kwargs.get("hf_token")
        while True:
            submission = await submission_queue.get()
            if submission.operation.op_type == main_mod.OpType.SHUTDOWN:
                return

    async def fake_event_listener(
        _event_queue,
        _submission_queue,
        _turn_complete_event,
        ready_event,
        _prompt_session,
        _config,
        *,
        session_holder=None,
    ):
        ready_event.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return

    async def fake_get_user_input(_prompt_session):
        return "/quit"

    from agent.core import hf_router_catalog

    monkeypatch.setattr(main_mod, "_clear_terminal", lambda: None)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: "hf-token")
    monkeypatch.setattr(main_mod, "_get_hf_identity", fake_get_hf_identity)
    monkeypatch.setattr(main_mod, "print_banner", lambda **_kwargs: None)
    monkeypatch.setattr(main_mod, "NotificationGateway", FakeGateway)
    monkeypatch.setattr(main_mod, "ToolRouter", FakeToolRouter)
    monkeypatch.setattr(main_mod, "submission_loop", fake_submission_loop)
    monkeypatch.setattr(main_mod, "event_listener", fake_event_listener)
    monkeypatch.setattr(main_mod, "get_user_input", fake_get_user_input)
    monkeypatch.setattr(hf_router_catalog, "prewarm", lambda: None)
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="anthropic/claude-opus-4.8:fal-ai",
            mcpServers={},
            messaging=SimpleNamespace(default_auto_destinations=lambda: []),
            tool_runtime="local",
        ),
    )

    await main_mod.main()

    assert seen == {
        "plan_token": "hf-token",
        "user_plan": "free",
        "hf_token": "hf-token",
    }


@pytest.mark.asyncio
async def test_headless_main_passes_user_plan_to_submission_loop(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_get_hf_identity(token):
        seen["plan_token"] = token
        return "tester", "pro"

    class FakeGateway:
        def __init__(self, _config):
            pass

        async def start(self):
            pass

        async def close(self):
            pass

    class FakeToolRouter:
        def __init__(self, _mcp_servers, *, hf_token=None, local_mode=True):
            self.hf_token = hf_token
            self.local_mode = local_mode

        async def __aexit__(self, *_args):
            pass

    async def fake_submission_loop(submission_queue, event_queue, **kwargs):
        seen["user_plan"] = kwargs.get("user_plan")
        seen["hf_token"] = kwargs.get("hf_token")
        await event_queue.put(Event(event_type="ready"))
        submission = await submission_queue.get()
        seen["prompt"] = submission.operation.data["text"]
        await event_queue.put(
            Event(event_type="turn_complete", data={"history_size": 0})
        )
        shutdown = await submission_queue.get()
        assert shutdown.operation.op_type == main_mod.OpType.SHUTDOWN

    monkeypatch.setattr(main_mod, "_clear_terminal", lambda: None)
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: "hf-token")
    monkeypatch.setattr(main_mod, "_get_hf_identity", fake_get_hf_identity)
    monkeypatch.setattr(main_mod, "NotificationGateway", FakeGateway)
    monkeypatch.setattr(main_mod, "ToolRouter", FakeToolRouter)
    monkeypatch.setattr(main_mod, "submission_loop", fake_submission_loop)
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="anthropic/claude-opus-4.8:fal-ai",
            mcpServers={},
            messaging=SimpleNamespace(default_auto_destinations=lambda: []),
            tool_runtime="local",
            max_iterations=3,
        ),
    )

    await main_mod.headless_main("train a model")

    assert seen == {
        "plan_token": "hf-token",
        "user_plan": "pro",
        "hf_token": "hf-token",
        "prompt": "train a model",
    }


@pytest.mark.asyncio
async def test_initial_sandbox_preload_waits_before_prompt():
    waited = False

    async def preload():
        nonlocal waited
        await asyncio.sleep(0)
        waited = True

    task = asyncio.create_task(preload())
    await main_mod._wait_for_initial_sandbox_preload(
        [SimpleNamespace(sandbox_preload_task=task)]
    )

    assert waited is True
