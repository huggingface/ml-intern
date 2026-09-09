import asyncio
from types import SimpleNamespace

import pytest

from agent.core.codex_runtime import (
    CODEX_TOOL_NAMESPACE,
    CodexAppServerRuntime,
    CodexRuntimeError,
    build_dynamic_tool_namespace,
    codex_login_status,
    codex_model_catalog,
)
from agent.core.tools import ToolSpec


class StubRouter:
    def __init__(self):
        self.tools = {
            "bash": ToolSpec(
                name="bash",
                description="shell",
                parameters={"type": "object"},
                handler=None,
            ),
            "research": ToolSpec(
                name="research",
                description="nested research",
                parameters={"type": "object"},
                handler=None,
            ),
            "hf_papers": ToolSpec(
                name="hf_papers",
                description="papers",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                handler=None,
            ),
            "invalid.tool": ToolSpec(
                name="invalid.tool",
                description="invalid name",
                parameters={"type": "object"},
                handler=None,
            ),
        }


def test_dynamic_namespace_keeps_ml_tools_and_skips_codex_duplicates():
    namespace, dispatch = build_dynamic_tool_namespace(
        StubRouter(),
        local_mode=True,
    )

    assert namespace is not None
    assert namespace["name"] == CODEX_TOOL_NAMESPACE
    assert [tool["name"] for tool in namespace["tools"]] == ["hf_papers"]
    assert dispatch == {"hf_papers": "hf_papers"}


def test_remote_sandbox_tools_can_expose_namespaced_bash():
    namespace, dispatch = build_dynamic_tool_namespace(
        StubRouter(),
        local_mode=False,
    )

    assert namespace is not None
    assert [tool["name"] for tool in namespace["tools"]] == [
        "bash",
        "hf_papers",
    ]
    assert dispatch == {"bash": "bash", "hf_papers": "hf_papers"}


@pytest.mark.asyncio
async def test_codex_login_status_reports_missing_cli(monkeypatch):
    monkeypatch.setattr(
        "agent.core.codex_runtime.shutil.which",
        lambda _binary: None,
    )

    with pytest.raises(CodexRuntimeError, match="Codex CLI is not installed"):
        await codex_login_status()


@pytest.mark.asyncio
async def test_codex_login_status_uses_public_cli_status(monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self):
            return b"Logged in using ChatGPT\n", b""

    async def fake_subprocess(*_args, **_kwargs):
        return Process()

    monkeypatch.setattr(
        "agent.core.codex_runtime.shutil.which",
        lambda _binary: "/usr/local/bin/codex",
    )
    monkeypatch.setattr(
        "agent.core.codex_runtime.asyncio.create_subprocess_exec",
        fake_subprocess,
    )

    assert await codex_login_status() == "Logged in using ChatGPT"


@pytest.mark.asyncio
async def test_codex_model_catalog_uses_account_visible_models(monkeypatch):
    writes = []

    class Stdin:
        def write(self, payload):
            writes.append(payload.decode())

        async def drain(self):
            return None

    class Stdout:
        def __init__(self):
            self.lines = [
                b'{"id":1,"result":{"userAgent":"Codex"}}\n',
                b'{"method":"remoteControl/status/changed","params":{}}\n',
                (
                    b'{"id":2,"result":{"data":[{"id":"gpt-5.6-sol",'
                    b'"model":"gpt-5.6-sol","displayName":"GPT-5.6-Sol",'
                    b'"defaultReasoningEffort":"low",'
                    b'"supportedReasoningEfforts":[{"reasoningEffort":"low"}],'
                    b'"isDefault":true}],"nextCursor":null}}\n'
                ),
            ]

        async def readline(self):
            return self.lines.pop(0)

    class Stderr:
        async def read(self):
            return b""

    class Process:
        def __init__(self):
            self.stdin = Stdin()
            self.stdout = Stdout()
            self.stderr = Stderr()
            self.returncode = None

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def fake_login_status(_codex_bin):
        return "Logged in using ChatGPT"

    async def fake_subprocess(*_args, **_kwargs):
        return Process()

    monkeypatch.setattr(
        "agent.core.codex_runtime.codex_login_status",
        fake_login_status,
    )
    monkeypatch.setattr(
        "agent.core.codex_runtime.shutil.which",
        lambda _binary: "/usr/local/bin/codex",
    )
    monkeypatch.setattr(
        "agent.core.codex_runtime.asyncio.create_subprocess_exec",
        fake_subprocess,
    )

    models = await codex_model_catalog()

    assert models[0]["model"] == "gpt-5.6-sol"
    assert models[0]["defaultReasoningEffort"] == "low"
    assert '"method":"model/list"' in writes[-1]
    assert '"includeHidden":false' in writes[-1]


@pytest.mark.asyncio
async def test_run_turn_sends_selected_reasoning_effort(tmp_path):
    requests = []
    runtime = CodexAppServerRuntime(
        config=SimpleNamespace(
            model_name="codex/gpt-5.6-sol",
            reasoning_effort="max",
        ),
        tool_router=SimpleNamespace(),
        hf_token=None,
        local_mode=False,
        cwd=tmp_path,
        autonomous_mode=False,
    )
    runtime.thread_id = "thread-1"

    async def fake_request(method, params):
        requests.append((method, params))
        return {"turn": {"id": "turn-1"}}

    runtime._request = fake_request
    await runtime._notifications.put(
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "done",
                },
            },
        }
    )
    await runtime._notifications.put(
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "turn": {"id": "turn-1", "status": "completed"},
            },
        }
    )

    result = await runtime.run_turn("hello")

    assert result == "done"
    assert requests == [
        (
            "turn/start",
            {
                "threadId": "thread-1",
                "input": [{"type": "text", "text": "hello"}],
                "effort": "max",
            },
        )
    ]


@pytest.mark.asyncio
async def test_run_turn_exits_when_session_is_cancelled(tmp_path):
    runtime = CodexAppServerRuntime(
        config=SimpleNamespace(model_name="codex/default"),
        tool_router=SimpleNamespace(),
        hf_token=None,
        local_mode=False,
        cwd=tmp_path,
        autonomous_mode=False,
    )
    runtime.thread_id = "thread-1"
    runtime._tool_session = SimpleNamespace(is_cancelled=True)

    async def fake_request(_method, _params):
        return {"turn": {"id": "turn-1"}}

    runtime._request = fake_request

    assert await runtime.run_turn("hello") == ""
    assert runtime.active_turn_id is None


@pytest.mark.asyncio
async def test_run_turn_times_out_when_codex_stops_emitting_activity(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "agent.core.codex_runtime._CODEX_NOTIFICATION_POLL_S",
        0.005,
    )
    monkeypatch.setattr(
        "agent.core.codex_runtime._CODEX_TURN_IDLE_TIMEOUT_S",
        0.01,
    )
    runtime = CodexAppServerRuntime(
        config=SimpleNamespace(model_name="codex/default"),
        tool_router=SimpleNamespace(),
        hf_token=None,
        local_mode=False,
        cwd=tmp_path,
        autonomous_mode=False,
    )
    runtime.thread_id = "thread-1"

    async def fake_request(_method, _params):
        return {"turn": {"id": "turn-1"}}

    runtime._request = fake_request

    with pytest.raises(CodexRuntimeError, match="no activity"):
        await runtime.run_turn("hello")


@pytest.mark.asyncio
async def test_interrupt_does_not_wait_forever_for_app_server(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent.core.codex_runtime._CODEX_INTERRUPT_TIMEOUT_S",
        0.01,
    )
    runtime = CodexAppServerRuntime(
        config=SimpleNamespace(model_name="codex/default"),
        tool_router=SimpleNamespace(),
        hf_token=None,
        local_mode=False,
        cwd=tmp_path,
        autonomous_mode=False,
    )
    runtime.thread_id = "thread-1"
    runtime.active_turn_id = "turn-1"

    async def hanging_request(_method, _params):
        await asyncio.Event().wait()

    runtime._request = hanging_request

    await runtime.interrupt()


@pytest.mark.asyncio
async def test_dynamic_tool_failure_is_returned_to_codex(tmp_path):
    class FailingRouter:
        async def call_tool(self, *_args, **_kwargs):
            raise RuntimeError("probe failed")

    responses = []
    runtime = CodexAppServerRuntime(
        config=SimpleNamespace(model_name="codex/default"),
        tool_router=FailingRouter(),
        hf_token=None,
        local_mode=True,
        cwd=tmp_path,
        autonomous_mode=False,
    )
    runtime._dispatch = {"hf_papers": "hf_papers"}
    runtime._tool_session = object()

    async def capture_response(message):
        responses.append(message)

    runtime._write = capture_response

    await runtime._handle_server_request(
        {
            "id": 7,
            "method": "item/tool/call",
            "params": {
                "namespace": CODEX_TOOL_NAMESPACE,
                "tool": "hf_papers",
                "arguments": {"operation": "search", "query": "LoRA"},
            },
        }
    )

    assert responses[0]["id"] == 7
    assert responses[0]["result"]["success"] is False
    assert "probe failed" in responses[0]["result"]["contentItems"][0]["text"]


@pytest.mark.asyncio
async def test_builtin_command_item_uses_tool_callback(tmp_path):
    calls = []

    async def on_tool(name, arguments, output, success, tool_call_id):
        calls.append((name, arguments, output, success, tool_call_id))

    runtime = CodexAppServerRuntime(
        config=SimpleNamespace(model_name="codex/default"),
        tool_router=SimpleNamespace(),
        hf_token=None,
        local_mode=False,
        cwd=tmp_path,
        autonomous_mode=False,
        on_tool=on_tool,
    )

    item = {
        "id": "cmd-1",
        "type": "commandExecution",
        "command": "pwd",
        "cwd": str(tmp_path),
        "status": "completed",
        "exitCode": 0,
        "aggregatedOutput": str(tmp_path),
    }
    await runtime._emit_builtin_item(item, completed=False)
    await runtime._emit_builtin_item(item, completed=True)

    assert calls[0] == (
        "codex_command",
        {"command": "pwd", "cwd": str(tmp_path)},
        None,
        None,
        "cmd-1",
    )
    assert calls[1][0] == "codex_command"
    assert calls[1][2] == str(tmp_path)
    assert calls[1][3] is True
