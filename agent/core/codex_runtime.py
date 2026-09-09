"""OpenAI-authenticated Codex runtime for ML Intern.

This module talks to the locally installed ``codex app-server`` over its stdio
JSON-RPC transport. Authentication remains fully owned by Codex. In
particular, ML Intern never reads ``~/.codex/auth.json`` or treats a ChatGPT
session token as an OpenAI API key.

Codex remains the agent loop in this mode. ML Intern's Hugging Face research,
Hub, Jobs, and sandbox tools are exposed to it as an experimental dynamic-tool
namespace supported by the Codex app-server.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import shutil
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent.config import Config
from agent.core.agent_loop import _base_needs_approval
from agent.core.codex_models import codex_model_name
from agent.core.session import Event, Session
from agent.core.tools import ToolRouter

logger = logging.getLogger(__name__)

CODEX_TOOL_NAMESPACE = "ml_intern"
_DYNAMIC_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_CODEX_BUILTIN_LOCAL_TOOLS = {"bash", "read", "write", "edit"}
_CODEX_INTERRUPT_TIMEOUT_S = 3.0
_CODEX_NOTIFICATION_POLL_S = 1.0
_CODEX_TURN_IDLE_TIMEOUT_S = 300.0
_UNSUPPORTED_CODEX_TOOLS = {
    # This tool creates a nested LiteLLM research loop using the active model.
    # Codex already has its own independent context and can call the underlying
    # research tools directly.
    "research",
}

ToolApprovalCallback = Callable[
    [str, dict[str, Any], str],
    Awaitable[bool] | bool,
]
DeltaCallback = Callable[[str], Awaitable[None] | None]
ToolCallback = Callable[
    [str, dict[str, Any], str | None, bool | None, str],
    Awaitable[None] | None,
]
EventCallback = Callable[[Event], Awaitable[None] | None]


class CodexRuntimeError(RuntimeError):
    """Raised when Codex authentication or app-server execution fails."""


async def _call_maybe_async(callback: Callable[..., Any] | None, *args: Any) -> Any:
    if callback is None:
        return None
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def codex_login_status(codex_bin: str = "codex") -> str:
    """Return Codex's public login status without reading cached credentials."""
    resolved = shutil.which(codex_bin)
    if resolved is None:
        raise CodexRuntimeError(
            "Codex CLI is not installed. Install it, then run `codex login`."
        )

    process = await asyncio.create_subprocess_exec(
        resolved,
        "login",
        "status",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    text = (stdout or stderr).decode(errors="replace").strip()
    if process.returncode != 0:
        detail = text or "no active Codex login"
        raise CodexRuntimeError(
            f"Codex authentication is unavailable: {detail}. Run `codex login`."
        )
    return text or "Codex authentication active"


async def codex_model_catalog(
    codex_bin: str = "codex",
    *,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Return the picker-visible model catalog for the active Codex account."""
    await codex_login_status(codex_bin)
    resolved = shutil.which(codex_bin)
    assert resolved is not None

    process = await asyncio.create_subprocess_exec(
        resolved,
        "app-server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    request_id = 0

    async def write(message: dict[str, Any]) -> None:
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        process.stdin.write(payload)
        await process.stdin.drain()

    async def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal request_id
        request_id += 1
        current_id = request_id
        await write({"method": method, "id": current_id, "params": params})
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout_s)
            if not line:
                raise CodexRuntimeError(
                    "Codex app-server closed while loading its model catalog."
                )
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != current_id:
                continue
            if "error" in message:
                error = message.get("error") or {}
                raise CodexRuntimeError(
                    str(error.get("message") or f"Codex {method} request failed.")
                )
            return message.get("result") or {}

    try:
        await request(
            "initialize",
            {
                "clientInfo": {
                    "name": "ml_intern",
                    "title": "ML Intern",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await write({"method": "initialized", "params": {}})

        models: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "limit": 100,
                "includeHidden": False,
            }
            if cursor:
                params["cursor"] = cursor
            result = await request("model/list", params)
            models.extend(
                model for model in result.get("data") or [] if isinstance(model, dict)
            )
            cursor = result.get("nextCursor")
            if not cursor:
                return models
    except TimeoutError as exc:
        raise CodexRuntimeError("Timed out while loading Codex models.") from exc
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        stderr = (await stderr_task).decode(errors="replace").strip()
        if process.returncode not in {None, 0, -15} and stderr:
            logger.debug("Codex model catalog stderr: %s", stderr[-1000:])


def build_dynamic_tool_namespace(
    tool_router: ToolRouter,
    *,
    local_mode: bool,
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """Build the Codex dynamic-tool namespace and its dispatch name map."""
    tools: list[dict[str, Any]] = []
    dispatch: dict[str, str] = {}

    skipped = set(_UNSUPPORTED_CODEX_TOOLS)
    if local_mode:
        # Codex already provides sandboxed local shell and file tools. Keeping
        # a second copy would create ambiguous or reserved tool names.
        skipped.update(_CODEX_BUILTIN_LOCAL_TOOLS)

    for tool in tool_router.tools.values():
        if tool.name in skipped or not _DYNAMIC_TOOL_NAME.fullmatch(tool.name):
            continue
        tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
        )
        dispatch[tool.name] = tool.name

    if not tools:
        return None, dispatch

    return (
        {
            "type": "namespace",
            "name": CODEX_TOOL_NAMESPACE,
            "description": (
                "ML Intern tools for Hugging Face documentation, papers, "
                "datasets, Hub repositories, Jobs, web research, notifications, "
                "planning, and optional remote sandbox execution."
            ),
            "tools": tools,
        },
        dispatch,
    )


class CodexAppServerRuntime:
    """Manage one authenticated Codex app-server process and thread."""

    def __init__(
        self,
        *,
        config: Config,
        tool_router: ToolRouter,
        hf_token: str | None,
        local_mode: bool,
        cwd: str | Path,
        autonomous_mode: bool,
        approve_tool: ToolApprovalCallback | None = None,
        on_delta: DeltaCallback | None = None,
        on_tool: ToolCallback | None = None,
        on_event: EventCallback | None = None,
        codex_bin: str = "codex",
        manage_tool_router: bool = True,
        tool_session: Session | None = None,
    ) -> None:
        self.config = config
        self.tool_router = tool_router
        self.hf_token = hf_token
        self.local_mode = local_mode
        self.cwd = str(Path(cwd).resolve())
        self.autonomous_mode = autonomous_mode
        self.approve_tool = approve_tool
        self.on_delta = on_delta
        self.on_tool = on_tool
        self.on_event = on_event
        self.codex_bin = codex_bin
        self.manage_tool_router = manage_tool_router
        self._provided_tool_session = tool_session

        self.auth_status: str | None = None
        self.thread_id: str | None = None
        self.active_turn_id: str | None = None

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future] = {}
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._request_id = 0
        self._server_request_tasks: set[asyncio.Task] = set()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._dispatch: dict[str, str] = {}
        self._session_events: asyncio.Queue[Event] = asyncio.Queue()
        self._tool_session: Session | None = None
        self._tool_router_entered = False

    async def __aenter__(self) -> "CodexAppServerRuntime":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        """Start Codex, initialize tools, and create an ephemeral thread."""
        if self._process is not None:
            return

        self.auth_status = await codex_login_status(self.codex_bin)
        resolved = shutil.which(self.codex_bin)
        assert resolved is not None

        try:
            if self.manage_tool_router:
                await self.tool_router.__aenter__()
                self._tool_router_entered = True
            if self._provided_tool_session is not None:
                self._tool_session = self._provided_tool_session
            else:
                self._tool_session = Session(
                    self._session_events,
                    self.config,
                    tool_router=self.tool_router,
                    hf_token=self.hf_token,
                    hf_username="unknown",
                    local_mode=self.local_mode,
                    autonomous_mode=self.autonomous_mode,
                    stream=True,
                )

            namespace, self._dispatch = build_dynamic_tool_namespace(
                self.tool_router,
                local_mode=self.local_mode,
            )

            self._process = await asyncio.create_subprocess_exec(
                resolved,
                "app-server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            self._stderr_task = asyncio.create_task(self._read_stderr())
            if self._provided_tool_session is None:
                self._event_task = asyncio.create_task(self._drain_session_events())

            await self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "ml_intern",
                        "title": "ML Intern",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._notify("initialized", {})

            params: dict[str, Any] = {
                "cwd": self.cwd,
                "ephemeral": True,
                "approvalPolicy": "never",
                "sandbox": "workspace-write" if self.local_mode else "read-only",
                "serviceName": "ml-intern",
                "developerInstructions": self._developer_instructions(),
            }
            requested_model = codex_model_name(self.config.model_name)
            if requested_model is not None:
                params["model"] = requested_model
            if namespace is not None:
                params["dynamicTools"] = [namespace]

            response = await self._request("thread/start", params)
            thread = response.get("thread") or {}
            self.thread_id = thread.get("id")
            if not self.thread_id:
                raise CodexRuntimeError("Codex app-server did not return a thread id.")
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Close the Codex process and ML Intern tool resources."""
        for task in tuple(self._server_request_tasks):
            task.cancel()
        if self._server_request_tasks:
            await asyncio.gather(
                *self._server_request_tasks,
                return_exceptions=True,
            )
        self._server_request_tasks.clear()

        background_tasks = (
            self._event_task,
            self._reader_task,
            self._stderr_task,
        )
        for task in background_tasks:
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in background_tasks if task is not None),
            return_exceptions=True,
        )
        self._event_task = None
        self._reader_task = None
        self._stderr_task = None

        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if self._tool_router_entered:
            await self.tool_router.__aexit__(None, None, None)
            self._tool_router_entered = False
        self._fail_pending(CodexRuntimeError("Codex app-server closed."))
        self.thread_id = None
        self.active_turn_id = None
        self._tool_session = None

    def _developer_instructions(self) -> str:
        if self.local_mode:
            tool_guidance = "Use your built-in Codex tools for local repository work. "
        else:
            tool_guidance = (
                "The host repository is read-only. Use the ml_intern namespace "
                "for remote sandbox execution and ML research tools. "
            )
        return (
            "You are the OpenAI-authenticated Codex runtime inside ML Intern. "
            f"{tool_guidance}"
            "Use the ml_intern namespace for Hugging Face docs, papers, datasets, "
            "Hub repositories, Jobs, and web research. Never claim that a ChatGPT "
            "login is an OpenAI API key. If an ML Intern tool is denied or fails, "
            "report that result instead of silently retrying a billable or "
            "destructive operation."
        )

    async def new_thread(self) -> None:
        """Start a fresh ephemeral Codex thread with the same runtime."""
        if self._process is None:
            raise CodexRuntimeError("Codex app-server is not running.")
        # Recreate the process-level integration to ensure the same dynamic
        # tools and model settings are applied consistently.
        await self.close()
        self.thread_id = None
        self.active_turn_id = None
        await self.start()

    async def interrupt(self) -> None:
        """Interrupt the active Codex turn, if one is running."""
        if not self.thread_id or not self.active_turn_id:
            return
        try:
            await asyncio.wait_for(
                self._request(
                    "turn/interrupt",
                    {"threadId": self.thread_id, "turnId": self.active_turn_id},
                ),
                timeout=_CODEX_INTERRUPT_TIMEOUT_S,
            )
        except TimeoutError:
            logger.warning(
                "Codex app-server did not acknowledge interrupt within %.0fs",
                _CODEX_INTERRUPT_TIMEOUT_S,
            )
        except Exception:
            logger.debug("Failed to interrupt Codex turn", exc_info=True)

    async def run_turn(self, prompt: str, *, stream: bool = True) -> str:
        """Run one user turn and return the final Codex answer."""
        if not self.thread_id:
            raise CodexRuntimeError("Codex runtime has not been started.")

        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
        }
        reasoning_effort = getattr(self.config, "reasoning_effort", None)
        if reasoning_effort:
            params["effort"] = reasoning_effort
        response = await self._request(
            "turn/start",
            params,
        )
        turn = response.get("turn") or {}
        self.active_turn_id = turn.get("id")
        final_text = ""
        loop = asyncio.get_running_loop()
        last_activity_at = loop.time()

        try:
            while True:
                if getattr(self._tool_session, "is_cancelled", False):
                    return final_text

                idle_for = loop.time() - last_activity_at
                remaining_idle_s = _CODEX_TURN_IDLE_TIMEOUT_S - idle_for
                if remaining_idle_s <= 0:
                    raise CodexRuntimeError(
                        "Codex produced no activity for 5 minutes, so the stuck "
                        "turn was stopped. Retry or choose a lower Thinking level."
                    )
                try:
                    message = await asyncio.wait_for(
                        self._notifications.get(),
                        timeout=min(
                            _CODEX_NOTIFICATION_POLL_S,
                            remaining_idle_s,
                        ),
                    )
                except TimeoutError:
                    continue
                last_activity_at = loop.time()
                method = message.get("method")
                params = message.get("params") or {}

                if params.get("threadId") not in {None, self.thread_id}:
                    continue
                notification_turn_id = params.get("turnId")
                if (
                    notification_turn_id
                    and self.active_turn_id
                    and notification_turn_id != self.active_turn_id
                ):
                    continue

                if method == "item/agentMessage/delta" and stream:
                    delta = str(params.get("delta") or "")
                    if delta:
                        await _call_maybe_async(self.on_delta, delta)
                elif method == "item/started":
                    await self._emit_builtin_item(
                        params.get("item") or {},
                        completed=False,
                    )
                elif method == "item/completed":
                    item = params.get("item") or {}
                    if item.get("type") == "agentMessage":
                        text = str(item.get("text") or "")
                        if item.get("phase") == "final_answer" or not final_text:
                            final_text = text
                    else:
                        await self._emit_builtin_item(item, completed=True)
                elif method == "turn/completed":
                    completed_turn = params.get("turn") or {}
                    if (
                        self.active_turn_id
                        and completed_turn.get("id") != self.active_turn_id
                    ):
                        continue
                    if completed_turn.get("status") == "failed":
                        error = completed_turn.get("error") or {}
                        message_text = error.get("message") or "Codex turn failed."
                        raise CodexRuntimeError(str(message_text))
                    return final_text
        finally:
            self.active_turn_id = None

    async def _emit_builtin_item(
        self,
        item: dict[str, Any],
        *,
        completed: bool,
    ) -> None:
        """Surface Codex built-in tools through the same callback as ML tools."""
        item_type = str(item.get("type") or "")
        if item_type == "commandExecution":
            name = "codex_command"
            arguments = {
                "command": item.get("command"),
                "cwd": item.get("cwd"),
            }
            output = str(item.get("aggregatedOutput") or "") if completed else None
            success = (
                item.get("status") == "completed" and item.get("exitCode") in {None, 0}
                if completed
                else None
            )
        elif item_type == "fileChange":
            name = "codex_file_change"
            arguments = {"changes": item.get("changes") or []}
            output = json.dumps(item.get("changes") or [], ensure_ascii=False)
            success = item.get("status") == "completed" if completed else None
            if not completed:
                output = None
        elif item_type == "mcpToolCall":
            name = f"mcp:{item.get('server')}.{item.get('tool')}"
            arguments = item.get("arguments") or {}
            result = item.get("result")
            error = item.get("error")
            output = (
                json.dumps(result if result is not None else error, ensure_ascii=False)
                if completed
                else None
            )
            success = (
                item.get("status") == "completed" and not error if completed else None
            )
        else:
            return

        await _call_maybe_async(
            self.on_tool,
            name,
            arguments,
            output,
            success,
            str(item.get("id") or f"codex-{self._request_id}"),
        )

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexRuntimeError("Codex app-server is not running.")
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"method": method, "id": request_id, "params": params})
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"method": method, "params": params})

    async def _write(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexRuntimeError("Codex app-server is not running.")
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            process.stdin.write(payload)
            await process.stdin.drain()

    async def _read_loop(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while line := await process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Ignoring non-JSON Codex app-server output: %r",
                        line[:200],
                    )
                    continue

                request_id = message.get("id")
                if request_id in self._pending and (
                    "result" in message or "error" in message
                ):
                    future = self._pending[request_id]
                    if "error" in message:
                        error = message.get("error") or {}
                        future.set_exception(
                            CodexRuntimeError(
                                str(error.get("message") or "Codex request failed.")
                            )
                        )
                    else:
                        future.set_result(message.get("result") or {})
                    continue

                if message.get("method") and request_id is not None:
                    task = asyncio.create_task(self._handle_server_request(message))
                    self._server_request_tasks.add(task)
                    task.add_done_callback(self._server_request_tasks.discard)
                    continue

                if message.get("method"):
                    await self._notifications.put(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(CodexRuntimeError(f"Codex read loop failed: {exc}"))
        finally:
            if self._process is not None and self._process.returncode is not None:
                detail = "\n".join(self._stderr_tail)
                suffix = f"\n{detail}" if detail else ""
                self._fail_pending(
                    CodexRuntimeError(f"Codex app-server exited unexpectedly.{suffix}")
                )

    async def _read_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        try:
            while line := await process.stderr.readline():
                text = line.decode(errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("codex app-server: %s", text)
        except asyncio.CancelledError:
            raise

    async def _drain_session_events(self) -> None:
        try:
            while True:
                event = await self._session_events.get()
                await _call_maybe_async(self.on_event, event)
        except asyncio.CancelledError:
            raise

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        method = message.get("method")
        if method != "item/tool/call":
            await self._write(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported Codex server request: {method}",
                    },
                }
            )
            return

        params = message.get("params") or {}
        namespace = params.get("namespace")
        codex_tool_name = str(params.get("tool") or "")
        tool_name = self._dispatch.get(codex_tool_name)
        arguments = params.get("arguments") or {}
        tool_call_id = str(params.get("callId") or request_id)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if namespace != CODEX_TOOL_NAMESPACE or not tool_name:
            await self._dynamic_tool_response(
                request_id,
                f"Unknown ML Intern tool: {namespace}.{codex_tool_name}",
                False,
            )
            return

        await _call_maybe_async(
            self.on_tool,
            tool_name,
            arguments,
            None,
            None,
            tool_call_id,
        )

        if _base_needs_approval(tool_name, arguments, self.config):
            approved = bool(
                await _call_maybe_async(
                    self.approve_tool,
                    tool_name,
                    arguments,
                    tool_call_id,
                )
            )
            if not approved:
                output = f"User denied ML Intern tool call: {tool_name}"
                await _call_maybe_async(
                    self.on_tool,
                    tool_name,
                    arguments,
                    output,
                    False,
                    tool_call_id,
                )
                await self._dynamic_tool_response(request_id, output, False)
                return

        assert self._tool_session is not None
        try:
            output, success = await self.tool_router.call_tool(
                tool_name,
                arguments,
                session=self._tool_session,
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            logger.exception("ML Intern tool failed in Codex runtime: %s", tool_name)
            output = f"ML Intern tool failed: {exc}"
            success = False
        await _call_maybe_async(
            self.on_tool,
            tool_name,
            arguments,
            output,
            success,
            tool_call_id,
        )
        await self._dynamic_tool_response(request_id, output, success)

    async def _dynamic_tool_response(
        self,
        request_id: int | str,
        output: str,
        success: bool,
    ) -> None:
        await self._write(
            {
                "id": request_id,
                "result": {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": str(output),
                        }
                    ],
                    "success": bool(success),
                },
            }
        )

    def _fail_pending(self, error: Exception) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)
