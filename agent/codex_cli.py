"""CLI presentation for the OpenAI-authenticated Codex runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession

from agent.config import Config
from agent.core.approval_policy import is_scheduled_operation
from agent.core.codex_runtime import CodexAppServerRuntime, CodexRuntimeError
from agent.core.session import Event
from agent.core.tools import ToolRouter
from agent.utils.terminal_display import (
    get_console,
    print_banner,
    print_error,
    print_markdown,
    print_tool_call,
    print_tool_log,
    print_tool_output,
)


def _is_scheduled_job(tool_name: str, arguments: dict[str, Any]) -> bool:
    return tool_name == "hf_jobs" and is_scheduled_operation(arguments.get("operation"))


def _available_mcp_servers(config: Config, hf_token: str | None) -> dict:
    """Skip the default HF OAuth MCP when no local HF identity exists."""
    if hf_token:
        return config.mcpServers

    available = {}
    for name, server in config.mcpServers.items():
        data = server.model_dump()
        url = str(data.get("url") or "")
        if "huggingface.co/mcp" in url:
            continue
        available[name] = server
    return available


async def run_codex_interactive(
    *,
    config: Config,
    prompt_session: PromptSession,
    hf_token: str | None,
    hf_user: str | None,
    local_mode: bool,
) -> None:
    """Run the interactive ML Intern CLI on the authenticated Codex runtime."""
    console = get_console()
    print_banner(
        model=config.model_name,
        hf_user=hf_user,
        tool_runtime="local filesystem" if local_mode else "HF sandbox",
    )

    streamed = [False]

    async def on_delta(delta: str) -> None:
        streamed[0] = True
        console.file.write(delta)
        console.file.flush()

    async def on_tool(
        name: str,
        arguments: dict[str, Any],
        output: str | None,
        success: bool | None,
        _tool_call_id: str,
    ) -> None:
        if output is None:
            print_tool_call(name, json.dumps(arguments)[:120])
        else:
            print_tool_output(output, bool(success), truncate=True)

    async def on_event(event: Event) -> None:
        if event.event_type == "tool_log" and event.data:
            print_tool_log(
                str(event.data.get("tool") or ""),
                str(event.data.get("log") or ""),
            )

    async def approve_tool(
        name: str,
        arguments: dict[str, Any],
        _tool_call_id: str,
    ) -> bool:
        console.print(f"\n[bold yellow]Approval required:[/bold yellow] {name}")
        console.print_json(data=arguments)
        answer = await prompt_session.prompt_async("Approve this tool call? [y/N] ")
        return answer.strip().lower() in {"y", "yes"}

    tool_router = ToolRouter(
        _available_mcp_servers(config, hf_token),
        hf_token=hf_token,
        local_mode=local_mode,
    )

    try:
        async with CodexAppServerRuntime(
            config=config,
            tool_router=tool_router,
            hf_token=hf_token,
            local_mode=local_mode,
            cwd=Path.cwd(),
            autonomous_mode=False,
            approve_tool=approve_tool,
            on_delta=on_delta,
            on_tool=on_tool,
            on_event=on_event,
        ) as runtime:
            console.print(f"[dim]{runtime.auth_status}[/dim]")
            while True:
                try:
                    user_input = await prompt_session.prompt_async("\nYou: ")
                except (EOFError, KeyboardInterrupt):
                    break

                stripped = user_input.strip()
                if not stripped:
                    continue
                if stripped.lower() in {"exit", "quit", "/quit", "/exit"}:
                    break
                if stripped == "/status":
                    console.print(f"[bold]Model:[/bold] {config.model_name}")
                    console.print(f"[bold]Auth:[/bold] {runtime.auth_status}")
                    continue
                if stripped == "/new":
                    await runtime.new_thread()
                    console.print("[green]Started a new Codex conversation.[/green]")
                    continue
                if stripped.startswith("/model"):
                    console.print(
                        "[dim]Codex runtime models are selected at startup. "
                        "Restart with `ml-intern --model codex/default` or "
                        "`ml-intern --model codex/<model>`.[/dim]"
                    )
                    continue

                streamed[0] = False
                try:
                    final_text = await runtime.run_turn(stripped, stream=True)
                except KeyboardInterrupt:
                    await runtime.interrupt()
                    console.print("\n[yellow]Interrupted.[/yellow]")
                    continue
                if streamed[0]:
                    console.file.write("\n")
                    console.file.flush()
                elif final_text:
                    await print_markdown(final_text, instant=True)
    except CodexRuntimeError as exc:
        print_error(str(exc))
    finally:
        console.print("\n[dim]Bye.[/dim]\n")


async def run_codex_headless(
    prompt: str,
    *,
    config: Config,
    hf_token: str | None,
    local_mode: bool,
    stream: bool,
) -> None:
    """Run one prompt through Codex and exit."""
    streamed = [False]

    async def on_delta(delta: str) -> None:
        streamed[0] = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    async def on_tool(
        name: str,
        arguments: dict[str, Any],
        output: str | None,
        success: bool | None,
        _tool_call_id: str,
    ) -> None:
        if output is None:
            print_tool_call(name, json.dumps(arguments)[:120])
        else:
            print_tool_output(output, bool(success), truncate=True)

    async def approve_tool(
        name: str,
        arguments: dict[str, Any],
        _tool_call_id: str,
    ) -> bool:
        # Match the existing headless policy: scheduled Jobs never receive
        # automatic approval because they can create recurring spend.
        return not _is_scheduled_job(name, arguments)

    tool_router = ToolRouter(
        _available_mcp_servers(config, hf_token),
        hf_token=hf_token,
        local_mode=local_mode,
    )
    async with CodexAppServerRuntime(
        config=config,
        tool_router=tool_router,
        hf_token=hf_token,
        local_mode=local_mode,
        cwd=Path.cwd(),
        autonomous_mode=True,
        approve_tool=approve_tool,
        on_delta=on_delta,
        on_tool=on_tool,
    ) as runtime:
        print(f"Codex auth: {runtime.auth_status}", file=sys.stderr)
        final_text = await runtime.run_turn(prompt, stream=stream)
        if streamed[0]:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif final_text:
            await print_markdown(final_text, instant=True)
