from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


from agent.config import Config, load_config
from agent.core.agent_loop import Handlers
from agent.core.session import Session
from agent.core.tools import ToolRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


class AgentResponseGenerator:
    """
    Thin async wrapper that executes the existing agent loop once and
    returns the assistant's final message.
    """

    def __init__(self, config_path: str | Path, max_iterations: int = 10) -> None:
        self.config_path = _resolve_project_path(config_path)
        self.config: Config = load_config(str(self.config_path))
        self.max_iterations = max_iterations

    @property
    def model_name(self) -> str:
        """Expose the agent model name for downstream logging."""
        return self.config.model_name

    async def run(self, prompt: str) -> str:
        """
        Execute the agent loop for a single prompt and return the assistant reply.
        """
        tool_router = ToolRouter(self.config.mcpServers)

        async with tool_router:
            session = Session(asyncio.Queue(), config=self.config)
            session.tool_router = tool_router
            await Handlers.run_agent(
                session,
                prompt,
                max_iterations=self.max_iterations,
            )
            return self._latest_assistant_response(session)

    def _latest_assistant_response(self, session: Session) -> str:
        """
        Extract the final assistant response from the session history.
        """
        for message in reversed(session.context_manager.items):
            if getattr(message, "role", None) == "assistant":
                return _content_to_text(getattr(message, "content", ""))

        raise RuntimeError("Agent did not produce an assistant message.")


def _content_to_text(content: Any) -> str:
    """
    Convert LiteLLM content payloads (str or list[dict]) into plain text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    return str(content)
