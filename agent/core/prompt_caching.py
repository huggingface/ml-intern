"""Prompt-caching hook for outgoing LLM requests.

HF Router-only inference does not use provider-native Anthropic cache-control
blocks, so this hook currently leaves messages and tool specs unchanged.
"""

from typing import Any


def with_prompt_caching(
    messages: list[Any],
    tools: list[dict] | None,
    model_name: str | None,
) -> tuple[list[Any], list[dict] | None]:
    """Return messages and tools unchanged."""
    _ = model_name
    return messages, tools
