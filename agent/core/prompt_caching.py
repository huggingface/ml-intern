"""Prompt-cache breakpoints for HF Router FAL requests.

The HF Router/OpenRouter path accepts Anthropic prompt caching as JSON
``cache_control`` content blocks. Headers like ``X-OpenRouter-Cache`` do not
produce cache usage counters through this route.
"""

from typing import Any

from agent.core.model_ids import HF_ROUTER_BASE_URL

_CACHE_CONTROL = {"type": "ephemeral"}
_CACHEABLE_ROLES = {"system", "user"}


def _is_fal_router_request(llm_params: dict[str, Any]) -> bool:
    model = str(llm_params.get("model") or "")
    api_base = str(llm_params.get("api_base") or "").rstrip("/")
    return api_base == HF_ROUTER_BASE_URL and ":fal" in model


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
    else:
        role = getattr(message, "role", None)
    return role if isinstance(role, str) else None


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported message type for prompt caching: {type(message)!r}")


def _has_cacheable_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and bool(block.get("text"))
        for block in content
    )


def _cache_target_index(messages: list[Any]) -> int | None:
    if len(messages) < 2:
        return None

    for idx in range(len(messages) - 2, -1, -1):
        message = messages[idx]
        if _message_role(message) not in _CACHEABLE_ROLES:
            continue
        if _has_cacheable_text(_message_content(message)):
            return idx
    return None


def _content_with_cache_control(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [
            {"type": "text", "text": content, "cache_control": dict(_CACHE_CONTROL)}
        ]

    blocks = [dict(block) if isinstance(block, dict) else block for block in content]
    for idx in range(len(blocks) - 1, -1, -1):
        block = blocks[idx]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and bool(block.get("text"))
        ):
            cached = dict(block)
            cached["cache_control"] = dict(_CACHE_CONTROL)
            blocks[idx] = cached
            break
    return blocks


def _tools_with_cache_control(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return tools

    cached_tools = list(tools)
    last_tool = dict(cached_tools[-1])
    last_tool["cache_control"] = dict(_CACHE_CONTROL)
    cached_tools[-1] = last_tool
    return cached_tools


def with_prompt_caching(
    messages: list[Any],
    tools: list[dict] | None,
    llm_params: dict[str, Any],
) -> tuple[list[Any], list[dict] | None]:
    """Return outgoing messages with a cache breakpoint for HF Router FAL.

    The newest message is treated as dynamic. The cache breakpoint is placed
    on the closest earlier system/user text block so provider-side caching
    covers the stable prefix without changing persisted conversation history.
    The final tool spec is also marked so stable tool definitions are cached.
    """
    if not _is_fal_router_request(llm_params):
        return messages, tools

    cached_tools = _tools_with_cache_control(tools)
    idx = _cache_target_index(messages)
    if idx is None:
        return messages, cached_tools

    cached_message = _message_to_dict(messages[idx])
    cached_message["content"] = _content_with_cache_control(
        cached_message.get("content")
    )

    cached_messages = list(messages)
    cached_messages[idx] = cached_message
    return cached_messages, cached_tools
