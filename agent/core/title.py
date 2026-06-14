"""Conversation title generation for CLI sessions.

A session gets a short human-readable title so it can be found again in the
``/resume`` picker and in its on-disk filename, the way ChatGPT and Claude name
conversations. The title is generated once, right after the first turn:

* ``generate_conversation_title`` asks the active model for a 3-6 word title.
* On any failure (network, billing, empty reply) it falls back to
  ``fallback_title`` — a pure-Python summary of the first user message, so a
  session always ends up with *some* readable name and the call never blocks or
  breaks a turn.

The call deliberately does NOT pass a ``session`` to telemetry: it uses a tiny
token budget and no reasoning effort so it can't meaningfully move cost or trip
the YOLO/usage-pause logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keep titles short and skimmable in a one-line picker row.
MAX_TITLE_WORDS = 6
MAX_TITLE_CHARS = 50

# Strip anything that looks like a long opaque secret (tokens, keys, base64
# blobs) before it can land in a persisted title or a filename. The title is
# derived from the first user message, which may contain a pasted credential.
_SECRET_RUN = re.compile(r"\b[A-Za-z0-9_\-]{30,}\b")


def _collapse(text: str) -> str:
    """Collapse all whitespace to single spaces and trim."""
    return " ".join(str(text).split())


def _strip_secrets(text: str) -> str:
    """Drop long credential-like tokens so they never reach disk via a title."""
    return _collapse(_SECRET_RUN.sub("", text))


def _cap(text: str, max_words: int = MAX_TITLE_WORDS, max_chars: int = MAX_TITLE_CHARS) -> str:
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def fallback_title(first_user_text: str | None) -> str:
    """Readable title derived from the first user message, no LLM required.

    Returns an empty string when there's nothing usable to title from.
    """
    cleaned = _strip_secrets(first_user_text or "")
    if not cleaned:
        return ""
    return _cap(cleaned)


def extract_first_user_text(items: Any) -> str:
    """Pull the first user message's text out of a context-manager item list.

    Handles both dict-shaped messages and litellm ``Message`` objects, and
    string or block-list ``content``.
    """
    for item in items or []:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role != "user":
            continue
        content = (
            item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        )
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    value = block.get("text") or block.get("content")
                    if isinstance(value, str):
                        parts.append(value)
                elif isinstance(block, str):
                    parts.append(block)
            text = " ".join(parts)
        else:
            text = ""
        text = _collapse(text)
        if text:
            return text
    return ""


def _clean_model_title(raw: str) -> str:
    """Normalize a model-produced title: one line, no quotes/trailing punctuation."""
    title = _collapse(raw)
    title = title.splitlines()[0] if title else ""
    title = title.strip().strip("\"'“”‘’`")
    title = title.rstrip(".!?,;: ")
    return _cap(_strip_secrets(title))


_SYSTEM_PROMPT = (
    "You generate a very short title summarizing a conversation. "
    "Reply with ONLY the title: 3 to 6 words, Title Case, no quotes, "
    "no trailing punctuation."
)


async def generate_conversation_title(
    model_name: str,
    hf_token: str | None,
    first_user_text: str,
    first_assistant_text: str | None = None,
) -> str:
    """Ask the active model for a short title for the conversation.

    Falls back to ``fallback_title(first_user_text)`` on any error or empty
    result. Never raises.
    """
    fallback = fallback_title(first_user_text)
    user_text = _collapse(first_user_text or "")
    if not user_text:
        return fallback

    try:
        from litellm import acompletion

        from agent.core.llm_params import _resolve_llm_params

        params = _resolve_llm_params(model_name, hf_token, reasoning_effort=None)
        assistant = _collapse(first_assistant_text or "") or "(no reply yet)"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"First user message:\n{user_text[:1000]}\n\n"
                    f"Assistant reply:\n{assistant[:1000]}\n\n"
                    "Return ONLY the title."
                ),
            },
        ]
        response = await acompletion(
            messages=messages,
            max_completion_tokens=24,
            stream=False,
            **params,
        )
        raw = ""
        if getattr(response, "choices", None):
            raw = response.choices[0].message.content or ""
        title = _clean_model_title(raw)
        return title or fallback
    except Exception as e:  # noqa: BLE001 — a bad title must never break a turn
        try:
            from agent.core.hf_access import is_inference_billing_error

            if is_inference_billing_error(e):
                logger.debug("Auto-title skipped (billing): %s", e)
            else:
                logger.debug("Auto-title generation failed: %s", e)
        except Exception:
            logger.debug("Auto-title generation failed: %s", e)
        return fallback
