"""Reload a previously saved session log into the active CLI session."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from litellm import Message

from agent.core.model_ids import strip_huggingface_model_prefix
from agent.core.model_switcher import is_valid_model_id

logger = logging.getLogger(__name__)

_REDACTED_MARKER = re.compile(r"\[REDACTED_[A-Z_]+\]")


@dataclass
class SessionLogEntry:
    """Metadata for a locally saved session log."""

    path: Path
    session_id: str
    session_start_time: str | None
    session_end_time: str | None
    model_name: str | None
    message_count: int
    preview: str
    mtime: float
    session_title: str | None = None


def _message_preview(content: Any, max_chars: int = 72) -> str:
    """Return a one-line preview for string or OpenAI-style block content."""
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
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _first_user_preview(messages: list[Any]) -> str:
    for raw in messages:
        if isinstance(raw, dict) and raw.get("role") == "user":
            preview = _message_preview(raw.get("content"))
            # Skip slash commands (e.g. a leading "/model ...") so the preview
            # reflects a real prompt rather than command noise.
            if preview and not preview.startswith("/"):
                return preview
    return "(no user prompt preview)"


def _sort_timestamp(entry: "SessionLogEntry") -> datetime:
    """Single canonical, tz-aware timestamp for both sorting and display.

    Prefers ``session_end_time``, then ``session_start_time``; naive values are
    normalized to local tz. Falls back to the file mtime when neither parses, so
    unparseable entries keep a sensible (non-collapsed) order.
    """
    for ts in (entry.session_end_time, entry.session_start_time):
        if isinstance(ts, str) and ts:
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            return dt if dt.tzinfo else dt.astimezone()
    return datetime.fromtimestamp(entry.mtime).astimezone()


def _read_session_log_entries(directory: Path) -> list[SessionLogEntry]:
    """Read every readable ``*.json`` log in one directory (no sort/dedupe)."""
    if not directory.exists():
        return []

    entries: list[SessionLogEntry] = []
    for path in directory.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        messages = data.get("messages") or []
        if not isinstance(messages, list):
            continue

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            # Namespace the fallback so a corrupted/legacy log with no
            # session_id can never collide with another file's real id when
            # the listing is deduped by session_id below.
            session_id = f"legacy:{path.stem}"

        stat = path.stat()
        title = data.get("session_title")
        entries.append(
            SessionLogEntry(
                path=path,
                session_id=session_id,
                session_start_time=data.get("session_start_time"),
                session_end_time=data.get("session_end_time"),
                model_name=data.get("model_name"),
                message_count=len(messages),
                preview=_first_user_preview(messages),
                mtime=stat.st_mtime,
                session_title=title if isinstance(title, str) and title else None,
            )
        )
    return entries


def list_session_logs(
    directory: Path,
    *,
    extra_dirs: Iterable[Path] = (),
) -> list[SessionLogEntry]:
    """Return readable session logs, newest first, deduped by ``session_id``.

    ``extra_dirs`` are union-read alongside ``directory`` (e.g. a legacy
    cwd-relative ``./session_logs`` left behind by the XDG migration) so old
    sessions stay visible regardless of launch cwd. A session present in more
    than one dir collapses to its newest file via the dedupe below.
    """
    entries: list[SessionLogEntry] = []
    seen_dirs: set[Path] = set()
    for d in [directory, *extra_dirs]:
        try:
            key = d.resolve()
        except OSError:
            key = d
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        entries.extend(_read_session_log_entries(d))

    # Sort and display use the SAME timestamp so the visible order never looks
    # scrambled (the old code sorted by mtime but displayed session_end_time).
    # mtime is only a tiebreaker for entries sharing a timestamp.
    entries.sort(key=lambda e: (_sort_timestamp(e), e.mtime), reverse=True)

    # Collapse multiple on-disk files for the same conversation to one entry.
    # A resumed continuation reuses its session_id but forks the save path, so
    # continuing writes a new-timestamp file while the original remains —
    # without this, /resume shows the same conversation two (or more) times.
    # Entries are newest-first, so the first seen per id is the latest state.
    deduped: dict[str, SessionLogEntry] = {}
    for entry in entries:
        deduped.setdefault(entry.session_id, entry)
    return list(deduped.values())


def format_session_log_entry(index: int, entry: SessionLogEntry) -> str:
    from rich.markup import escape

    label = _sort_timestamp(entry).astimezone().strftime("%Y-%m-%d %H:%M")
    short_id = entry.session_id[:8]
    model = entry.model_name or "unknown model"
    # Lead with the human-readable title; fall back to the first-prompt preview
    # for older logs that predate session titles. Escape so a title/preview
    # containing Rich markup (e.g. "[red]") can't inject styling or raise when
    # this string is printed via the console.
    heading = escape(entry.session_title or entry.preview)
    return (
        f"{index:>2}. {heading}\n"
        f"    {label}  {short_id}  {entry.message_count} msgs  {model}"
    )


def resolve_session_log_arg(
    arg: str,
    entries: list[SessionLogEntry],
    directory: Path,
) -> Path | None:
    """Resolve ``/resume <arg>`` as index, path, filename, or session id prefix."""
    value = arg.strip()
    if not value:
        return None

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(entries):
            return entries[idx - 1].path

    candidate = Path(value).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(directory / candidate)
        if candidate.suffix != ".json":
            candidates.append(directory / f"{value}.json")

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    matches = [
        entry.path
        for entry in entries
        if entry.session_id.startswith(value) or entry.path.name.startswith(value)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _turn_count_from_messages(messages: list[Any]) -> int:
    return sum(
        1 for raw in messages if isinstance(raw, dict) and raw.get("role") == "user"
    )


def _has_redacted_content(messages: list[Any]) -> bool:
    """Whether any message body contains a ``[REDACTED_*]`` marker."""
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content")
        if isinstance(content, str) and _REDACTED_MARKER.search(content):
            return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str) and _REDACTED_MARKER.search(text):
                        return True
    return False


def restore_session_from_log(session: Any, path: Path) -> dict[str, Any]:
    """Replace the active session context with messages from ``path``.

    Continues the saved session (reusing its id and on-disk save path) when
    the log's ``user_id`` matches the current session, and forks otherwise:
    the caller's session id stays put and future heartbeat saves go to a
    fresh file rather than overwriting the source log.

    Returns metadata for the ``resume_complete`` event.
    """
    with open(path) as f:
        data = json.load(f)

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Selected log does not contain a messages array")

    restored_messages: list[Message] = []
    dropped_count = 0
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") == "system":
            continue
        try:
            restored_messages.append(Message.model_validate(raw))
        except Exception as e:
            dropped_count += 1
            logger.warning("Dropping malformed message from %s: %s", path, e)

    if not restored_messages:
        raise ValueError("Selected log has no restorable non-system messages")

    cm = session.context_manager
    system_msg = cm.items[0] if cm.items and cm.items[0].role == "system" else None
    cm.items = ([system_msg] if system_msg else []) + restored_messages

    # Validate the saved model id before switching. ``update_model`` doesn't
    # check availability; an unrecognised id silently sticks and the next LLM
    # call fails with a cryptic routing error. Logs from a different
    # deployment, an older catalog, or a removed model land here.
    saved_model = data.get("model_name")
    invalid_saved_model: str | None = None
    if isinstance(saved_model, str) and saved_model:
        normalized_model = strip_huggingface_model_prefix(saved_model)
        if normalized_model and is_valid_model_id(normalized_model):
            session.update_model(normalized_model)
        else:
            invalid_saved_model = saved_model
            logger.warning(
                "Saved log model %r failed format validation; keeping %r",
                saved_model,
                session.config.model_name,
            )

    cm._recompute_usage(session.config.model_name)

    saved_session_id = data.get("session_id")
    saved_user_id = data.get("user_id")
    is_continuation = saved_user_id == session.user_id

    # Rotate the conversation epoch so an auto-title task still in flight for
    # the pre-resume conversation bails instead of stamping its title onto this
    # one (a forked resume keeps no field the title-task guard would otherwise
    # catch). getattr keeps test doubles without the attr working.
    session._conversation_epoch = getattr(session, "_conversation_epoch", 0) + 1

    # Carry the saved title across the resume so the conversation keeps its
    # name and auto-titling doesn't re-fire on it.
    saved_title = data.get("session_title")
    if isinstance(saved_title, str) and saved_title:
        session.session_title = saved_title
        session._title_user_set = True

    if is_continuation:
        if isinstance(saved_session_id, str) and saved_session_id:
            session.session_id = saved_session_id
        session.session_start_time = (
            data.get("session_start_time") or session.session_start_time
        )

    # Always fork the on-disk save path. The source log is treated as an
    # immutable snapshot: ``logged_events`` is reset to a single
    # ``resumed_from`` marker below for cost accounting, so reusing the
    # source path would let the next heartbeat save destroy the original
    # ``llm_call``/event history on disk. The next save will pick a fresh
    # filename instead.
    session._local_save_path = None

    saved_event_count = (
        len(data.get("events", [])) if isinstance(data.get("events"), list) else 0
    )
    session.logged_events = [
        {
            "timestamp": datetime.now().isoformat(),
            "event_type": "resumed_from",
            "data": {
                "path": str(path),
                "original_session_id": (
                    saved_session_id if isinstance(saved_session_id, str) else None
                ),
                "original_event_count": saved_event_count,
                "forked": not is_continuation,
            },
        }
    ]
    session.turn_count = _turn_count_from_messages(raw_messages)
    session.last_auto_save_turn = session.turn_count
    session.pending_approval = None

    return {
        "path": str(path),
        "restored_count": len(restored_messages),
        "dropped_count": dropped_count,
        "model_name": session.config.model_name,
        "session_title": session.session_title,
        "invalid_saved_model": invalid_saved_model,
        "forked": not is_continuation,
        "had_redacted_content": _has_redacted_content(raw_messages),
    }
