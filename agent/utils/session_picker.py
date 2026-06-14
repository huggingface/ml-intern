"""Type-to-filter interactive picker for ``/resume``.

A small fzf-style picker built on prompt_toolkit (already a dependency): type to
filter the saved sessions by title/preview/model, arrow keys to move, Enter to
select, Esc/Ctrl+C to cancel. Falls back are handled by the caller; this module
only renders the picker and returns the chosen ``Path`` (or ``None``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core.session_resume import SessionLogEntry

_MAX_VISIBLE = 8


def _entry_haystack(entry: "SessionLogEntry") -> str:
    """Lowercased text a filter query is matched against."""
    parts = [
        entry.session_title or "",
        entry.preview or "",
        entry.model_name or "",
        entry.session_id or "",
    ]
    return " ".join(parts).lower()


def filter_session_entries(
    entries: list["SessionLogEntry"], query: str
) -> list["SessionLogEntry"]:
    """Return entries matching ``query`` (order preserved).

    Whitespace-separated terms are ANDed; each must be a substring of the
    entry's title/preview/model/id. An empty query matches everything.
    """
    terms = query.lower().split()
    if not terms:
        return list(entries)
    out = []
    for entry in entries:
        hay = _entry_haystack(entry)
        if all(term in hay for term in terms):
            out.append(entry)
    return out


def _row_label(entry: "SessionLogEntry") -> tuple[str, str]:
    """Return (heading, meta) display strings for one entry."""
    from agent.core.session_resume import _sort_timestamp

    heading = entry.session_title or entry.preview or "(untitled session)"
    label = _sort_timestamp(entry).astimezone().strftime("%Y-%m-%d %H:%M")
    model = entry.model_name or "unknown model"
    meta = f"{label} · {entry.message_count} msgs · {model}"
    return heading, meta


async def pick_session_interactive(
    entries: list["SessionLogEntry"],
) -> Path | None:
    """Run the type-to-filter picker; return the chosen path or None."""
    if not entries:
        return None

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    state = {"query": "", "sel": 0, "offset": 0}

    def matches() -> list["SessionLogEntry"]:
        return filter_session_entries(entries, state["query"])

    def get_fragments():
        rows = matches()
        if state["sel"] >= len(rows):
            state["sel"] = max(0, len(rows) - 1)
        # Keep the selection inside the visible viewport.
        if state["sel"] < state["offset"]:
            state["offset"] = state["sel"]
        elif state["sel"] >= state["offset"] + _MAX_VISIBLE:
            state["offset"] = state["sel"] - _MAX_VISIBLE + 1

        frags: list[tuple[str, str]] = []
        frags.append(
            ("class:header", "Resume a session  (type to filter · ↑↓ · Enter · Esc)\n")
        )
        frags.append(("class:filter", f"\nfilter > {state['query']}\n\n"))

        if not rows:
            frags.append(("class:meta", "  no matching sessions\n"))
        else:
            window = rows[state["offset"] : state["offset"] + _MAX_VISIBLE]
            if state["offset"] > 0:
                frags.append(("class:meta", "  ↑ more\n"))
            for i, entry in enumerate(window):
                row_idx = state["offset"] + i
                selected = row_idx == state["sel"]
                heading, meta = _row_label(entry)
                marker = "❯ " if selected else "  "
                head_style = "class:selected" if selected else ""
                frags.append((head_style, f"{marker}{heading}\n"))
                frags.append(("class:meta", f"    {meta}\n"))
            if state["offset"] + _MAX_VISIBLE < len(rows):
                frags.append(("class:meta", "  ↓ more\n"))

        frags.append(("class:meta", f"\n{len(rows)} of {len(entries)} match\n"))
        return frags

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _(event):
        state["sel"] = max(0, state["sel"] - 1)

    @kb.add("down")
    @kb.add("c-n")
    def _(event):
        rows = matches()
        state["sel"] = min(len(rows) - 1, state["sel"] + 1) if rows else 0

    @kb.add("enter")
    def _(event):
        rows = matches()
        result = rows[state["sel"]].path if rows else None
        event.app.exit(result=result)

    @kb.add("c-c")
    @kb.add("escape")
    def _(event):
        event.app.exit(result=None)

    @kb.add("backspace")
    def _(event):
        state["query"] = state["query"][:-1]
        state["sel"] = 0

    @kb.add("c-u")
    def _(event):
        state["query"] = ""
        state["sel"] = 0

    @kb.add(Keys.Any)
    def _(event):
        if event.data and event.data.isprintable():
            state["query"] += event.data
            state["sel"] = 0

    style = Style.from_dict(
        {
            "header": "bold",
            "filter": "ansicyan",
            "selected": "bold ansicyan",
            "meta": "ansibrightblack",
        }
    )

    control = FormattedTextControl(get_fragments, focusable=True, show_cursor=False)
    layout = Layout(HSplit([Window(content=control, wrap_lines=True)]))
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    return await app.run_async()
