"""Chunk text into source-addressable passages for search tools."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    title: str
    text: str
    line_start: int
    line_end: int


def chunk_markdown(content: str, *, max_chars: int = 1800) -> list[TextChunk]:
    """Split markdown into heading-aware chunks with line ranges."""
    lines = content.splitlines()
    chunks: list[TextChunk] = []
    heading = "Introduction"
    buffer: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(line for _, line in buffer).strip()
        if text:
            chunks.extend(_split_oversized(heading, buffer, max_chars=max_chars))
        buffer = []

    for index, line in enumerate(lines, 1):
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            flush()
            heading = heading_match.group(2).strip()
            buffer.append((index, line))
            continue
        buffer.append((index, line))
        if sum(len(part) + 1 for _, part in buffer) >= max_chars:
            flush()
    flush()

    if chunks:
        return chunks
    text = content.strip()
    if not text:
        return []
    return [TextChunk(title="Content", text=text[:max_chars], line_start=1, line_end=len(lines) or 1)]


def chunk_code(content: str, *, window: int = 80, overlap: int = 15) -> list[TextChunk]:
    """Split source code into overlapping line windows."""
    lines = content.splitlines()
    if not lines:
        return []
    chunks: list[TextChunk] = []
    step = max(1, window - overlap)
    for start in range(0, len(lines), step):
        end = min(len(lines), start + window)
        chunk_lines = lines[start:end]
        title = _guess_code_title(chunk_lines) or f"Lines {start + 1}-{end}"
        chunks.append(
            TextChunk(
                title=title,
                text="\n".join(chunk_lines).strip(),
                line_start=start + 1,
                line_end=end,
            )
        )
        if end == len(lines):
            break
    return [chunk for chunk in chunks if chunk.text]


def _split_oversized(
    heading: str, buffer: list[tuple[int, str]], *, max_chars: int
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for item in buffer:
        line_len = len(item[1]) + 1
        if current and current_chars + line_len > max_chars:
            chunks.append(_make_chunk(heading, current))
            current = []
            current_chars = 0
        current.append(item)
        current_chars += line_len
    if current:
        chunks.append(_make_chunk(heading, current))
    return chunks


def _make_chunk(heading: str, items: list[tuple[int, str]]) -> TextChunk:
    return TextChunk(
        title=heading,
        text="\n".join(line for _, line in items).strip(),
        line_start=items[0][0],
        line_end=items[-1][0],
    )


def _guess_code_title(lines: list[str]) -> str | None:
    for line in lines:
        stripped = line.strip()
        match = re.match(r"(async\s+def|def|class)\s+([A-Za-z_][\w]*)", stripped)
        if match:
            return stripped.rstrip(":")
        if stripped.startswith("if __name__"):
            return "Script entrypoint"
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//", "/*", "*")):
            return stripped[:80]
    return None
