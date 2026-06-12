#!/usr/bin/env python3
"""Measure the per-turn cost of the two session-persistence paths.

Replaces reasoning with numbers: drives a synthetic but realistic multi-turn
session and times the two paths using the project's *real* serializers, so the
results reflect what ``save_snapshot`` and ``save_trajectory_local`` actually do
in production.

  • Mongo snapshot — ``save_snapshot`` re-serializes ``context_manager.items``
    (``model_dump`` + per-message ``_safe_message_doc``/BSON size-check) every
    turn. ``items`` is bounded by compaction, so this stays roughly flat.

  • HF/disk trajectory — ``save_trajectory_local`` builds ``get_trajectory``
    (``model_dump`` of every item plus the full ``logged_events`` list), scrubs
    it, ``json.dump``s it to disk, and the uploader then reads it back and
    converts the whole thing to JSONL. ``logged_events`` is never trimmed within
    a session (only reset on ``/new``), so this grows with the session.

Both paths are measured here for CPU + local disk only — the real Mongo writes
and HF Hub uploads add network/server time on top, so these numbers are a lower
bound on the true cost.

Usage:
    uv run python scripts/bench_persistence.py
    uv run python scripts/bench_persistence.py --turns 200 --tokens-per-turn 800

Outputs:
    A per-checkpoint table for the compaction-on and compaction-off cases, plus
    the cumulative cost and the share of a representative turn's wall-clock.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

from litellm import ChatCompletionMessageToolCall as ToolCall
from litellm import Message

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.redact import scrub  # noqa: E402
from agent.core.session_persistence import _doc_id, _safe_message_doc  # noqa: E402

logger = logging.getLogger("bench_persistence")

# Workload shape, tuned to an autonomous ML-agent turn. Overridable via argparse.
DEFAULT_TURNS = 100
DEFAULT_MSGS_PER_TURN = 8  # 1 user + assistant(s) w/ tool_calls + tool results
DEFAULT_TOKENS_PER_TURN = 1500  # streamed assistant tokens -> that many chunk events
DEFAULT_TOOL_RESULT_BYTES = 4000  # logs / code / job output per tool message
DEFAULT_ASSISTANT_BYTES = 800
DEFAULT_CHUNK_BYTES = 6  # avg per-token chunk payload

# A representative turn's wall-clock denominator. An ML-agent turn is LLM
# streaming plus tool/job latency; we deliberately pick a small value so the
# persistence share isn't flattered.
DEFAULT_TURN_WALLCLOCK_S = 20.0

# Compaction proxy: the real code compacts on a token threshold; we cap the live
# message window. When items exceed the cap, keep head(3) + recent(KEEP_RECENT),
# mirroring ``ContextManager.compact`` (system + first user + summary + recent).
COMPACT_CAP = 120
KEEP_RECENT = 40
COMPACT_HEAD = 3

# Fixed so the synthetic workload is reproducible run to run.
SEED = 1234


def _blob(n: int) -> str:
    """Non-trivial, scrubbable-looking text so the serializers do real work."""
    return "".join(random.choice("abcdefghij klmnop ") for _ in range(n))


def _make_turn_messages(turn: int, cfg: argparse.Namespace) -> list[Message]:
    """Build one turn's worth of messages: a user prompt, an assistant message
    with tool calls, and the matching tool-result messages."""
    msgs = [Message(role="user", content=f"[turn {turn}] {_blob(120)}")]
    n_tools = cfg.msgs_per_turn - 2
    tool_calls = [
        ToolCall(
            id=f"call_{turn}_{i}",
            type="function",
            function={"name": "sandbox_exec", "arguments": json.dumps({"cmd": _blob(60)})},
        )
        for i in range(n_tools)
    ]
    msgs.append(
        Message(role="assistant", content=_blob(cfg.assistant_bytes), tool_calls=tool_calls)
    )
    for i in range(n_tools):
        msgs.append(
            Message(
                role="tool",
                content=_blob(cfg.tool_result_bytes),
                tool_call_id=f"call_{turn}_{i}",
                name="sandbox_exec",
            )
        )
    return msgs


def _make_turn_events(cfg: argparse.Namespace) -> list[dict]:
    """Build one turn's worth of logged events: the per-token assistant chunks
    plus an ``llm_call`` and a ``tool_log`` event, matching ``send_event``."""
    events = [
        {"timestamp": "t", "event_type": "assistant_chunk", "data": {"content": _blob(cfg.chunk_bytes)}}
        for _ in range(cfg.tokens_per_turn)
    ]
    events.append({"timestamp": "t", "event_type": "llm_call", "data": {"cost_usd": 0.01, "model": "x"}})
    events.append({"timestamp": "t", "event_type": "tool_log", "data": {"tool": "sandbox_exec", "log": _blob(200)}})
    return events


def _compact(items: list[Message]) -> list[Message]:
    """Bound the live message window the way ``ContextManager.compact`` does."""
    if len(items) <= COMPACT_CAP:
        return items
    head = items[:COMPACT_HEAD]
    summary = Message(role="assistant", content="[summary of older turns] " + _blob(400))
    return head + [summary] + items[-KEEP_RECENT:]


def _cost_mongo_snapshot(items: list[Message]) -> tuple[float, int]:
    """Mirror ``save_snapshot``: ``_serialize_messages`` plus a per-message
    ``_safe_message_doc`` and upsert payload. Returns (seconds, n_writes)."""
    start = time.perf_counter()
    serialized = [m.model_dump(mode="json") for m in items]  # _serialize_messages
    ops = []
    for idx, raw in enumerate(serialized):
        doc = _safe_message_doc(raw)  # BSON size-check per message
        ops.append({"_id": _doc_id("sess", idx), "idx": idx, "message": doc, "updated_at": "t"})
    return time.perf_counter() - start, len(ops)


def _cost_hf_trajectory(items: list[Message], events: list[dict], tmpdir: str) -> tuple[float, int]:
    """Mirror ``save_trajectory_local`` plus the uploader read-back/JSONL
    convert. Returns (seconds, blob_bytes_written)."""
    start = time.perf_counter()
    trajectory = {
        "session_id": "sess",
        "messages": [m.model_dump() for m in items],
        "events": events,
        "tools": [],
    }
    for key in ("messages", "events", "tools"):
        trajectory[key] = scrub(trajectory[key])
    fp = Path(tmpdir) / "sess.json"
    tmp = fp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(trajectory, f, indent=2)
    tmp.replace(fp)
    nbytes = fp.stat().st_size
    # Uploader: read the whole blob back and convert every message to JSONL.
    with open(fp) as f:
        data = json.load(f)
    for m in data["messages"]:
        json.dumps(m)
    return time.perf_counter() - start, nbytes


def _run(compaction: bool, cfg: argparse.Namespace) -> None:
    """Drive a full session and log the per-checkpoint and cumulative costs."""
    items: list[Message] = [Message(role="system", content=_blob(1500))]
    events: list[dict] = []
    cum_mongo_s = cum_hf_s = 0.0
    cum_writes = 0
    last_bytes = 0

    logger.info("=== %s ===", "COMPACTION ON" if compaction else "COMPACTION OFF")
    logger.info(
        "%5s %6s %7s %9s %8s %11s %11s %13s",
        "turn", "items", "events", "mongo_ms", "hf_ms", "cum_writes", "hf_blob_MB", "persist/turn%",
    )
    with tempfile.TemporaryDirectory() as tmp:
        for turn in range(1, cfg.turns + 1):
            items += _make_turn_messages(turn, cfg)
            events += _make_turn_events(cfg)
            if compaction:
                items = _compact(items)

            mongo_s, n_writes = _cost_mongo_snapshot(items)
            hf_s, last_bytes = _cost_hf_trajectory(items, events, tmp)
            cum_mongo_s += mongo_s
            cum_hf_s += hf_s
            cum_writes += n_writes

            if turn % max(1, cfg.turns // 4) == 0 or turn == cfg.turns:
                per_turn_pct = 100.0 * (mongo_s + hf_s) / cfg.turn_wallclock_s
                logger.info(
                    "%5d %6d %7d %9.1f %8.1f %11s %11.2f %12.2f%%",
                    turn, len(items), len(events), mongo_s * 1000, hf_s * 1000,
                    f"{cum_writes:,}", last_bytes / 1e6, per_turn_pct,
                )

    total_persist_s = cum_mongo_s + cum_hf_s
    total_wall_s = cfg.turns * cfg.turn_wallclock_s
    msgs_created = cfg.turns * cfg.msgs_per_turn
    logger.info(
        "cumulative: mongo=%.2fs hf=%.2fs total=%.2fs", cum_mongo_s, cum_hf_s, total_persist_s
    )
    logger.info(
        "cumulative message-writes (Mongo): %s (vs %s messages actually created)",
        f"{cum_writes:,}", f"{msgs_created:,}",
    )
    logger.info(
        "persistence share of wall-clock: %.2f%% (turn budget=%.0fs, %d turns)",
        100.0 * total_persist_s / total_wall_s, cfg.turn_wallclock_s, cfg.turns,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    ap.add_argument("--msgs-per-turn", type=int, default=DEFAULT_MSGS_PER_TURN)
    ap.add_argument("--tokens-per-turn", type=int, default=DEFAULT_TOKENS_PER_TURN)
    ap.add_argument("--tool-result-bytes", type=int, default=DEFAULT_TOOL_RESULT_BYTES)
    ap.add_argument("--assistant-bytes", type=int, default=DEFAULT_ASSISTANT_BYTES)
    ap.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    ap.add_argument("--turn-wallclock-s", type=float, default=DEFAULT_TURN_WALLCLOCK_S)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = parse_args(argv)
    random.seed(SEED)
    logger.info(
        "Session persistence cost benchmark (CPU+disk only; lower bound).",
    )
    logger.info(
        "Workload: %d turns, %d msgs/turn, %d chunk-events/turn, tool_result=%dB",
        cfg.turns, cfg.msgs_per_turn, cfg.tokens_per_turn, cfg.tool_result_bytes,
    )
    _run(compaction=True, cfg=cfg)
    _run(compaction=False, cfg=cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
