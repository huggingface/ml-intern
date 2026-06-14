"""Tests for ``agent.core.session_resume``."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from litellm import Message

from agent.core import session_resume

ROUTER_GPT_55 = "openai/gpt-5.5:fal-ai"


def _write_session_log(
    directory: Path,
    name: str,
    *,
    session_id: str,
    content: str,
    mtime: float,
    user_id: str | None = "user-a",
    extra_messages: list[dict] | None = None,
    events: list[dict] | None = None,
    session_title: str | None = None,
    end_time: str = "2026-01-01T00:05:00",
) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / name
    payload = {
        "session_id": session_id,
        "session_title": session_title,
        "user_id": user_id,
        "session_start_time": "2026-01-01T00:00:00",
        "session_end_time": end_time,
        "model_name": ROUTER_GPT_55,
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": content},
            *(extra_messages or []),
        ],
        "events": events
        if events is not None
        else [{"event_type": "turn_complete", "data": {}}],
    }
    path.write_text(json.dumps(payload))
    os.utime(path, (mtime, mtime))
    return path


class _FakeContext:
    def __init__(self) -> None:
        self.items = [Message(role="system", content="current system")]
        self.running_context_usage = 0
        self.recompute_calls: list[str] = []

    def _recompute_usage(self, model_name: str) -> None:
        self.recompute_calls.append(model_name)
        self.running_context_usage = 123


class _FakeSession:
    def __init__(self, *, user_id: str | None = "user-a") -> None:
        self.context_manager = _FakeContext()
        self.config = SimpleNamespace(model_name="moonshotai/Kimi-K2.6")
        self.session_id = "current-session"
        self.session_start_time = "2026-01-02T00:00:00"
        self.user_id = user_id
        self.session_title: str | None = None
        self._title_user_set = False
        self.logged_events: list[dict] = []
        self._local_save_path: str | None = None
        self.turn_count = 0
        self.last_auto_save_turn = 0
        self.pending_approval: dict | None = {"tool_calls": ["pending"]}

    def update_model(self, model_name: str) -> None:
        self.config.model_name = model_name


def test_session_log_listing_newest_first(tmp_path):
    log_dir = tmp_path / "session_logs"
    older = _write_session_log(
        log_dir,
        "older.json",
        session_id="older-session",
        content="older prompt",
        mtime=time.time() - 10,
    )
    newer = _write_session_log(
        log_dir,
        "newer.json",
        session_id="newer-session",
        content="newer prompt",
        mtime=time.time(),
    )

    entries = session_resume.list_session_logs(log_dir)

    assert [entry.path for entry in entries] == [newer, older]
    assert entries[0].session_id == "newer-session"
    assert entries[0].preview == "newer prompt"


def test_restore_continues_when_user_id_matches(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="continue this work",
        mtime=time.time(),
        user_id="user-a",
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["restored_count"] == 1
    assert result["dropped_count"] == 0
    assert result["forked"] is False
    assert result["model_name"] == ROUTER_GPT_55
    assert result["had_redacted_content"] is False
    assert result["invalid_saved_model"] is None
    assert session.config.model_name == ROUTER_GPT_55
    assert session.session_id == "saved-session"
    # Source log path is never reused: future heartbeat saves write to a
    # fresh file so the snapshot stays intact (regression: see source-log
    # round-trip test below).
    assert session._local_save_path is None
    assert session.turn_count == 1
    assert session.last_auto_save_turn == 1
    assert session.pending_approval is None
    assert [msg.role for msg in session.context_manager.items] == ["system", "user"]
    assert session.context_manager.items[0].content == "current system"
    assert session.context_manager.items[1].content == "continue this work"
    assert session.context_manager.running_context_usage == 123
    assert session.context_manager.recompute_calls == [ROUTER_GPT_55]
    assert len(session.logged_events) == 1
    marker = session.logged_events[0]
    assert marker["event_type"] == "resumed_from"
    assert marker["data"]["forked"] is False
    assert marker["data"]["original_session_id"] == "saved-session"
    assert marker["data"]["original_event_count"] == 1


def test_restore_forks_when_user_id_differs(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="someone else's chat",
        mtime=time.time(),
        user_id="user-a",
    )

    session = _FakeSession(user_id="user-b")
    original_session_id = session.session_id
    original_start_time = session.session_start_time

    result = session_resume.restore_session_from_log(session, path)

    assert result["forked"] is True
    assert session.session_id == original_session_id
    assert session.session_start_time == original_start_time
    assert session._local_save_path is None
    marker = session.logged_events[0]
    assert marker["event_type"] == "resumed_from"
    assert marker["data"]["forked"] is True
    assert marker["data"]["original_session_id"] == "saved-session"


def test_restore_forks_when_one_side_is_anonymous(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="anonymous save",
        mtime=time.time(),
        user_id=None,
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["forked"] is True
    assert session._local_save_path is None


def test_restore_continues_when_both_sides_anonymous(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="local-only chat",
        mtime=time.time(),
        user_id=None,
    )

    session = _FakeSession(user_id=None)

    result = session_resume.restore_session_from_log(session, path)

    assert result["forked"] is False
    assert session.session_id == "saved-session"
    assert session._local_save_path is None


def test_restore_rejects_invalid_saved_model(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = log_dir / "session.json"
    log_dir.mkdir()
    path.write_text(
        json.dumps(
            {
                "session_id": "saved",
                "user_id": "user-a",
                "model_name": "not a real id with spaces",
                "messages": [{"role": "user", "content": "hello"}],
                "events": [],
            }
        )
    )

    session = _FakeSession(user_id="user-a")
    original_model = session.config.model_name

    result = session_resume.restore_session_from_log(session, path)

    assert result["invalid_saved_model"] == "not a real id with spaces"
    assert result["model_name"] == original_model
    assert session.config.model_name == original_model


def test_restore_counts_dropped_messages(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = log_dir / "session.json"
    log_dir.mkdir()
    path.write_text(
        json.dumps(
            {
                "session_id": "saved",
                "user_id": "user-a",
                "model_name": ROUTER_GPT_55,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": 12345},  # invalid content type
                ],
                "events": [],
            }
        )
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["restored_count"] == 1
    assert result["dropped_count"] == 1


def test_restore_does_not_overwrite_source_log_on_save(tmp_path, monkeypatch):
    """Regression: resuming + saving must not destroy the source log on disk.

    Without the always-fork ``_local_save_path`` reset, the next heartbeat
    save would rewrite the source file with ``events=[resumed_from]`` and
    ``total_cost_usd=0``, wiping the original audit trail. This builds a
    real ``Session`` and exercises the round-trip.
    """
    monkeypatch.chdir(tmp_path)

    from agent.context_manager.manager import ContextManager
    from agent.core.session import Session

    log_dir = tmp_path / "session_logs"
    log_dir.mkdir()
    src_path = log_dir / "src.json"
    src_payload = {
        "session_id": "saved-session",
        "user_id": "user-a",
        "session_start_time": "2026-01-01T00:00:00",
        "session_end_time": "2026-01-01T00:05:00",
        "model_name": ROUTER_GPT_55,
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "earlier work"},
        ],
        "events": [
            {"event_type": "llm_call", "data": {"cost_usd": 0.42}},
            {"event_type": "turn_complete", "data": {}},
        ],
    }
    src_path.write_text(json.dumps(src_payload, indent=2))
    src_bytes_before = src_path.read_bytes()

    class _Cfg:
        model_name = ROUTER_GPT_55
        save_sessions = True
        session_dataset_repo = None
        auto_save_interval = 1
        heartbeat_interval_s = 60
        max_iterations = 10
        yolo_mode = False
        confirm_cpu_jobs = False
        auto_file_upload = False
        reasoning_effort = None
        share_traces = False
        personal_trace_repo_template = None
        mcpServers: dict = {}

    cm = ContextManager.__new__(ContextManager)
    cm.items = [Message(role="system", content="current system")]
    cm.tool_specs = []
    cm.model_max_tokens = 200_000
    cm.running_context_usage = 0
    cm.compact_size = 0.1
    cm.untouched_messages = 5
    cm.hf_token = None
    cm.local_mode = True
    cm.system_prompt = "current system"
    cm.on_message_added = None

    import asyncio as _asyncio

    session = Session(
        event_queue=_asyncio.Queue(),
        config=_Cfg(),
        tool_router=None,
        context_manager=cm,
        hf_token=None,
        user_id="user-a",
        local_mode=True,
    )

    session_resume.restore_session_from_log(session, src_path)
    assert session._local_save_path is None

    saved_path = session.save_trajectory_local(directory=str(log_dir))

    assert saved_path is not None
    assert Path(saved_path) != src_path
    assert src_path.read_bytes() == src_bytes_before


def test_restore_flags_redacted_messages(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="my token is [REDACTED_HF_TOKEN]",
        mtime=time.time(),
        user_id="user-a",
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["had_redacted_content"] is True


def test_resolve_session_log_arg_accepts_index_and_id_prefix(tmp_path):
    log_dir = tmp_path / "session_logs"
    older = _write_session_log(
        log_dir,
        "older.json",
        session_id="abcdef-older",
        content="x",
        mtime=time.time() - 10,
    )
    newer = _write_session_log(
        log_dir,
        "newer.json",
        session_id="123456-newer",
        content="y",
        mtime=time.time(),
    )
    entries = session_resume.list_session_logs(log_dir)

    assert session_resume.resolve_session_log_arg("1", entries, log_dir) == newer
    assert session_resume.resolve_session_log_arg("abc", entries, log_dir) == older
    assert session_resume.resolve_session_log_arg("nope", entries, log_dir) is None


def test_list_populates_session_title_with_preview_fallback(tmp_path):
    log_dir = tmp_path / "session_logs"
    _write_session_log(
        log_dir, "titled.json", session_id="s1", content="train a model",
        mtime=time.time(), session_title="Fine-Tune Llama", end_time="2026-02-02T10:00:00",
    )
    _write_session_log(
        log_dir, "untitled.json", session_id="s2", content="process data",
        mtime=time.time() - 5, end_time="2026-02-01T10:00:00",
    )
    by_id = {e.session_id: e for e in session_resume.list_session_logs(log_dir)}
    assert by_id["s1"].session_title == "Fine-Tune Llama"
    assert by_id["s2"].session_title is None
    assert by_id["s2"].preview == "process data"  # preview still available


def test_preview_skips_slash_command_first_message(tmp_path):
    log_dir = tmp_path / "session_logs"
    _write_session_log(
        log_dir, "cmd.json", session_id="s1", content="/model openai/gpt-5.5",
        mtime=time.time(),
        extra_messages=[{"role": "user", "content": "actually fine-tune llama"}],
    )
    entry = session_resume.list_session_logs(log_dir)[0]
    assert entry.preview == "actually fine-tune llama"


def test_sort_and_display_agree_newest_first(tmp_path):
    log_dir = tmp_path / "session_logs"
    # Deliberately give the OLDER end_time a NEWER mtime to expose the old
    # mtime-sort-vs-end_time-display mismatch; the fix sorts by end_time.
    _write_session_log(
        log_dir, "a.json", session_id="old", content="x",
        mtime=time.time(), end_time="2026-01-01T09:00:00",
    )
    _write_session_log(
        log_dir, "b.json", session_id="new", content="y",
        mtime=time.time() - 100, end_time="2026-03-01T09:00:00",
    )
    entries = session_resume.list_session_logs(log_dir)
    # Newest end_time first regardless of mtime.
    assert [e.session_id for e in entries] == ["new", "old"]
    # Displayed labels are in the same (descending) order.
    labels = [session_resume._sort_timestamp(e) for e in entries]
    assert labels == sorted(labels, reverse=True)


def test_sort_timestamp_falls_back_to_mtime(tmp_path):
    entry = session_resume.SessionLogEntry(
        path=Path("x.json"), session_id="s", session_start_time="not-a-date",
        session_end_time="also-bad", model_name=None, message_count=1,
        preview="p", mtime=1_700_000_000.0,
    )
    ts = session_resume._sort_timestamp(entry)
    assert ts.tzinfo is not None  # tz-aware, no TypeError


def test_format_entry_shows_title(tmp_path):
    log_dir = tmp_path / "session_logs"
    _write_session_log(
        log_dir, "t.json", session_id="s1", content="train a model",
        mtime=time.time(), session_title="My Cool Run",
    )
    entry = session_resume.list_session_logs(log_dir)[0]
    out = session_resume.format_session_log_entry(1, entry)
    assert "My Cool Run" in out


def test_restore_carries_session_title(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir, "t.json", session_id="s1", content="hello",
        mtime=time.time(), session_title="Resumed Run", user_id="user-a",
    )
    session = _FakeSession(user_id="user-a")
    result = session_resume.restore_session_from_log(session, path)
    assert session.session_title == "Resumed Run"
    assert session._title_user_set is True
    assert result["session_title"] == "Resumed Run"


def test_list_dedupes_same_session_id_keeping_newest(tmp_path):
    log_dir = tmp_path / "session_logs"
    # Same session_id (a resumed continuation), two files, different end_times.
    _write_session_log(
        log_dir, "old.json", session_id="dup", content="first",
        mtime=time.time() - 100, session_title="Old Title",
        end_time="2026-01-01T09:00:00",
    )
    _write_session_log(
        log_dir, "new.json", session_id="dup", content="continued",
        mtime=time.time(), session_title="New Title",
        end_time="2026-03-01T09:00:00",
    )
    # A genuinely different session that happens to share a title.
    _write_session_log(
        log_dir, "other.json", session_id="other", content="z",
        mtime=time.time() - 50, session_title="New Title",
        end_time="2026-02-01T09:00:00",
    )
    entries = session_resume.list_session_logs(log_dir)
    ids = [e.session_id for e in entries]
    assert ids.count("dup") == 1  # collapsed to one
    assert "other" in ids  # distinct session kept
    # The kept "dup" entry is the newest (New Title).
    dup = next(e for e in entries if e.session_id == "dup")
    assert dup.session_title == "New Title"
