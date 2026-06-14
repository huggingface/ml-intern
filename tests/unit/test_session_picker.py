from pathlib import Path

from agent.core.session_resume import SessionLogEntry
from agent.utils.session_picker import _row_label, filter_session_entries


def _entry(session_id, title=None, preview="", model=None):
    return SessionLogEntry(
        path=Path(f"{session_id}.json"),
        session_id=session_id,
        session_start_time="2026-01-01T00:00:00",
        session_end_time="2026-01-01T00:05:00",
        model_name=model,
        message_count=3,
        preview=preview,
        mtime=1_700_000_000.0,
        session_title=title,
    )


def _entries():
    return [
        _entry("a", title="Fine-Tune Llama On SQuAD", model="gpt-4o-mini"),
        _entry("b", title="Dataset Audit Helper", model="gpt-5.5"),
        _entry("c", preview="APK threat analysis platform", model="gpt-5.5"),
    ]


def test_empty_query_returns_all():
    entries = _entries()
    assert filter_session_entries(entries, "") == entries
    assert filter_session_entries(entries, "   ") == entries


def test_filter_matches_title_case_insensitive():
    out = filter_session_entries(_entries(), "llama")
    assert [e.session_id for e in out] == ["a"]


def test_filter_matches_preview_and_model():
    assert [e.session_id for e in filter_session_entries(_entries(), "apk")] == ["c"]
    # model substring
    assert {e.session_id for e in filter_session_entries(_entries(), "gpt-5.5")} == {
        "b",
        "c",
    }


def test_filter_terms_are_anded():
    out = filter_session_entries(_entries(), "dataset helper")
    assert [e.session_id for e in out] == ["b"]
    assert filter_session_entries(_entries(), "dataset llama") == []


def test_row_label_uses_title_then_preview():
    head, meta = _row_label(_entry("a", title="My Title", model="m"))
    assert head == "My Title"
    head2, _ = _row_label(_entry("b", preview="some prompt", model="m"))
    assert head2 == "some prompt"
    head3, _ = _row_label(_entry("c", model="m"))
    assert head3 == "(untitled session)"
