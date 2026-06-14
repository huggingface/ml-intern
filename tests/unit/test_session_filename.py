import asyncio
import re
from pathlib import Path

from litellm import Message

from agent.core.session import Session
from agent.core.session_resume import list_session_logs
from agent.core.title import slugify


class _FakeConfig:
    model_name = "openai/gpt-5.5"
    save_sessions = False
    session_dataset_repo = "fake/repo"
    session_log_dir = None
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


class _FakeContext:
    def __init__(self) -> None:
        self.items = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="train a model"),
        ]
        self.model_max_tokens = 200_000
        self.running_context_usage = 0
        self.on_message_added = None


def _make_session() -> Session:
    return Session(
        event_queue=asyncio.Queue(),
        config=_FakeConfig(),
        tool_router=None,
        context_manager=_FakeContext(),
        hf_token=None,
        user_id="user-a",
        local_mode=True,
    )


_TITLED = re.compile(r"^session_[a-z0-9-]+_[0-9a-f]{8}_\d{8}_\d{6}\.json$")
_LEGACY = re.compile(r"^session_[0-9a-f-]{36}_\d{8}_\d{6}\.json$")


def test_titled_filename_when_title_set(tmp_path):
    session = _make_session()
    session.session_title = "Fine-Tune Llama On SQuAD"
    saved = session.save_trajectory_local(directory=str(tmp_path))
    name = Path(saved).name
    assert _TITLED.match(name), name
    assert slugify(session.session_title) in name
    assert name.startswith("session_") and name.endswith(".json")


def test_legacy_filename_when_no_title(tmp_path):
    session = _make_session()
    saved = session.save_trajectory_local(directory=str(tmp_path))
    name = Path(saved).name
    assert _LEGACY.match(name), name


def test_unsafe_or_empty_title_falls_back_to_legacy(tmp_path):
    for bad in ("!!!", "✨✨✨", "   "):
        session = _make_session()
        session.session_title = bad
        saved = session.save_trajectory_local(directory=str(tmp_path))
        name = Path(saved).name
        assert _LEGACY.match(name), (bad, name)


def test_title_change_renames_existing_file_in_place(tmp_path):
    session = _make_session()
    first = session.save_trajectory_local(directory=str(tmp_path))
    assert Path(first).exists()

    # Setting a title renames the existing file in place (no orphan, no dup).
    session.session_title = "my run"
    session.apply_title_to_local_file()
    renamed = session._local_save_path

    assert renamed != first
    assert not Path(first).exists()
    assert Path(renamed).exists()
    assert "my-run" in Path(renamed).name
    assert len(list(tmp_path.glob("session_*.json"))) == 1

    # Further saves overwrite the same titled file.
    again = session.save_trajectory_local(directory=str(tmp_path))
    assert again == renamed
    assert len(list(tmp_path.glob("session_*.json"))) == 1


def test_rename_refreshes_persisted_title(tmp_path):
    import json

    from agent.core.session_resume import list_session_logs

    session = _make_session()
    session.save_trajectory_local(directory=str(tmp_path))

    session.session_title = "my run"
    session.apply_title_to_local_file()
    renamed = session._local_save_path

    # The JSON content's session_title is updated, not just the filename.
    data = json.loads(Path(renamed).read_text())
    assert data["session_title"] == "my run"
    # And /resume's listing reads that refreshed title.
    entry = next(
        e for e in list_session_logs(tmp_path) if Path(e.path) == Path(renamed)
    )
    assert entry.session_title == "my run"


def test_rename_preserves_original_timestamp(tmp_path):
    session = _make_session()
    first = session.save_trajectory_local(directory=str(tmp_path))
    ts = "_".join(Path(first).stem.split("_")[-2:])
    session.session_title = "my run"
    session.apply_title_to_local_file()
    assert ts in Path(session._local_save_path).name


def test_apply_title_no_file_yet_is_noop(tmp_path):
    session = _make_session()
    session.session_title = "my run"
    session.apply_title_to_local_file()  # nothing saved yet -> no crash
    assert session._local_save_path is None
    # The next save still produces a titled filename.
    saved = session.save_trajectory_local(directory=str(tmp_path))
    assert "my-run" in Path(saved).name


def test_list_session_logs_parses_titled_files(tmp_path):
    session = _make_session()
    session.session_title = "Fine-Tune Llama On SQuAD"
    saved = session.save_trajectory_local(directory=str(tmp_path))
    entries = list_session_logs(tmp_path)
    assert any(Path(e.path) == Path(saved) for e in entries)


def test_persist_title_creates_file_when_none_exists(tmp_path):
    # Mirrors /rename right after a resume: no file cached, but content exists.
    session = _make_session()
    session.config.save_sessions = True
    session.config.session_log_dir = str(tmp_path)
    session.session_title = "demo session"
    session._title_user_set = True
    session.persist_title()
    files = list(tmp_path.glob("session_*.json"))
    assert len(files) == 1
    assert "demo-session" in files[0].name
    import json
    assert json.loads(files[0].read_text())["session_title"] == "demo session"


def test_persist_title_renames_existing_file(tmp_path):
    session = _make_session()
    session.config.save_sessions = True
    session.config.session_log_dir = str(tmp_path)
    session.save_trajectory_local(directory=str(tmp_path))
    session.session_title = "renamed run"
    session.persist_title()
    files = list(tmp_path.glob("session_*.json"))
    assert len(files) == 1  # renamed in place, no second file
    assert "renamed-run" in files[0].name


def test_persist_title_noop_for_empty_session(tmp_path):
    from litellm import Message

    session = _make_session()
    session.config.save_sessions = True
    session.config.session_log_dir = str(tmp_path)
    session.context_manager.items = [Message(role="system", content="sys")]
    session.session_title = "nothing here"
    session.persist_title()
    assert list(tmp_path.glob("session_*.json")) == []
