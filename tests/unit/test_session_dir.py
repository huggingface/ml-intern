import asyncio
from pathlib import Path
from types import SimpleNamespace

from litellm import Message

from agent.core.session import (
    DEFAULT_SESSION_LOG_DIR,
    SESSION_DIR_ENV_VAR,
    Session,
    resolve_session_log_dir,
)
from agent.core.session_resume import list_session_logs


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


def _clear_env(monkeypatch):
    monkeypatch.delenv(SESSION_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


def test_resolve_uses_config_value_first(monkeypatch, tmp_path):
    monkeypatch.setenv(SESSION_DIR_ENV_VAR, str(tmp_path / "env"))
    config = SimpleNamespace(session_log_dir=str(tmp_path / "cfg"))
    assert resolve_session_log_dir(config) == tmp_path / "cfg"


def test_resolve_uses_env_when_config_unset(monkeypatch, tmp_path):
    monkeypatch.setenv(SESSION_DIR_ENV_VAR, str(tmp_path / "env"))
    config = SimpleNamespace(session_log_dir=None)
    assert resolve_session_log_dir(config) == tmp_path / "env"


def test_resolve_uses_legacy_dir_when_present(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "session_logs").mkdir()
    # config unset, no env, legacy ./session_logs exists in cwd
    assert resolve_session_log_dir(SimpleNamespace(session_log_dir=None)) == (
        DEFAULT_SESSION_LOG_DIR
    )


def test_resolve_falls_back_to_xdg(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)  # no ./session_logs here
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
    expected = tmp_path / "xdgdata" / "ml-intern" / "sessions"
    assert resolve_session_log_dir(None) == expected


def test_resolve_xdg_default_home(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    expected = tmp_path / "home" / ".local" / "share" / "ml-intern" / "sessions"
    assert resolve_session_log_dir(None) == expected


def test_save_then_list_round_trip_via_resolver(monkeypatch, tmp_path):
    # No legacy dir, XDG pointed at tmp -> save writes there, list finds it.
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))

    session = _make_session()
    saved = session.save_trajectory_local()  # directory=None -> resolver
    resolved = resolve_session_log_dir(session.config)

    assert saved is not None
    assert Path(saved).parent == resolved
    assert resolved == tmp_path / "xdgdata" / "ml-intern" / "sessions"

    entries = list_session_logs(resolved)
    assert any(Path(e.path) == Path(saved) for e in entries)


def test_save_respects_explicit_directory(monkeypatch, tmp_path):
    # Back-compat: an explicit directory still works and overrides resolution.
    _clear_env(monkeypatch)
    session = _make_session()
    explicit = tmp_path / "explicit_logs"
    saved = session.save_trajectory_local(directory=str(explicit))
    assert saved is not None
    assert Path(saved).parent == explicit


def test_precedence_config_over_env_over_legacy(monkeypatch, tmp_path):
    # All three present -> config wins
    monkeypatch.chdir(tmp_path)
    (tmp_path / "session_logs").mkdir()
    monkeypatch.setenv(SESSION_DIR_ENV_VAR, str(tmp_path / "env"))
    config = SimpleNamespace(session_log_dir=str(tmp_path / "cfg"))
    assert resolve_session_log_dir(config) == tmp_path / "cfg"
    # Remove config -> env wins over legacy
    config = SimpleNamespace(session_log_dir=None)
    assert resolve_session_log_dir(config) == tmp_path / "env"
    # Remove env -> legacy wins
    monkeypatch.delenv(SESSION_DIR_ENV_VAR)
    assert resolve_session_log_dir(config) == DEFAULT_SESSION_LOG_DIR
