"""Tests for the embedded sandbox FastAPI server's bearer-token auth (issue #78)."""

import importlib.util
import subprocess

from fastapi.testclient import TestClient

from agent.tools.sandbox_client import _SANDBOX_SERVER


def _load_server(tmp_path, monkeypatch, token):
    """Write the embedded server source to disk and importlib-load it.

    Module-level `_AUTH_TOKEN` is bound at import time from `os.environ`, so
    `monkeypatch.setenv` before import is what makes each test isolated.
    """
    monkeypatch.setenv("HF_TOKEN", token)
    path = tmp_path / "sandbox_server.py"
    path.write_text(_SANDBOX_SERVER)
    spec = importlib.util.spec_from_file_location("sandbox_server_under_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_authorization_header_rejects(tmp_path, monkeypatch):
    mod = _load_server(tmp_path, monkeypatch, "secret-xyz")
    client = TestClient(mod.app)
    assert client.get("/api/health").status_code == 401


def test_bearer_wrong_token_rejects(tmp_path, monkeypatch):
    mod = _load_server(tmp_path, monkeypatch, "secret-xyz")
    client = TestClient(mod.app)
    r = client.get("/api/health", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_bearer_correct_token_passes(tmp_path, monkeypatch):
    mod = _load_server(tmp_path, monkeypatch, "secret-xyz")
    client = TestClient(mod.app)
    r = client.get("/api/health", headers={"Authorization": "Bearer secret-xyz"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_bash_unauthenticated_never_executes(tmp_path, monkeypatch):
    """/api/bash must 401 before subprocess.Popen is invoked."""
    mod = _load_server(tmp_path, monkeypatch, "secret-xyz")

    def _fail(*_a, **_kw):
        raise AssertionError("subprocess.Popen invoked without auth")

    monkeypatch.setattr(subprocess, "Popen", _fail)
    client = TestClient(mod.app)
    r = client.post(
        "/api/bash",
        headers={"Authorization": "Bearer wrong"},
        json={"command": "id", "work_dir": "/app", "timeout": 10},
    )
    assert r.status_code == 401


def test_fail_closed_when_hf_token_unset(tmp_path, monkeypatch):
    """With no HF_TOKEN in the env, every request must 401 — including ones
    that present an empty Bearer value."""
    mod = _load_server(tmp_path, monkeypatch, "")
    client = TestClient(mod.app)
    assert client.get("/api/health").status_code == 401
    r = client.get("/api/health", headers={"Authorization": "Bearer "})
    assert r.status_code == 401


def test_write_endpoint_also_protected(tmp_path, monkeypatch):
    """Spot-check that POST routes beyond /api/bash are covered by the
    app-wide dependency (write/edit/read/kill/exists all share it)."""
    mod = _load_server(tmp_path, monkeypatch, "secret-xyz")
    client = TestClient(mod.app)
    target = tmp_path / "should_not_exist.txt"
    r = client.post(
        "/api/write",
        headers={"Authorization": "Bearer wrong"},
        json={"path": str(target), "content": "pwned"},
    )
    assert r.status_code == 401
    assert not target.exists()


def test_bash_with_valid_auth_executes(tmp_path, monkeypatch):
    """Positive-path check: with the correct Bearer, /api/bash actually runs
    the command and returns its output. Balances the auth-only negative tests."""
    mod = _load_server(tmp_path, monkeypatch, "secret-xyz")
    client = TestClient(mod.app)
    r = client.post(
        "/api/bash",
        headers={"Authorization": "Bearer secret-xyz"},
        json={"command": "echo hello-sandbox", "work_dir": str(tmp_path), "timeout": 10},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["success"] is True
    assert "hello-sandbox" in payload["output"]
