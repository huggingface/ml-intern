"""Regression tests for ``_bash_handler`` argument handling.

The bash tool's JSON schema declares ``timeout`` as an integer, but model
providers occasionally serialize schema-typed integers as JSON strings
(``"120"``). The handler must coerce such values instead of letting
``min(...)`` raise an uncaught ``TypeError`` that kills the agent turn.
"""

from types import SimpleNamespace

from agent.tools import local_tools
from agent.tools.local_tools import DEFAULT_TIMEOUT, MAX_TIMEOUT


def _fake_run(seen: dict):
    """Return a ``subprocess.run`` stand-in that records its kwargs."""

    def run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    return run


async def test_bash_timeout_string_arg_does_not_crash(monkeypatch):
    """A string ``timeout`` must be coerced, not crash the handler."""
    seen: dict = {}
    monkeypatch.setattr(local_tools.subprocess, "run", _fake_run(seen))

    output, ok = await local_tools._bash_handler(
        {"command": "echo hi", "timeout": "30"}
    )

    assert ok is True
    assert output == "ok"
    assert seen["kwargs"]["timeout"] == 30
    assert isinstance(seen["kwargs"]["timeout"], int)


async def test_bash_timeout_none_uses_default(monkeypatch):
    """A missing ``timeout`` falls back to ``DEFAULT_TIMEOUT``."""
    seen: dict = {}
    monkeypatch.setattr(local_tools.subprocess, "run", _fake_run(seen))

    output, ok = await local_tools._bash_handler({"command": "echo hi"})

    assert ok is True
    assert seen["kwargs"]["timeout"] == DEFAULT_TIMEOUT


async def test_bash_timeout_integer_arg_is_unchanged(monkeypatch):
    """A valid integer ``timeout`` is passed through unchanged."""
    seen: dict = {}
    monkeypatch.setattr(local_tools.subprocess, "run", _fake_run(seen))

    await local_tools._bash_handler({"command": "echo hi", "timeout": 300})

    assert seen["kwargs"]["timeout"] == 300


async def test_bash_timeout_oversized_is_clamped(monkeypatch):
    """An oversized ``timeout`` is clamped to ``MAX_TIMEOUT``."""
    seen: dict = {}
    monkeypatch.setattr(local_tools.subprocess, "run", _fake_run(seen))

    await local_tools._bash_handler(
        {"command": "echo hi", "timeout": MAX_TIMEOUT + 100_000}
    )

    assert seen["kwargs"]["timeout"] == MAX_TIMEOUT


async def test_bash_timeout_oversized_string_is_clamped(monkeypatch):
    """An oversized string ``timeout`` is coerced *and* clamped."""
    seen: dict = {}
    monkeypatch.setattr(local_tools.subprocess, "run", _fake_run(seen))

    await local_tools._bash_handler(
        {"command": "echo hi", "timeout": str(MAX_TIMEOUT + 100_000)}
    )

    assert seen["kwargs"]["timeout"] == MAX_TIMEOUT


async def test_bash_timeout_non_numeric_falls_back_to_default(monkeypatch):
    """A non-numeric ``timeout`` falls back to ``DEFAULT_TIMEOUT``."""
    seen: dict = {}
    monkeypatch.setattr(local_tools.subprocess, "run", _fake_run(seen))

    output, ok = await local_tools._bash_handler(
        {"command": "echo hi", "timeout": "not-a-number"}
    )

    assert ok is True
    assert seen["kwargs"]["timeout"] == DEFAULT_TIMEOUT
