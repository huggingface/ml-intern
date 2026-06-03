"""Tests for model access handling in backend/routes/agent.py."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routes import agent  # noqa: E402


def _session(model: str = agent.DEFAULT_FREE_MODEL_ID):
    return SimpleNamespace(
        session=SimpleNamespace(config=SimpleNamespace(model_name=model)),
    )


def test_premium_model_predicate_uses_router_ids_only():
    assert not agent._is_premium_model(agent.DEFAULT_FREE_MODEL_ID)
    assert agent._is_premium_model(agent.DEFAULT_OPUS_MODEL_ID)
    assert agent._is_premium_model(agent.DEFAULT_GPT_MODEL_ID)
    assert agent._is_pro_only_premium_model(agent.DEFAULT_OPUS_MODEL_ID)
    assert agent._is_pro_only_premium_model(agent.DEFAULT_GPT_MODEL_ID)
    assert not agent._is_premium_model("moonshotai/Kimi-K2.6")
    assert not agent._is_premium_model("unsupported/model")


def test_available_models_mark_kimi_default_and_opus_gpt_as_pro_only():
    models = {model["id"]: model for model in agent.AVAILABLE_MODELS}

    assert agent.DEFAULT_FREE_MODEL_ID == "moonshotai/Kimi-K2.6"
    assert models[agent.DEFAULT_FREE_MODEL_ID]["label"] == "Kimi K2.6"
    assert models[agent.DEFAULT_FREE_MODEL_ID]["recommended"] is True
    assert models[agent.DEFAULT_FREE_MODEL_ID]["minimum_plan"] == "free"
    assert all("Claude Sonnet" not in model["label"] for model in models.values())
    assert models[agent.DEFAULT_OPUS_MODEL_ID]["minimum_plan"] == "pro"
    assert models[agent.DEFAULT_GPT_MODEL_ID]["minimum_plan"] == "pro"


@pytest.mark.asyncio
async def test_default_session_uses_configured_default_model():
    model = await agent._model_override_for_new_session(None, None)

    assert model is None


@pytest.mark.asyncio
async def test_explicit_default_model_request_is_allowed():
    model = await agent._model_override_for_new_session(
        None,
        agent.DEFAULT_FREE_MODEL_ID,
    )

    assert model == agent.DEFAULT_FREE_MODEL_ID


@pytest.mark.asyncio
async def test_switching_to_default_model_is_allowed_for_free_user(monkeypatch):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(user_id="u1")

    async def fake_update_session_model(session_id, model_id):
        updated.append((session_id, model_id))

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_model",
        fake_update_session_model,
    )

    response = await agent.set_session_model(
        "s1",
        {"model": agent.DEFAULT_FREE_MODEL_ID},
        request=None,
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {"session_id": "s1", "model": agent.DEFAULT_FREE_MODEL_ID}
    assert updated == [("s1", agent.DEFAULT_FREE_MODEL_ID)]


@pytest.mark.asyncio
async def test_switching_to_pro_only_model_is_allowed_for_pro_user(monkeypatch):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(user_id="u1")

    async def fake_update_session_model(session_id, model_id):
        updated.append((session_id, model_id))

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_model",
        fake_update_session_model,
    )

    response = await agent.set_session_model(
        "s1",
        {"model": agent.DEFAULT_GPT_MODEL_ID},
        request=None,
        user={"user_id": "u1", "plan": "pro"},
    )

    assert response == {"session_id": "s1", "model": agent.DEFAULT_GPT_MODEL_ID}
    assert updated == [("s1", agent.DEFAULT_GPT_MODEL_ID)]


@pytest.mark.asyncio
async def test_switching_to_pro_only_model_is_rejected_for_free_user(monkeypatch):
    async def fake_check_session_access(session_id, user, request=None):
        return SimpleNamespace(user_id=user["user_id"])

    async def fail_if_updated(session_id, model_id):
        raise AssertionError("free users should not switch to pro-only models")

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(agent.session_manager, "update_session_model", fail_if_updated)

    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": agent.DEFAULT_GPT_MODEL_ID},
            request=None,
            user={"user_id": "u1", "plan": "free"},
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "model_requires_pro"


@pytest.mark.asyncio
async def test_switching_to_unknown_model_id_is_rejected(monkeypatch):
    async def fake_check_session_access(session_id, user, request=None):
        return SimpleNamespace(user_id=user["user_id"])

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)

    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": "unsupported/model"},
            request=None,
            user={"user_id": "u1", "plan": "free"},
        )

    assert exc_info.value.status_code == 400
    assert "Unknown model" in exc_info.value.detail


@pytest.mark.asyncio
async def test_default_model_submit_has_no_daily_limit():
    await agent._enforce_model_plan_access(
        {"user_id": "u1", "plan": "free"},
        _session(agent.DEFAULT_FREE_MODEL_ID),
    )


@pytest.mark.asyncio
async def test_free_user_cannot_submit_with_pro_only_model():
    with pytest.raises(HTTPException) as exc_info:
        await agent._enforce_model_plan_access(
            {"user_id": "u1", "plan": "free"},
            _session(agent.DEFAULT_OPUS_MODEL_ID),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "model_requires_pro"


@pytest.mark.asyncio
async def test_pro_user_can_submit_with_pro_only_model():
    await agent._enforce_model_plan_access(
        {"user_id": "u1", "plan": "pro"},
        _session(agent.DEFAULT_OPUS_MODEL_ID),
    )


@pytest.mark.asyncio
async def test_restore_summary_enforces_model_plan_before_seed(monkeypatch):
    events = []
    agent_session = _session()

    class Request:
        headers = {}
        cookies = {}

    async def fake_create_session(**kwargs):
        events.append(("create", kwargs["model"]))
        return "s1"

    async def fake_check_session_access(
        session_id, user, request, preload_sandbox=True
    ):
        events.append(("check", session_id, preload_sandbox))
        return agent_session

    async def fake_enforce_plan(user, session):
        assert user["user_id"] == "u1"
        assert session is agent_session
        events.append(("gate", session.session.config.model_name))

    async def fake_seed(session_id, messages):
        events.append(("seed", session_id))
        return len(messages)

    monkeypatch.setattr(agent.session_manager, "create_session", fake_create_session)
    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(agent, "_enforce_model_plan_access", fake_enforce_plan)
    monkeypatch.setattr(agent.session_manager, "seed_from_summary", fake_seed)

    response = await agent.restore_session_summary(
        Request(),
        {"messages": [{"role": "user", "content": "resume this"}]},
        {"user_id": "u1", "plan": "free"},
    )

    assert response.session_id == "s1"
    assert events == [
        ("create", None),
        ("check", "s1", False),
        ("gate", agent.DEFAULT_FREE_MODEL_ID),
        ("seed", "s1"),
    ]


@pytest.mark.asyncio
async def test_set_session_yolo_calls_manager_with_cap_presence(monkeypatch):
    async def fake_check_session_access(session_id, user, request=None):
        assert session_id == "s1"
        assert user["user_id"] == "u1"
        return object()

    calls = []

    async def fake_update_session_auto_approval(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {
            "enabled": kwargs["enabled"],
            "cost_cap_usd": 7.5,
            "estimated_spend_usd": 0.0,
            "remaining_usd": 7.5,
        }

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_auto_approval",
        fake_update_session_auto_approval,
    )

    response = await agent.set_session_yolo(
        "s1",
        agent.SessionYoloRequest(enabled=True, cost_cap_usd=7.5),
        {"user_id": "u1"},
    )

    assert response["enabled"] is True
    assert response["remaining_usd"] == 7.5
    assert calls == [
        (
            "s1",
            {
                "enabled": True,
                "cost_cap_usd": 7.5,
                "cap_provided": True,
            },
        )
    ]
