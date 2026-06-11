"""Tests for hosted model handling in backend/routes/agent.py."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routes import agent  # noqa: E402
from dependencies import INTERNAL_HF_TOKEN_KEY  # noqa: E402


def test_available_models_exclude_sonnet_and_have_no_pro_gate():
    models = {model["id"]: model for model in agent.AVAILABLE_MODELS}

    assert models[agent.DEFAULT_OPUS_MODEL_ID]["label"] == "Claude Opus 4.8"
    assert models[agent.DEFAULT_OPUS_MODEL_ID]["recommended"] is True
    assert "recommended" not in models[agent.DEFAULT_FREE_MODEL_ID]
    assert all("minimum_plan" not in model for model in models.values())
    assert all("tier" not in model for model in models.values())


def test_default_model_for_user_is_plan_aware():
    assert agent._default_model_for_user({"plan": "pro"}) == agent.DEFAULT_OPUS_MODEL_ID
    assert (
        agent._default_model_for_user({"plan": "free"}) == agent.DEFAULT_FREE_MODEL_ID
    )
    assert agent._default_model_for_user({}) == agent.DEFAULT_FREE_MODEL_ID


@pytest.mark.asyncio
async def test_llm_health_uses_request_hf_token(monkeypatch):
    class Request:
        headers = {"Authorization": "Bearer user-token"}
        cookies = {}

    resolved = []
    completions = []

    def fake_resolve_llm_params(
        model_name,
        session_hf_token=None,
        reasoning_effort=None,
        strict=False,
    ):
        resolved.append((model_name, session_hf_token, reasoning_effort, strict))
        return {
            "model": f"openai/{model_name}",
            "api_base": "https://router.huggingface.co/v1",
            "api_key": session_hf_token,
        }

    async def fake_acompletion(**kwargs):
        completions.append(kwargs)

    monkeypatch.setattr(
        agent.session_manager,
        "config",
        SimpleNamespace(model_name=agent.DEFAULT_FREE_MODEL_ID),
    )
    monkeypatch.setattr(agent, "_resolve_llm_params", fake_resolve_llm_params)
    monkeypatch.setattr(agent, "acompletion", fake_acompletion)

    response = await agent.llm_health_check(Request())

    assert response.status == "ok"
    assert resolved == [(agent.DEFAULT_FREE_MODEL_ID, "user-token", "high", False)]
    assert completions[0]["api_key"] == "user-token"


@pytest.mark.asyncio
async def test_llm_health_skips_router_probe_without_token(monkeypatch):
    class Request:
        headers = {}
        cookies = {}

    def fail_resolve_llm_params(*args, **kwargs):
        raise AssertionError(
            "health check should not resolve router params without token"
        )

    async def fail_acompletion(**kwargs):
        raise AssertionError("health check should not call LiteLLM without token")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        agent.session_manager,
        "config",
        SimpleNamespace(model_name=agent.DEFAULT_FREE_MODEL_ID),
    )
    monkeypatch.setattr(agent, "_resolve_llm_params", fail_resolve_llm_params)
    monkeypatch.setattr(agent, "acompletion", fail_acompletion)

    response = await agent.llm_health_check(Request())

    assert response.status == "skipped"
    assert response.model == agent.DEFAULT_FREE_MODEL_ID


@pytest.mark.asyncio
async def test_generate_title_sends_session_id_to_hf_router(monkeypatch):
    completions = []
    titles = []

    def fake_resolve_llm_params(
        model_name,
        session_hf_token=None,
        reasoning_effort=None,
        strict=False,
    ):
        return {
            "model": f"openai/{model_name}",
            "api_base": "https://router.huggingface.co/v1",
            "api_key": session_hf_token,
            "extra_body": {"reasoning_effort": reasoning_effort},
        }

    async def fake_acompletion(**kwargs):
        completions.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Clean title"),
                )
            ]
        )

    async def fake_check_session_access(session_id, user):
        assert session_id == "session-1"
        assert user["user_id"] == "u1"
        return SimpleNamespace(
            session=SimpleNamespace(
                session_id="session-1",
                inference_billing_session_id="session-1:usage:window-1",
            )
        )

    async def fake_update_session_title(session_id, title):
        titles.append((session_id, title))

    monkeypatch.setattr(agent, "_resolve_llm_params", fake_resolve_llm_params)
    monkeypatch.setattr(agent, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent.session_manager,
        "update_session_title",
        fake_update_session_title,
    )

    response = await agent.generate_title(
        agent.SubmitRequest(session_id="session-1", text="Make something useful"),
        {"user_id": "u1", INTERNAL_HF_TOKEN_KEY: "hf_fake"},
    )

    assert response == {"title": "Clean title"}
    assert completions[0]["extra_body"] == {
        "reasoning_effort": "low",
        "session_id": "session-1:usage:window-1",
    }
    assert titles == [("session-1", "Clean title")]


@pytest.mark.asyncio
async def test_empty_session_model_uses_plan_default():
    assert (
        await agent._model_override_for_new_session(None, {"plan": "pro"})
        == agent.DEFAULT_OPUS_MODEL_ID
    )
    assert (
        await agent._model_override_for_new_session(None, {"plan": "free"})
        == agent.DEFAULT_FREE_MODEL_ID
    )


@pytest.mark.asyncio
async def test_explicit_session_model_is_honored():
    model = await agent._model_override_for_new_session(
        agent.DEFAULT_GPT_MODEL_ID,
        {"plan": "free"},
    )

    assert model == agent.DEFAULT_GPT_MODEL_ID


@pytest.mark.asyncio
async def test_switching_to_opus_is_allowed_for_free_user(monkeypatch):
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
        {"model": agent.DEFAULT_OPUS_MODEL_ID},
        request=None,
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {"session_id": "s1", "model": agent.DEFAULT_OPUS_MODEL_ID}
    assert updated == [("s1", agent.DEFAULT_OPUS_MODEL_ID)]


@pytest.mark.asyncio
async def test_switching_to_gpt_is_allowed_for_free_user(monkeypatch):
    updated = []

    async def fake_check_session_access(session_id, user, request=None):
        return SimpleNamespace(user_id=user["user_id"])

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
        user={"user_id": "u1", "plan": "free"},
    )

    assert response == {"session_id": "s1", "model": agent.DEFAULT_GPT_MODEL_ID}
    assert updated == [("s1", agent.DEFAULT_GPT_MODEL_ID)]


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
async def test_restore_summary_uses_default_model_without_quota_gate(monkeypatch):
    events = []

    class Request:
        headers = {}
        cookies = {}

    async def fake_create_session(**kwargs):
        events.append(("create", kwargs["model"], kwargs.get("user_plan")))
        return "s1"

    async def fake_check_session_access(
        session_id, user, request, preload_sandbox=True
    ):
        events.append(("check", session_id, preload_sandbox))
        return SimpleNamespace(session=SimpleNamespace(config=SimpleNamespace()))

    async def fake_seed(session_id, messages):
        events.append(("seed", session_id))
        return len(messages)

    monkeypatch.setattr(agent.session_manager, "create_session", fake_create_session)
    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(agent.session_manager, "seed_from_summary", fake_seed)

    response = await agent.restore_session_summary(
        Request(),
        {"messages": [{"role": "user", "content": "resume this"}]},
        {"user_id": "u1", "plan": "free"},
    )

    assert response.session_id == "s1"
    assert response.model == agent.DEFAULT_FREE_MODEL_ID
    assert events == [
        ("create", agent.DEFAULT_FREE_MODEL_ID, "free"),
        ("check", "s1", False),
        ("seed", "s1"),
    ]


@pytest.mark.asyncio
async def test_check_session_access_passes_user_plan(monkeypatch):
    seen = {}
    expected_session = SimpleNamespace(user_id="u1")

    class Request:
        headers = {"Authorization": "Bearer user-token"}
        cookies = {}

    async def fake_ensure_session_loaded(session_id, user_id, **kwargs):
        seen["session_id"] = session_id
        seen["user_id"] = user_id
        seen.update(kwargs)
        return expected_session

    monkeypatch.setattr(
        agent.session_manager,
        "ensure_session_loaded",
        fake_ensure_session_loaded,
    )

    result = await agent._check_session_access(
        "s1",
        {"user_id": "u1", "username": "tester", "plan": "pro"},
        Request(),
    )

    assert result is expected_session
    assert seen == {
        "session_id": "s1",
        "user_id": "u1",
        "hf_token": "user-token",
        "hf_username": "tester",
        "user_plan": "pro",
        "preload_sandbox": True,
    }
