import pytest

from agent.core.hf_access import (
    fetch_hf_user_plan,
    is_billing_error,
    is_inference_billing_error,
    jobs_access_from_whoami,
    normalize_hf_user_plan,
)


def test_personal_user_lists_username_namespace():
    access = jobs_access_from_whoami(
        {
            "name": "alice",
            "orgs": [],
        }
    )
    assert access.username == "alice"
    assert access.org_names == []
    assert access.eligible_namespaces == ["alice"]
    assert access.default_namespace == "alice"


def test_user_with_orgs_lists_all_namespaces_regardless_of_plan():
    # Plan/tier is ignored — credits live on the namespace itself, so any
    # org the user belongs to is eligible.  We sort orgs alphabetically and
    # always put the personal namespace first so the picker default is the
    # user's own account.
    access = jobs_access_from_whoami(
        {
            "name": "alice",
            "orgs": [
                {"name": "team-a", "plan": "team"},
                {"name": "oss-friends", "plan": "free"},
            ],
        }
    )
    assert access.username == "alice"
    assert access.org_names == ["oss-friends", "team-a"]
    assert access.eligible_namespaces == ["alice", "oss-friends", "team-a"]
    assert access.default_namespace == "alice"


def test_free_user_without_org_still_eligible_under_personal_namespace():
    # Pro is no longer required — the user is offered their personal
    # namespace; whether they actually have credits is decided at job
    # creation time when HF returns a 402 / billing error.
    access = jobs_access_from_whoami(
        {
            "name": "alice",
            "orgs": [],
        }
    )
    assert access.eligible_namespaces == ["alice"]
    assert access.default_namespace == "alice"


def test_org_only_token_falls_back_to_first_org():
    access = jobs_access_from_whoami(
        {
            "name": None,
            "orgs": [{"name": "team-a"}, {"name": "team-b"}],
        }
    )
    assert access.username is None
    assert access.eligible_namespaces == ["team-a", "team-b"]
    assert access.default_namespace == "team-a"


def test_is_billing_error_detects_402_and_credit_phrasing():
    assert is_billing_error("402 Payment Required")
    assert is_billing_error("Insufficient credits on namespace foo")
    assert is_billing_error("This namespace requires credits to run jobs")
    assert is_billing_error("Out of credit, please add billing")
    assert not is_billing_error("Internal server error")
    assert not is_billing_error("")


def test_is_inference_billing_error_detects_credit_and_quota_phrasing():
    assert is_inference_billing_error("402 Payment Required")
    assert is_inference_billing_error("exhausted monthly credits")
    assert is_inference_billing_error("insufficient_quota")
    assert is_inference_billing_error("monthly credits exhausted")
    assert not is_inference_billing_error("503 service unavailable")


def test_normalize_hf_user_plan_uses_ispro_only():
    assert normalize_hf_user_plan({"isPro": True}) == "pro"
    assert normalize_hf_user_plan({"isPro": False}) == "free"
    assert normalize_hf_user_plan({"plan": "HF Pro"}) == "free"
    assert normalize_hf_user_plan(None) is None


@pytest.mark.asyncio
async def test_fetch_hf_user_plan_returns_unknown_without_token():
    assert await fetch_hf_user_plan(None) == "unknown"


@pytest.mark.asyncio
async def test_fetch_hf_user_plan_returns_unknown_when_whoami_unavailable(monkeypatch):
    async def fake_fetch_whoami_v2(_token, timeout=5.0):
        return None

    monkeypatch.setattr("agent.core.hf_access.fetch_whoami_v2", fake_fetch_whoami_v2)

    assert await fetch_hf_user_plan("hf-token") == "unknown"


@pytest.mark.asyncio
async def test_fetch_hf_user_plan_normalizes_whoami(monkeypatch):
    async def fake_fetch_whoami_v2(_token, timeout=5.0):
        return {"isPro": True}

    monkeypatch.setattr("agent.core.hf_access.fetch_whoami_v2", fake_fetch_whoami_v2)

    assert await fetch_hf_user_plan("hf-token") == "pro"
