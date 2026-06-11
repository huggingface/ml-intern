from types import SimpleNamespace

import pytest

from agent.core import effort_probe
from agent.core.prompt_caching import HF_ROUTER_SESSION_ID_HEADER

BILLING_SESSION_ID = "00000000-0000-4000-8000-000000000001"


@pytest.mark.asyncio
async def test_probe_effort_sends_session_id_to_hf_router(monkeypatch):
    completions = []

    async def fake_acompletion(**kwargs):
        completions.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=None,
        )

    monkeypatch.setattr(effort_probe, "acompletion", fake_acompletion)

    outcome = await effort_probe.probe_effort(
        "moonshotai/Kimi-K2.6:novita",
        "high",
        "hf_fake",
        session=SimpleNamespace(
            session_id="session-1",
            inference_billing_session_id=BILLING_SESSION_ID,
        ),
    )

    assert outcome.effective_effort == "high"
    assert completions[0]["extra_body"] == {"reasoning_effort": "high"}
    assert completions[0]["extra_headers"] == {
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID
    }
