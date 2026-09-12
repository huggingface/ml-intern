"""Unit tests for KPI dataset v2 facts and rollups."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


SALT = "stable-test-salt"


def _load():
    path = Path(__file__).parent.parent.parent / "scripts" / "build_kpis.py"
    spec = importlib.util.spec_from_file_location("build_kpis", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_kpis"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _ev(event_type, data=None, ts="2026-06-01T10:00:00+00:00"):
    return {"timestamp": ts, "event_type": event_type, "data": data or {}}


def _usage_metrics(
    *,
    total_usd=0.0,
    inference_usd=0.0,
    jobs_usd=0.0,
    sandboxes_usd=0.0,
    prompt=0,
    completion=0,
    cache_read=0,
    cache_creation=0,
    total_tokens=0,
):
    return {
        "total_usd": total_usd,
        "total_usd_source": "hf_billing_plus_sandbox_estimate",
        "app_total_usd": total_usd,
        "app_telemetry": {
            "inference_usd": inference_usd,
            "hf_jobs_estimated_usd": jobs_usd,
            "sandbox_estimated_usd": sandboxes_usd,
        },
        "hf_billing": {
            "available": True,
            "current_session": {
                "inference_providers_usd": inference_usd,
                "hf_jobs_usd": jobs_usd,
            },
        },
        "llm": {
            "calls": 1,
            "calls_by_model": {"model-a": 1},
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "total_tokens": total_tokens,
        },
        "hf_jobs": {"submits": 0, "estimated_usd": jobs_usd},
        "sandboxes": {"creates": 0, "estimated_usd": sandboxes_usd},
    }


def _session(
    *,
    session_id="session-1",
    user_id="user-1",
    user_plan="free",
    start="2026-06-01T10:00:00+00:00",
    end="2026-06-01T10:05:00+00:00",
    events=None,
    messages=None,
    usage_metrics=None,
):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "user_plan": user_plan,
        "session_start_time": start,
        "session_end_time": end,
        "model_name": "model-a",
        "messages": messages or [{"role": "user", "content": "hi"}],
        "events": events or [],
        "usage_metrics": usage_metrics or _usage_metrics(),
    }


class FakeHfApi:
    def __init__(self, files):
        self.files = files
        self.uploads = []

    def list_repo_files(self, *, repo_id, repo_type, token):
        return list(self.files)

    def create_repo(self, **kwargs):
        return None

    def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
        with open(path_or_fileobj, "rb") as f:
            content = f.read()
        self.uploads.append({"path_in_repo": path_in_repo, "content": content})


def test_session_fact_hashes_ids_preserves_plan_and_uses_usage_metrics_for_billing():
    mod = _load()
    session = _session(
        session_id="raw-session",
        user_id="raw-user",
        user_plan="pro",
        start="2026-06-01T23:50:00+00:00",
        end="2026-06-02T00:10:00+00:00",
        events=[
            _ev(
                "llm_call",
                {
                    "model": "model-a",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cache_read_tokens": 2,
                    "cache_creation_tokens": 3,
                    "total_tokens": 20,
                    "cost_usd": 999.0,
                },
                ts="2026-06-01T23:59:00+00:00",
            ),
            _ev("turn_complete", ts="2026-06-02T00:01:00+00:00"),
        ],
        usage_metrics=_usage_metrics(
            total_usd=3.0,
            inference_usd=1.0,
            jobs_usd=0.5,
            sandboxes_usd=0.25,
            prompt=10,
            completion=5,
            cache_read=2,
            cache_creation=3,
            total_tokens=20,
        ),
    )

    fact = mod._session_fact(session, salt=SALT)
    second = mod._session_fact(session, salt=SALT)

    assert fact["session_id_hash"] == second["session_id_hash"]
    assert fact["user_id_hash"] == second["user_id_hash"]
    assert fact["session_id_hash"] != "raw-session"
    assert fact["user_id_hash"] != "raw-user"
    assert fact["user_plan"] == "pro"
    assert fact["active_dates"] == ["2026-06-01", "2026-06-02"]
    assert fact["active_hours"] == ["2026-06-01T23", "2026-06-02T00"]
    assert fact["input_tokens"] == 15
    assert fact["output_tokens"] == 5
    assert fact["total_tokens"] == 20
    assert fact["usage_total_usd"] == 3.0
    assert fact["usage_inference_providers_usd"] == 1.0
    assert fact["usage_hf_jobs_usd"] == 0.5
    assert fact["usage_sandboxes_usd"] == 0.25

    model_usage = json.loads(fact["model_usage_json"])
    assert model_usage == {
        "model-a": {
            "sessions": 1,
            "calls": 1,
            "input_tokens": 15,
            "output_tokens": 5,
            "total_tokens": 20,
        }
    }


def test_session_fact_requires_hash_salt():
    mod = _load()

    with pytest.raises(RuntimeError):
        mod._session_fact(_session(), salt="")


def test_daily_rollup_dedupes_users_and_uses_latest_plan():
    mod = _load()
    facts = [
        mod._session_fact(
            _session(
                session_id="s1",
                user_id="u1",
                user_plan="free",
                end="2026-06-01T10:00:00+00:00",
                messages=[{"role": "user"}],
                usage_metrics=_usage_metrics(
                    total_usd=1.0,
                    inference_usd=0.5,
                    prompt=10,
                    completion=5,
                    total_tokens=15,
                ),
            ),
            salt=SALT,
        ),
        mod._session_fact(
            _session(
                session_id="s2",
                user_id="u1",
                user_plan="pro",
                end="2026-06-01T11:00:00+00:00",
                messages=[{"role": "user"}, {"role": "user"}, {"role": "user"}],
                usage_metrics=_usage_metrics(
                    total_usd=3.0,
                    inference_usd=1.5,
                    prompt=30,
                    completion=10,
                    total_tokens=40,
                ),
            ),
            salt=SALT,
        ),
        mod._session_fact(
            _session(
                session_id="s3",
                user_id="u2",
                user_plan=None,
                messages=[{"role": "user"}, {"role": "user"}],
                usage_metrics=_usage_metrics(
                    total_usd=2.0,
                    inference_usd=1.0,
                    prompt=20,
                    completion=5,
                    total_tokens=25,
                ),
            ),
            salt=SALT,
        ),
    ]

    row = mod._daily_rollup(facts, "2026-06-01")

    assert row["active_users"] == 2
    assert row["active_sessions"] == 3
    assert row["free_users"] == 0
    assert row["pro_users"] == 1
    assert row["unknown_plan_users"] == 1
    assert row["turns_min"] == 1
    assert row["turns_max"] == 3
    assert row["turns_avg"] == 2.0
    assert row["input_tokens_min"] == 10
    assert row["input_tokens_max"] == 30
    assert row["total_usage_total_usd"] == 6.0
    assert row["usage_total_usd_avg"] == 2.0


def test_job_artifact_and_model_rollups_normalize_and_dedupe():
    mod = _load()
    model_hash = "hash-model"
    dataset_hash = "hash-dataset"
    sandbox_space_hash = "hash-sandbox-space"
    trackio_space_hash = "hash-trackio-space"
    events = [
        _ev("hf_job_submit", {"job_id": "j1"}),
        _ev("hf_job_submit", {"job_id": "j2"}),
        _ev("hf_job_complete", {"job_id": "j1", "final_status": "COMPLETED"}),
        _ev("hf_job_complete", {"job_id": "j2", "final_status": "FAILED"}),
        _ev("hf_job_complete", {"job_id": "j3", "final_status": "CANCELED"}),
        _ev("sandbox_create", {"hardware": "cpu-basic"}),
        _ev("sandbox_create", {"hardware": "a10g-large"}),
        _ev(
            "hub_artifact",
            {"repo_type": "model", "artifact_hash": model_hash, "success": True},
        ),
        _ev(
            "hub_artifact",
            {"repo_type": "model", "artifact_hash": model_hash, "success": True},
        ),
        _ev(
            "hub_artifact",
            {"repo_type": "dataset", "artifact_hash": dataset_hash, "success": True},
        ),
        _ev(
            "hub_artifact",
            {
                "repo_type": "space",
                "artifact_hash": sandbox_space_hash,
                "is_sandbox": True,
                "success": True,
            },
        ),
        _ev(
            "hub_artifact",
            {
                "repo_type": "space",
                "artifact_hash": trackio_space_hash,
                "is_sandbox": False,
                "success": True,
            },
        ),
        _ev(
            "llm_call",
            {
                "model": "model-b",
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
        ),
    ]
    fact = mod._session_fact(
        _session(
            events=events,
            usage_metrics=_usage_metrics(prompt=4, completion=6, total_tokens=10),
        ),
        salt=SALT,
    )

    assert fact["hf_jobs_submitted"] == 2
    assert fact["hf_jobs_completed"] == 1
    assert fact["hf_jobs_failed"] == 1
    assert fact["hf_jobs_cancelled"] == 1
    assert fact["sandboxes_created"] == 2
    assert fact["sandboxes_cpu"] == 1
    assert fact["sandboxes_gpu"] == 1
    assert fact["hub_models_created"] == 1
    assert fact["hub_datasets_created"] == 1
    assert fact["hub_spaces_created"] == 2
    assert fact["hub_non_sandbox_spaces_created"] == 1

    row = mod._daily_rollup([fact], "2026-06-01")
    assert row["total_hf_jobs_completed"] == 1
    assert row["total_hub_models_created"] == 1
    assert row["total_hub_non_sandbox_spaces_created"] == 1
    assert json.loads(row["model_usage_json"]) == {
        "model-b": {
            "sessions": 1,
            "calls": 1,
            "input_tokens": 4,
            "output_tokens": 6,
            "total_tokens": 10,
        }
    }


def test_monthly_rollup_filters_by_active_month():
    mod = _load()
    may_fact = mod._session_fact(
        _session(
            session_id="may",
            start="2026-05-31T23:55:00+00:00",
            end="2026-06-01T00:05:00+00:00",
            events=[_ev("turn_complete", ts="2026-06-01T00:01:00+00:00")],
        ),
        salt=SALT,
    )
    july_fact = mod._session_fact(
        _session(
            session_id="july",
            start="2026-07-01T00:00:00+00:00",
            end="2026-07-01T00:05:00+00:00",
        ),
        salt=SALT,
    )

    row = mod._monthly_rollup([may_fact, july_fact], "2026-06")

    assert row["active_sessions"] == 1
    assert row["completed_sessions"] == 1


def test_run_for_hour_uploads_session_facts_only_once_for_current_day(monkeypatch):
    mod = _load()
    files = [
        "sessions/2026-06-01/previous.jsonl",
        "sessions/2026-06-02/current.jsonl",
    ]
    sessions = {
        "sessions/2026-06-01/previous.jsonl": _session(
            session_id="previous",
            start="2026-06-01T23:55:00+00:00",
            end="2026-06-02T00:05:00+00:00",
        ),
        "sessions/2026-06-02/current.jsonl": _session(
            session_id="current",
            start="2026-06-02T10:00:00+00:00",
            end="2026-06-02T10:10:00+00:00",
        ),
    }
    api = FakeHfApi(files)
    monkeypatch.setattr(
        mod,
        "_download_session",
        lambda repo_id, path, token: sessions[path],
    )

    row = mod.run_for_hour(
        api,
        source_repo="source",
        target_repo="target",
        hour_dt=datetime(2026, 6, 2, 10, tzinfo=timezone.utc),
        token="token",
        salt=SALT,
    )

    assert row["active_sessions"] == 1
    assert [upload["path_in_repo"] for upload in api.uploads] == [
        "v2/session_facts/2026-06-02.jsonl",
        "v2/hourly/2026-06-02/10.csv",
        "v2/daily/2026-06-02.csv",
        "v2/monthly/2026-06.csv",
    ]
