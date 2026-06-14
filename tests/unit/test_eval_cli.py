import json

import pytest


def test_eval_cli_writes_artifacts(monkeypatch, tmp_path):
    from eval import main
    from agent.eval.compare import ModelResult

    monkeypatch.setattr(
        "eval.load_examples",
        lambda task, split, limit: [{"sentence": "great", "label": 1}],
    )
    monkeypatch.setattr(
        "eval.evaluate_model",
        lambda task, model_id, examples, client=None: ModelResult(
            model_id=model_id,
            metrics={"accuracy": 1.0 if "candidate" in model_id else 0.0},
        ),
    )

    exit_code = main(
        [
            "--task", "glue_sst2",
            "--baseline-model", "baseline-model",
            "--candidate-model", "candidate-model",
            "--output-dir", str(tmp_path),
            "--limit", "1",
        ]
    )

    assert exit_code == 0
    leaderboard = (tmp_path / "leaderboard.jsonl").read_text().strip().splitlines()
    assert len(leaderboard) == 1
    row = json.loads(leaderboard[0])
    assert row["task"] == "glue_sst2"
    assert row["delta"] == 1.0


def test_eval_cli_validates_cost_file_before_model_evaluation(monkeypatch, tmp_path):
    from eval import main

    calls = {"count": 0}

    def fail_if_called(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("evaluate_model should not be called")

    monkeypatch.setattr("eval.load_examples", lambda task, split, limit: [{"sentence": "great", "label": 1}])
    monkeypatch.setattr("eval.evaluate_model", fail_if_called)

    missing_cost_file = tmp_path / "missing-costs.json"

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--task", "glue_sst2",
                "--baseline-model", "baseline-model",
                "--candidate-model", "candidate-model",
                "--output-dir", str(tmp_path),
                "--cost-file", str(missing_cost_file),
            ]
        )

    assert calls["count"] == 0


def test_eval_cli_rejects_non_positive_limit():
    from eval import main

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--task", "glue_sst2",
                "--baseline-model", "baseline-model",
                "--candidate-model", "candidate-model",
                "--output-dir", "eval_runs",
                "--limit", "0",
            ]
        )

    assert exc_info.value.code == 2
