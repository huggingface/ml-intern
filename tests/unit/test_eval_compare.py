import pytest

from agent.eval.compare import ModelResult, compare_results


def test_compare_results_computes_primary_metric_delta():
    baseline = ModelResult(
        model_id="baseline-model",
        metrics={"accuracy": 0.84},
    )
    candidate = ModelResult(
        model_id="candidate-model",
        metrics={"accuracy": 0.89},
    )

    comparison = compare_results(
        task_id="glue_sst2",
        primary_metric="accuracy",
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.primary_metric == "accuracy"
    assert comparison.baseline_score == 0.84
    assert comparison.candidate_score == 0.89
    assert comparison.delta == pytest.approx(0.05)
