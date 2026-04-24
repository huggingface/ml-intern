"""Comparison logic for baseline-vs-candidate evaluation results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class ComparisonResult:
    task_id: str
    primary_metric: str
    baseline: ModelResult
    candidate: ModelResult
    baseline_score: float
    candidate_score: float
    delta: float


def compare_results(
    task_id: str,
    primary_metric: str,
    baseline: ModelResult,
    candidate: ModelResult,
) -> ComparisonResult:
    baseline_score = baseline.metrics[primary_metric]
    candidate_score = candidate.metrics[primary_metric]
    return ComparisonResult(
        task_id=task_id,
        primary_metric=primary_metric,
        baseline=baseline,
        candidate=candidate,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        delta=candidate_score - baseline_score,
    )
