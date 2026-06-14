"""Artifact builders and writers for evaluation runs."""

from datetime import UTC, datetime
from copy import deepcopy
import json
from pathlib import Path

from agent.eval.compare import ComparisonResult


def build_run_record(
    run_id: str,
    comparison: ComparisonResult,
    dataset: str,
    split: str,
    parameters: dict,
    training_cost: float | None,
    eval_cost: float | None,
    notes: str | None,
) -> dict:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "task": comparison.task_id,
        "dataset": dataset,
        "split": split,
        "baseline_model": comparison.baseline.model_id,
        "candidate_model": comparison.candidate.model_id,
        "baseline_metrics": deepcopy(comparison.baseline.metrics),
        "candidate_metrics": deepcopy(comparison.candidate.metrics),
        "primary_metric": comparison.primary_metric,
        "primary_delta": comparison.delta,
        "training_cost": training_cost,
        "eval_cost": eval_cost,
        "notes": notes,
        "parameters": deepcopy(parameters),
    }


def build_leaderboard_row(record: dict) -> dict:
    metric = record["primary_metric"]
    return {
        "run_id": record["run_id"],
        "created_at": record["created_at"],
        "task": record["task"],
        "baseline_model": record["baseline_model"],
        "candidate_model": record["candidate_model"],
        "primary_metric": metric,
        "baseline_score": record["baseline_metrics"][metric],
        "candidate_score": record["candidate_metrics"][metric],
        "delta": record["primary_delta"],
        "training_cost": record["training_cost"],
        "eval_cost": record["eval_cost"],
        "notes": record["notes"],
    }


def write_run_artifact(output_dir: Path, record: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['run_id']}.json"
    if path.exists():
        raise FileExistsError(f"Run artifact already exists: {path}")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_leaderboard_row(output_dir: Path, row: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "leaderboard.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path
