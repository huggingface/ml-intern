"""Local evaluation CLI for baseline-vs-candidate model comparisons."""

import argparse
import json
from pathlib import Path
import uuid

from agent.eval.artifacts import (
    append_leaderboard_row,
    build_leaderboard_row,
    build_run_record,
    write_run_artifact,
)
from agent.eval.compare import compare_results
from agent.eval.registry import get_task
from agent.eval.runner import evaluate_model, load_examples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline model against a candidate model.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--cost-file", default=None)
    parser.add_argument("--notes", default=None)
    return parser.parse_args(argv)


def load_cost_metadata(path: str | None) -> tuple[float | None, float | None]:
    if path is None:
        return None, None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("training_cost"), payload.get("eval_cost")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task = get_task(args.task)
    examples = load_examples(task, split=args.split, limit=args.limit)

    baseline = evaluate_model(task, args.baseline_model, examples)
    candidate = evaluate_model(task, args.candidate_model, examples)

    comparison = compare_results(
        task_id=task.task_id,
        primary_metric=task.primary_metric,
        baseline=baseline,
        candidate=candidate,
    )
    training_cost, eval_cost = load_cost_metadata(args.cost_file)
    output_dir = Path(args.output_dir)
    run_record = build_run_record(
        run_id=str(uuid.uuid4()),
        comparison=comparison,
        dataset=f"{task.dataset_name}/{task.dataset_config}",
        split=args.split or task.default_split,
        parameters={"limit": args.limit},
        training_cost=training_cost,
        eval_cost=eval_cost,
        notes=args.notes,
    )
    write_run_artifact(output_dir, run_record)
    append_leaderboard_row(output_dir, build_leaderboard_row(run_record))

    print(
        f"{comparison.primary_metric}: "
        f"{comparison.baseline_score:.4f} -> {comparison.candidate_score:.4f} "
        f"(delta {comparison.delta:+.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
