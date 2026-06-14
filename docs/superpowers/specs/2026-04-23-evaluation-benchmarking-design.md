# Evaluation And Benchmarking Design

Date: 2026-04-23
Issue: `#84`
Scope: First PR only

## Goal

Add a reproducible local evaluation pipeline for comparing a baseline model
against a trained model on a task-specific benchmark. The first PR should
produce machine-readable artifacts and a leaderboard-style summary without
changing the live agent runtime or the web UI.

## Problem

The repo can help users train and iterate on models, but it does not yet offer
a standard way to answer a basic question: did the trained model improve over
the baseline, and was the gain worth the cost?

Issue `#84` asks for:

- task-specific benchmarks
- baseline vs agent-generated model comparison
- training cost vs performance gain tracking
- a reproducible `eval.py`
- leaderboard-style logging

This design narrows that request into a first contribution that is small enough
for one PR and reusable by later work.

## First-PR Scope

This PR will add:

- a local CLI entrypoint, `eval.py`
- a small evaluation utility layer separate from `agent/core/*`
- support for comparing exactly two models in one run:
  - baseline model
  - candidate model
- a task registry for explicit benchmark definitions
- machine-readable run artifacts
- a leaderboard-style artifact for cross-run comparison
- unit tests for registry resolution, comparison logic, and artifact writing
- contributor documentation for how to run the evaluation pipeline

This PR will not add:

- frontend leaderboard views
- backend endpoints for evaluation storage
- automatic agent-triggered evaluation
- broad benchmark support for every task family
- heavyweight end-to-end remote evaluation tests

## Design Summary

The new feature should live as a standalone evaluation workflow that can be run
by contributors from the command line. The evaluation logic must be isolated
from the live chat session path so that the first PR stays low-risk and easy to
review.

The system will:

1. parse CLI arguments
2. resolve a named evaluation task
3. evaluate the baseline model
4. evaluate the candidate model
5. compute metric deltas and cost metadata
6. write a reproducible run record
7. update a leaderboard-style artifact

## Proposed File Layout

Exact names may shift slightly to match repo conventions, but the structure
should stay close to this:

```text
eval.py
agent/
  eval/
    __init__.py
    registry.py
    runner.py
    compare.py
    artifacts.py
tests/
  unit/
    test_eval_registry.py
    test_eval_compare.py
    test_eval_artifacts.py
```

Rationale:

- `eval.py` stays as the thin executable entrypoint requested by the issue
- `agent/eval/` keeps the logic reusable without coupling it to `agent/core/*`
- tests focus on deterministic behavior instead of real remote inference

## CLI Contract

The script should accept explicit, reproducible inputs instead of hidden
defaults. The minimal interface should include:

- `--task`: task id from the registry
- `--baseline-model`: baseline model id
- `--candidate-model`: model id to compare against the baseline
- `--output-dir`: where artifacts are written

Optional but useful first-PR inputs:

- `--limit`: cap number of evaluation samples for quick runs
- `--split`: override the configured dataset split when supported
- `--cost-file`: path to cost metadata provided by the user
- `--notes`: free-form text stored in the run artifact

The script should print a concise terminal summary and write the canonical
result to disk.

## Task Registry

The first PR should use an explicit task registry instead of a generic
"benchmark anything" abstraction.

Each task definition should declare:

- task id
- dataset or benchmark source
- default split
- primary metric name
- any secondary metrics
- any evaluation parameters needed by the runner

Why a registry:

- keeps supported benchmarks obvious
- gives each task one clear configuration source
- makes tests deterministic
- provides a clean extension point for future tasks

The initial implementation should support exactly one built-in task:
`glue_sst2`.

Why `glue_sst2` first:

- it matches the issue's request for task-specific benchmarks
- it is small enough for a focused first PR
- it uses a well-known benchmark
- the primary metric is straightforward accuracy
- it can be evaluated without inventing a broad benchmark framework first

## Evaluation Boundaries

The evaluator should be structured around small interfaces:

- task resolution
- model evaluation
- result comparison
- artifact writing

This separation matters because the first PR should prove the data model and
workflow before taking on complex provider-specific evaluation plumbing.

The core comparison layer should not depend on how metrics were produced. That
lets unit tests inject stub metric outputs while future integrations can plug
in real benchmark runners.

## Artifact Model

Two artifact types should be written.

### 1. Per-run artifact

This is the source of truth for one evaluation run. It should be machine
readable and detailed enough to reproduce the setup.

Format: one JSON file per successful run.

Recommended fields:

- `run_id`
- `created_at`
- `task`
- `dataset`
- `split`
- `baseline_model`
- `candidate_model`
- `baseline_metrics`
- `candidate_metrics`
- `primary_metric`
- `primary_delta`
- `training_cost`
- `eval_cost`
- `notes`
- `parameters`

The `parameters` object should capture the exact evaluation configuration used,
such as sample limit, split, and any task-specific settings.

### 2. Leaderboard artifact

This is a compact comparison view across runs.

Format: append-only JSONL, one row per successful run.

Recommended fields:

- `run_id`
- `task`
- `baseline_model`
- `candidate_model`
- `primary_metric`
- `baseline_score`
- `candidate_score`
- `delta`
- `training_cost`
- `eval_cost`
- `created_at`
- `notes`

The leaderboard should favor readability and scanning over completeness.
Detailed debugging and reproducibility should stay in the per-run artifact.

## Cost Tracking

Issue `#84` explicitly asks for training cost vs performance gain tracking.
For the first PR, cost data should be accepted as explicit metadata rather than
calculated automatically from platform logs.

This keeps scope under control while still establishing the schema we need for
future automation.

Behavior:

- if a cost file or explicit cost metadata is provided, store it in the run
  artifact and leaderboard row
- if no cost metadata is provided, write `null` values rather than guessing

This makes the data model honest and avoids silently incorrect cost numbers.

## Error Handling

The script should fail early and clearly for invalid setup.

Expected early validation errors:

- unknown task id
- missing model ids
- unsupported output directory state
- malformed cost metadata

Runtime evaluation failures should:

- surface enough context to debug which model and task failed
- avoid producing a misleading "successful" leaderboard entry
- avoid partially written artifacts that look complete

If one model evaluation fails, the command should exit non-zero and skip both
the per-run artifact and the leaderboard update. The first PR should not invent
a failed-run artifact format.

## Testing Strategy

This PR should focus on deterministic unit tests.

Required coverage:

- registry resolution for known and unknown tasks
- comparison logic for baseline vs candidate metrics
- derived delta calculation
- per-run artifact writing
- leaderboard append or rewrite behavior
- invalid input handling

Tests should use stubbed evaluation outputs rather than real model calls.
That keeps the test suite fast, reliable, and suitable for CI.

Real remote evaluation can be added later once the core data flow is proven.

## Why This Slice First

This slice lands the most valuable infrastructure from the issue without
mixing multiple risky changes together.

It gives the project:

- a reproducible evaluation entrypoint
- a standard comparison record
- a leaderboard-ready data format
- a clean extension point for future tasks and UI work

It avoids:

- changing live agent behavior
- shipping partial backend storage design
- taking on UI requirements before the underlying data model is stable

## Follow-up Work After This PR

Natural next steps once this lands:

- add more task definitions
- integrate real benchmark runners where appropriate
- let the agent trigger the evaluation pipeline
- surface leaderboard results in the web UI
- automate cost collection from training/eval jobs

## Acceptance Criteria

This first PR is successful if:

- contributors can run `eval.py` locally with a task, baseline model, and
  candidate model
- the script writes a reproducible run artifact
- the script writes or updates a leaderboard-style artifact
- the comparison includes primary metric deltas
- optional cost metadata is recorded without guessing
- unit tests cover the core workflow logic
- no live agent, backend, or frontend behavior changes are required
