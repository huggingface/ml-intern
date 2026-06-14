"""Task registry for local evaluation workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    dataset_name: str
    dataset_config: str | None
    default_split: str
    text_column: str
    label_column: str
    primary_metric: str


_TASKS: dict[str, EvalTask] = {
    "glue_sst2": EvalTask(
        task_id="glue_sst2",
        dataset_name="glue",
        dataset_config="sst2",
        default_split="validation",
        text_column="sentence",
        label_column="label",
        primary_metric="accuracy",
    )
}


def get_task(task_id: str) -> EvalTask:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation task: {task_id}") from exc
