import pytest

from agent.eval import EvalTask, get_task


def test_get_task_returns_glue_sst2_definition():
    task = get_task("glue_sst2")

    assert isinstance(task, EvalTask)
    assert task.task_id == "glue_sst2"
    assert task.dataset_name == "glue"
    assert task.dataset_config == "sst2"
    assert task.default_split == "validation"
    assert task.text_column == "sentence"
    assert task.label_column == "label"
    assert task.primary_metric == "accuracy"


def test_get_task_raises_for_unknown_task():
    with pytest.raises(ValueError, match="^Unknown evaluation task: unknown_task$"):
        get_task("unknown_task")
