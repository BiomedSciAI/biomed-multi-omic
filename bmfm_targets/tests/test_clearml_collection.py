from datetime import datetime
from types import SimpleNamespace

from bmfm_targets.evaluation import clearml_utils


def fake_task(task_id, name, day, f1):
    return SimpleNamespace(
        id=task_id,
        name=name,
        data=SimpleNamespace(created=datetime(2026, 1, day)),
        get_last_scalar_metrics=lambda: {"Summary": {"f1": {"last": f1}}},
    )


def test_newest_task_wins_whatever_the_order(monkeypatch):
    tasks = [
        fake_task("old", "dc_ft", 1, 0.1),
        fake_task("new", "dc_ft", 3, 0.9),
        fake_task("mid", "dc_ft", 2, 0.5),
        fake_task("other", "mca_ft", 1, 0.4),
    ]
    for ordering in (tasks, tasks[::-1]):
        monkeypatch.setattr(
            clearml_utils.Task, "get_tasks", lambda ordering=ordering, **_: ordering
        )
        table = clearml_utils.collect_project_metrics("project")
        assert table["task_id"].tolist() == ["new", "other"]
        assert table["value"].tolist() == [0.9, 0.4]


def test_missing_section_gives_empty_table(monkeypatch):
    monkeypatch.setattr(
        clearml_utils.Task, "get_tasks", lambda **_: [fake_task("a", "x", 1, 0.5)]
    )
    table = clearml_utils.collect_project_metrics("project", section="absent")
    assert table.empty
    assert "task_id" in table
