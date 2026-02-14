from __future__ import annotations

import importlib
import os

import pytest

pytest.importorskip("langgraph")


def test_dataset_has_required_fields():
    from src.evals.dataset import GOLDEN_SAMPLES

    assert GOLDEN_SAMPLES, "Golden dataset must not be empty."
    for case in GOLDEN_SAMPLES:
        assert "input" in case
        assert "expected_intent" in case
        assert "expected_tool" in case
        assert "expected_entities" in case


def test_evaluation_runner_outputs_report(tmp_path):
    os.environ["PM_AI_DB_PATH"] = str(tmp_path / "eval_pm_ai.db")

    from src.db.database import initialize_database, seed_database
    import src.evals.runner as runner

    importlib.reload(runner)

    initialize_database(reset=True)
    seed_database(force=True)

    runner.GOLDEN_SAMPLES = [
        {
            "input": "List projects",
            "expected_intent": "read",
            "expected_tool": "get_projects",
            "expected_entities": {},
        },
        {
            "input": "Update task 5 to done",
            "expected_intent": "write",
            "expected_tool": "update_task_status",
            "expected_entities": {"task_id": 5, "status": "Done"},
        },
    ]

    report = runner.evaluate_golden_samples()

    assert report["summary"]["total_cases"] == 2
    assert isinstance(report["summary"]["reliability_score"], float)
    assert len(report["rows"]) == 2
    assert all(isinstance(row["passed"], bool) for row in report["rows"])
    assert "Case" in report["table"]
