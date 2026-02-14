from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

langgraph = pytest.importorskip("langgraph")


def _bootstrap(tmp_path: Path):
    os.environ["PM_AI_DB_PATH"] = str(tmp_path / "orchestrator_pm_ai.db")

    from src.db.database import initialize_database, seed_database
    from src.orchestrator.graph import build_graph, run_turn

    checkpoint_path = str(tmp_path / "checkpoints.db")
    initialize_database(reset=True)
    seed_database(force=True)

    app = build_graph(checkpoint_path=checkpoint_path)
    return app, run_turn


def test_read_request_runs_direct_tool(tmp_path: Path):
    app, run_turn = _bootstrap(tmp_path)

    state = run_turn(app, "List projects", thread_id="thread-read")

    assistant_messages = [m for m in state.get("messages", []) if m.get("role") == "assistant"]
    assert any("Read result from get_projects" in m.get("content", "") for m in assistant_messages)


def test_write_request_interrupts_for_approval(tmp_path: Path):
    app, run_turn = _bootstrap(tmp_path)

    state = run_turn(app, "Update task 1 to done", thread_id="thread-write")

    assert state.get("next_action", {}).get("name") == "update_task_status"
    assert "approval" in state.get("explanation", "").lower()


def test_approved_write_executes_pending_action(tmp_path: Path):
    app, run_turn = _bootstrap(tmp_path)

    run_turn(app, "Update task 1 to done", thread_id="thread-approve")
    final_state = run_turn(app, "yes", thread_id="thread-approve")

    assert final_state.get("next_action") is None

    tool_messages = [m for m in final_state.get("messages", []) if m.get("role") == "tool"]
    payload = json.loads(tool_messages[-1]["content"])
    assert payload["task_id"] == 1
    assert payload["new_status"] == "Done"


def test_multistep_request_searches_member_then_proposes_workflow(tmp_path: Path):
    app, run_turn = _bootstrap(tmp_path)

    first = run_turn(
        app,
        "Create a project and assign the first task to Sarah.",
        thread_id="thread-multi",
    )
    assert first.get("next_action") is None

    second = run_turn(
        app,
        "Create a project and assign the first task to Sarah.",
        thread_id="thread-multi",
    )

    action = second.get("next_action")
    assert action is not None
    assert action["name"] == "create_project_with_tasks"
    assert action["args"]["tasks"][0]["assignee_id"] == 1
