from __future__ import annotations

import importlib
import json
import os
from pathlib import Path


def _load_modules(tmp_path: Path):
    os.environ["PM_AI_DB_PATH"] = str(tmp_path / "test_pm_ai.db")

    import src.db.database as database
    import src.tools as tools

    importlib.reload(database)
    importlib.reload(tools)

    database.initialize_database(reset=True)
    database.seed_database(force=True)

    return database, tools


def test_get_projects_returns_seeded_projects(tmp_path: Path):
    _, tools = _load_modules(tmp_path)

    raw = tools.get_projects()
    payload = json.loads(raw)

    assert len(payload) == 5
    assert payload[0]["name"] == "Mobile App Redesign"


def test_get_tasks_filters_by_project_and_assignee(tmp_path: Path):
    _, tools = _load_modules(tmp_path)

    raw = tools.get_tasks(project_id=1, assignee_id=1)
    payload = json.loads(raw)

    assert len(payload) == 1
    assert payload[0]["title"] == "Finalize navigation prototype"


def test_update_task_status_returns_not_found_error(tmp_path: Path):
    _, tools = _load_modules(tmp_path)

    message = tools.update_task_status(task_id=9999, status="Done")

    assert message == "Task with id=9999 was not found."


def test_create_project_with_tasks_creates_workflow_records(tmp_path: Path):
    database, tools = _load_modules(tmp_path)

    raw = tools.create_project_with_tasks(
        name="AI Copilot Pilot",
        owner_id=1,
        tasks=[
            {
                "title": "Define success metrics",
                "description": "Establish qualitative and quantitative outcomes.",
                "status": "Not Started",
                "assignee_id": 2,
                "due_date": "2026-03-10",
            },
            {
                "title": "Draft rollout plan",
                "description": "Prepare phased launch plan.",
            },
        ],
    )
    payload = json.loads(raw)

    assert payload["project_name"] == "AI Copilot Pilot"
    assert len(payload["created_task_ids"]) == 2

    with database.get_connection() as connection:
        project = connection.execute(
            "SELECT name FROM Projects WHERE id = ?",
            (payload["project_id"],),
        ).fetchone()
    assert project["name"] == "AI Copilot Pilot"


def test_search_team_members_returns_matching_member(tmp_path: Path):
    _, tools = _load_modules(tmp_path)

    raw = tools.search_team_members("Sarah")
    payload = json.loads(raw)

    assert len(payload) == 1
    assert payload[0]["email"] == "sarah.kim@saturnpm.com"
