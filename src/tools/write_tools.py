"""Write and workflow tool functions for Saturn PM assistant."""

from __future__ import annotations

import json
import logging

from src.tools.base import ALLOWED_TASK_STATUSES, ensure_database_ready
from src.db.database import get_connection

logger = logging.getLogger(__name__)


def update_task_status(task_id: int, status: str) -> str:
    """Update a task status.

    Args:
        task_id: Identifier of the task to update.
        status: New status value. Supported values are Not Started, In Progress,
            In Review, Blocked, and Done.

    Returns:
        JSON string of the updated task summary.
        String error message when the task is not found or status is invalid.
    """
    try:
        ensure_database_ready()
        normalized_status = status.strip()
        if normalized_status not in ALLOWED_TASK_STATUSES:
            return "Invalid status. Allowed values: " + ", ".join(sorted(ALLOWED_TASK_STATUSES))

        with get_connection() as connection:
            current = connection.execute(
                "SELECT id, title, status FROM Tasks WHERE id = ?",
                (task_id,),
            ).fetchone()

            if current is None:
                return f"Task with id={task_id} was not found."

            connection.execute(
                "UPDATE Tasks SET status = ? WHERE id = ?",
                (normalized_status, task_id),
            )

        payload = {
            "task_id": task_id,
            "title": current["title"],
            "old_status": current["status"],
            "new_status": normalized_status,
        }
        return json.dumps(payload, ensure_ascii=True)
    except Exception as exc:
        logger.exception("Failed to update task status.")
        return f"Error updating task status: {exc}"


def create_project_with_tasks(name: str, owner_id: int, tasks: list[dict]) -> str:
    """Create a new project and associated tasks in one transaction.

    Args:
        name: Name for the new project.
        owner_id: Team member id who owns the project.
        tasks: List of dictionaries, where each task can include title, description,
            status, assignee_id, and due_date.

    Returns:
        JSON string containing the created project id and task ids.
        String error message when validation fails or owner does not exist.
    """
    try:
        ensure_database_ready()
        project_name = name.strip()
        if not project_name:
            return "Project name cannot be empty."

        if tasks is None:
            tasks = []

        with get_connection() as connection:
            owner = connection.execute(
                "SELECT id FROM TeamMembers WHERE id = ?",
                (owner_id,),
            ).fetchone()
            if owner is None:
                return f"Owner with id={owner_id} was not found."

            project_cursor = connection.execute(
                "INSERT INTO Projects (name, status, owner_id) VALUES (?, ?, ?)",
                (project_name, "Planning", owner_id),
            )
            project_id = int(project_cursor.lastrowid)

            created_task_ids: list[int] = []
            for idx, task in enumerate(tasks, start=1):
                if not isinstance(task, dict):
                    return f"Task item #{idx} must be a dictionary."

                title = str(task.get("title", "")).strip()
                if not title:
                    return f"Task item #{idx} is missing a non-empty title."

                status = str(task.get("status", "Not Started")).strip() or "Not Started"
                if status not in ALLOWED_TASK_STATUSES:
                    return (
                        f"Task item #{idx} has invalid status '{status}'. "
                        f"Allowed: {', '.join(sorted(ALLOWED_TASK_STATUSES))}"
                    )

                assignee_id = task.get("assignee_id")
                if assignee_id is not None:
                    assignee = connection.execute(
                        "SELECT id FROM TeamMembers WHERE id = ?",
                        (assignee_id,),
                    ).fetchone()
                    if assignee is None:
                        return f"Task item #{idx} references unknown assignee id={assignee_id}."

                task_cursor = connection.execute(
                    """
                    INSERT INTO Tasks (project_id, title, description, status, assignee_id, due_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        title,
                        str(task.get("description", "")).strip(),
                        status,
                        assignee_id,
                        task.get("due_date"),
                    ),
                )
                created_task_ids.append(int(task_cursor.lastrowid))

        response = {
            "project_id": project_id,
            "project_name": project_name,
            "created_task_ids": created_task_ids,
        }
        return json.dumps(response, ensure_ascii=True)
    except Exception as exc:
        logger.exception("Failed to create project with tasks.")
        return f"Error creating project with tasks: {exc}"
