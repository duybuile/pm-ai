"""Tool functions for the Saturn PM assistant."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.db.database import get_connection, initialize_database, seed_database

logger = logging.getLogger(__name__)

_ALLOWED_TASK_STATUSES = {"Not Started", "In Progress", "In Review", "Blocked", "Done"}


def _ensure_database_ready() -> None:
    """Create schema and seed mock data if needed."""
    initialize_database(reset=False)
    seed_database(force=False)


def _serialize_rows(rows: list[Any]) -> str:
    """Convert sqlite rows into JSON text for LLM consumption."""
    return json.dumps([dict(row) for row in rows], ensure_ascii=True)


def get_projects() -> str:
    """Return all projects.

    Returns:
        JSON string list of project objects.

    Notes:
        This read-only tool is intended for LLM planning and retrieval prompts.
    """
    try:
        _ensure_database_ready()
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT id, name, status, owner_id FROM Projects ORDER BY id"
            ).fetchall()
        return _serialize_rows(rows)
    except Exception as exc:
        logger.exception("Failed to fetch projects.")
        return f"Error retrieving projects: {exc}"


def get_tasks(project_id: int | None = None, assignee_id: int | None = None) -> str:
    """Return tasks with optional filtering by project and assignee.

    Args:
        project_id: Optional project identifier to scope tasks.
        assignee_id: Optional team member identifier to scope tasks.

    Returns:
        JSON string list of task objects when tasks are found.
        String error message when no records match the filters.
    """
    try:
        _ensure_database_ready()
        query = (
            "SELECT id, project_id, title, description, status, assignee_id, due_date "
            "FROM Tasks"
        )
        params: list[int] = []
        filters: list[str] = []

        if project_id is not None:
            filters.append("project_id = ?")
            params.append(project_id)

        if assignee_id is not None:
            filters.append("assignee_id = ?")
            params.append(assignee_id)

        if filters:
            query += " WHERE " + " AND ".join(filters)

        query += " ORDER BY id"

        with get_connection() as connection:
            rows = connection.execute(query, params).fetchall()

        if not rows:
            return "No tasks found matching the provided filters."
        return _serialize_rows(rows)
    except Exception as exc:
        logger.exception("Failed to fetch tasks.")
        return f"Error retrieving tasks: {exc}"


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
        _ensure_database_ready()
        normalized_status = status.strip()
        if normalized_status not in _ALLOWED_TASK_STATUSES:
            return (
                "Invalid status. Allowed values: "
                + ", ".join(sorted(_ALLOWED_TASK_STATUSES))
            )

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
        _ensure_database_ready()
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
                if status not in _ALLOWED_TASK_STATUSES:
                    return (
                        f"Task item #{idx} has invalid status '{status}'. "
                        f"Allowed: {', '.join(sorted(_ALLOWED_TASK_STATUSES))}"
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


def search_team_members(query: str) -> str:
    """Search team members by partial name or email.

    Args:
        query: Partial text used to match member names and emails.

    Returns:
        JSON string list containing id, name, and email for matching team members.
        String error message when no team members match the query.
    """
    try:
        _ensure_database_ready()
        search_term = query.strip()
        if not search_term:
            return "Search query cannot be empty."

        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, name, email
                FROM TeamMembers
                WHERE LOWER(name) LIKE LOWER(?) OR LOWER(email) LIKE LOWER(?)
                ORDER BY id
                """,
                (f"%{search_term}%", f"%{search_term}%"),
            ).fetchall()

        if not rows:
            return f"No team members found for query '{search_term}'."
        return _serialize_rows(rows)
    except Exception as exc:
        logger.exception("Failed to search team members.")
        return f"Error searching team members: {exc}"
