"""Read-only tool functions for Saturn PM assistant."""

from __future__ import annotations

import logging

from src.tools.base import ensure_database_ready, serialize_rows
from src.db.database import get_connection

logger = logging.getLogger(__name__)


def get_projects() -> str:
    """Return all projects.

    Returns:
        JSON string list of project objects.

    Notes:
        This read-only tool is intended for LLM planning and retrieval prompts.
    """
    try:
        ensure_database_ready()
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT id, name, status, owner_id FROM Projects ORDER BY id"
            ).fetchall()
        return serialize_rows(rows)
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
        ensure_database_ready()
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
        return serialize_rows(rows)
    except Exception as exc:
        logger.exception("Failed to fetch tasks.")
        return f"Error retrieving tasks: {exc}"


def search_team_members(query: str) -> str:
    """Search team members by partial name or email.

    Args:
        query: Partial text used to match member names and emails.

    Returns:
        JSON string list containing id, name, and email for matching team members.
        String error message when no team members match the query.
    """
    try:
        ensure_database_ready()
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
        return serialize_rows(rows)
    except Exception as exc:
        logger.exception("Failed to search team members.")
        return f"Error searching team members: {exc}"
