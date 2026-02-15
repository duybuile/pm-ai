"""Tool registry and metadata helpers for Saturn PM assistant."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from src.tools.read_tools import get_projects, get_tasks, search_team_members
from src.tools.write_tools import create_project_with_tasks, update_task_status

ToolFn = Callable[..., str]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "get_projects": get_projects,
    "get_tasks": get_tasks,
    "search_team_members": search_team_members,
    "update_task_status": update_task_status,
    "create_project_with_tasks": create_project_with_tasks,
}

READ_TOOL_NAMES = {"get_projects", "get_tasks", "search_team_members"}
WRITE_TOOL_NAMES = {"update_task_status", "create_project_with_tasks"}


def get_tool_registry() -> dict[str, ToolFn]:
    """Return the canonical registry of callable tool functions."""
    return dict(TOOL_REGISTRY)


def get_tool_definitions(as_text: bool = True) -> str | list[dict[str, Any]]:
    """Return tool metadata used by the LLM oracle system prompt.

    Args:
        as_text: When True, return a single human-readable manual string.
            When False, return structured metadata records.
    """
    definitions: list[dict[str, Any]] = []
    for name, fn in TOOL_REGISTRY.items():
        signature = str(inspect.signature(fn))
        doc = inspect.getdoc(fn) or ""
        summary = doc.splitlines()[0] if doc else ""
        mode = "write" if name in WRITE_TOOL_NAMES else "read"
        definitions.append(
            {
                "name": name,
                "signature": f"{name}{signature}",
                "mode": mode,
                "description": summary,
            }
        )

    if not as_text:
        return definitions

    lines = ["Available tools:"]
    for item in definitions:
        lines.append(
            f"- {item['signature']} [{item['mode']}] - {item['description']}"
        )
    return "\n".join(lines)
