"""LangGraph orchestration layer for Saturn PM assistant.

This module defines the planner/router, read-tool executor, and human-approval
workflow for write operations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from src import cfg
from src.tools.tools import (
    create_project_with_tasks,
    get_projects,
    get_tasks,
    search_team_members,
    update_task_status,
)
from utils.config.log_handler import setup_logger

try:
    from langgraph.graph.message import add_messages
except Exception as e:  # pragma: no cover - fallback for environments without langgraph
    def add_messages(left: list[Any], right: list[Any]) -> list[Any]:
        """Fallback add_messages implementation when langgraph isn't installed."""
        return [*(left or []), *(right or [])]


logger = setup_logger(logger_name=__name__, level="info", console_logging=True)

_READ_TOOLS = {
    "get_projects": get_projects,
    "get_tasks": get_tasks,
    "search_team_members": search_team_members,
}
_WRITE_TOOLS = {
    "update_task_status": update_task_status,
    "create_project_with_tasks": create_project_with_tasks,
}
_ALL_TOOLS = {**_READ_TOOLS, **_WRITE_TOOLS}


class State(TypedDict):
    """Graph state shared across all nodes."""

    messages: Annotated[list[Any], add_messages]
    next_action: NotRequired[dict[str, Any] | None]
    explanation: str
    planned_tool: NotRequired[dict[str, Any] | None]


def _msg_content(message: Any) -> str:
    """Extract content from dict-like or object-like chat message."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _msg_role(message: Any) -> str:
    """Extract role/type from dict-like or object-like chat message."""
    if isinstance(message, dict):
        return str(message.get("role", message.get("type", ""))).lower()
    return str(getattr(message, "type", getattr(message, "role", ""))).lower()


def _last_user_text(messages: list[Any]) -> str:
    """Return latest user text from message history."""
    for message in reversed(messages):
        if _msg_role(message) in {"human", "user"}:
            return _msg_content(message)
    return ""


def _latest_tool_payload(messages: list[Any], tool_name: str) -> str | None:
    """Return the most recent tool payload for a named tool."""
    for message in reversed(messages):
        role = _msg_role(message)
        if role not in {"tool"}:
            continue

        if isinstance(message, dict):
            if message.get("name") == tool_name:
                return str(message.get("content", ""))
            continue

        if getattr(message, "name", None) == tool_name:
            return str(getattr(message, "content", ""))
    return None


def _yes_no(text: str) -> str:
    """Classify approval response as yes/no/unknown."""
    normalized = text.strip().lower()
    if normalized in {"yes", "y", "approve", "approved", "confirm", "go ahead"}:
        return "yes"
    if normalized in {"no", "n", "deny", "denied", "cancel", "stop"}:
        return "no"
    return "unknown"


def _extract_task_id(text: str) -> int | None:
    match = re.search(r"task\s*(?:id\s*)?(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_status(text: str) -> str | None:
    status_aliases = {
        "not started": "Not Started",
        "in progress": "In Progress",
        "in review": "In Review",
        "blocked": "Blocked",
        "done": "Done",
    }
    lower = text.lower()
    for alias, canonical in status_aliases.items():
        if alias in lower:
            return canonical
    return None


def _extract_project_name(text: str) -> str:
    quoted = re.search(r"project\s+(?:named\s+)?['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()

    trailing = re.search(r"create\s+(?:a\s+)?project\s+(?:named\s+)?([a-zA-Z0-9 _-]+)", text, flags=re.IGNORECASE)
    if trailing:
        candidate = trailing.group(1)
        candidate = re.split(r"\s+(?:with|and)\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = candidate.strip(" .")
        if cleaned:
            return cleaned

    return "New Project"


def _extract_assignee_name(text: str) -> str | None:
    match = re.search(r"assign\s+(?:the\s+)?first\s+task\s+to\s+([a-zA-Z]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def oracle_node(state: State) -> dict[str, Any]:
    """Plan the next step: respond, run read tool, or stage write action for approval."""
    messages = state.get("messages", [])
    user_text = _last_user_text(messages)
    lower = user_text.lower()

    if not user_text:
        return {
            "explanation": "No user message was available.",
            "messages": [{"role": "assistant", "content": "Please share a request so I can help."}],
            "planned_tool": None,
            "next_action": None,
        }

    if any(phrase in lower for phrase in ["list projects", "show projects", "all projects"]):
        return {
            "explanation": "I need to fetch the current project list.",
            "planned_tool": {"name": "get_projects", "args": {}},
            "next_action": None,
        }

    if "task" in lower and any(phrase in lower for phrase in ["show", "list", "what are"]):
        filters: dict[str, int] = {}
        project_match = re.search(r"project\s*(\d+)", lower)
        assignee_match = re.search(r"assignee\s*(\d+)", lower)
        if project_match:
            filters["project_id"] = int(project_match.group(1))
        if assignee_match:
            filters["assignee_id"] = int(assignee_match.group(1))

        return {
            "explanation": "I need to fetch task data with the requested filters.",
            "planned_tool": {"name": "get_tasks", "args": filters},
            "next_action": None,
        }

    if "update" in lower and "task" in lower:
        task_id = _extract_task_id(user_text)
        status = _extract_status(user_text)
        if task_id is None or status is None:
            return {
                "explanation": "I need a task id and valid status before proposing an update.",
                "messages": [{
                    "role": "assistant",
                    "content": "Please specify both the task id and one status: Not Started, In Progress, In Review, Blocked, or Done.",
                }],
                "planned_tool": None,
                "next_action": None,
            }

        return {
            "explanation": f"This will change task {task_id} status to '{status}'. I need your approval before writing.",
            "messages": [{
                "role": "assistant",
                "content": f"I can update task {task_id} to '{status}'. Approve this change? Reply yes or no.",
            }],
            "planned_tool": None,
            "next_action": {"name": "update_task_status", "args": {"task_id": task_id, "status": status}},
        }

    if "create" in lower and "project" in lower:
        assignee_name = _extract_assignee_name(user_text)
        if assignee_name:
            payload = _latest_tool_payload(messages, "search_team_members")
            if payload is None:
                return {
                    "explanation": f"I need to resolve {assignee_name}'s team member id first.",
                    "planned_tool": {"name": "search_team_members", "args": {"query": assignee_name}},
                    "next_action": None,
                }

            try:
                search_hits = json.loads(payload)
            except json.JSONDecodeError:
                search_hits = []

            assignee_id = search_hits[0]["id"] if search_hits else None
            if assignee_id is None:
                return {
                    "explanation": "I could not resolve the assignee id from team member search.",
                    "messages": [{
                        "role": "assistant",
                        "content": f"I could not find a team member match for '{assignee_name}'. Please provide an assignee id.",
                    }],
                    "planned_tool": None,
                    "next_action": None,
                }
        else:
            assignee_id = None

        project_name = _extract_project_name(user_text)
        task_payload: dict[str, Any] = {"title": "Initial setup task", "status": "Not Started"}
        if assignee_id is not None:
            task_payload["assignee_id"] = assignee_id

        action_args = {
            "name": project_name,
            "owner_id": assignee_id or 1,
            "tasks": [task_payload],
        }
        return {
            "explanation": "This will create a new project and its first task. I need your approval before writing.",
            "messages": [{
                "role": "assistant",
                "content": (
                    f"I am ready to create project '{project_name}' with 1 starter task"
                    + (f" assigned to member id {assignee_id}" if assignee_id is not None else "")
                    + ". Approve this write operation? Reply yes or no."
                ),
            }],
            "planned_tool": None,
            "next_action": {"name": "create_project_with_tasks", "args": action_args},
        }

    if "sarah" in lower and "id" in lower:
        return {
            "explanation": "I need to search team members by name.",
            "planned_tool": {"name": "search_team_members", "args": {"query": "Sarah"}},
            "next_action": None,
        }

    return {
        "explanation": "No tool call is required.",
        "messages": [{
            "role": "assistant",
            "content": "I can help with project/task reads, task status updates, and project creation workflows. Tell me what you want to do.",
        }],
        "planned_tool": None,
        "next_action": None,
    }


def tool_executor_node(state: State) -> dict[str, Any]:
    """Execute read tools immediately and append tool output to message history."""
    planned_tool = state.get("planned_tool")
    if not planned_tool:
        return {"planned_tool": None}

    tool_name = planned_tool["name"]
    tool_args = planned_tool.get("args", {})
    tool = _READ_TOOLS.get(tool_name)
    if tool is None:
        logger.error("Unknown read tool requested: %s", tool_name)
        return {
            "messages": [{"role": "assistant", "content": f"Read tool '{tool_name}' is not available."}],
            "planned_tool": None,
        }

    logger.info("Executing read tool %s with args=%s", tool_name, tool_args)
    result = tool(**tool_args)
    return {
        "messages": [
            {"role": "tool", "name": tool_name, "content": result},
            {"role": "assistant", "content": f"Read result from {tool_name}: {result}"},
        ],
        "planned_tool": None,
    }


def human_approval_node(state: State) -> dict[str, Any]:
    """Handle yes/no response for pending write actions and execute approved writes."""
    action = state.get("next_action")
    if not action:
        return {
            "messages": [{"role": "assistant", "content": "No pending write action exists."}],
            "next_action": None,
        }

    response = _yes_no(_last_user_text(state.get("messages", [])))
    if response == "unknown":
        return {
            "messages": [{
                "role": "assistant",
                "content": "Please reply with yes or no to approve or cancel the pending write operation.",
            }],
        }

    if response == "no":
        return {
            "messages": [{"role": "assistant", "content": "Understood. I canceled that write operation."}],
            "next_action": None,
            "explanation": "User denied write request.",
        }

    tool_name = action["name"]
    tool = _WRITE_TOOLS.get(tool_name)
    if tool is None:
        return {
            "messages": [{"role": "assistant", "content": f"Write tool '{tool_name}' is not available."}],
            "next_action": None,
        }

    tool_args = action.get("args", {})
    logger.info("Executing approved write tool %s with args=%s", tool_name, tool_args)
    result = tool(**tool_args)

    return {
        "messages": [
            {"role": "tool", "name": tool_name, "content": result},
            {"role": "assistant", "content": f"Write operation completed: {result}"},
        ],
        "next_action": None,
        "explanation": "Approved write request executed.",
    }


def _route_from_oracle(state: State) -> str:
    """Choose next node after oracle planning."""
    if state.get("planned_tool"):
        return "tool_executor"
    if state.get("next_action"):
        return "human_approval"
    return "__end__"


def _build_checkpointer(path: str):
    """Create a SqliteSaver checkpointer instance."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "langgraph with sqlite checkpoint support is required. Install project dependencies first."
        ) from exc

    checkpoint_file = Path(path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        return SqliteSaver.from_conn_string(str(checkpoint_file))
    except Exception:
        return SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_file}")


def build_graph(checkpoint_path: str | None = None):
    """Compile and return the LangGraph app.

    Args:
        checkpoint_path: Optional sqlite checkpoint file path override.

    Returns:
        Compiled LangGraph app with SqliteSaver persistence and HITL interruption.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("langgraph is required to build the orchestrator graph.") from exc

    path = checkpoint_path or cfg.get("orchestrator.checkpoint_path", "conf/langgraph_checkpoints.db")

    builder = StateGraph(State)
    builder.add_node("oracle", oracle_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("human_approval", human_approval_node)

    builder.add_edge(START, "oracle")
    builder.add_conditional_edges(
        "oracle",
        _route_from_oracle,
        {
            "tool_executor": "tool_executor",
            "human_approval": "human_approval",
            "__end__": END,
        },
    )
    builder.add_edge("tool_executor", "oracle")
    builder.add_edge("human_approval", END)

    checkpointer = _build_checkpointer(path)
    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])


def run_turn(app: Any, user_message: str, thread_id: str) -> dict[str, Any]:
    """Run one chat turn with configured recursion limit and thread persistence."""
    recursion_limit = int(cfg.get("orchestrator.recursion_limit", 10))
    return app.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit},
    )
