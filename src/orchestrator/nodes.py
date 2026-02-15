"""LangGraph node implementations for Saturn PM orchestrator."""

from __future__ import annotations

import json
import logging

from src.orchestrator.message_utils import last_user_text, latest_tool_payload, yes_no
from src.orchestrator.planner import (
    extract_assignee_name,
    extract_project_name,
    extract_status,
    extract_task_id,
    fallback_oracle_response,
    plan_with_llm,
)
from src.orchestrator.state import State
from src.tools.tools_registry import (
    READ_TOOL_NAMES,
    TOOL_REGISTRY,
    WRITE_TOOL_NAMES,
)

logger = logging.getLogger(__name__)
# Backward-compatible mutable alias used by graph facade/tests.
_TOOL_REGISTRY = TOOL_REGISTRY
_STATUS_MAP = {
    "not started": "Not Started",
    "in progress": "In Progress",
    "in review": "In Review",
    "blocked": "Blocked",
    "done": "Done",
}


def oracle_node(state: State) -> State:
    """LLM-backed planner that chooses a tool or direct response."""
    messages = state.get("messages", [])
    user_text = last_user_text(messages)
    lower = user_text.lower()

    if state.get("next_action"):
        return {
            "explanation": "Pending write action requires approval handling.",
            "planned_tool": None,
        }

    if not user_text:
        return {
            "explanation": "No user message was available.",
            "messages": [{"role": "assistant", "content": "Please share a request so I can help."}],
            "planned_tool": None,
            "next_action": None,
            "last_tool_name": None,
            "last_tool_mode": None,
            "last_tool_result": None,
        }

    # Deterministic write parsing for stable safety-gate behavior.
    if "update" in lower and "task" in lower:
        task_id = extract_task_id(user_text)
        status = extract_status(user_text)
        if task_id is not None and status is not None:
            explanation = f"I will update task {task_id} to '{status}'. This requires your approval."
            return {
                "explanation": explanation,
                "messages": [{"role": "assistant", "content": f"{explanation} Approve this write action? Reply yes or no."}],
                "planned_tool": None,
                "next_action": {"name": "update_task_status", "args": {"task_id": task_id, "status": status}},
                "last_tool_name": None,
                "last_tool_mode": None,
                "last_tool_result": None,
            }

    # Deterministic multi-step handling: create project + assign first task to a member.
    if "create" in lower and "project" in lower and "assign" in lower and "first task" in lower:
        assignee_name = extract_assignee_name(user_text)
        if assignee_name:
            payload = latest_tool_payload(messages, "search_team_members")
            if payload is None:
                return {
                    "explanation": f"I need {assignee_name}'s team member id before creating the project.",
                    "planned_tool": {"name": "search_team_members", "args": {"query": assignee_name}},
                    "next_action": None,
                    "last_tool_name": None,
                    "last_tool_mode": None,
                    "last_tool_result": None,
                }

            # Prevent same-turn immediate write proposal after a just-executed read; finish this turn first.
            if state.get("last_tool_mode") == "read":
                return {
                    "explanation": "I found the assignee id from team data. I can now prepare the project creation step.",
                    "messages": [{"role": "assistant", "content": "I found the assignee id. Repeat the request to proceed with the write action."}],
                    "planned_tool": None,
                    "next_action": None,
                    "last_tool_name": None,
                    "last_tool_mode": None,
                    "last_tool_result": None,
                }

            try:
                search_hits = json.loads(payload)
            except json.JSONDecodeError:
                search_hits = []

            assignee_id = search_hits[0]["id"] if search_hits else None
            if assignee_id is not None:
                project_name = extract_project_name(user_text)
                explanation = f"I will create project '{project_name}' and assign the first task to member id {assignee_id}. This requires your approval."
                return {
                    "explanation": explanation,
                    "messages": [{"role": "assistant", "content": f"{explanation} Approve this write action? Reply yes or no."}],
                    "planned_tool": None,
                    "next_action": {
                        "name": "create_project_with_tasks",
                        "args": {
                            "name": project_name,
                            "owner_id": assignee_id,
                            "tasks": [{"title": "Initial setup task", "status": "Not Started", "assignee_id": assignee_id}],
                        },
                    },
                    "last_tool_name": None,
                    "last_tool_mode": None,
                    "last_tool_result": None,
                }

    try:
        decision = plan_with_llm(user_text, messages)
        logger.info("Oracle LLM decision: %s", decision)
    except Exception as exc:
        logger.warning("Oracle LLM unavailable, using fallback planner: %s", exc, exc_info=True)
        decision = fallback_oracle_response(user_text, messages)

    tool_name = decision.get("tool")
    args = decision.get("args", {})
    explanation = str(decision.get("explanation", "")).strip() or "No explanation provided."

    if tool_name == "update_task_status" and isinstance(args, dict):
        status_raw = str(args.get("status", "")).strip()
        if status_raw:
            normalized = _STATUS_MAP.get(status_raw.lower())
            if normalized:
                args["status"] = normalized

    if not tool_name or str(tool_name).lower() == "none":
        if state.get("last_tool_mode") == "read" and state.get("last_tool_name") and state.get("last_tool_result"):
            explanation = f"Read result from {state['last_tool_name']}: {state['last_tool_result']}"
        return {
            "explanation": explanation,
            "messages": [{"role": "assistant", "content": explanation}],
            "planned_tool": None,
            "next_action": None,
            "last_tool_name": None,
            "last_tool_mode": None,
            "last_tool_result": None,
        }

    if tool_name in WRITE_TOOL_NAMES:
        if "approval" not in explanation.lower():
            explanation = f"{explanation} This requires your approval."
        return {
            "explanation": explanation,
            "messages": [{"role": "assistant", "content": f"{explanation} Approve this write action? Reply yes or no."}],
            "planned_tool": None,
            "next_action": {"name": tool_name, "args": args},
            "last_tool_name": None,
            "last_tool_mode": None,
            "last_tool_result": None,
        }

    if tool_name in READ_TOOL_NAMES:
        return {
            "explanation": explanation,
            "planned_tool": {"name": tool_name, "args": args},
            "next_action": None,
            "last_tool_name": None,
            "last_tool_mode": None,
            "last_tool_result": None,
        }

    return {
        "explanation": f"Tool '{tool_name}' is not available.",
        "messages": [{"role": "assistant", "content": f"I could not find tool '{tool_name}'. Please rephrase your request."}],
        "planned_tool": None,
        "next_action": None,
        "last_tool_name": None,
        "last_tool_mode": None,
        "last_tool_result": None,
    }


def execute_tool_node(state: State) -> State:
    """Execute selected tool dynamically from registry using planned or pending action."""
    action = state.get("planned_tool") or state.get("next_action")
    if not action:
        return {"planned_tool": None, "last_tool_name": None, "last_tool_mode": None, "last_tool_result": None}

    tool_name = action.get("name")
    tool_args = action.get("args", {})
    tool = _TOOL_REGISTRY.get(str(tool_name))
    if tool is None:
        logger.error("Unknown tool requested: %s", tool_name)
        return {
            "messages": [{"role": "assistant", "content": f"Tool '{tool_name}' is not available."}],
            "planned_tool": None,
            "last_tool_name": None,
            "last_tool_mode": None,
            "last_tool_result": None,
        }

    logger.info("Executing tool %s with args=%s", tool_name, tool_args)
    result = tool(**tool_args)
    mode = "read" if tool_name in READ_TOOL_NAMES else "write"
    tool_call_id = f"{tool_name}_{mode}_call"
    tool_message = {"role": "tool", "name": tool_name, "content": result, "tool_call_id": tool_call_id}
    if mode == "read":
        return {
            "messages": [tool_message],
            "planned_tool": None,
            "last_tool_name": str(tool_name),
            "last_tool_mode": mode,
            "last_tool_result": result,
        }

    return {
        "messages": [
            tool_message,
            {"role": "assistant", "content": f"Write operation completed from {tool_name}: {result}"},
        ],
        "planned_tool": None,
        "last_tool_name": str(tool_name),
        "last_tool_mode": mode,
        "last_tool_result": result,
    }


def human_approval_node(state: State) -> State:
    """Handle yes/no response for pending write actions and stage approved writes."""
    action = state.get("next_action")
    if not action:
        return {
            "messages": [{"role": "assistant", "content": "No pending write action exists."}],
            "next_action": None,
        }

    response = yes_no(last_user_text(state.get("messages", [])))
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
            "planned_tool": None,
        }

    tool_name = action["name"]
    if tool_name not in WRITE_TOOL_NAMES:
        return {
            "messages": [{"role": "assistant", "content": f"Write tool '{tool_name}' is not available."}],
            "next_action": None,
            "planned_tool": None,
        }

    return {
        "messages": [{"role": "assistant", "content": "Approval received. Executing write action now."}],
        "planned_tool": action,
        "next_action": None,
        "explanation": "Approved write request executed.",
    }
