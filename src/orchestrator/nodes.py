"""LangGraph node implementations for Saturn PM orchestrator."""

from __future__ import annotations

import logging

from src.orchestrator.message_utils import last_user_text, yes_no
from src.orchestrator.planner import fallback_oracle_response, plan_with_llm
from src.orchestrator.state import State
from src.tools.tools_registry import (
    READ_TOOL_NAMES,
    TOOL_REGISTRY,
    WRITE_TOOL_NAMES,
)

logger = logging.getLogger(__name__)
# Backward-compatible mutable alias used by graph facade/tests.
_TOOL_REGISTRY = TOOL_REGISTRY


def oracle_node(state: State) -> State:
    """LLM-backed planner that chooses a tool or direct response."""
    messages = state.get("messages", [])
    user_text = last_user_text(messages)

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
        }

    try:
        decision = plan_with_llm(user_text, messages)
        logger.info("Oracle LLM decision: %s", decision)
    except Exception as exc:
        logger.warning("Oracle LLM unavailable, using fallback planner: %s", exc)
        decision = fallback_oracle_response(user_text, messages)

    tool_name = decision.get("tool")
    args = decision.get("args", {})
    explanation = str(decision.get("explanation", "")).strip() or "No explanation provided."

    if not tool_name or str(tool_name).lower() == "none":
        return {
            "explanation": explanation,
            "messages": [{"role": "assistant", "content": explanation}],
            "planned_tool": None,
            "next_action": None,
        }

    if tool_name in WRITE_TOOL_NAMES:
        return {
            "explanation": explanation,
            "messages": [{"role": "assistant", "content": f"{explanation} Approve this write action? Reply yes or no."}],
            "planned_tool": None,
            "next_action": {"name": tool_name, "args": args},
        }

    if tool_name in READ_TOOL_NAMES:
        return {
            "explanation": explanation,
            "planned_tool": {"name": tool_name, "args": args},
            "next_action": None,
        }

    return {
        "explanation": f"Tool '{tool_name}' is not available.",
        "messages": [{"role": "assistant", "content": f"I could not find tool '{tool_name}'. Please rephrase your request."}],
        "planned_tool": None,
        "next_action": None,
    }


def execute_tool_node(state: State) -> State:
    """Execute selected tool dynamically from registry using planned or pending action."""
    action = state.get("planned_tool") or state.get("next_action")
    if not action:
        return {"planned_tool": None}

    tool_name = action.get("name")
    tool_args = action.get("args", {})
    tool = _TOOL_REGISTRY.get(str(tool_name))
    if tool is None:
        logger.error("Unknown tool requested: %s", tool_name)
        return {
            "messages": [{"role": "assistant", "content": f"Tool '{tool_name}' is not available."}],
            "planned_tool": None,
        }

    logger.info("Executing tool %s with args=%s", tool_name, tool_args)
    result = tool(**tool_args)
    mode = "read" if tool_name in READ_TOOL_NAMES else "write"
    tool_call_id = f"{tool_name}_{mode}_call"
    completion_text = "Read result" if mode == "read" else "Write operation completed"
    return {
        "messages": [
            {"role": "tool", "name": tool_name, "content": result, "tool_call_id": tool_call_id},
            {"role": "assistant", "content": f"{completion_text} from {tool_name}: {result}"},
        ],
        "planned_tool": None,
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
