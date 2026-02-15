"""Compatibility facade for orchestrator modules.

The orchestrator implementation is split across:
- state.py
- message_utils.py
- planner.py
- nodes.py
- routing.py
- runtime.py
"""

from __future__ import annotations

from src.orchestrator import nodes as _nodes
from src.orchestrator import planner as _planner
from src.orchestrator.message_utils import (
    last_user_text as _last_user_text,
    latest_tool_payload as _latest_tool_payload,
    message_to_dict as _message_to_dict,
    msg_content as _msg_content,
    msg_role as _msg_role,
    yes_no as _yes_no,
)
from src.orchestrator.runtime import (
    app,
    build_checkpointer as _build_checkpointer,
    build_graph,
    run_turn,
)
from src.orchestrator.state import State
from src.tools.tools_registry import READ_TOOL_NAMES, WRITE_TOOL_NAMES

# Backward-compatible symbols expected by some tests/call-sites.
_extract_task_id = _planner.extract_task_id
_extract_status = _planner.extract_status
_extract_project_name = _planner.extract_project_name
_extract_assignee_name = _planner.extract_assignee_name
_load_oracle_prompt = _planner.load_oracle_prompt
_safe_json_object = _planner.safe_json_object
_fallback_oracle_response = _planner.fallback_oracle_response
_plan_with_llm = _planner.plan_with_llm

_TOOL_REGISTRY = _nodes._TOOL_REGISTRY



def oracle_node(state: State) -> State:
    """Compatibility wrapper around the modular oracle node."""
    original_plan = _nodes.plan_with_llm
    original_fallback = _nodes.fallback_oracle_response
    _nodes.plan_with_llm = _plan_with_llm
    _nodes.fallback_oracle_response = _fallback_oracle_response
    try:
        return _nodes.oracle_node(state)
    finally:
        _nodes.plan_with_llm = original_plan
        _nodes.fallback_oracle_response = original_fallback


def execute_tool_node(state: State) -> State:
    """Compatibility wrapper around the modular execute-tool node."""
    original_registry = _nodes._TOOL_REGISTRY
    original_read = _nodes.READ_TOOL_NAMES
    original_write = _nodes.WRITE_TOOL_NAMES
    _nodes._TOOL_REGISTRY = _TOOL_REGISTRY
    _nodes.READ_TOOL_NAMES = READ_TOOL_NAMES
    _nodes.WRITE_TOOL_NAMES = WRITE_TOOL_NAMES
    try:
        return _nodes.execute_tool_node(state)
    finally:
        _nodes._TOOL_REGISTRY = original_registry
        _nodes.READ_TOOL_NAMES = original_read
        _nodes.WRITE_TOOL_NAMES = original_write


def human_approval_node(state: State) -> State:
    """Compatibility wrapper around the modular approval node."""
    return _nodes.human_approval_node(state)
