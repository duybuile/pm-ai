"""Routing helpers for graph edge decisions."""

from __future__ import annotations

from src.orchestrator.state import State


def route_from_oracle(state: State) -> str:
    """Choose next node after oracle planning."""
    if state.get("planned_tool"):
        return "execute_tool"
    if state.get("next_action"):
        return "human_approval"
    return "__end__"


def route_from_approval(state: State) -> str:
    """Choose next step after human approval handling."""
    return "execute_tool" if state.get("planned_tool") else "__end__"
