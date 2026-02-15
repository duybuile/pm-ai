"""State schema for Saturn PM orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    def add_messages(left: list[Any], right: list[Any]) -> list[Any]:
        """Type-checking stub for LangGraph reducer."""
        return [*(left or []), *(right or [])]
else:
    try:
        from langgraph.graph.message import add_messages
    except ModuleNotFoundError:
        logger.warning("Could not import langgraph.add_messages")

        def add_messages(left: list[Any], right: list[Any]) -> list[Any]:
            """Fallback add_messages implementation when langgraph isn't installed."""
            return [*(left or []), *(right or [])]


class State(TypedDict, total=False):
    """Graph state shared across nodes."""

    messages: Annotated[list[Any], add_messages]
    next_action: dict[str, Any] | None
    explanation: str
    planned_tool: dict[str, Any] | None
    last_tool_name: str | None
    last_tool_mode: str | None
    last_tool_result: str | None
