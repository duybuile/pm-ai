"""Graph construction and runtime entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from src import cfg
from src.orchestrator.message_utils import message_to_dict
from src.orchestrator.nodes import execute_tool_node, human_approval_node, oracle_node
from src.orchestrator.routing import (
    route_from_approval,
    route_from_execute_tool,
    route_from_oracle,
)
from src.orchestrator.state import State

logger = logging.getLogger(__name__)
_CHECKPOINTER_CONTEXTS: list[Any] = []


def build_checkpointer(path: str):
    """Create a SqliteSaver checkpointer instance."""
    try:
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError:  # pragma: no cover
        raise ModuleNotFoundError(
            "langgraph with sqlite checkpoint support is required. Install project dependencies first."
        )

    checkpoint_file = Path(path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    conn_candidates = (str(checkpoint_file), f"sqlite:///{checkpoint_file}")
    last_error: Exception | None = None

    for conn_string in conn_candidates:
        try:
            maybe_saver = SqliteSaver.from_conn_string(conn_string)
            if isinstance(maybe_saver, BaseCheckpointSaver):
                return maybe_saver

            if hasattr(maybe_saver, "__enter__") and hasattr(maybe_saver, "__exit__"):
                saver = maybe_saver.__enter__()
                if isinstance(saver, BaseCheckpointSaver):
                    _CHECKPOINTER_CONTEXTS.append(maybe_saver)
                    return saver
                maybe_saver.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise TypeError(
        "Unable to create a valid SqliteSaver checkpointer from connection string."
    ) from last_error


def build_graph(checkpoint_path: str | None = None) -> Any:
    """Compile and return the LangGraph app."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("langgraph is required to build the orchestrator graph.") from exc

    path = checkpoint_path or cfg.get("orchestrator.checkpoint_path", "conf/langgraph_checkpoints.db")

    builder = StateGraph(State)  # type: ignore[arg-type]
    builder.add_node("oracle", cast(Any, oracle_node))
    builder.add_node("execute_tool", cast(Any, execute_tool_node))
    builder.add_node("human_approval", cast(Any, human_approval_node))

    builder.add_edge(START, "oracle")
    builder.add_conditional_edges(
        "oracle",
        route_from_oracle,
        {
            "execute_tool": "execute_tool",
            "human_approval": "human_approval",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "human_approval",
        route_from_approval,
        {
            "execute_tool": "execute_tool",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "execute_tool",
        route_from_execute_tool,
        {
            "oracle": "oracle",
            "__end__": END,
        },
    )

    checkpointer = build_checkpointer(path)
    return builder.compile(checkpointer=checkpointer)


def run_turn(app: Any, user_message: str, thread_id: str) -> State:
    """Run one chat turn with configured recursion limit and thread persistence."""
    recursion_limit = int(cfg.get("orchestrator.recursion_limit", 10))
    raw_state = app.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit},
    )
    messages = raw_state.get("messages", [])
    raw_state["messages"] = [message_to_dict(message) for message in messages]
    return raw_state


try:
    app = build_graph()
except Exception as exc:  # pragma: no cover
    logger.warning("Could not initialize compiled LangGraph app at import time: %s", exc)
    app = None
