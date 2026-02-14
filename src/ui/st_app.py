"""Streamlit frontend for Saturn PM Assistant.

Features:
- Real-time sidebar dashboard for Projects and Tasks.
- Persistent chat session with LangGraph thread id.
- Human approval gate for write operations.
- Technical logs panel for demo visibility.
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from src import cfg
from src.db.database import get_connection, initialize_database, seed_database
from src.orchestrator.graph import app as compiled_app
from src.orchestrator.graph import build_graph, run_turn
from utils.config.log_handler import setup_logger

logger = setup_logger(
    logger_name=__name__,
    level=cfg.get("logging.level", "info"),
    console_logging=cfg.get("logging.console_logging", True)
)


def _safe_graph_app() -> Any:
    """Return a compiled graph instance, rebuilding if import-time init failed."""
    return compiled_app or build_graph()


def _load_table(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute a read query and return rows as dictionaries."""
    with get_connection() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def _dashboard() -> None:
    """Render the real-time database dashboard in sidebar."""
    st.sidebar.header("Real-time Dashboard")

    projects = _load_table("SELECT id, name, status, owner_id FROM Projects ORDER BY id")
    st.sidebar.subheader("Projects")
    st.sidebar.dataframe(projects, width="stretch", hide_index=True)

    project_filter_options = ["All"] + [str(project["id"]) for project in projects]
    selected_project = st.sidebar.selectbox("Task Filter (Project ID)", project_filter_options, index=0)

    if selected_project == "All":
        tasks = _load_table(
            "SELECT id, project_id, title, status, assignee_id, due_date FROM Tasks ORDER BY id"
        )
    else:
        tasks = _load_table(
            "SELECT id, project_id, title, status, assignee_id, due_date FROM Tasks WHERE project_id = ? ORDER BY id",
            (int(selected_project),),
        )

    st.sidebar.subheader("Tasks")
    st.sidebar.dataframe(tasks, width="stretch", hide_index=True)


def _init_session_state() -> None:
    """Initialize chat and workflow session state."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "latest_state" not in st.session_state:
        st.session_state.latest_state = {}


def _display_chat_messages() -> None:
    """Render the current chat transcript."""
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        if role == "tool":
            continue

        with st.chat_message(role):
            st.write(message.get("content", ""))


def _refresh_messages_from_state(state: dict[str, Any]) -> None:
    """Project LangGraph messages into Streamlit-visible chat history."""
    normalized = state.get("messages", [])
    if normalized:
        st.session_state.messages = normalized


def _handle_user_prompt(prompt: str) -> None:
    """Send a user prompt through the orchestrator and refresh UI state."""
    app = _safe_graph_app()
    if app is None:
        st.error("Graph could not be initialized. Check LangGraph dependencies.")
        return

    logger.info("Processing user prompt for thread_id=%s", st.session_state.thread_id)
    state = run_turn(app, prompt, st.session_state.thread_id)
    st.session_state.latest_state = state
    _refresh_messages_from_state(state)


def _render_approval_gate() -> None:
    """Render approval controls when a pending write action exists."""
    state = st.session_state.latest_state or {}
    next_action = state.get("next_action")
    if not next_action:
        return

    explanation = state.get("explanation", "The assistant proposed a write operation.")
    st.warning(explanation)

    col1, col2 = st.columns(2)
    if col1.button("Confirm Action", type="primary", width="stretch"):
        _handle_user_prompt("yes")
        st.rerun()
    if col2.button("Cancel", width="stretch"):
        _handle_user_prompt("no")
        st.rerun()


def _technical_logs() -> None:
    """Render technical state for interviewer visibility."""
    with st.expander("Technical Logs", expanded=False):
        st.json(st.session_state.latest_state or {})


def run_ui() -> None:
    """Run the Streamlit app."""
    page_title = cfg.get("streamlit.page_title", "Saturn PM Assistant")
    page_icon = cfg.get("streamlit.page_icon", "🪐")
    layout = cfg.get("streamlit.layout", "wide")
    chat_title = cfg.get("streamlit.chat_title", "Saturn PM Assistant")

    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    st.title(chat_title)

    initialize_database(reset=False)
    seed_database(force=False)

    _init_session_state()
    _dashboard()
    _display_chat_messages()
    _render_approval_gate()

    prompt = st.chat_input("Ask about projects, tasks, or request write actions...")
    if prompt:
        _handle_user_prompt(prompt)
        st.rerun()

    _technical_logs()
