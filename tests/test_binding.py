from __future__ import annotations

from src.tools.tools_registry import get_tool_definitions


def test_get_tool_definitions_contains_known_tools():
    manual = get_tool_definitions(as_text=True)

    assert "get_projects" in manual
    assert "update_task_status" in manual
    assert "create_project_with_tasks" in manual


def test_oracle_node_uses_llm_decision_for_read(monkeypatch):
    from src.orchestrator import graph

    monkeypatch.setattr(
        graph,
        "_plan_with_llm",
        lambda _user, _messages: {
            "tool": "get_projects",
            "args": {},
            "explanation": "Need to read project data.",
        },
    )

    state = {"messages": [{"role": "user", "content": "List projects"}]}
    out = graph.oracle_node(state)

    assert out["planned_tool"]["name"] == "get_projects"
    assert out["next_action"] is None


def test_oracle_node_stages_write_for_approval(monkeypatch):
    from src.orchestrator import graph

    monkeypatch.setattr(
        graph,
        "_plan_with_llm",
        lambda _user, _messages: {
            "tool": "update_task_status",
            "args": {"task_id": 3, "status": "Done"},
            "explanation": "I should update task 3.",
        },
    )

    state = {"messages": [{"role": "user", "content": "Update task 3 to done"}]}
    out = graph.oracle_node(state)

    assert out["next_action"]["name"] == "update_task_status"
    assert "Approve" in out["messages"][0]["content"]


def test_execute_tool_node_calls_registry(monkeypatch):
    from src.orchestrator import graph

    monkeypatch.setitem(graph._TOOL_REGISTRY, "fake_tool", lambda **_kwargs: '{"ok": true}')
    monkeypatch.setattr(graph, "READ_TOOL_NAMES", {"fake_tool"})

    state = {"planned_tool": {"name": "fake_tool", "args": {}}}
    out = graph.execute_tool_node(state)

    assert out["messages"][0]["role"] == "tool"
    assert out["messages"][0]["name"] == "fake_tool"
