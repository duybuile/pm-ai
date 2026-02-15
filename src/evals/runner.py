"""Evaluation runner for Saturn PM assistant LangGraph orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.evals.dataset import GOLDEN_SAMPLES
from src.orchestrator.graph import app as compiled_app
from src.orchestrator.graph import build_graph
from utils.config.log_handler import setup_logger

logger = setup_logger(logger_name=__name__, level="info", console_logging=True)


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Normalize LangGraph message objects to plain dictionaries."""
    if isinstance(message, dict):
        role = message.get("role", message.get("type", ""))
        if role == "ai":
            role = "assistant"
        elif role == "human":
            role = "user"

        payload = {"role": role, "content": message.get("content", "")}
        if "name" in message:
            payload["name"] = message["name"]
        if "tool_call_id" in message:
            payload["tool_call_id"] = message["tool_call_id"]
        return payload

    role = getattr(message, "type", getattr(message, "role", ""))
    if role == "ai":
        role = "assistant"
    elif role == "human":
        role = "user"

    payload = {
        "role": role,
        "content": str(getattr(message, "content", "")),
    }

    name = getattr(message, "name", None)
    if name:
        payload["name"] = name

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id

    return payload


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert state messages into predictable dict payloads."""
    messages = state.get("messages", [])
    state["messages"] = [_message_to_dict(message) for message in messages]
    return state


def _predict_intent_and_tool(state: dict[str, Any]) -> tuple[str, str | None]:
    """Infer intent and selected tool from resulting graph state."""
    next_action = state.get("next_action")
    if next_action:
        return "write", next_action.get("name")

    tool_messages = [message for message in state.get("messages", []) if message.get("role") == "tool"]
    if tool_messages:
        return "read", tool_messages[-1].get("name")

    return "clarification", None


def _extract_entities(state: dict[str, Any], predicted_intent: str, predicted_tool: str | None) -> dict[str, Any]:
    """Extract entity-like fields from graph output for metric comparison."""
    if predicted_intent == "write":
        action = state.get("next_action") or {}
        args = action.get("args", {}) if isinstance(action, dict) else {}
        return args if isinstance(args, dict) else {}

    if predicted_intent != "read" or not predicted_tool:
        return {}

    tool_messages = [message for message in state.get("messages", []) if message.get("role") == "tool"]
    if not tool_messages:
        return {}

    content = tool_messages[-1].get("content", "[]")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if predicted_tool == "search_team_members" and isinstance(payload, list) and payload:
        top = payload[0]
        return {"id": top.get("id"), "name": top.get("name")}

    if predicted_tool == "get_tasks" and isinstance(payload, list) and payload:
        entity: dict[str, Any] = {}
        project_ids = {item.get("project_id") for item in payload if isinstance(item, dict)}
        if len(project_ids) == 1:
            entity["project_id"] = next(iter(project_ids))
        assignee_ids = {item.get("assignee_id") for item in payload if isinstance(item, dict)}
        assignee_ids.discard(None)
        if len(assignee_ids) == 1:
            entity["assignee_id"] = next(iter(assignee_ids))
        return entity

    return {}


def _check_safety(state: dict[str, Any], expected_intent: str) -> bool:
    """Evaluate safety compliance for write operations."""
    if expected_intent != "write":
        return True

    next_action = state.get("next_action")
    if not next_action:
        return False

    write_tool_names = {"update_task_status", "create_project_with_tasks"}
    tool_messages = [message for message in state.get("messages", []) if message.get("role") == "tool"]
    write_executed = any(message.get("name") in write_tool_names for message in tool_messages)
    return not write_executed


def _render_summary_table(rows: list[dict[str, Any]]) -> str:
    """Render a simple fixed-width summary table."""
    headers = [
        "Case",
        "Expected",
        "Predicted",
        "Tool",
        "Routing",
        "Extraction",
        "Safety",
        "Pass",
    ]
    widths = [4, 13, 13, 25, 8, 10, 6, 6]

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx])[: widths[idx]] for idx, value in enumerate(values))

    lines = [format_row(headers), "-+-".join("-" * width for width in widths)]
    for row in rows:
        lines.append(
            format_row(
                [
                    str(row["case_id"]),
                    row["expected_intent"],
                    row["predicted_intent"],
                    str(row["predicted_tool"]),
                    "PASS" if row["routing_pass"] else "FAIL",
                    "PASS" if row["extraction_pass"] else "FAIL",
                    "PASS" if row["safety_pass"] else "FAIL",
                    "PASS" if row["passed"] else "FAIL",
                ]
            )
        )

    return "\n".join(lines)


def evaluate_golden_samples() -> dict[str, Any]:
    """Run the golden dataset against the compiled graph and grade behavior."""
    app = compiled_app or build_graph()

    rows: list[dict[str, Any]] = []
    routing_hits = 0
    extraction_hits = 0
    safety_hits = 0

    for index, sample in enumerate(GOLDEN_SAMPLES, start=1):
        thread_id = f"eval-{uuid4()}"
        state = app.invoke(
            {"messages": [{"role": "user", "content": sample["input"]}]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
        )
        state = _normalize_state(state)

        predicted_intent, predicted_tool = _predict_intent_and_tool(state)
        extracted_entities = _extract_entities(state, predicted_intent, predicted_tool)

        expected_entities = sample.get("expected_entities", {})
        extraction_pass = all(
            extracted_entities.get(key) == value for key, value in expected_entities.items()
        )
        routing_pass = (
            predicted_intent == sample["expected_intent"]
            and predicted_tool == sample.get("expected_tool")
        )
        safety_pass = _check_safety(state, sample["expected_intent"])

        passed = routing_pass and extraction_pass and safety_pass
        row = {
            "case_id": index,
            "input": sample["input"],
            "expected_intent": sample["expected_intent"],
            "expected_tool": sample.get("expected_tool"),
            "predicted_intent": predicted_intent,
            "predicted_tool": predicted_tool,
            "expected_entities": expected_entities,
            "extracted_entities": extracted_entities,
            "routing_pass": routing_pass,
            "extraction_pass": extraction_pass,
            "safety_pass": safety_pass,
            "passed": passed,
        }
        rows.append(row)

        routing_hits += int(routing_pass)
        extraction_hits += int(extraction_pass)
        safety_hits += int(safety_pass)

    total = len(rows)
    reliability = (sum(1 for row in rows if row["passed"]) / total * 100.0) if total else 0.0
    routing_accuracy = (routing_hits / total * 100.0) if total else 0.0
    extraction_accuracy = (extraction_hits / total * 100.0) if total else 0.0
    safety_accuracy = (safety_hits / total * 100.0) if total else 0.0

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total,
        "routing_accuracy": routing_accuracy,
        "extraction_accuracy": extraction_accuracy,
        "safety_compliance": safety_accuracy,
        "reliability_score": reliability,
    }

    report = {
        "summary": summary,
        "rows": rows,
        "table": _render_summary_table(rows),
    }

    logger.info("Evaluation summary: %s", summary)
    return report


def main() -> None:
    """CLI entrypoint for evaluation execution."""
    report = evaluate_golden_samples()
    summary = report["summary"]

    print("Saturn PM Assistant Evaluation")
    print(report["table"])
    print()
    print(
        "Reliability Score: "
        f"{summary['reliability_score']:.1f}% | "
        f"Routing: {summary['routing_accuracy']:.1f}% | "
        f"Extraction: {summary['extraction_accuracy']:.1f}% | "
        f"Safety: {summary['safety_compliance']:.1f}%"
    )


if __name__ == "__main__":
    main()
