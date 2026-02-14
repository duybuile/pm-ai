"""Message normalization and extraction helpers for orchestrator."""

from __future__ import annotations

from typing import Any


def msg_content(message: Any) -> str:
    """Extract content from dict-like or object-like chat message."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def msg_role(message: Any) -> str:
    """Extract role/type from dict-like or object-like chat message."""
    if isinstance(message, dict):
        return str(message.get("role", message.get("type", ""))).lower()
    return str(getattr(message, "type", getattr(message, "role", ""))).lower()


def message_to_dict(message: Any) -> dict[str, Any]:
    """Convert LangChain/BaseMessage objects or dict-like messages to plain dicts."""

    def normalize_role(role: str) -> str:
        if role == "ai":
            return "assistant"
        if role == "human":
            return "user"
        return role

    if isinstance(message, dict):
        role = normalize_role(msg_role(message))
        payload: dict[str, Any] = {"role": role, "content": message.get("content", "")}
        if "name" in message:
            payload["name"] = message["name"]
        if "tool_call_id" in message:
            payload["tool_call_id"] = message["tool_call_id"]
        return payload

    role = normalize_role(msg_role(message))
    payload = {"role": role, "content": msg_content(message)}
    name = getattr(message, "name", None)
    tool_call_id = getattr(message, "tool_call_id", None)
    if name:
        payload["name"] = name
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def last_user_text(messages: list[Any]) -> str:
    """Return latest user text from message history."""
    for message in reversed(messages):
        if msg_role(message) in {"human", "user"}:
            return msg_content(message)
    return ""


def latest_tool_payload(messages: list[Any], tool_name: str) -> str | None:
    """Return the most recent tool payload for a named tool."""
    for message in reversed(messages):
        role = msg_role(message)
        if role not in {"tool"}:
            continue

        if isinstance(message, dict):
            if message.get("name") == tool_name:
                return str(message.get("content", ""))
            continue

        if getattr(message, "name", None) == tool_name:
            return str(getattr(message, "content", ""))
    return None


def yes_no(text: str) -> str:
    """Classify approval response as yes/no/unknown."""
    normalized = text.strip().lower()
    if normalized in {"yes", "y", "approve", "approved", "confirm", "go ahead"}:
        return "yes"
    if normalized in {"no", "n", "deny", "denied", "cancel", "stop"}:
        return "no"
    return "unknown"
