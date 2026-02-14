"""LLM oracle planning and fallback heuristics."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src import cfg
from src.tools.tools_registry import get_tool_definitions
from utils.llms.llm_client import LLMClient

from src.orchestrator.message_utils import latest_tool_payload, msg_content, msg_role


def extract_task_id(text: str) -> int | None:
    match = re.search(r"task\s*(?:id\s*)?(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_status(text: str) -> str | None:
    status_aliases = {
        "not started": "Not Started",
        "in progress": "In Progress",
        "in review": "In Review",
        "blocked": "Blocked",
        "done": "Done",
    }
    lower = text.lower()
    for alias, canonical in status_aliases.items():
        if alias in lower:
            return canonical
    return None


def extract_project_name(text: str) -> str:
    quoted = re.search(r"project\s+(?:named\s+)?['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()

    trailing = re.search(r"create\s+(?:a\s+)?project\s+(?:named\s+)?([a-zA-Z0-9 _-]+)", text, flags=re.IGNORECASE)
    if trailing:
        candidate = trailing.group(1)
        candidate = re.split(r"\s+(?:with|and)\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = candidate.strip(" .")
        if cleaned and not cleaned.lower().startswith("and "):
            return cleaned

    return "New Project"


def extract_assignee_name(text: str) -> str | None:
    match = re.search(r"assign\s+(?:the\s+)?first\s+task\s+to\s+([a-zA-Z]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def load_oracle_prompt() -> str:
    """Load the configured oracle prompt template from the prompt directory."""
    prompt_version = cfg.get("orchestrator.prompt_version", "v1")
    prompt_dir = cfg.get("orchestrator.prompt_dir", "prompt")
    prompt_file = Path(prompt_dir) / f"oracle_{prompt_version}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Oracle prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def safe_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw model output, including fenced-code responses."""
    if not text:
        return None
    candidate = text.strip()
    candidate = candidate.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def fallback_oracle_response(user_text: str, messages: list[Any]) -> dict[str, Any]:
    """Heuristic fallback used when LLM planning is unavailable."""
    lower = user_text.lower()

    if any(phrase in lower for phrase in ["list projects", "show projects", "all projects"]):
        return {"tool": "get_projects", "args": {}, "explanation": "I should fetch the current project list."}

    if "task" in lower and any(phrase in lower for phrase in ["show", "list", "what are"]):
        filters: dict[str, int] = {}
        project_match = re.search(r"project\s*(\d+)", lower)
        assignee_match = re.search(r"assignee\s*(\d+)", lower)
        if project_match:
            filters["project_id"] = int(project_match.group(1))
        if assignee_match:
            filters["assignee_id"] = int(assignee_match.group(1))
        return {"tool": "get_tasks", "args": filters, "explanation": "I should read tasks with the requested filters."}

    if "update" in lower and "task" in lower:
        task_id = extract_task_id(user_text)
        status = extract_status(user_text)
        if task_id is None or status is None:
            return {
                "tool": None,
                "args": {},
                "explanation": "Please provide task id and one status: Not Started, In Progress, In Review, Blocked, or Done.",
            }
        return {
            "tool": "update_task_status",
            "args": {"task_id": task_id, "status": status},
            "explanation": f"I want to update task {task_id} to '{status}' and need your approval.",
        }

    if "create" in lower and "project" in lower:
        assignee_name = extract_assignee_name(user_text)
        if assignee_name:
            payload = latest_tool_payload(messages, "search_team_members")
            if payload is None:
                return {
                    "tool": "search_team_members",
                    "args": {"query": assignee_name},
                    "explanation": f"I should find {assignee_name}'s team member id first.",
                }
            try:
                search_hits = json.loads(payload)
            except json.JSONDecodeError:
                search_hits = []
            assignee_id = search_hits[0]["id"] if search_hits else None
            if assignee_id is None:
                return {
                    "tool": None,
                    "args": {},
                    "explanation": f"I could not find '{assignee_name}'. Please provide an assignee id.",
                }
        else:
            assignee_id = None

        project_name = extract_project_name(user_text)
        task_payload: dict[str, Any] = {"title": "Initial setup task", "status": "Not Started"}
        if assignee_id is not None:
            task_payload["assignee_id"] = assignee_id
        return {
            "tool": "create_project_with_tasks",
            "args": {"name": project_name, "owner_id": assignee_id or 1, "tasks": [task_payload]},
            "explanation": f"I want to create project '{project_name}' and need your approval before writing.",
        }

    if "sarah" in lower and "id" in lower:
        return {
            "tool": "search_team_members",
            "args": {"query": "Sarah"},
            "explanation": "I should search the team members by name.",
        }

    return {
        "tool": None,
        "args": {},
        "explanation": "I can help with project/task reads, updates, and project creation workflows.",
    }


def plan_with_llm(user_text: str, messages: list[Any]) -> dict[str, Any]:
    """Get oracle plan from LLM client using strict JSON contract."""
    prompt_template = load_oracle_prompt()
    tools_manual = get_tool_definitions(as_text=True)
    history_lines = []
    for message in messages[-6:]:
        role = msg_role(message)
        if role not in {"user", "human", "assistant", "ai", "tool"}:
            continue
        history_lines.append(f"{role}: {msg_content(message)}")
    history_text = "\n".join(history_lines) if history_lines else "(no prior messages)"
    full_prompt = prompt_template.format(
        tools_manual=tools_manual,
        conversation_history=history_text,
        user_input=user_text,
    )

    llm_api = cfg.get("orchestrator.llm.api", "openai")
    llm_base = cfg.get("orchestrator.llm.base", "https://api.openai.com/v1")
    llm_model = cfg.get("orchestrator.llm.model", "gpt-5-mini")
    llm_temperature = float(cfg.get("orchestrator.llm.temperature", 0.0))
    api_key_env = cfg.get("orchestrator.llm.api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "")

    client = LLMClient(
        api=llm_api,
        base=llm_base,
        model=llm_model,
        temperature=llm_temperature,
        api_key=api_key,
    )
    raw = client.call(full_prompt)
    parsed = safe_json_object(raw or "")
    if not parsed:
        raise ValueError(f"Oracle LLM returned non-JSON content: {raw}")

    return {
        "tool": parsed.get("tool"),
        "args": parsed.get("args", {}) if isinstance(parsed.get("args", {}), dict) else {},
        "explanation": str(parsed.get("explanation", "")).strip() or "No explanation provided by oracle.",
    }
