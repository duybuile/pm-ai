"""Shared helpers for Saturn PM tool implementations."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.db.database import initialize_database, seed_database

logger = logging.getLogger(__name__)

ALLOWED_TASK_STATUSES = {"Not Started", "In Progress", "In Review", "Blocked", "Done"}


def ensure_database_ready() -> None:
    """Create schema and seed mock data if needed."""
    initialize_database(reset=False)
    seed_database(force=False)


def serialize_rows(rows: list[Any]) -> str:
    """Convert sqlite rows into JSON text for LLM consumption."""
    return json.dumps([dict(row) for row in rows], ensure_ascii=True)
