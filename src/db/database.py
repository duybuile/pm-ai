"""Database setup and seed utilities for the Saturn PM assistant."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BASE_DIR / "conf" / "pm_ai.db"
MIGRATION_FILE = BASE_DIR / "migration" / "001_init.sql"


def get_database_path() -> Path:
    """Return the SQLite database path, allowing an environment override for tests."""
    override = os.getenv("PM_AI_DB_PATH")
    return Path(override) if override else DEFAULT_DB_PATH


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a managed SQLite connection with transaction safety and row mapping."""
    db_path = get_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        logger.exception("Transaction rolled back due to an error.")
        raise
    finally:
        connection.close()


def initialize_database(reset: bool = False) -> None:
    """Create database tables from migration DDL.

    Args:
        reset: When True, delete the database file before recreating tables.
    """
    db_path = get_database_path()
    if reset and db_path.exists():
        db_path.unlink()
        logger.info("Deleted database file at %s", db_path)

    if not MIGRATION_FILE.exists():
        raise FileNotFoundError(f"Migration file not found: {MIGRATION_FILE}")

    ddl_sql = MIGRATION_FILE.read_text(encoding="utf-8")
    with get_connection() as connection:
        connection.executescript(ddl_sql)

    logger.info("Database initialized at %s", db_path)


def seed_database(force: bool = False) -> dict[str, int]:
    """Insert realistic mock PM data.

    Args:
        force: When True, clear existing records before seeding.

    Returns:
        A dictionary with seeded record counts.
    """
    initialize_database(reset=False)

    team_members = [
        (1, "Sarah Kim", "sarah.kim@saturnpm.com"),
        (2, "Leo Martinez", "leo.martinez@saturnpm.com"),
        (3, "Priya Nair", "priya.nair@saturnpm.com"),
        (4, "Avery Chen", "avery.chen@saturnpm.com"),
    ]

    projects = [
        (1, "Mobile App Redesign", "In Progress", 1),
        (2, "Q2 Growth Campaign", "Planning", 2),
        (3, "Data Warehouse Migration", "In Progress", 3),
        (4, "Customer Onboarding Revamp", "Blocked", 1),
        (5, "Security Compliance Audit", "Completed", 4),
    ]

    tasks = [
        (1, 1, "Finalize navigation prototype", "Create Figma flows for all key journeys.", "In Progress", 1, "2026-02-20"),
        (2, 1, "Run usability interviews", "Interview 8 beta users for pain points.", "Not Started", 2, "2026-02-24"),
        (3, 1, "Implement design system tokens", "Map new typography and spacing tokens.", "In Review", 4, "2026-02-19"),
        (4, 2, "Define campaign KPIs", "Align on conversion and CAC targets.", "Done", 2, "2026-02-05"),
        (5, 2, "Create ad creative briefs", "Draft creative direction for paid channels.", "In Progress", 1, "2026-02-18"),
        (6, 2, "Set up attribution dashboard", "Connect ad spend and lead events.", "Not Started", 3, "2026-02-28"),
        (7, 3, "Inventory legacy ETL jobs", "Document ownership and dependencies.", "Done", 3, "2026-01-31"),
        (8, 3, "Provision warehouse schemas", "Create staging and mart schemas.", "In Progress", 4, "2026-02-17"),
        (9, 3, "Migrate finance pipelines", "Port monthly close transformations.", "Blocked", 2, "2026-03-04"),
        (10, 4, "Map onboarding funnel", "Identify drop-off points from signup to activation.", "In Progress", 1, "2026-02-22"),
        (11, 4, "Draft lifecycle email sequence", "Define first-30-day engagement emails.", "Not Started", 2, "2026-02-25"),
        (12, 4, "Implement in-app checklist", "Build guided checklist for new accounts.", "Blocked", 4, "2026-03-01"),
        (13, 5, "Collect SOC2 evidence", "Gather and organize required artifacts.", "Done", 4, "2026-01-25"),
        (14, 5, "Review access controls", "Audit IAM roles and least privilege.", "Done", 3, "2026-01-28"),
        (15, 5, "Remediate findings", "Close medium-priority security gaps.", "Done", 2, "2026-02-02"),
    ]

    comments = [
        (1, 1, "Prototype v2 looks solid. Need tablet breakpoints.", 4, "2026-02-11T09:15:00Z"),
        (2, 2, "Recruiting participants this week.", 2, "2026-02-12T14:20:00Z"),
        (3, 9, "Waiting on data engineering capacity.", 3, "2026-02-10T17:05:00Z"),
        (4, 12, "Blocked by dependency on checklist API endpoint.", 4, "2026-02-13T11:33:00Z"),
        (5, 10, "Funnel analysis draft shared in Notion.", 1, "2026-02-12T08:05:00Z"),
    ]

    with get_connection() as connection:
        if force:
            connection.execute("DELETE FROM Comments")
            connection.execute("DELETE FROM Tasks")
            connection.execute("DELETE FROM Projects")
            connection.execute("DELETE FROM TeamMembers")

        existing = connection.execute("SELECT COUNT(*) AS count FROM Projects").fetchone()["count"]
        if existing and not force:
            logger.info("Skipping seed; data already exists.")
            return {"projects": existing}

        connection.executemany(
            "INSERT INTO TeamMembers (id, name, email) VALUES (?, ?, ?)",
            team_members,
        )
        connection.executemany(
            "INSERT INTO Projects (id, name, status, owner_id) VALUES (?, ?, ?, ?)",
            projects,
        )
        connection.executemany(
            """
            INSERT INTO Tasks (id, project_id, title, description, status, assignee_id, due_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tasks,
        )
        connection.executemany(
            "INSERT INTO Comments (id, task_id, message, user_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            comments,
        )

    logger.info(
        "Seeded database with %d team members, %d projects, %d tasks, and %d comments.",
        len(team_members),
        len(projects),
        len(tasks),
        len(comments),
    )

    return {
        "team_members": len(team_members),
        "projects": len(projects),
        "tasks": len(tasks),
        "comments": len(comments),
    }
