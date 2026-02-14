"""Pydantic schemas for Saturn PM assistant entities and updates."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TeamMember(BaseModel):
    """Team member record."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    email: str


class Project(BaseModel):
    """Project record."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    status: str
    owner_id: int


class Task(BaseModel):
    """Task record."""

    model_config = ConfigDict(extra="forbid")

    id: int
    project_id: int
    title: str
    description: str
    status: str
    assignee_id: int | None
    due_date: str | None


class Comment(BaseModel):
    """Comment record."""

    model_config = ConfigDict(extra="forbid")

    id: int
    task_id: int
    message: str
    user_id: int
    timestamp: str


class ProjectUpdate(BaseModel):
    """Partial project update payload for write operations."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    status: str | None = None
    owner_id: int | None = None


class TaskUpdate(BaseModel):
    """Partial task update payload for write operations."""

    model_config = ConfigDict(extra="forbid")

    project_id: int | None = None
    title: str | None = None
    description: str | None = None
    status: str | None = None
    assignee_id: int | None = None
    due_date: str | None = None
