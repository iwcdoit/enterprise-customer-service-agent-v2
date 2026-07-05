from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


PlanActionType = Literal["tool", "rag", "llm", "handoff", "final"]
PlanStepStatus = Literal["pending", "running", "completed", "failed", "blocked", "skipped"]
PlanSource = Literal["llm", "rule"]


class PlanStep(BaseModel):
    """One bounded step in a multi-intent customer-service plan."""

    id: str
    title: str
    action_type: PlanActionType
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    status: PlanStepStatus = "pending"
    attempts: int = 0
    observation: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    """A bounded execution plan generated before running tools or graph nodes."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_goal: str
    max_steps: int = 6
    steps: list[PlanStep] = Field(default_factory=list)
    source: PlanSource = "rule"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanExecutionResult(BaseModel):
    """Summary of a plan execution attempt."""

    plan: AgentPlan
    completed_step_ids: list[str] = Field(default_factory=list)
    blocked_step_ids: list[str] = Field(default_factory=list)
    failed_step_ids: list[str] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
