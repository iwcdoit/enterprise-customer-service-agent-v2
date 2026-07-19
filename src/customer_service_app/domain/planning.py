from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PlanActionType = Literal["rag", "tool", "llm", "confirm", "handoff"]
PlanStepStatus = Literal["pending", "running", "success", "failed", "skipped"]
PlanDecision = Literal["direct", "planned"]


class PlanStep(BaseModel):
    """One executable step in a multi-intent user request."""

    step_id: str
    title: str
    goal: str
    action_type: PlanActionType
    tool_name: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: PlanStepStatus = "pending"
    requires_confirmation: bool = False
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    """Structured execution plan generated for complex requests."""

    plan_id: str
    conversation_id: str
    user_goal: str
    decision: PlanDecision = "planned"
    steps: list[PlanStep] = Field(default_factory=list)
    max_steps: int = 6


class PlanExecutionResult(BaseModel):
    """Result produced by executing a plan until finish or interruption."""

    plan: AgentPlan | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list)
    waiting_confirmation_id: str | None = None
    final_answer: str | None = None
