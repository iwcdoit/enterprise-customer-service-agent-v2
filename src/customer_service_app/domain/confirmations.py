from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


PendingActionStatus = Literal["pending", "approved", "rejected", "executed", "expired", "failed"]
RiskLevel = Literal["low", "medium", "high"]


class PendingActionView(BaseModel):
    """User-facing view of an operation waiting for confirmation."""

    id: str
    tenant_id: str
    user_id: str
    conversation_id: str | None = None
    action_type: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: PendingActionStatus
    risk_level: RiskLevel = "low"
    confirmation_prompt: str
    thread_id: str | None = None
    confirmation_id: str | None = None
    comment: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    expired: bool = False


class ConfirmationDecisionRequest(BaseModel):
    """Request body for approve/reject confirmation APIs."""

    tenant_id: str = "default"
    user_id: str
    reason: str | None = None


class ConfirmationDecisionResponse(BaseModel):
    """Response returned after approving or rejecting a pending action."""

    confirmation_id: str
    status: PendingActionStatus
    message: str
    result: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    thread_id: str | None = None
    run_id: str | None = None
    answer: str = ""
    graph_status: str = "completed"
    next_confirmation: PendingActionView | None = None
    plan: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
