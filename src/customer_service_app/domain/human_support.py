from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


HumanHandoffStatus = Literal[
    "waiting_assignment",
    "assigned",
    "in_service",
    "resolution_submitted",
    "resolved",
    "cancelled",
]


class HumanHandoffView(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    conversation_id: str
    support_ticket_id: str | None = None
    origin_thread_id: str | None = None
    status: HumanHandoffStatus
    queue_name: str
    priority: str
    reason: str
    assigned_agent_id: str | None = None
    resolution_code: str | None = None
    resolution_summary: str | None = None
    next_mode: str | None = None
    version: int
    requested_at: str
    updated_at: str


class HumanAssignmentRequest(BaseModel):
    tenant_id: str
    agent_id: str = Field(min_length=1, max_length=64)
    expected_version: int | None = None


class HumanMessageRequest(BaseModel):
    tenant_id: str
    agent_id: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=8000)


class HumanResolutionRequest(BaseModel):
    tenant_id: str
    agent_id: str = Field(min_length=1, max_length=64)
    resolution_code: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=8000)
    next_mode: Literal["resume_bot", "close_conversation"] = "resume_bot"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanResolutionConfirmationRequest(BaseModel):
    tenant_id: str
    operator_id: str = Field(min_length=1, max_length=64)
    expected_version: int | None = None
