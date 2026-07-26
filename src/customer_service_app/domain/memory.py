from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_service_app.domain.schemas import ChatMessage


MemoryType = Literal["profile", "task", "event", "risk"]
MemoryVerification = Literal[
    "explicit_user",
    "verified_tool",
    "business_system",
    "human_confirmed",
    "risk_engine",
]
MemorySensitivity = Literal["public", "internal", "sensitive"]


class ShortTermContext(BaseModel):
    """Selected context used for one model call."""

    recent_messages: list[ChatMessage] = Field(default_factory=list)
    summary: str | None = None
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)
    memories: list["CustomerMemoryView"] = Field(default_factory=list)


class CustomerMemoryView(BaseModel):
    """Long-term memory selected for the current conversation."""

    id: str
    memory_type: MemoryType
    memory_key: str
    memory_value: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: str = "system"
    verification_status: MemoryVerification
    evidence_ids: list[str] = Field(default_factory=list)
    sensitivity: MemorySensitivity = "internal"
    expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryWriteCommand(BaseModel):
    """Command object used when a service wants to write long-term memory."""

    tenant_id: str
    user_id: str
    memory_type: MemoryType
    memory_key: str
    memory_value: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: str = "agent"
    verification_status: MemoryVerification
    evidence_ids: list[str] = Field(default_factory=list)
    sensitivity: MemorySensitivity = "internal"
    expires_at: str | None = None


class MemoryUpdateRequest(BaseModel):
    """User-controlled update for one owned memory."""

    memory_value: dict[str, Any]
    expires_at: datetime | None = None


class MemoryPreferenceUpdateRequest(BaseModel):
    """Enable or disable long-term memory for one user."""

    enabled: bool


class MemoryPreferenceView(BaseModel):
    """Current long-term memory consent returned to the client."""

    tenant_id: str
    user_id: str
    enabled: bool
    updated_at: datetime | None = None


class MemoryCleanupView(BaseModel):
    """Result of deleting expired long-term memories."""

    deleted_count: int
