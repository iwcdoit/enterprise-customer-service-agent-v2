from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TraceStatus = Literal["started", "success", "failed", "skipped", "waiting"]


class TraceEvent(BaseModel):
    """Internal trace event persisted for an Agent run."""

    stage: str
    name: str
    status: TraceStatus
    detail: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
