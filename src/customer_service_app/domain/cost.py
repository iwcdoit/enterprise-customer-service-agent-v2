from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


TenantTier = Literal["basic", "standard", "premium"]


class CostStrategy(BaseModel):
    """Runtime strategy selected for one tenant request."""

    tenant_id: str
    tier: TenantTier
    model: str
    rag_top_k: int
    history_turns: int
    use_rerank: bool = False
    cache_first: bool = False
    degraded: bool = False
    budget_tokens: int = 0
    used_tokens: int = 0
    remaining_tokens: int = 0
    usage_ratio: float = 0.0
    usage_percent: float = 0.0
    budget_warning: bool = False
    budget_exceeded: bool = False
    degradation_reason: str | None = None


class TokenUsage(BaseModel):
    """Token usage normalized from an LLM provider response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
