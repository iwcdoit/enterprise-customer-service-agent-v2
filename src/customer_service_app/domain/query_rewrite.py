from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RewriteSource = Literal["original", "rule", "llm", "fallback"]


class QueryRewriteResult(BaseModel):
    """问题理解与检索改写的受控结果。"""

    original_question: str
    standalone_question: str
    dense_query: str
    sparse_query: str
    intent: str = "unknown"
    entities: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: str | None = None
    confidence: float = 1.0
    source: RewriteSource = "original"
    reason: str = ""


class RetrievalQuality(BaseModel):
    """检索后的质量门禁，决定继续回答、重试还是降级。"""

    sufficient: bool
    chunk_count: int
    max_score: float = 0.0
    retry_recommended: bool = False
    reason: str = ""
