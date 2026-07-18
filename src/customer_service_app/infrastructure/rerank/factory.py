from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.rerank.base import KnowledgeReranker
from customer_service_app.infrastructure.rerank.http_reranker import HttpKnowledgeReranker


def build_reranker(settings: Settings) -> KnowledgeReranker | None:
    if not settings.rerank_enabled:
        return None
    settings.require("RERANK_BASE_URL", settings.rerank_base_url)
    return HttpKnowledgeReranker(settings)
