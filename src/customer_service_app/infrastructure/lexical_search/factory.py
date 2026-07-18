from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.lexical_search.base import LexicalKnowledgeRetriever
from customer_service_app.infrastructure.lexical_search.opensearch_bm25 import (
    OpenSearchBM25Retriever,
)


def build_lexical_retriever(settings: Settings) -> LexicalKnowledgeRetriever | None:
    """未开启 OpenSearch 时返回 None，RAG 会自动退化为纯向量检索。"""

    if not settings.opensearch_enabled:
        return None
    return OpenSearchBM25Retriever(settings)
