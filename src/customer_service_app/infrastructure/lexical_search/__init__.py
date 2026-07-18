"""Keyword-based knowledge retrieval adapters."""

from customer_service_app.infrastructure.lexical_search.base import LexicalKnowledgeRetriever
from customer_service_app.infrastructure.lexical_search.factory import build_lexical_retriever

__all__ = ["LexicalKnowledgeRetriever", "build_lexical_retriever"]
