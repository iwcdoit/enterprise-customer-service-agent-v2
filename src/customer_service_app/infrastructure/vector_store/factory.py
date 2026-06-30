from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ConfigurationError
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore
from customer_service_app.infrastructure.vector_store.milvus_store import MilvusKnowledgeVectorStore
from customer_service_app.infrastructure.vector_store.qdrant_store import QdrantKnowledgeVectorStore


def build_vector_store(settings: Settings) -> KnowledgeVectorStore:
    """Build the configured vector store implementation."""
    provider = settings.vector_store_provider.strip().lower()

    if provider == "qdrant":
        return QdrantKnowledgeVectorStore(settings)

    if provider == "milvus":
        return MilvusKnowledgeVectorStore(settings)

    raise ConfigurationError(
        "Unsupported VECTOR_STORE_PROVIDER. Expected one of: qdrant, milvus."
    )
