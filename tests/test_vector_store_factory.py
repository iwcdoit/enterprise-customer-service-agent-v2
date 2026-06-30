from __future__ import annotations

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ConfigurationError
from customer_service_app.infrastructure.vector_store.factory import build_vector_store
from customer_service_app.infrastructure.vector_store.milvus_store import MilvusKnowledgeVectorStore
from customer_service_app.infrastructure.vector_store.qdrant_store import QdrantKnowledgeVectorStore


def test_build_vector_store_defaults_to_qdrant() -> None:
    settings = Settings(vector_store_provider="qdrant")

    vector_store = build_vector_store(settings)

    assert isinstance(vector_store, QdrantKnowledgeVectorStore)


def test_build_vector_store_supports_milvus() -> None:
    settings = Settings(vector_store_provider="milvus", milvus_uri="https://milvus.example.com")

    vector_store = build_vector_store(settings)

    assert isinstance(vector_store, MilvusKnowledgeVectorStore)


def test_build_vector_store_rejects_unknown_provider() -> None:
    settings = Settings(vector_store_provider="unknown")

    with pytest.raises(ConfigurationError):
        build_vector_store(settings)
