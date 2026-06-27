from __future__ import annotations

from typing import Protocol

from customer_service_app.domain.schemas import KnowledgeChunk


class KnowledgeVectorStore(Protocol):
    """Interface for knowledge vector stores."""

    async def search(
        self,
        *,
        tenant_id: str,
        query_vector: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[KnowledgeChunk]:
        """Search the most relevant knowledge chunks by query vector."""
        ...

    async def upsert_chunks(
        self,
        *,
        tenant_id: str,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
    ) -> None:
        """Upsert knowledge chunks and their embeddings into the vector store."""
        ...
