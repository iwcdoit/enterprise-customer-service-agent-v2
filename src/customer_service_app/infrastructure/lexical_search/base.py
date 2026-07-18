from __future__ import annotations

from typing import Protocol

from customer_service_app.domain.schemas import KnowledgeChunk


class LexicalKnowledgeRetriever(Protocol):
    """BM25 等关键词检索实现需要遵循的边界。"""

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        top_k: int,
    ) -> list[KnowledgeChunk]: ...

    async def upsert_chunks(
        self,
        *,
        tenant_id: str,
        chunks: list[KnowledgeChunk],
    ) -> None: ...

    async def close(self) -> None: ...
