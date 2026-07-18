from __future__ import annotations

from typing import Protocol

from customer_service_app.domain.schemas import KnowledgeChunk


class KnowledgeReranker(Protocol):
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[KnowledgeChunk],
        top_k: int,
    ) -> list[KnowledgeChunk]: ...

    async def close(self) -> None: ...
