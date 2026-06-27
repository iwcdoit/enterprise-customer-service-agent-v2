from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    """Interface for embedding model clients."""

    async def embed_query(self, text: str) -> list[float]:
        """把单个用户问题转成一个向量。"""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把多个知识片段批量转成多个向量。"""
        ...
