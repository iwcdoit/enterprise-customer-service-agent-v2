from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore


class QdrantKnowledgeVectorStore(KnowledgeVectorStore):
    """Qdrant implementation for RAG knowledge storage and retrieval."""

    def __init__(self, settings: Settings):
        """Store settings and create the Qdrant client lazily."""
        self._settings = settings
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Return a lazily initialized Qdrant client."""
        if self._client is None:
            url = self._settings.require("QDRANT_URL", self._settings.qdrant_url)
            self._client = AsyncQdrantClient(url=url, api_key=self._settings.qdrant_api_key or None)
        return self._client

    async def ensure_collection(self) -> None:
        """Ensure the configured Qdrant collection exists."""
        try:
            exists = await self.client.collection_exists(self._settings.qdrant_collection)
            if not exists:
                await self.client.create_collection(
                    collection_name=self._settings.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self._settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise ExternalServiceError(f"Qdrant collection initialization failed: {exc}") from exc

    async def search(
        self,
        *,
        tenant_id: str,
        query_vector: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[KnowledgeChunk]:
        """用问题 embedding 到 Qdrant 检索相似知识。

        生产逻辑：
        1. 先确保 collection 存在。
        2. 用 `tenant_id` 做过滤，避免不同租户的数据混在一起。
        3. 用 cosine 相似度查 top_k 个最接近的问题知识块。
        4. 把 Qdrant 返回的 payload 转成项目内部的 `KnowledgeChunk`。
        """
        await self.ensure_collection()
        query_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )
        try:
            response = await self.client.query_points(
                collection_name=self._settings.qdrant_collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise ExternalServiceError(f"Qdrant search failed: {exc}") from exc

        chunks: list[KnowledgeChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunks.append(
                KnowledgeChunk(
                    id=str(point.id),
                    source=str(payload.get("source", "")),
                    title=str(payload.get("title", "")),
                    content=str(payload.get("content", "")),
                    score=float(point.score),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return chunks

    async def upsert_chunks(
        self,
        *,
        tenant_id: str,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
    ) -> None:
        """写入或更新知识库向量。

        `upsert` = update + insert：
        - id 已存在就更新
        - id 不存在就插入

        `zip(chunks, vectors, strict=True)` 会把知识片段和向量一一配对；
        `strict=True` 表示两边数量不一致时直接报错，避免数据错位。
        """
        await self.ensure_collection()
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {
                "tenant_id": tenant_id,
                "source": chunk.source,
                "title": chunk.title,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            points.append(PointStruct(id=chunk.id, vector=vector, payload=payload))

        try:
            await self.client.upsert(
                collection_name=self._settings.qdrant_collection,
                points=points,
            )
        except Exception as exc:
            raise ExternalServiceError(f"Qdrant upsert failed: {exc}") from exc
