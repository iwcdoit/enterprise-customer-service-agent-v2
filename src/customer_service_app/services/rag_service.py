from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore


class RagService:
    """RAG 检索服务。

    它负责在调用大模型之前，先根据用户问题去知识库找相关资料。
    注意：RAG 是应用代码主动检索资料，不是大模型自己去查数据库。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_client: EmbeddingClient,
        vector_store: KnowledgeVectorStore,
    ):
        """保存 RAG 所需依赖。

        embedding_client 用于把用户问题变成向量；
        vector_store 用于根据向量从 Qdrant、Milvus 等向量库中找相似知识块。
        """

        self._settings = settings
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    async def retrieve(
        self,
        *,
        tenant_id: str,
        question: str,
        top_k: int | None = None,
    ) -> list[KnowledgeChunk]:
        """根据用户问题检索知识库。

        流程是：问题文本 -> embedding 向量 -> 向量库相似度检索 -> KnowledgeChunk。
        如果配置关闭了 RAG，就直接返回空列表。
        """

        if not self._settings.rag_enabled:
            return []

        query_vector = await self._embedding_client.embed_query(question)
        return await self._vector_store.search(
            tenant_id=tenant_id,
            query_vector=query_vector,
            top_k=top_k or self._settings.rag_top_k,
            score_threshold=self._settings.rag_score_threshold,
        )
