from __future__ import annotations

import asyncio
import logging
import re

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient
from customer_service_app.infrastructure.lexical_search.base import LexicalKnowledgeRetriever
from customer_service_app.infrastructure.rerank.base import KnowledgeReranker
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore


class RagService:
    """租户隔离的向量/BM25 混合检索，支持 RRF 融合和可选 Reranker。"""

    _realtime_words = ("订单状态", "物流到哪", "快递到哪", "退款进度", "工单进度")
    _policy_words = ("规则", "政策", "条件", "期限", "是否可以", "怎么处理", "流程")

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_client: EmbeddingClient,
        vector_store: KnowledgeVectorStore,
        lexical_retriever: LexicalKnowledgeRetriever | None = None,
        reranker: KnowledgeReranker | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._lexical_retriever = lexical_retriever
        self._reranker = reranker

    async def retrieve(
        self,
        *,
        tenant_id: str,
        question: str,
        top_k: int | None = None,
        use_rerank: bool = False,
    ) -> list[KnowledgeChunk]:
        """并发召回两路候选，融合后只把最终 top_k 交给大模型。"""

        if not self._settings.rag_enabled:
            return []
        resolved_top_k = top_k or self._settings.rag_top_k
        mode = self._route(question)
        if mode == "none":
            return []

        candidate_k = max(
            resolved_top_k,
            resolved_top_k * max(self._settings.hybrid_candidate_multiplier, 1),
        )
        dense_task = None
        lexical_task = None
        if mode in {"dense", "hybrid"}:
            dense_task = asyncio.create_task(
                self._dense_search(
                    tenant_id=tenant_id,
                    question=question,
                    top_k=candidate_k,
                )
            )
        if mode in {"lexical", "hybrid"} and self._lexical_retriever is not None:
            lexical_task = asyncio.create_task(
                self._lexical_retriever.search(
                    tenant_id=tenant_id,
                    query=question,
                    top_k=candidate_k,
                )
            )

        tasks = [task for task in (dense_task, lexical_task) if task is not None]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dense: list[KnowledgeChunk] = []
        lexical: list[KnowledgeChunk] = []
        result_index = 0
        if dense_task is not None:
            value = results[result_index]
            result_index += 1
            if isinstance(value, Exception):
                logging.warning("dense retrieval degraded: %s", value)
            else:
                dense = value
        if lexical_task is not None:
            value = results[result_index]
            if isinstance(value, Exception):
                logging.warning("BM25 retrieval degraded: %s", value)
            else:
                lexical = value

        # 两路都失败时必须显式报错；只有一路失败时使用另一条结果降级返回。
        if not dense and not lexical and any(isinstance(item, Exception) for item in results):
            raise next(item for item in results if isinstance(item, Exception))

        candidates = self._rrf(dense=dense, lexical=lexical)
        if use_rerank and self._reranker is not None and candidates:
            try:
                return await self._reranker.rerank(
                    query=question,
                    chunks=candidates,
                    top_k=resolved_top_k,
                )
            except Exception as exc:
                # Reranker 是质量增强，不应成为客服链路单点。
                logging.warning("rerank degraded to RRF order: %s", exc)
        return candidates[:resolved_top_k]

    async def _dense_search(
        self,
        *,
        tenant_id: str,
        question: str,
        top_k: int,
    ) -> list[KnowledgeChunk]:
        query_vector = await self._embedding_client.embed_query(question)
        chunks = await self._vector_store.search(
            tenant_id=tenant_id,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=self._settings.rag_score_threshold,
        )
        for item in chunks:
            item.metadata = {**item.metadata, "retriever": "dense"}
        return chunks

    def _route(self, question: str) -> str:
        """实时业务状态交给工具，稳定政策知识才进入 RAG。"""

        value = question.strip()
        # 不用 `\b`：Python 会把中文也当作单词字符，“订单202607180001现在”会匹配失败。
        has_business_id = bool(
            re.search(
                r"(?<![A-Za-z0-9])(?:[A-Za-z]{2,8}-?)?\d{8,24}(?![A-Za-z0-9])",
                value,
            )
        )
        realtime_query = any(word in value for word in self._realtime_words) or (
            any(word in value for word in ("订单", "物流", "退款", "工单"))
            and any(word in value for word in ("查", "查询", "状态", "进度", "到哪"))
        )
        if has_business_id and realtime_query and not any(
            word in value for word in self._policy_words
        ):
            return "none"

        configured = self._settings.retrieval_mode.lower()
        if configured in {"lexical", "hybrid"} and self._lexical_retriever is None:
            return "dense"
        if configured not in {"dense", "lexical", "hybrid"}:
            return "dense"
        return configured

    def _rrf(
        self,
        *,
        dense: list[KnowledgeChunk],
        lexical: list[KnowledgeChunk],
    ) -> list[KnowledgeChunk]:
        """用排名融合而非原始分数融合，避免 BM25 与余弦分数尺度不一致。"""

        rrf_k = max(self._settings.hybrid_rrf_k, 1)
        scores: dict[str, float] = {}
        chunks: dict[str, KnowledgeChunk] = {}
        sources: dict[str, set[str]] = {}
        rankings = (
            (dense, self._settings.hybrid_dense_weight, "dense"),
            (lexical, self._settings.hybrid_lexical_weight, "bm25"),
        )
        for ranking, weight, source in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                key = chunk.id or f"{chunk.source}:{chunk.title}:{hash(chunk.content)}"
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
                chunks.setdefault(key, chunk.model_copy(deep=True))
                sources.setdefault(key, set()).add(source)

        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        result: list[KnowledgeChunk] = []
        for key in ordered:
            chunk = chunks[key]
            chunk.score = scores[key]
            chunk.metadata = {
                **chunk.metadata,
                "retrievers": sorted(sources[key]),
                "fusion": "rrf",
            }
            result.append(chunk)
        return result
