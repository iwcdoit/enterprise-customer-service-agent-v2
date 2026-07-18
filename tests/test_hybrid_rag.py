from __future__ import annotations

from typing import Any

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.services.rag_service import RagService


def _chunk(chunk_id: str, score: float, source: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        source=source,
        title=source,
        content=f"{source} content",
        score=score,
    )


class FakeEmbeddingClient:
    async def embed_query(self, _: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, chunks: list[KnowledgeChunk] | None = None, error: Exception | None = None):
        self.chunks = chunks or []
        self.error = error
        self.calls = 0

    async def search(self, **_: Any) -> list[KnowledgeChunk]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.chunks


class FakeLexicalRetriever:
    def __init__(self, chunks: list[KnowledgeChunk] | None = None, error: Exception | None = None):
        self.chunks = chunks or []
        self.error = error
        self.calls = 0

    async def search(self, **_: Any) -> list[KnowledgeChunk]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.chunks


class ReverseReranker:
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[KnowledgeChunk],
        top_k: int,
    ) -> list[KnowledgeChunk]:
        del query
        return list(reversed(chunks))[:top_k]


def _settings(**updates: Any) -> Settings:
    return Settings(
        rag_enabled=True,
        retrieval_mode="hybrid",
        hybrid_rrf_k=60,
        hybrid_dense_weight=1.0,
        hybrid_lexical_weight=1.0,
        **updates,
    )


async def test_hybrid_rag_rrf_favors_chunk_recalled_by_both_routes() -> None:
    vector = FakeVectorStore(
        [_chunk("dense-only", 0.95, "dense.md"), _chunk("both", 0.80, "both.md")]
    )
    lexical = FakeLexicalRetriever(
        [_chunk("both", 9.2, "both.md"), _chunk("bm25-only", 8.5, "bm25.md")]
    )
    service = RagService(
        settings=_settings(),
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_retriever=lexical,  # type: ignore[arg-type]
    )

    result = await service.retrieve(tenant_id="tenant-a", question="七天退货规则", top_k=3)

    assert result[0].id == "both"
    assert result[0].metadata["retrievers"] == ["bm25", "dense"]
    assert {item.id for item in result} == {"both", "dense-only", "bm25-only"}


async def test_hybrid_rag_can_apply_optional_reranker() -> None:
    service = RagService(
        settings=_settings(),
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_store=FakeVectorStore(  # type: ignore[arg-type]
            [_chunk("first", 0.9, "first.md"), _chunk("second", 0.8, "second.md")]
        ),
        lexical_retriever=FakeLexicalRetriever([]),  # type: ignore[arg-type]
        reranker=ReverseReranker(),  # type: ignore[arg-type]
    )

    result = await service.retrieve(
        tenant_id="tenant-a",
        question="退款条件",
        top_k=1,
        use_rerank=True,
    )

    assert [item.id for item in result] == ["second"]


async def test_hybrid_rag_skips_knowledge_for_realtime_order_lookup() -> None:
    vector = FakeVectorStore([_chunk("policy", 0.9, "policy.md")])
    lexical = FakeLexicalRetriever([_chunk("policy", 9.0, "policy.md")])
    service = RagService(
        settings=_settings(),
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_store=vector,  # type: ignore[arg-type]
        lexical_retriever=lexical,  # type: ignore[arg-type]
    )

    result = await service.retrieve(
        tenant_id="tenant-a",
        question="帮我查订单202607180001现在的物流状态",
    )

    assert result == []
    assert vector.calls == 0
    assert lexical.calls == 0


async def test_hybrid_rag_degrades_to_dense_when_bm25_fails() -> None:
    service = RagService(
        settings=_settings(),
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_store=FakeVectorStore([_chunk("dense", 0.9, "dense.md")]),  # type: ignore[arg-type]
        lexical_retriever=FakeLexicalRetriever(error=RuntimeError("search unavailable")),  # type: ignore[arg-type]
    )

    result = await service.retrieve(tenant_id="tenant-a", question="退货政策")

    assert [item.id for item in result] == ["dense"]


async def test_hybrid_rag_raises_when_all_retrieval_routes_fail() -> None:
    service = RagService(
        settings=_settings(),
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_store=FakeVectorStore(error=RuntimeError("vector unavailable")),  # type: ignore[arg-type]
        lexical_retriever=FakeLexicalRetriever(error=RuntimeError("search unavailable")),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError):
        await service.retrieve(tenant_id="tenant-a", question="退货政策")
