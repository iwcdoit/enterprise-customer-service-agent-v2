from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.domain.knowledge_lifecycle import KnowledgeSourceDocument
from customer_service_app.infrastructure.knowledge_ingestion import (
    ChunkingConfig,
    MarkdownKnowledgeChunker,
)
from customer_service_app.services.knowledge_lifecycle_service import (
    KnowledgeLifecycleService,
    _parse_lifecycle_datetime,
)


class FakeKnowledgeRepository:
    def __init__(self) -> None:
        self.documents: dict[tuple[str, str], SimpleNamespace] = {}

    async def get_by_source(self, *, tenant_id: str, source: str):
        return self.documents.get((tenant_id, source))

    async def list_by_tenant(self, *, tenant_id: str):
        return [
            value
            for (stored_tenant, _), value in self.documents.items()
            if stored_tenant == tenant_id
        ]

    async def save(self, **values):
        document = self.documents.get((values["tenant_id"], values["source"]))
        if document is None:
            document = SimpleNamespace()
        for key, value in values.items():
            if key == "metadata":
                key = "metadata_json"
            elif key == "chunk_ids":
                key = "chunk_ids_json"
            setattr(document, key, value)
        self.documents[(values["tenant_id"], values["source"])] = document
        return document


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[list[str]] = []
        self.deletes: list[list[str]] = []

    async def upsert_chunks(self, *, tenant_id: str, chunks, vectors=None) -> None:
        self.upserts.append([chunk.id for chunk in chunks])

    async def delete_chunks(self, *, tenant_id: str, chunk_ids: list[str]) -> None:
        self.deletes.append(list(chunk_ids))


class FakeSemanticCache:
    def __init__(self) -> None:
        self.invalidations: list[list[str]] = []

    async def invalidate_by_chunk_ids(self, *, tenant_id: str, chunk_ids: list[str]) -> int:
        self.invalidations.append(list(chunk_ids))
        return 1 if chunk_ids else 0


def _service():
    repository = FakeKnowledgeRepository()
    embedding = FakeEmbeddingClient()
    vector = FakeIndex()
    lexical = FakeIndex()
    cache = FakeSemanticCache()
    service = KnowledgeLifecycleService(
        settings=Settings(knowledge_corpus_version="release-1", knowledge_ingest_batch_size=10),
        repository=repository,
        chunker=MarkdownKnowledgeChunker(
            ChunkingConfig(max_chars=300, min_chars=20, overlap_chars=30)
        ),
        embedding_client=embedding,
        vector_store=vector,
        lexical_retriever=lexical,
        semantic_cache=cache,
    )
    return service, repository, embedding, vector, lexical, cache


@pytest.mark.asyncio
async def test_sync_skips_unchanged_document_and_reindexes_changed_content() -> None:
    service, repository, embedding, vector, lexical, cache = _service()
    source = "policies/refund.md"
    first = KnowledgeSourceDocument(
        source=source,
        text="# 退款政策\n\n商品签收七天内可以申请退货。",
    )

    first_report = await service.sync(
        tenant_id="tenant-a",
        documents=[first],
        now=datetime(2026, 7, 1, tzinfo=UTC),
    )
    unchanged_report = await service.sync(
        tenant_id="tenant-a",
        documents=[first],
        now=datetime(2026, 7, 2, tzinfo=UTC),
    )
    changed_report = await service.sync(
        tenant_id="tenant-a",
        documents=[
            KnowledgeSourceDocument(
                source=source,
                text="# 退款政策\n\n商品签收七天内可以申请退货，质量问题免运费。",
            )
        ],
        now=datetime(2026, 7, 3, tzinfo=UTC),
    )

    assert first_report.indexed == [source]
    assert unchanged_report.unchanged == [source]
    assert changed_report.indexed == [source]
    assert embedding.calls == 2
    assert len(vector.upserts) == 2
    assert len(lexical.upserts) == 2
    assert cache.invalidations
    assert repository.documents[("tenant-a", source)].status == "active"


@pytest.mark.asyncio
async def test_expired_and_missing_documents_are_removed_from_both_indexes() -> None:
    service, repository, _, vector, lexical, _ = _service()
    source = "announcements/old.md"
    await service.sync(
        tenant_id="tenant-a",
        documents=[
            KnowledgeSourceDocument(
                source=source,
                text="# 活动公告\n\n活动期间可以申请价保。",
            )
        ],
        now=datetime(2026, 7, 1, tzinfo=UTC),
    )
    old_ids = list(repository.documents[("tenant-a", source)].chunk_ids_json)

    expired = await service.sync(
        tenant_id="tenant-a",
        documents=[
            KnowledgeSourceDocument(
                source=source,
                text=(
                    "---\nexpires_at: 2026-07-01\n---\n"
                    "# 活动公告\n\n活动期间可以申请价保。"
                ),
            )
        ],
        now=datetime(2026, 7, 3, tzinfo=UTC),
    )
    pruned = await service.sync(
        tenant_id="tenant-a",
        documents=[],
        prune_missing=True,
        now=datetime(2026, 7, 4, tzinfo=UTC),
    )

    assert expired.expired == [source]
    assert old_ids in vector.deletes
    assert old_ids in lexical.deletes
    assert pruned.deleted == [source]
    assert repository.documents[("tenant-a", source)].status == "deleted"


def test_storage_chunk_ids_are_namespaced_by_tenant() -> None:
    service, _, _, _, _, _ = _service()
    chunk = service._chunker.chunk(
        text="# 物流\n\n订单发货后可以查询物流。",
        source="guide/logistics.md",
    )[0]

    tenant_a = service._namespace_chunk("tenant-a", chunk)
    tenant_b = service._namespace_chunk("tenant-b", chunk)

    assert tenant_a.id != tenant_b.id


def test_date_only_expiration_remains_valid_through_declared_day() -> None:
    expires_at = _parse_lifecycle_datetime("2026-07-01", end_of_day=True)

    assert expires_at == datetime(2026, 7, 2, tzinfo=UTC)
