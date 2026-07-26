from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from customer_service_app.core.config import Settings
from customer_service_app.domain.knowledge_lifecycle import (
    KnowledgeSourceDocument,
    KnowledgeSyncReport,
)
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.db.models import KnowledgeDocument
from customer_service_app.infrastructure.db.repositories import KnowledgeDocumentRepository
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient
from customer_service_app.infrastructure.knowledge_ingestion import MarkdownKnowledgeChunker
from customer_service_app.infrastructure.lexical_search.base import LexicalKnowledgeRetriever
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore


class KnowledgeLifecycleService:
    """Reconcile source documents, retrieval indexes, manifests, and semantic cache."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: KnowledgeDocumentRepository,
        chunker: MarkdownKnowledgeChunker,
        embedding_client: EmbeddingClient,
        vector_store: KnowledgeVectorStore,
        lexical_retriever: LexicalKnowledgeRetriever | None,
        semantic_cache: RedisSemanticCache | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repository
        self._chunker = chunker
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._lexical_retriever = lexical_retriever
        self._semantic_cache = semantic_cache

    async def sync(
        self,
        *,
        tenant_id: str,
        documents: list[KnowledgeSourceDocument],
        prune_missing: bool = False,
        now: datetime | None = None,
    ) -> KnowledgeSyncReport:
        """Apply changed documents and optionally remove manifests absent from the source set."""

        synchronized_at = now or datetime.now(UTC)
        report = KnowledgeSyncReport()
        supplied_sources: set[str] = set()
        for document in documents:
            supplied_sources.add(document.source)
            await self._sync_document(
                tenant_id=tenant_id,
                document=document,
                synchronized_at=synchronized_at,
                report=report,
            )

        if prune_missing:
            manifests = await self._repo.list_by_tenant(tenant_id=tenant_id)
            for manifest in manifests:
                if manifest.source in supplied_sources or manifest.status == "deleted":
                    continue
                await self._remove_indexed_chunks(
                    tenant_id=tenant_id,
                    chunk_ids=list(manifest.chunk_ids_json or []),
                    report=report,
                )
                await self._save_manifest(
                    tenant_id=tenant_id,
                    source=manifest.source,
                    chunks=[],
                    status="deleted",
                    synchronized_at=synchronized_at,
                    previous=manifest,
                )
                report.deleted.append(manifest.source)
        return report

    async def _sync_document(
        self,
        *,
        tenant_id: str,
        document: KnowledgeSourceDocument,
        synchronized_at: datetime,
        report: KnowledgeSyncReport,
    ) -> None:
        raw_chunks = self._chunker.chunk(
            text=document.text,
            source=document.source,
            document_metadata={
                **document.metadata,
                "corpus_version": document.metadata.get(
                    "corpus_version",
                    self._settings.knowledge_corpus_version,
                ),
            },
        )
        previous = await self._repo.get_by_source(
            tenant_id=tenant_id,
            source=document.source,
        )
        if not raw_chunks:
            if previous is None:
                report.skipped.append(document.source)
                return
            await self._remove_indexed_chunks(
                tenant_id=tenant_id,
                chunk_ids=list(previous.chunk_ids_json or []),
                report=report,
            )
            await self._save_manifest(
                tenant_id=tenant_id,
                source=document.source,
                chunks=[],
                status="deleted",
                synchronized_at=synchronized_at,
                previous=previous,
            )
            report.deleted.append(document.source)
            return

        chunks = [self._namespace_chunk(tenant_id, chunk) for chunk in raw_chunks]
        metadata = chunks[0].metadata
        effective_at = _parse_lifecycle_datetime(metadata.get("effective_at"))
        expires_at = _parse_lifecycle_datetime(metadata.get("expires_at"), end_of_day=True)
        status = _lifecycle_status(
            now=synchronized_at,
            effective_at=effective_at,
            expires_at=expires_at,
        )

        if status != "active":
            await self._remove_indexed_chunks(
                tenant_id=tenant_id,
                chunk_ids=list(previous.chunk_ids_json or []) if previous else [],
                report=report,
            )
            await self._save_manifest(
                tenant_id=tenant_id,
                source=document.source,
                chunks=chunks,
                status=status,
                synchronized_at=synchronized_at,
                previous=previous,
            )
            getattr(report, status).append(document.source)
            return

        chunk_ids = [chunk.id for chunk in chunks]
        if self._is_unchanged(previous=previous, chunks=chunks, chunk_ids=chunk_ids):
            report.unchanged.append(document.source)
            return

        batch_size = max(self._settings.knowledge_ingest_batch_size, 1)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = await self._embedding_client.embed_documents(
                [chunk.content for chunk in batch]
            )
            await self._vector_store.upsert_chunks(
                tenant_id=tenant_id,
                chunks=batch,
                vectors=vectors,
            )
            if self._lexical_retriever is not None:
                await self._lexical_retriever.upsert_chunks(
                    tenant_id=tenant_id,
                    chunks=batch,
                )

        old_ids = set(previous.chunk_ids_json or []) if previous else set()
        stale_ids = sorted(old_ids.difference(chunk_ids))
        await self._delete_from_indexes(tenant_id=tenant_id, chunk_ids=stale_ids)
        await self._invalidate_cache(
            tenant_id=tenant_id,
            chunk_ids=sorted(old_ids.union(chunk_ids)),
            report=report,
        )
        await self._save_manifest(
            tenant_id=tenant_id,
            source=document.source,
            chunks=chunks,
            status="active",
            synchronized_at=synchronized_at,
            previous=previous,
        )
        report.indexed.append(document.source)

    async def _remove_indexed_chunks(
        self,
        *,
        tenant_id: str,
        chunk_ids: list[str],
        report: KnowledgeSyncReport,
    ) -> None:
        await self._delete_from_indexes(tenant_id=tenant_id, chunk_ids=chunk_ids)
        await self._invalidate_cache(
            tenant_id=tenant_id,
            chunk_ids=chunk_ids,
            report=report,
        )

    async def _delete_from_indexes(self, *, tenant_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        await self._vector_store.delete_chunks(
            tenant_id=tenant_id,
            chunk_ids=chunk_ids,
        )
        if self._lexical_retriever is not None:
            await self._lexical_retriever.delete_chunks(
                tenant_id=tenant_id,
                chunk_ids=chunk_ids,
            )

    async def _invalidate_cache(
        self,
        *,
        tenant_id: str,
        chunk_ids: list[str],
        report: KnowledgeSyncReport,
    ) -> None:
        if self._semantic_cache is None or not chunk_ids:
            return
        report.invalidated_cache_entries += await self._semantic_cache.invalidate_by_chunk_ids(
            tenant_id=tenant_id,
            chunk_ids=chunk_ids,
        )

    async def _save_manifest(
        self,
        *,
        tenant_id: str,
        source: str,
        chunks: list[KnowledgeChunk],
        status: str,
        synchronized_at: datetime,
        previous: KnowledgeDocument | None,
    ) -> KnowledgeDocument:
        metadata = chunks[0].metadata if chunks else dict(previous.metadata_json if previous else {})
        document_id = str(
            metadata.get("document_id") or (previous.document_id if previous else "")
        )
        content_hash = str(
            metadata.get("document_content_hash") or (previous.content_hash if previous else "")
        )
        version = str(metadata.get("version") or (previous.version if previous else "1"))
        corpus_version = str(
            metadata.get("corpus_version")
            or (
                previous.corpus_version
                if previous
                else self._settings.knowledge_corpus_version
            )
        )
        return await self._repo.save(
            tenant_id=tenant_id,
            source=source,
            document_id=document_id,
            content_hash=content_hash,
            version=version,
            corpus_version=corpus_version,
            status=status,
            chunk_ids=[chunk.id for chunk in chunks] if status == "active" else [],
            metadata=metadata,
            effective_at=_parse_lifecycle_datetime(metadata.get("effective_at")),
            expires_at=_parse_lifecycle_datetime(metadata.get("expires_at"), end_of_day=True),
            synced_at=synchronized_at,
        )

    @staticmethod
    def _namespace_chunk(tenant_id: str, chunk: KnowledgeChunk) -> KnowledgeChunk:
        """Prevent the same source path in two tenants from sharing a storage ID."""

        storage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}:{chunk.id}"))
        return chunk.model_copy(update={"id": storage_id})

    @staticmethod
    def _is_unchanged(
        *,
        previous: KnowledgeDocument | None,
        chunks: list[KnowledgeChunk],
        chunk_ids: list[str],
    ) -> bool:
        if previous is None or previous.status != "active":
            return False
        metadata = chunks[0].metadata
        return (
            previous.content_hash == str(metadata.get("document_content_hash") or "")
            and previous.version == str(metadata.get("version") or "1")
            and previous.corpus_version
            == str(metadata.get("corpus_version") or "unversioned")
            and list(previous.chunk_ids_json or []) == chunk_ids
        )


def _parse_lifecycle_datetime(value: Any, *, end_of_day: bool = False) -> datetime | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    if "T" not in text and " " not in text:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        parsed = datetime.combine(parsed_date, time.min, tzinfo=UTC)
        if end_of_day:
            parsed += timedelta(days=1)
    else:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _lifecycle_status(
    *,
    now: datetime,
    effective_at: datetime | None,
    expires_at: datetime | None,
) -> str:
    if expires_at is not None and expires_at <= now:
        return "expired"
    if effective_at is not None and effective_at > now:
        return "scheduled"
    return "active"
