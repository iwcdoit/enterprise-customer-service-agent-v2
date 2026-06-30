from __future__ import annotations

import asyncio
import json
from typing import Any

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore


class MilvusKnowledgeVectorStore(KnowledgeVectorStore):
    """Milvus implementation for larger-scale RAG knowledge storage."""

    _VECTOR_FIELD = "vector"
    _MAX_ID_LENGTH = 128
    _MAX_TENANT_LENGTH = 128
    _MAX_SOURCE_LENGTH = 512
    _MAX_TITLE_LENGTH = 512
    _MAX_CONTENT_LENGTH = 65535
    _MAX_METADATA_LENGTH = 8192

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Return a lazily initialized Milvus client."""
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise ExternalServiceError("Python package `pymilvus` is required") from exc

            uri = self._settings.require("MILVUS_URI", self._settings.milvus_uri)
            self._client = MilvusClient(
                uri=uri,
                token=self._settings.milvus_token or None,
            )
        return self._client

    async def ensure_collection(self) -> None:
        """Ensure the configured Milvus collection exists and is loaded."""
        try:
            exists = await asyncio.to_thread(
                self.client.has_collection,
                collection_name=self._settings.milvus_collection,
            )
            if not exists:
                await asyncio.to_thread(self._create_collection)
            else:
                await asyncio.to_thread(
                    self.client.load_collection,
                    collection_name=self._settings.milvus_collection,
                )
        except Exception as exc:
            raise ExternalServiceError(f"Milvus collection initialization failed: {exc}") from exc

    def _create_collection(self) -> None:
        """Create Milvus schema and vector index."""
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self._MAX_ID_LENGTH,
        )
        schema.add_field(
            field_name="tenant_id",
            datatype=DataType.VARCHAR,
            max_length=self._MAX_TENANT_LENGTH,
        )
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=self._MAX_SOURCE_LENGTH,
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=self._MAX_TITLE_LENGTH,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=self._MAX_CONTENT_LENGTH,
        )
        schema.add_field(
            field_name="metadata_json",
            datatype=DataType.VARCHAR,
            max_length=self._MAX_METADATA_LENGTH,
        )
        schema.add_field(
            field_name=self._VECTOR_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=self._settings.embedding_dimension,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self._VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self._settings.milvus_collection,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name=self._settings.milvus_collection)

    async def search(
        self,
        *,
        tenant_id: str,
        query_vector: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[KnowledgeChunk]:
        """Search relevant knowledge chunks from Milvus with tenant isolation."""
        await self.ensure_collection()
        try:
            result_sets = await asyncio.to_thread(
                self.client.search,
                collection_name=self._settings.milvus_collection,
                data=[query_vector],
                anns_field=self._VECTOR_FIELD,
                limit=top_k,
                filter=f"tenant_id == {json.dumps(tenant_id)}",
                output_fields=["id", "source", "title", "content", "metadata_json"],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as exc:
            raise ExternalServiceError(f"Milvus search failed: {exc}") from exc

        chunks: list[KnowledgeChunk] = []
        for hit in _first_result_set(result_sets):
            entity = _hit_entity(hit)
            score = _hit_score(hit)
            if score < score_threshold:
                continue
            chunks.append(
                KnowledgeChunk(
                    id=_hit_id(hit, entity),
                    source=str(entity.get("source", "")),
                    title=str(entity.get("title", "")),
                    content=str(entity.get("content", "")),
                    score=score,
                    metadata=_metadata_from_json(entity.get("metadata_json")),
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
        """Insert or update knowledge chunks and vectors in Milvus."""
        await self.ensure_collection()
        rows: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows.append(
                {
                    "id": _truncate(chunk.id, self._MAX_ID_LENGTH),
                    "tenant_id": _truncate(tenant_id, self._MAX_TENANT_LENGTH),
                    "source": _truncate(chunk.source, self._MAX_SOURCE_LENGTH),
                    "title": _truncate(chunk.title, self._MAX_TITLE_LENGTH),
                    "content": _truncate(chunk.content, self._MAX_CONTENT_LENGTH),
                    "metadata_json": _truncate(
                        json.dumps(chunk.metadata, ensure_ascii=False, default=str),
                        self._MAX_METADATA_LENGTH,
                    ),
                    self._VECTOR_FIELD: vector,
                }
            )

        try:
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self._settings.milvus_collection,
                data=rows,
            )
        except Exception as exc:
            raise ExternalServiceError(f"Milvus upsert failed: {exc}") from exc


def _first_result_set(result_sets: Any) -> list[Any]:
    if not result_sets:
        return []
    first = result_sets[0]
    return list(first) if first else []


def _hit_entity(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity") or {}
        return dict(entity) if isinstance(entity, dict) else {}
    entity = getattr(hit, "entity", {}) or {}
    return dict(entity) if isinstance(entity, dict) else {}


def _hit_id(hit: Any, entity: dict[str, Any]) -> str:
    if isinstance(hit, dict):
        return str(hit.get("id") or entity.get("id") or "")
    return str(getattr(hit, "id", None) or entity.get("id") or "")


def _hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit.get("distance", hit.get("score", 0.0)) or 0.0)
    return float(getattr(hit, "distance", getattr(hit, "score", 0.0)) or 0.0)


def _metadata_from_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truncate(value: str, max_length: int) -> str:
    return value if len(value) <= max_length else value[:max_length]
