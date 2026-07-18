from __future__ import annotations

import json
from typing import Any

import httpx

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.domain.schemas import KnowledgeChunk


class OpenSearchBM25Retriever:
    """通过 OpenSearch BM25 检索关键词，并强制按 tenant_id 隔离。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.require("OPENSEARCH_URL", settings.opensearch_url).rstrip("/")
        self._index = settings.opensearch_index
        auth = None
        if settings.opensearch_username:
            auth = (settings.opensearch_username, settings.opensearch_password)
        headers = {"Content-Type": "application/json"}
        if settings.opensearch_api_key:
            headers["Authorization"] = f"ApiKey {settings.opensearch_api_key}"
        self._client = httpx.AsyncClient(
            timeout=settings.opensearch_timeout_seconds,
            headers=headers,
            auth=auth,
        )

    async def ensure_index(self) -> None:
        """创建包含租户、标题、正文和结构元数据的知识索引。"""

        try:
            response = await self._client.head(f"{self._base_url}/{self._index}")
            if response.status_code == 200:
                return
            if response.status_code != 404:
                self._raise(response, "check OpenSearch index")
            response = await self._client.put(
                f"{self._base_url}/{self._index}",
                json={
                    "mappings": {
                        "dynamic": "strict",
                        "properties": {
                            "tenant_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "document_type": {"type": "keyword"},
                            "heading_path": {"type": "keyword"},
                            "title": {
                                "type": "text",
                                "analyzer": self._settings.opensearch_analyzer,
                            },
                            "content": {
                                "type": "text",
                                "analyzer": self._settings.opensearch_analyzer,
                            },
                            "metadata": {"type": "object", "enabled": False},
                        },
                    }
                },
            )
            if response.status_code not in {200, 201}:
                self._raise(response, "create OpenSearch index")
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"OpenSearch index initialization failed: {exc}") from exc

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        top_k: int,
    ) -> list[KnowledgeChunk]:
        """查询标题和正文，标题命中的 BM25 权重更高。"""

        await self.ensure_index()
        try:
            response = await self._client.post(
                f"{self._base_url}/{self._index}/_search",
                json={
                    "size": top_k,
                    "track_total_hits": False,
                    "query": {
                        "bool": {
                            "filter": [{"term": {"tenant_id": tenant_id}}],
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["title^2.5", "content"],
                                        "type": "best_fields",
                                    }
                                }
                            ],
                        }
                    },
                },
            )
            if response.status_code != 200:
                self._raise(response, "search OpenSearch")
            hits = response.json().get("hits", {}).get("hits", [])
            return [self._to_chunk(hit) for hit in hits]
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(f"OpenSearch search failed: {exc}") from exc

    async def upsert_chunks(
        self,
        *,
        tenant_id: str,
        chunks: list[KnowledgeChunk],
    ) -> None:
        """使用和向量库相同的 chunk_id 批量写入 BM25 索引。"""

        if not chunks:
            return
        await self.ensure_index()
        lines: list[str] = []
        for chunk in chunks:
            lines.append(json.dumps({"index": {"_index": self._index, "_id": chunk.id}}))
            lines.append(
                json.dumps(
                    {
                        "tenant_id": tenant_id,
                        "source": chunk.source,
                        "document_type": str(chunk.metadata.get("document_type") or "knowledge"),
                        "heading_path": chunk.metadata.get("heading_path") or [],
                        "title": chunk.title,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
            )
        try:
            response = await self._client.post(
                f"{self._base_url}/_bulk?refresh=wait_for",
                content="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
            payload = response.json()
            if response.status_code != 200 or payload.get("errors"):
                self._raise(response, "bulk upsert OpenSearch")
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(f"OpenSearch bulk upsert failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _to_chunk(hit: dict[str, Any]) -> KnowledgeChunk:
        source = hit.get("_source") or {}
        return KnowledgeChunk(
            id=str(hit.get("_id") or ""),
            source=str(source.get("source") or ""),
            title=str(source.get("title") or ""),
            content=str(source.get("content") or ""),
            score=float(hit.get("_score") or 0.0),
            metadata={**(source.get("metadata") or {}), "retriever": "bm25"},
        )

    @staticmethod
    def _raise(response: httpx.Response, operation: str) -> None:
        raise ExternalServiceError(
            f"Failed to {operation}: HTTP {response.status_code} {response.text[:300]}"
        )
