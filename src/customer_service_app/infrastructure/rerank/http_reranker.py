from __future__ import annotations

import httpx

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.domain.schemas import KnowledgeChunk


class HttpKnowledgeReranker:
    """适配 BGE/Cohere 风格 HTTP Rerank 接口。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        headers = {}
        if settings.rerank_api_key:
            headers["Authorization"] = f"Bearer {settings.rerank_api_key}"
        self._client = httpx.AsyncClient(
            timeout=settings.rerank_timeout_seconds,
            headers=headers,
        )

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[KnowledgeChunk],
        top_k: int,
    ) -> list[KnowledgeChunk]:
        if not chunks:
            return []
        try:
            response = await self._client.post(
                self._settings.rerank_base_url,
                json={
                    "model": self._settings.rerank_model,
                    "query": query,
                    "documents": [item.content for item in chunks],
                    "top_n": min(top_k, len(chunks)),
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(f"Rerank request failed: {exc}") from exc

        ranked: list[KnowledgeChunk] = []
        for item in results:
            index = int(item["index"])
            if index < 0 or index >= len(chunks):
                continue
            chunk = chunks[index].model_copy(deep=True)
            chunk.score = float(item.get("relevance_score") or item.get("score") or 0.0)
            chunk.metadata = {**chunk.metadata, "reranked": True}
            ranked.append(chunk)
        return ranked

    async def close(self) -> None:
        await self._client.aclose()
