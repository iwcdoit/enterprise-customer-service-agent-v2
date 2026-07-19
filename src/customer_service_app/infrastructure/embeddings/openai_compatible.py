from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Embedding client for OpenAI-compatible providers."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Create the embedding HTTP client lazily."""

        if self._client is None:
            api_key = self._settings.embedding_api_key or self._settings.llm_api_key
            self._settings.require("EMBEDDING_API_KEY or LLM_API_KEY", api_key)
            base_url = self._settings.require(
                "EMBEDDING_BASE_URL",
                self._settings.embedding_base_url,
            )
            self._settings.require("EMBEDDING_MODEL", self._settings.embedding_model)
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._settings.embedding_timeout_seconds,
            )
        return self._client

    async def embed_query(self, text: str) -> list[float]:
        """Turn one query text into one vector."""

        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本，输出顺序与输入顺序一致。"""

        payload = self._request_payload(texts)
        try:
            response = await self.client.embeddings.create(**payload)
        except Exception as exec:
            logging.info("embed_documents error: %s", exec)
            raise ExternalServiceError(f"llm stream request error: {exec}") from exec
        return [item.embedding for item in response.data]

    def _request_payload(self, texts: list[str]) -> dict[str, Any]:
        """Build provider-specific embedding request payload."""

        payload: dict[str, Any] = {
            "model": self._settings.embedding_model,
            "input": texts,
        }
        provider = self._settings.embedding_provider.lower()
        model = self._settings.embedding_model.lower()
        if provider in {"dashscope", "bailian", "aliyun"} or model.startswith(
            "text-embedding-v4"
        ):
            payload["dimensions"] = self._settings.embedding_dimension
        return payload

    async def close(self) -> None:
        """Close the shared embedding HTTP client."""

        if self._client is not None:
            await self._client.close()
