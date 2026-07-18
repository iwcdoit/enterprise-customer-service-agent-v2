from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """OpenAI 兼容格式的 Embedding 客户端。

    这里虽然类名带 OpenAI，但它调用的是“OpenAI 兼容协议”：
    阿里百炼 DashScope、部分国产大模型平台也支持类似的 HTTP 格式。
    """

    def __init__(self, settings: Settings):
        """构造方法：保存配置，客户端稍后按需创建。"""
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """懒加载 OpenAI 兼容客户端。

        这里优先读取 `EMBEDDING_API_KEY`，如果没填就复用 `LLM_API_KEY`。
        这样阿里百炼这类平台可以只配一套 key。
        """
        if self._client is None:
            api_key = self._settings.embedding_api_key or self._settings.llm_api_key
            self._settings.require("EMBEDDING_API_KEY or LLM_API_KEY", api_key)
            base_url = self._settings.require("EMBEDDING_BASE_URL", self._settings.embedding_base_url)
            self._settings.require("EMBEDDING_MODEL", self._settings.embedding_model)
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._settings.embedding_timeout_seconds,
            )
        return self._client

    async def embed_query(self, text: str) -> list[float]:
        """把一个问题转成向量。

        这里复用 `embed_documents([text])`，因为很多平台单条和批量接口是同一个。
        """
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量调用 embedding 接口，把文本列表转成向量列表。"""
        try:
            response = await self.client.embeddings.create(**self._request_payload(texts))
        except Exception as exc:
            raise ExternalServiceError(f"Embedding request failed: {exc}") from exc
        return [item.embedding for item in response.data]

    def _request_payload(self, texts: list[str]) -> dict[str, Any]:
        """Build provider-specific embedding request payload."""
        payload: dict[str, Any] = {
            "model": self._settings.embedding_model,
            "input": texts,
        }
        provider = self._settings.embedding_provider.lower()
        model = self._settings.embedding_model.lower()
        if provider in {"dashscope", "bailian", "aliyun"} or model.startswith("text-embedding-v4"):
            payload["dimensions"] = self._settings.embedding_dimension
        return payload

    async def close(self) -> None:
        """释放 Embedding HTTP 连接池。"""

        if self._client is not None:
            await self._client.close()
