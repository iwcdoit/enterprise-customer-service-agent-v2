from __future__ import annotations

import httpx

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client for a local-compatible HTTP embedding endpoint."""

    def __init__(self, settings: Settings):
        """保存配置，请求时用 httpx 创建临时 HTTP 客户端。"""
        self._settings = settings

    async def embed_query(self, text: str) -> list[float]:
        """把单个问题转成向量。"""
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量调用 embedding 接口。

        `async with httpx.AsyncClient(...) as client` 是异步上下文管理器，
        类似 Java `try-with-resources`：代码块结束后自动释放连接资源。
        """
        base_url = self._settings.require("EMBEDDING_BASE_URL", self._settings.embedding_base_url)
        model = self._settings.require("EMBEDDING_MODEL", self._settings.embedding_model)
        try:
            async with httpx.AsyncClient(timeout=self._settings.embedding_timeout_seconds) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}/api/embed",
                    json={"model": model, "input": texts},
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _safe_error_detail(exc.response)
            if "does not support embeddings" in detail:
                raise ExternalServiceError(
                    "Configured embedding model does not support vector generation. "
                    f"Current EMBEDDING_MODEL={model!r}."
                ) from exc
            raise ExternalServiceError(
                f"Embedding request failed: {exc}. response={detail}"
            ) from exc
        except Exception as exc:
            raise ExternalServiceError(f"Embedding request failed: {exc}") from exc
        return payload["embeddings"]


def _safe_error_detail(response: httpx.Response) -> str:
    """尽量从响应里提取可读的错误信息。"""
    try:
        return str(response.json().get("error", ""))
    except Exception:
        return response.text[:300]
