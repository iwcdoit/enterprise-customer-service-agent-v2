from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ConfigurationError
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient
from customer_service_app.infrastructure.embeddings.ollama import OllamaEmbeddingClient
from customer_service_app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingClient,
)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """根据配置创建 Embedding 客户端。

    这是工厂方法，类似 Java 里常见的 `EmbeddingClientFactory.create(settings)`。
    上层业务只依赖 `EmbeddingClient` 接口，不需要关心底层模型服务供应商。
    """
    provider = settings.embedding_provider.lower()
    if provider == "ollama":
        return OllamaEmbeddingClient(settings)
    if provider in {
        "openai_compatible",
        "deepseek",
        "openai",
        "dashscope",
        "bailian",
        "aliyun",
    }:
        return OpenAICompatibleEmbeddingClient(settings)
    raise ConfigurationError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
