from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ConfigurationError
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient
from customer_service_app.infrastructure.embeddings.ollama import OllamaEmbeddingClient
from customer_service_app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingClient,
)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """根据配置创建 Embedding 客户端。"""
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
