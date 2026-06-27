from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from customer_service_app.core.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Enterprise Customer Service"
    runtime_env: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    allowed_origins: str = ""

    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 60

    embedding_provider: str = "openai_compatible"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 1024
    embedding_timeout_seconds: int = 60

    database_url: str = ""

    semantic_cache_enabled: bool = False
    redis_url: str = ""
    semantic_cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.90

    rag_enabled: bool = True
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "customer_service_knowledge"
    rag_top_k: int = 5
    rag_score_threshold: float = 0.35

    serpapi_key: str = ""
    search_result_count: int = 5

    mcp_after_sales_enabled: bool = False
    mcp_after_sales_transport: str = "streamable_http"
    mcp_after_sales_url: str = "https://mcp-after-sales.example.com/mcp"
    mcp_after_sales_command: str = "python"
    mcp_after_sales_args: str = "scripts/run_after_sales_mcp_server.py"
    mcp_timeout_seconds: int = 20
    mcp_approval_signing_secret: str = ""
    mcp_approval_issuer: str = "customer-service-agent"
    mcp_approval_token_ttl_seconds: int = 600

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @property
    def cors_origins(self) -> list[str]:
        """把逗号分隔的跨域白名单字符串转成列表。"""
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    def require(self, name: str, value: str | None) -> str:
        """读取必填配置；为空就抛出清晰错误。"""
        if not value or not value.strip():
            raise ConfigurationError(f"Missing required configuration: {name}")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
