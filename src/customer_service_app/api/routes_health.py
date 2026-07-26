from __future__ import annotations

from fastapi import APIRouter

from customer_service_app.core.config import get_settings
from customer_service_app.domain.schemas import HealthResponse


router = APIRouter(tags=["health"])
"""健康检查接口路由组。"""


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    """存活检查：只说明应用进程还活着。"""
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, runtime_env=settings.runtime_env)


@router.get("/health/ready")
async def ready() -> dict:
    """就绪检查：检查关键配置是否已经填写。"""
    settings = get_settings()
    vector_provider = settings.vector_store_provider.lower()
    vector_configured = {
        "qdrant": bool(settings.qdrant_url),
        "milvus": bool(settings.milvus_uri),
    }.get(vector_provider, False)
    checks = {
        "llm_configured": bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model),
        "database_configured": bool(settings.database_url),
        "rag_configured": (
            bool(
                vector_configured
                and settings.embedding_base_url
                and settings.embedding_model
            )
            if settings.rag_enabled
            else None
        ),
        "lexical_search_configured": (
            bool(settings.opensearch_url) if settings.opensearch_enabled else None
        ),
        "redis_configured": bool(settings.redis_url) if settings.semantic_cache_enabled else None,
        "checkpoint_configured": (
            bool(settings.graph_checkpoint_postgres_url)
            if settings.graph_checkpointer == "postgres"
            else True
        ),
        "mcp_configured": (
            bool(
                settings.mcp_after_sales_url
                and settings.mcp_approval_signing_secret
            )
            if settings.mcp_after_sales_enabled
            else None
        ),
        "langsmith_configured": (
            bool(settings.langsmith_api_key and settings.langsmith_project)
            if settings.langsmith_tracing
            else None
        ),
        "security_configured": (
            bool(
                settings.jwt_secret_key
                and settings.jwt_issuer
                and settings.jwt_audience
            )
            if settings.security_enabled
            else None
        ),
        "otel_configured": (
            bool(settings.otel_exporter_otlp_endpoint) if settings.otel_enabled else None
        ),
    }
    return {"status": "ok" if all(v is not False for v in checks.values()) else "not_ready", "checks": checks}
