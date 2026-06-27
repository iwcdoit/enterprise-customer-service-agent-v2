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
    checks = {
        "llm_configured": bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model),
        "database_configured": bool(settings.database_url),
        "rag_configured": bool(settings.qdrant_url and settings.embedding_base_url and settings.embedding_model),
        "redis_configured": bool(settings.redis_url) if settings.semantic_cache_enabled else None,
    }
    return {"status": "ok" if all(v is not False for v in checks.values()) else "not_ready", "checks": checks}
