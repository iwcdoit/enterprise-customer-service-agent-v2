from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import get_settings
from customer_service_app.domain.schemas import (
    PendingActionSummaryView,
    RuntimeConfigView,
    TenantStrategyView,
)
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.ops_service import OpsService

router = APIRouter(tags=["ops"])

OPS_CONSOLE_PATH = Path(__file__).resolve().parents[1] / "web" / "ops_console.html"
"""运营验证台 HTML 文件路径。"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/ops", response_class=HTMLResponse, include_in_schema=False)
async def ops_console() -> HTMLResponse:
    """返回本地运营验证台页面。

    一个函数可以挂多个路由装饰器，所以 `/` 和 `/ops` 都能打开同一个页面。
    """
    return HTMLResponse(OPS_CONSOLE_PATH.read_text(encoding="utf-8"))


@router.get("/ops/api/runtime", response_model=RuntimeConfigView)
async def runtime_config() -> RuntimeConfigView:
    """Return a safe runtime configuration snapshot for the ops console."""

    return OpsService(settings=get_settings()).runtime_config()


@router.get("/ops/api/tenant-strategy", response_model=TenantStrategyView)
async def tenant_strategy(
    tenant_id: str,
    used_tokens: int | None = Query(default=None, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> TenantStrategyView:
    """Preview the selected cost strategy for one tenant.

    `used_tokens` is optional and mainly used for运营验证：传入不同 token 用量，
    可以观察 basic/standard/premium 在预算接近阈值时如何降级。
    """

    return await OpsService(
        settings=get_settings(),
        session=session,
    ).tenant_strategy(
        tenant_id=tenant_id,
        used_tokens=used_tokens,
    )


@router.get("/ops/api/pending-actions", response_model=PendingActionSummaryView)
async def pending_action_summary(
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> PendingActionSummaryView:
    """Return pending confirmation summary for one tenant/user."""

    return await OpsService(
        settings=get_settings(),
        session=session,
    ).pending_action_summary(
        tenant_id=tenant_id,
        user_id=user_id,
    )
