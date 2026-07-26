from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings, get_settings
from customer_service_app.core.exceptions import AuthenticationError, PermissionDeniedError
from customer_service_app.core.security import (
    CurrentPrincipal,
    decode_access_token,
    principal_from_claims,
)
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.container import ApplicationContainer
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> CurrentPrincipal:
    """解析当前身份；关闭鉴权时返回仅用于本地开发的匿名主体。"""

    if not settings.security_enabled:
        return CurrentPrincipal(
            subject="",
            tenant_id="",
            roles=frozenset(),
            authenticated=False,
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Bearer token is required")
    claims = decode_access_token(credentials.credentials, settings)
    return principal_from_claims(claims)


def require_roles(*allowed_roles: str) -> Callable[[CurrentPrincipal], CurrentPrincipal]:
    """创建 FastAPI 角色依赖，鉴权关闭时保留本地开发兼容性。"""

    def dependency(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> CurrentPrincipal:
        if principal.authenticated and not principal.has_any_role(*allowed_roles):
            raise PermissionDeniedError("Current role is not allowed to access this resource")
        return principal

    return dependency


def get_customer_service_agent(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> CustomerServiceAgent:
    """使用请求级数据库会话构建客服 Agent。"""
    container: ApplicationContainer = request.app.state.container

    return container.build_agent(session)
