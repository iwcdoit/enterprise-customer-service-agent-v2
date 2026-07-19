from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.container import ApplicationContainer
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


def get_customer_service_agent(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> CustomerServiceAgent:
    """使用请求级数据库会话构建客服 Agent。"""
    container: ApplicationContainer = request.app.state.container

    return container.build_agent(session)
