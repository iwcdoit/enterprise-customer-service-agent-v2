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
    """FastAPI 依赖：为每个请求组装一个客服 Agent。

    `Depends(get_db_session)` 表示这个参数由 FastAPI 自动注入，
    类似 Spring 里通过容器注入 Bean，只是 FastAPI 更偏函数式。
    """
    container: ApplicationContainer = request.app.state.container
    return container.build_agent(session)
