from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from after_sales_mcp.database import Order, SupportTicket


class AfterSalesRepository:
    """Database adapter hidden behind MCP tools."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_order(
        self,
        *,
        tenant_id: str,
        user_id: str,
        order_id: str,
    ) -> Order | None:
        result = await self._session.execute(
            select(Order).where(
                Order.tenant_id == tenant_id,
                Order.user_id == user_id,
                Order.order_id == order_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_ticket(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        category: str,
        title: str,
        detail: str,
        priority: str,
        metadata: dict[str, Any],
        idempotency_key: str,
    ) -> SupportTicket:
        existing = await self._session.execute(
            select(SupportTicket).where(
                SupportTicket.idempotency_key == idempotency_key
            )
        )
        ticket = existing.scalar_one_or_none()
        if ticket is not None:
            return ticket
        ticket = SupportTicket(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category=category,
            title=title,
            detail=detail,
            priority=priority,
            metadata_json=metadata,
            idempotency_key=idempotency_key,
        )
        self._session.add(ticket)
        await self._session.flush()
        return ticket
