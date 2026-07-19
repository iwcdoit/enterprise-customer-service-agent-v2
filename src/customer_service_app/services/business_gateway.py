from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.infrastructure.mcp.base import MCPBusinessClient
from customer_service_app.infrastructure.db.repositories import OrderRepository, TicketRepository
from customer_service_app.services.human_support_service import HumanSupportService


class BusinessGateway:
    """Boundary between Agent orchestration and business systems.

    The current implementation uses local database repositories so the project can run locally.
    In production, these methods can call HTTP services, RPC services, or MCP server tools.
    """

    def __init__(
        self,
        session: AsyncSession,
        mcp_client: MCPBusinessClient | None = None,
        human_support_service: HumanSupportService | None = None,
    ):
        self._session = session
        self._mcp_client = mcp_client
        self._human_support_service = human_support_service or HumanSupportService(session)

    async def query_order_status(
        self,
        *,
        tenant_id: str,
        user_id: str,
        order_id: str,
    ) -> dict[str, Any]:
        """Query an order that belongs to the current tenant and user."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="query_order_status",
                arguments={"tenant_id": tenant_id, "user_id": user_id, "order_id": order_id},
            )
        order = await OrderRepository(self._session).get_by_order_id(
            tenant_id=tenant_id,
            user_id=user_id,
            order_id=order_id,
        )
        if order is None:
            return {"found": False, "order_id": order_id, "message": "未找到该用户名下的订单"}
        return {
            "found": True,
            "order_id": order.order_id,
            "status": order.status,
            "logistics_company": order.logistics_company,
            "tracking_number": order.tracking_number,
            "metadata": order.metadata_json,
        }

    async def query_logistics_status(
        self,
        *,
        tenant_id: str,
        user_id: str,
        order_id: str,
    ) -> dict[str, Any]:
        """Query logistics status based on the order record."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="query_logistics_status",
                arguments={"tenant_id": tenant_id, "user_id": user_id, "order_id": order_id},
            )
        order_payload = await self.query_order_status(
            tenant_id=tenant_id,
            user_id=user_id,
            order_id=order_id,
        )
        if not order_payload.get("found"):
            return order_payload
        return {
            "found": True,
            "order_id": order_id,
            "logistics_company": order_payload.get("logistics_company"),
            "tracking_number": order_payload.get("tracking_number"),
            "logistics_status": order_payload.get("metadata", {}).get(
                "logistics_status", order_payload.get("status")
            ),
        }

    async def create_refund_case(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        order_id: str,
        reason: str,
        priority: str = "normal",
        metadata: dict[str, Any] | None = None,
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Create a refund case in the after-sales system."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="create_refund_case",
                arguments={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "order_id": order_id,
                    "reason": reason,
                    "refund_type": (metadata or {}).get("refund_type", "return_refund"),
                    "priority": priority,
                    "approval_token": approval_token,
                },
            )
        ticket = await TicketRepository(self._session).create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category="refund",
            title=f"退款申请：{order_id}",
            detail=reason,
            priority=priority,
            metadata={"order_id": order_id, **(metadata or {})},
        )
        return {"ticket_id": ticket.id, "status": ticket.status, "category": ticket.category}

    async def create_compensation_case(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        order_id: str,
        reason: str,
        compensation_type: str,
        priority: str = "normal",
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Create a compensation case such as coupon, points, or partial refund."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="create_compensation_case",
                arguments={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "order_id": order_id,
                    "reason": reason,
                    "compensation_type": compensation_type,
                    "priority": priority,
                    "approval_token": approval_token,
                },
            )
        ticket = await TicketRepository(self._session).create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category="compensation",
            title=f"补偿申请：{order_id}",
            detail=reason,
            priority=priority,
            metadata={"order_id": order_id, "compensation_type": compensation_type},
        )
        return {"ticket_id": ticket.id, "status": ticket.status, "category": ticket.category}

    async def create_exchange_case(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        order_id: str,
        reason: str,
        priority: str = "normal",
        approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Create an exchange or reshipment case."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="create_exchange_case",
                arguments={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "order_id": order_id,
                    "reason": reason,
                    "priority": priority,
                    "approval_token": approval_token,
                },
            )
        ticket = await TicketRepository(self._session).create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category="exchange",
            title=f"换货/补发申请：{order_id}",
            detail=reason,
            priority=priority,
            metadata={"order_id": order_id},
        )
        return {"ticket_id": ticket.id, "status": ticket.status, "category": ticket.category}

    async def transfer_to_human(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        reason: str,
        priority: str = "high",
        approval_token: str | None = None,
        origin_thread_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create the external case and the local durable human-ownership session."""

        if self._mcp_client is not None:
            result = await self._mcp_client.call_tool(
                name="transfer_to_human",
                arguments={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "reason": reason,
                    "priority": priority,
                    "approval_token": approval_token,
                },
            )
        else:
            ticket = await TicketRepository(self._session).create(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                category="human_handoff",
                title="转人工处理",
                detail=reason,
                priority=priority,
            )
            result = {
                "ticket_id": ticket.id,
                "status": ticket.status,
                "message": "已创建人工客服工单",
            }

        if not conversation_id:
            return result
        ticket_id = str(result.get("ticket_id") or "") or None
        handoff = await self._human_support_service.start_handoff(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            support_ticket_id=ticket_id,
            origin_thread_id=origin_thread_id,
            reason=reason,
            priority=priority,
            queue_name="complaint" if priority == "urgent" else "general",
            idempotency_key=(
                f"handoff:{tenant_id}:{idempotency_key or ticket_id or conversation_id}"
            ),
        )
        return {
            **result,
            "handoff_id": handoff.id,
            "handoff_status": handoff.status,
            "service_mode": "waiting_human",
        }

    async def query_price_protection(
        self,
        *,
        tenant_id: str,
        user_id: str,
        order_id: str,
    ) -> dict[str, Any]:
        """Return price-protection information from order metadata."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="query_price_protection",
                arguments={"tenant_id": tenant_id, "user_id": user_id, "order_id": order_id},
            )
        order_payload = await self.query_order_status(
            tenant_id=tenant_id,
            user_id=user_id,
            order_id=order_id,
        )
        if not order_payload.get("found"):
            return order_payload
        metadata = order_payload.get("metadata", {})
        return {
            "found": True,
            "order_id": order_id,
            "eligible": bool(metadata.get("price_protection_eligible", False)),
            "paid_amount": metadata.get("paid_amount"),
            "current_amount": metadata.get("current_amount"),
            "policy": metadata.get("price_protection_policy", "以商户价保规则为准"),
        }

    async def query_customer_profile(self, *, tenant_id: str, user_id: str) -> dict[str, Any]:
        """Return a lightweight customer profile used by the Agent."""

        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="query_customer_profile",
                arguments={"tenant_id": tenant_id, "user_id": user_id},
            )
        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "vip_level": "standard",
            "risk_flags": [],
        }
