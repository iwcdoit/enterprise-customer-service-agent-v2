from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.db.repositories import OrderRepository, TicketRepository
from customer_service_app.infrastructure.mcp.approval import build_approval_token
from customer_service_app.infrastructure.mcp.base import MCPBusinessClient
from customer_service_app.services.human_support_service import HumanSupportService


class BusinessGateway:
    """Boundary for customer-service business capabilities.

    The agent calls this gateway instead of directly depending on order, ticket,
    CRM, ERP, or MCP implementation details.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        session: AsyncSession,
        mcp_client: MCPBusinessClient | None,
        human_support_service: HumanSupportService | None = None,
    ):
        self._settings = settings
        self._session = session
        self._mcp_client = mcp_client
        self._human_support_service = human_support_service

    async def query_order_status(
        self,
        *,
        tenant_id: str,
        user_id: str,
        order_id: str,
    ) -> dict[str, Any]:
        if self._mcp_client is not None:
            return await self._mcp_client.call_tool(
                name="query_order_status",
                arguments={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "order_id": order_id,
                },
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

    async def create_refund_ticket(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        order_id: str,
        reason: str,
        priority: str,
    ) -> dict[str, Any]:
        if self._mcp_client is not None:
            mcp_tool_name = "create_refund_case"
            arguments = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "order_id": order_id,
                "reason": reason,
                "refund_type": "return_refund",
                "priority": priority,
            }
            arguments["approval_token"] = build_approval_token(
                settings=self._settings,
                tool_name=mcp_tool_name,
                tenant_id=tenant_id,
                user_id=user_id,
                arguments=arguments,
            )
            return await self._mcp_client.call_tool(name=mcp_tool_name, arguments=arguments)

        ticket = await TicketRepository(self._session).create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category="refund",
            title=f"退款申请：{order_id}",
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
        priority: str,
    ) -> dict[str, Any]:
        if self._mcp_client is not None:
            mcp_tool_name = "transfer_to_human"
            arguments = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "reason": reason,
                "priority": priority,
            }
            arguments["approval_token"] = build_approval_token(
                settings=self._settings,
                tool_name=mcp_tool_name,
                tenant_id=tenant_id,
                user_id=user_id,
                arguments=arguments,
            )
            result = await self._mcp_client.call_tool(name=mcp_tool_name, arguments=arguments)
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

        # 外部工单只表示业务系统已受理；本地接管记录负责控制当前会话由 Bot 还是人工处理。
        if conversation_id and self._human_support_service is not None:
            ticket_id = str(result.get("ticket_id") or "") or None
            handoff = await self._human_support_service.start_handoff(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                support_ticket_id=ticket_id,
                origin_thread_id=None,
                reason=reason,
                priority=priority,
                queue_name="complaint" if priority == "urgent" else "general",
                idempotency_key=f"handoff:{tenant_id}:{ticket_id or conversation_id}",
            )
            result.update(
                handoff_id=handoff.id,
                handoff_status=handoff.status,
                service_mode="waiting_human",
            )
        return result
