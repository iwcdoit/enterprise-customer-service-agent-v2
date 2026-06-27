from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.db.repositories import OrderRepository, TicketRepository
from customer_service_app.infrastructure.mcp.base import MCPBusinessClient


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
    ):
        self._settings = settings
        self._session = session
        self._mcp_client = mcp_client

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
            arguments["approval_token"] = self._build_approval_token(
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
            arguments["approval_token"] = self._build_approval_token(
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
            category="human_handoff",
            title="转人工处理",
            detail=reason,
            priority=priority,
        )
        return {"ticket_id": ticket.id, "status": ticket.status, "message": "已创建人工客服工单"}

    def _build_approval_token(
        self,
        *,
        tool_name: str,
        tenant_id: str,
        user_id: str,
        arguments: dict[str, Any],
    ) -> str:
        secret = self._settings.require(
            "MCP_APPROVAL_SIGNING_SECRET",
            self._settings.mcp_approval_signing_secret,
        )
        now = datetime.now(timezone.utc)
        confirmation_id = self._confirmation_id(
            tool_name=tool_name,
            tenant_id=tenant_id,
            user_id=user_id,
            arguments=arguments,
        )
        claims = {
            "iss": self._settings.mcp_approval_issuer,
            "sub": user_id,
            "tenant_id": tenant_id,
            "tool_name": tool_name,
            "confirmation_id": confirmation_id,
            "iat": int(now.timestamp()),
            "exp": int(
                (
                    now
                    + timedelta(seconds=self._settings.mcp_approval_token_ttl_seconds)
                ).timestamp()
            ),
        }
        return jwt.encode(claims, secret, algorithm="HS256")

    @staticmethod
    def _confirmation_id(
        *,
        tool_name: str,
        tenant_id: str,
        user_id: str,
        arguments: dict[str, Any],
    ) -> str:
        raw = json.dumps(
            {
                "tool_name": tool_name,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "arguments": arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
