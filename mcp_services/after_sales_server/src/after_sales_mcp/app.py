from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from after_sales_mcp.config import get_settings
from after_sales_mcp.database import session_context
from after_sales_mcp.repository import AfterSalesRepository
from after_sales_mcp.security import verify_approval_token


settings = get_settings()
mcp = FastMCP(
    "after-sales-mcp-server",
    instructions=(
        "Tenant-isolated order and after-sales capabilities. "
        "Write tools require a signed approval token."
    ),
    host=settings.host,
    port=settings.port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@mcp.resource("after-sales://tool-contracts")
def tool_contracts() -> str:
    """Publish tool governance metadata separately from JSON Schema."""

    return json.dumps(
        [
            {
                "name": "query_order_status",
                "read_only": True,
                "requires_confirmation": False,
                "risk_level": "low",
                "action_type": "query",
            },
            {
                "name": "query_logistics_status",
                "read_only": True,
                "requires_confirmation": False,
                "risk_level": "low",
                "action_type": "query",
            },
            {
                "name": "query_price_protection",
                "read_only": True,
                "requires_confirmation": False,
                "risk_level": "low",
                "action_type": "query",
            },
            {
                "name": "query_customer_profile",
                "read_only": True,
                "requires_confirmation": False,
                "risk_level": "low",
                "action_type": "query",
            },
            {
                "name": "create_refund_case",
                "read_only": False,
                "requires_confirmation": True,
                "risk_level": "medium",
                "action_type": "refund",
            },
            {
                "name": "create_compensation_case",
                "read_only": False,
                "requires_confirmation": True,
                "risk_level": "medium",
                "action_type": "compensation",
            },
            {
                "name": "create_exchange_case",
                "read_only": False,
                "requires_confirmation": True,
                "risk_level": "medium",
                "action_type": "exchange",
            },
            {
                "name": "transfer_to_human",
                "read_only": False,
                "requires_confirmation": True,
                "risk_level": "medium",
                "action_type": "handoff",
            },
        ],
        ensure_ascii=False,
    )


async def _order_payload(tenant_id: str, user_id: str, order_id: str) -> dict[str, Any]:
    async with session_context() as session:
        order = await AfterSalesRepository(session).get_order(
            tenant_id=tenant_id,
            user_id=user_id,
            order_id=order_id,
        )
        if order is None:
            return {"found": False, "order_id": order_id}
        return {
            "found": True,
            "order_id": order.order_id,
            "status": order.status,
            "logistics_company": order.logistics_company,
            "tracking_number": order.tracking_number,
            "metadata": order.metadata_json,
        }


@mcp.tool()
async def query_order_status(
    tenant_id: str,
    user_id: str,
    order_id: str,
) -> dict[str, Any]:
    """Query one order owned by the current tenant and user."""

    return await _order_payload(tenant_id, user_id, order_id)


@mcp.tool()
async def query_logistics_status(
    tenant_id: str,
    user_id: str,
    order_id: str,
) -> dict[str, Any]:
    """Query logistics fields from the owned order."""

    payload = await _order_payload(tenant_id, user_id, order_id)
    if not payload.get("found"):
        return payload
    return {
        "found": True,
        "order_id": order_id,
        "logistics_company": payload.get("logistics_company"),
        "tracking_number": payload.get("tracking_number"),
        "logistics_status": payload.get("metadata", {}).get(
            "logistics_status",
            payload.get("status"),
        ),
    }


@mcp.tool()
async def query_price_protection(
    tenant_id: str,
    user_id: str,
    order_id: str,
) -> dict[str, Any]:
    """Query price-protection facts from the owned order."""

    payload = await _order_payload(tenant_id, user_id, order_id)
    if not payload.get("found"):
        return payload
    metadata = payload.get("metadata", {})
    return {
        "found": True,
        "order_id": order_id,
        "eligible": bool(metadata.get("price_protection_eligible", False)),
        "paid_amount": metadata.get("paid_amount"),
        "current_amount": metadata.get("current_amount"),
        "policy": metadata.get("price_protection_policy", "以商户价保规则为准"),
    }


@mcp.tool()
async def query_customer_profile(tenant_id: str, user_id: str) -> dict[str, Any]:
    """Return the minimal profile needed by after-sales policy."""

    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "vip_level": "standard",
        "risk_flags": [],
    }


async def _create_ticket(
    *,
    tool_name: str,
    approval_token: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str | None,
    category: str,
    title: str,
    detail: str,
    priority: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    claims = verify_approval_token(
        token=approval_token,
        tenant_id=tenant_id,
        user_id=user_id,
        tool_name=tool_name,
    )
    async with session_context() as session:
        ticket = await AfterSalesRepository(session).create_ticket(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category=category,
            title=title,
            detail=detail,
            priority=priority,
            metadata=metadata,
            idempotency_key=str(claims["confirmation_id"]),
        )
        await session.commit()
        return {
            "ticket_id": ticket.id,
            "status": ticket.status,
            "category": ticket.category,
            "confirmation_id": claims["confirmation_id"],
        }


@mcp.tool()
async def create_refund_case(
    tenant_id: str,
    user_id: str,
    order_id: str,
    reason: str,
    refund_type: str,
    approval_token: str,
    conversation_id: str | None = None,
    priority: str = "normal",
) -> dict[str, Any]:
    """Create an idempotent refund case after verified HIL approval."""

    return await _create_ticket(
        tool_name="create_refund_case",
        approval_token=approval_token,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        category="refund",
        title=f"退款申请：{order_id}",
        detail=reason,
        priority=priority,
        metadata={"order_id": order_id, "refund_type": refund_type},
    )


@mcp.tool()
async def create_compensation_case(
    tenant_id: str,
    user_id: str,
    order_id: str,
    reason: str,
    compensation_type: str,
    approval_token: str,
    conversation_id: str | None = None,
    priority: str = "normal",
) -> dict[str, Any]:
    """Create an idempotent compensation case after verified HIL approval."""

    return await _create_ticket(
        tool_name="create_compensation_case",
        approval_token=approval_token,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        category="compensation",
        title=f"补偿申请：{order_id}",
        detail=reason,
        priority=priority,
        metadata={"order_id": order_id, "compensation_type": compensation_type},
    )


@mcp.tool()
async def create_exchange_case(
    tenant_id: str,
    user_id: str,
    order_id: str,
    reason: str,
    approval_token: str,
    conversation_id: str | None = None,
    priority: str = "normal",
) -> dict[str, Any]:
    """Create an idempotent exchange case after verified HIL approval."""

    return await _create_ticket(
        tool_name="create_exchange_case",
        approval_token=approval_token,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        category="exchange",
        title=f"换货/补发申请：{order_id}",
        detail=reason,
        priority=priority,
        metadata={"order_id": order_id},
    )


@mcp.tool()
async def transfer_to_human(
    tenant_id: str,
    user_id: str,
    reason: str,
    approval_token: str,
    conversation_id: str | None = None,
    priority: str = "high",
) -> dict[str, Any]:
    """Create an idempotent human-handoff case after verified HIL approval."""

    return await _create_ticket(
        tool_name="transfer_to_human",
        approval_token=approval_token,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        category="human_handoff",
        title="转人工处理",
        detail=reason,
        priority=priority,
        metadata={},
    )


def main() -> None:
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
