from __future__ import annotations

from typing import Any

from customer_service_app.infrastructure.db.repositories import OrderRepository, TicketRepository
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolSpec


async def query_order_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """查询当前租户和用户拥有的订单。"""

    order_id = str(arguments["order_id"])
    if context.business_gateway is not None:
        return await context.business_gateway.query_order_status(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            order_id=order_id,
        )
    order = await OrderRepository(context.session).get_by_order_id(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
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
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """查询物流状态工具。"""

    order_id = str(arguments["order_id"])
    if context.business_gateway is not None:
        return await context.business_gateway.query_logistics_status(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            order_id=order_id,
        )
    order_payload = await query_order_status(arguments, context)
    if not order_payload.get("found"):
        return order_payload
    return {
        "found": True,
        "order_id": order_id,
        "logistics_company": order_payload.get("logistics_company"),
        "tracking_number": order_payload.get("tracking_number"),
        "logistics_status": order_payload.get("status"),
    }


async def query_price_protection(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """查询订单价保资格工具。"""

    order_id = str(arguments["order_id"])
    if context.business_gateway is not None:
        return await context.business_gateway.query_price_protection(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            order_id=order_id,
        )
    order_payload = await query_order_status(arguments, context)
    if not order_payload.get("found"):
        return order_payload
    metadata = order_payload.get("metadata", {})
    return {
        "found": True,
        "order_id": order_id,
        "eligible": bool(metadata.get("price_protection_eligible", False)),
        "paid_amount": metadata.get("paid_amount"),
        "current_amount": metadata.get("current_amount"),
    }


async def query_customer_profile(
    _: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """查询当前用户画像工具。"""

    if context.business_gateway is not None:
        return await context.business_gateway.query_customer_profile(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
        )
    return {
        "tenant_id": context.tenant_id,
        "user_id": context.user_id,
        "vip_level": "standard",
        "risk_flags": [],
    }


async def create_refund_ticket(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """创建退款工单工具。

    当模型判断用户明确要退款时，会返回 create_refund_ticket 的 tool_call。
    后端执行这里的代码，把退款申请写入 support_tickets 表。
    """

    order_id = str(arguments["order_id"])
    reason = str(arguments["reason"])
    ticket = await TicketRepository(context.session).create(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        category="refund",
        title=f"退款申请：{order_id}",
        detail=reason,
        priority=str(arguments.get("priority", "normal")),
        metadata={"order_id": order_id},
    )
    return {"ticket_id": ticket.id, "status": ticket.status, "category": ticket.category}


async def transfer_to_human(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """转人工工具。

    用户投诉、情绪强烈或模型无法继续处理时，创建人工客服工单。
    """

    reason = str(arguments["reason"])
    ticket = await TicketRepository(context.session).create(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        category="human_handoff",
        title="转人工处理",
        detail=reason,
        priority=str(arguments.get("priority", "high")),
    )
    return {"ticket_id": ticket.id, "status": ticket.status, "message": "已创建人工客服工单"}


ORDER_STATUS_TOOL = ToolSpec(
    name="query_order_status",
    description="查询当前用户名下指定订单的状态、物流公司和运单号。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1, "description": "用户要查询的订单号"}
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_order_status,
    read_only=True,
    requires_confirmation=False,
    risk_level="low",
    action_type="query",
    parallel_safe=True,
)


LOGISTICS_STATUS_TOOL = ToolSpec(
    name="query_logistics_status",
    description="查询当前用户名下指定订单的物流状态、物流公司和运单号。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1, "description": "用户要查询物流的订单号"}
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_logistics_status,
    read_only=True,
    requires_confirmation=False,
    risk_level="low",
    action_type="query",
    parallel_safe=True,
)


PRICE_PROTECTION_TOOL = ToolSpec(
    name="query_price_protection",
    description="查询指定订单是否满足价保或补差价条件。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1, "description": "用户要查询价保的订单号"}
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_price_protection,
    read_only=True,
    requires_confirmation=False,
    risk_level="low",
    action_type="query",
    parallel_safe=True,
)


CUSTOMER_PROFILE_TOOL = ToolSpec(
    name="query_customer_profile",
    description="查询当前用户的基础会员等级、风险标记和客服画像。",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    handler=query_customer_profile,
    read_only=True,
    requires_confirmation=False,
    risk_level="low",
    action_type="query",
    parallel_safe=True,
)


REFUND_TICKET_TOOL = ToolSpec(
    name="create_refund_ticket",
    description="当用户要申请退款或退货退款时，创建退款工单。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "申请退款的订单号"},
            "reason": {"type": "string", "description": "用户描述的退款原因"},
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "工单优先级",
            },
        },
        "required": ["order_id", "reason"],
        "additionalProperties": False,
    },
    handler=create_refund_ticket,
    read_only=False,
    requires_confirmation=True,
    risk_level="medium",
    action_type="refund",
)


HUMAN_HANDOFF_TOOL = ToolSpec(
    name="transfer_to_human",
    description="当用户明确要求人工客服、投诉升级、情绪强烈或模型无法解决时创建人工客服工单。",
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "转人工原因"},
            "priority": {
                "type": "string",
                "enum": ["normal", "high", "urgent"],
                "description": "人工处理优先级",
            },
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
    handler=transfer_to_human,
    read_only=False,
    requires_confirmation=True,
    risk_level="medium",
    action_type="handoff",
)
