from __future__ import annotations

from typing import Any

from customer_service_app.infrastructure.db.repositories import OrderRepository, TicketRepository
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolSpec


async def query_order_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """查询订单状态工具。

    这个函数是真正访问 MySQL 的地方。模型不能直接查数据库，只能请求调用这个工具。
    查询时必须带 tenant_id 和 user_id，避免用户查到别人的订单。
    """

    order_id = str(arguments["order_id"])
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
    # ToolSpec 这一段不是给 Python 执行工具逻辑用的，而是给大模型看的“工具说明书”。
    # 模型会根据 name、description、parameters 判断是否应该返回 tool_call。
    name="query_order_status",
    description="查询当前用户名下指定订单的状态、物流公司和运单号。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "用户要查询的订单号"}
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
    handler=query_order_status,
)


REFUND_TICKET_TOOL = ToolSpec(
    # 这个工具告诉模型：当用户明确要退款/退货退款时，可以调用 create_refund_ticket。
    name="create_refund_ticket",
    description="当用户要申请退款或退货退款时，创建退款工单。该工具属于高风险写操作，需要确认后执行。",
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
    requires_confirmation=True,
)


HUMAN_HANDOFF_TOOL = ToolSpec(
    # 这个工具告诉模型：投诉升级、要求人工、无法处理时，可以创建人工客服工单。
    name="transfer_to_human",
    description="当用户明确要求人工客服、投诉升级、情绪强烈或模型无法解决时创建人工客服工单。该工具需要确认后执行。",
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
    requires_confirmation=True,
)
