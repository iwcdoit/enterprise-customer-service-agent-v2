from __future__ import annotations

from typing import Any

from customer_service_app.core.exceptions import AppError
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolSpec


def _business_gateway(context: ToolExecutionContext):
    if context.business_gateway is None:
        raise AppError("Business gateway is not configured", code="business_gateway_missing")
    return context.business_gateway


async def query_order_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """查询订单状态工具。

    模型不能直接查数据库或 MCP，只能请求调用这个工具。
    具体是走本地仓库还是 MCP 服务，由 BusinessGateway 决定。
    """

    order_id = str(arguments["order_id"])
    return await _business_gateway(context).query_order_status(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        order_id=order_id,
    )


async def create_refund_ticket(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """创建退款工单工具。

    当模型判断用户明确要退款时，会返回 create_refund_ticket 的 tool_call。
    后端执行这里的代码，并通过 BusinessGateway 创建退款申请。
    """

    order_id = str(arguments["order_id"])
    reason = str(arguments["reason"])
    return await _business_gateway(context).create_refund_ticket(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        order_id=order_id,
        reason=reason,
        priority=str(arguments.get("priority", "normal")),
    )


async def transfer_to_human(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """转人工工具。

    用户投诉、情绪强烈或模型无法继续处理时，创建人工客服工单。
    """

    reason = str(arguments["reason"])
    return await _business_gateway(context).transfer_to_human(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        reason=reason,
        priority=str(arguments.get("priority", "high")),
    )


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
