from __future__ import annotations

from typing import Any

from customer_service_app.services.tool_registry import ToolExecutionContext, ToolSpec


async def create_refund_case(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """Create a refund case after user confirmation."""

    return await context.business_gateway.create_refund_case(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        order_id=str(arguments["order_id"]),
        reason=str(arguments["reason"]),
        priority=str(arguments.get("priority", "normal")),
        metadata={
            "refund_type": arguments.get("refund_type", "return_refund"),
            "evidence_urls": arguments.get("evidence_urls", []),
        },
        approval_token=context.approval_token,
    )


async def create_compensation_case(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """Create a compensation case after user confirmation."""

    return await context.business_gateway.create_compensation_case(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        order_id=str(arguments["order_id"]),
        reason=str(arguments["reason"]),
        compensation_type=str(arguments["compensation_type"]),
        priority=str(arguments.get("priority", "normal")),
        approval_token=context.approval_token,
    )


async def create_exchange_case(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """Create an exchange or reshipment case after user confirmation."""

    return await context.business_gateway.create_exchange_case(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        conversation_id=context.conversation_id,
        order_id=str(arguments["order_id"]),
        reason=str(arguments["reason"]),
        priority=str(arguments.get("priority", "normal")),
        approval_token=context.approval_token,
    )


REFUND_CASE_TOOL = ToolSpec(
    name="create_refund_case",
    description="用户确认后创建退货退款或仅退款售后申请。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1, "description": "申请售后的订单号"},
            "reason": {"type": "string", "minLength": 1, "description": "用户描述的退款原因"},
            "refund_type": {
                "type": "string",
                "enum": ["return_refund", "refund_only"],
                "description": "退货退款或仅退款",
            },
            "evidence_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用户上传的凭证图片地址",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "工单优先级",
            },
        },
        "required": ["order_id", "reason", "refund_type"],
        "additionalProperties": False,
    },
    handler=create_refund_case,
    read_only=False,
    requires_confirmation=True,
    risk_level="medium",
    action_type="refund",
)


COMPENSATION_CASE_TOOL = ToolSpec(
    name="create_compensation_case",
    description="用户确认后创建不退货补偿申请，例如优惠券、积分、部分退款或补发配件。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1, "description": "需要补偿的订单号"},
            "reason": {"type": "string", "minLength": 1, "description": "用户要求补偿的原因"},
            "compensation_type": {
                "type": "string",
                "enum": ["coupon", "points", "partial_refund", "reship_accessory", "freight_refund"],
                "description": "补偿类型",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "工单优先级",
            },
        },
        "required": ["order_id", "reason", "compensation_type"],
        "additionalProperties": False,
    },
    handler=create_compensation_case,
    read_only=False,
    requires_confirmation=True,
    risk_level="medium",
    action_type="compensation",
)


EXCHANGE_CASE_TOOL = ToolSpec(
    name="create_exchange_case",
    description="用户确认后创建换货、补发配件或错发补发申请。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "minLength": 1, "description": "需要换货或补发的订单号"},
            "reason": {"type": "string", "minLength": 1, "description": "换货或补发原因"},
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "工单优先级",
            },
        },
        "required": ["order_id", "reason"],
        "additionalProperties": False,
    },
    handler=create_exchange_case,
    read_only=False,
    requires_confirmation=True,
    risk_level="medium",
    action_type="exchange",
)
