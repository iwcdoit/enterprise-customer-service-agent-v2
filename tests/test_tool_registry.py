from __future__ import annotations

from typing import Any

import pytest

from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry, ToolSpec


async def sample_handler(arguments: dict[str, Any], _: ToolExecutionContext) -> dict[str, Any]:
    """测试用工具函数，返回传入文本。"""
    return {"echo": arguments["text"]}


@pytest.mark.asyncio
async def test_tool_registry_executes_registered_tool() -> None:
    """验证工具注册后可以按名称执行。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=sample_handler,
        )
    )

    result = await registry.execute(
        name="echo",
        arguments_json='{"text": "hello"}',
        context=None,
    )

    assert result == {"echo": "hello"}


@pytest.mark.asyncio
async def test_high_risk_tool_requires_confirmation_before_execution() -> None:
    """高风险工具未确认时不应该真正执行 handler。"""

    async def should_not_run(_: dict[str, Any], __: ToolExecutionContext) -> dict[str, Any]:
        raise AssertionError("high risk handler should not run before confirmation")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="create_refund_ticket",
            description="create refund ticket",
            parameters={"type": "object", "properties": {"order_id": {"type": "string"}}},
            handler=should_not_run,
            requires_confirmation=True,
        )
    )

    result = await registry.execute(
        name="create_refund_ticket",
        arguments_json='{"order_id": "202606040001"}',
        context=None,
    )

    assert result["requires_confirmation"] is True
    assert result["tool_name"] == "create_refund_ticket"
    assert result["arguments"] == {"order_id": "202606040001"}


@pytest.mark.asyncio
async def test_high_risk_tool_executes_after_confirmation() -> None:
    """显式确认后，高风险工具才会进入真实 handler。"""

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="create_refund_ticket",
            description="create refund ticket",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=sample_handler,
            requires_confirmation=True,
        )
    )
    context = ToolExecutionContext(
        tenant_id="default",
        user_id="u001",
        conversation_id="c001",
        session=None,
        search_client=None,
        confirmed_tools={"create_refund_ticket"},
    )

    result = await registry.execute(
        name="create_refund_ticket",
        arguments_json='{"text": "confirmed"}',
        context=context,
    )

    assert result == {"echo": "confirmed"}
