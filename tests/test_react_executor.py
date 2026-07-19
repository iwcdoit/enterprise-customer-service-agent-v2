from __future__ import annotations

from typing import Any

import pytest

from customer_service_app.domain.planning import PlanStep
from customer_service_app.services.react_executor import ReactExecutor
from customer_service_app.services.tool_registry import (
    ToolExecutionContext,
    ToolRegistry,
    ToolSpec,
)


class FakeRagService:
    async def retrieve(self, **_: Any) -> list:
        return []


async def echo_tool(
    arguments: dict[str, Any],
    _: ToolExecutionContext,
) -> dict[str, Any]:
    return {"value": arguments["value"]}


def build_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        tenant_id="tenant-1",
        user_id="user-1",
        conversation_id="conversation-1",
        session=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_react_executor_executes_one_tool_step() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={"type": "object"},
            handler=echo_tool,
        )
    )
    step = PlanStep(
        step_id="step-1",
        title="first",
        goal="echo value",
        action_type="tool",
        tool_name="echo",
        arguments={"value": 1},
    )
    executor = ReactExecutor(
        rag_service=FakeRagService(),  # type: ignore[arg-type]
        tool_registry=registry,
    )

    observation = await executor.execute_step(
        tenant_id="tenant-1",
        question="test",
        step=step,
        context=build_context(),
    )

    assert observation["step_id"] == "step-1"
    assert observation["result"] == {"value": 1}


@pytest.mark.asyncio
async def test_react_executor_returns_rag_observation() -> None:
    executor = ReactExecutor(
        rag_service=FakeRagService(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )
    step = PlanStep(
        step_id="step-rag",
        title="retrieve",
        goal="retrieve policy",
        action_type="rag",
    )

    observation = await executor.execute_step(
        tenant_id="tenant-1",
        question="test",
        step=step,
        context=build_context(),
    )

    assert observation == {
        "step_id": "step-rag",
        "action_type": "rag",
        "chunk_count": 0,
        "chunks": [],
    }
