from __future__ import annotations

from typing import Any

import pytest

from customer_service_app.domain.planning import AgentPlan, PlanStep
from customer_service_app.services.react_executor import ReactExecutor
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry, ToolSpec


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
async def test_react_executor_runs_steps_once_and_respects_dependencies() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={"type": "object"},
            handler=echo_tool,
        )
    )
    plan = AgentPlan(
        user_goal="execute two steps",
        max_steps=2,
        steps=[
            PlanStep(
                id="step-1",
                title="first",
                action_type="tool",
                tool_name="echo",
                arguments={"value": 1},
            ),
            PlanStep(
                id="step-2",
                title="second",
                action_type="tool",
                tool_name="echo",
                arguments={"value": 2},
                depends_on=["step-1"],
            ),
        ],
    )
    executor = ReactExecutor(
        rag_service=FakeRagService(),  # type: ignore[arg-type]
        tool_registry=registry,
    )

    result = await executor.execute(
        plan=plan,
        tenant_id="tenant-1",
        question="test",
        context=build_context(),
    )

    assert result.completed_step_ids == ["step-1", "step-2"]
    assert result.observations["step-2"]["value"] == 2
    assert [step.attempts for step in result.plan.steps] == [1, 1]


@pytest.mark.asyncio
async def test_react_executor_never_runs_steps_past_plan_limit() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={"type": "object"},
            handler=echo_tool,
        )
    )
    plan = AgentPlan(
        user_goal="bounded execution",
        max_steps=1,
        steps=[
            PlanStep(
                id=f"step-{index}",
                title="echo",
                action_type="tool",
                tool_name="echo",
                arguments={"value": index},
            )
            for index in range(1, 3)
        ],
    )
    executor = ReactExecutor(
        rag_service=FakeRagService(),  # type: ignore[arg-type]
        tool_registry=registry,
    )

    result = await executor.execute(
        plan=plan,
        tenant_id="tenant-1",
        question="test",
        context=build_context(),
    )

    assert result.completed_step_ids == ["step-1"]
    assert result.skipped_step_ids == ["step-2"]
    assert result.plan.steps[1].attempts == 0
