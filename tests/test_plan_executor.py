from __future__ import annotations

from typing import Any

import pytest

from customer_service_app.domain.planning import AgentPlan, PlanStep
from customer_service_app.services.plan_executor import PlanExecutor
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry, ToolSpec


async def echo_tool(arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    return {"ok": True, "arguments": arguments}


async def confirm_tool(arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    return {"requires_confirmation": True, "arguments": arguments}


def build_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        tenant_id="t1",
        user_id="u1",
        conversation_id="c1",
        session=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_plan_executor_completes_tool_step() -> None:
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
        user_goal="test",
        steps=[
            PlanStep(
                id="s1",
                title="echo",
                action_type="tool",
                tool_name="echo",
                arguments={"value": 1},
            )
        ],
    )

    result = await PlanExecutor(tool_registry=registry).execute(plan=plan, context=build_context())

    assert result.completed_step_ids == ["s1"]
    assert plan.steps[0].status == "completed"
    assert result.observations["s1"]["arguments"] == {"value": 1}


@pytest.mark.asyncio
async def test_plan_executor_blocks_confirmation_step() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="confirm",
            description="confirm",
            parameters={"type": "object"},
            handler=confirm_tool,
            requires_confirmation=True,
        )
    )
    plan = AgentPlan(
        user_goal="test",
        steps=[
            PlanStep(
                id="s1",
                title="confirm",
                action_type="tool",
                tool_name="confirm",
            )
        ],
    )

    result = await PlanExecutor(tool_registry=registry).execute(plan=plan, context=build_context())

    assert result.blocked_step_ids == ["s1"]
    assert plan.steps[0].status == "blocked"


@pytest.mark.asyncio
async def test_plan_executor_skips_missing_dependency() -> None:
    registry = ToolRegistry()
    plan = AgentPlan(
        user_goal="test",
        steps=[
            PlanStep(
                id="s1",
                title="depends",
                action_type="tool",
                tool_name="missing",
                depends_on=["not_done"],
            )
        ],
    )

    result = await PlanExecutor(tool_registry=registry).execute(plan=plan, context=build_context())

    assert result.skipped_step_ids == ["s1"]
    assert result.observations["s1"]["reason"] == "dependency_not_completed"
