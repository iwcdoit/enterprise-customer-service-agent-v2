from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from customer_service_app.domain.planning import AgentPlan, PlanStep
from customer_service_app.services.plan_executor import PlanExecutor
from customer_service_app.services.tool_registry import (
    ToolExecutionContext,
    ToolRegistry,
    ToolSpec,
)


async def echo_tool(
    arguments: dict[str, Any], context: ToolExecutionContext
) -> dict[str, Any]:
    return {"ok": True, "arguments": arguments}


class FakeReactExecutor:
    async def execute_step(self, *, tenant_id, question, step, context):
        return {"step_id": step.step_id, "result": {"value": step.arguments.get("value")}}


class FakeConfirmationService:
    async def create_for_tool(self, **kwargs):
        return SimpleNamespace(id="confirmation-1", confirmation_prompt="请确认执行")


def build_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        tenant_id="t1",
        user_id="u1",
        conversation_id="c1",
        session=object(),  # type: ignore[arg-type]
        search_client=object(),  # type: ignore[arg-type]
    )


def build_executor(registry: ToolRegistry) -> PlanExecutor:
    return PlanExecutor(
        tool_registry=registry,
        react_executor=FakeReactExecutor(),  # type: ignore[arg-type]
        confirmation_service=FakeConfirmationService(),  # type: ignore[arg-type]
    )


def build_plan(step: PlanStep) -> AgentPlan:
    return AgentPlan(
        plan_id="plan-1",
        conversation_id="c1",
        user_goal="test",
        steps=[step],
    )


@pytest.mark.asyncio
async def test_plan_executor_completes_tool_step() -> None:
    registry = ToolRegistry()
    plan = build_plan(
        PlanStep(
            step_id="s1",
            title="echo",
            goal="读取信息",
            action_type="tool",
            tool_name="echo",
            arguments={"value": 1},
        )
    )

    result = await build_executor(registry).execute(
        plan=plan,
        tenant_id="t1",
        user_id="u1",
        question="test",
        context=build_context(),
    )

    assert plan.steps[0].status == "success"
    assert result.observations[0]["step_id"] == "s1"


@pytest.mark.asyncio
async def test_plan_executor_stops_at_confirmation_step() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="confirm",
            description="confirm",
            parameters={"type": "object"},
            handler=echo_tool,
            requires_confirmation=True,
        )
    )
    plan = build_plan(
        PlanStep(
            step_id="s1",
            title="confirm",
            goal="执行高风险动作",
            action_type="confirm",
            tool_name="confirm",
        )
    )

    result = await build_executor(registry).execute(
        plan=plan,
        tenant_id="t1",
        user_id="u1",
        question="test",
        context=build_context(),
    )

    assert result.waiting_confirmation_id == "confirmation-1"
    assert result.final_answer == "请确认执行"
    assert plan.steps[0].status == "success"
