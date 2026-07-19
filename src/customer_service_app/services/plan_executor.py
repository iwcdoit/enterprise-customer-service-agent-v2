from __future__ import annotations

from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.react_executor import ReactExecutor
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry


class PlanExecutor:
    """Execute a plan until it finishes or reaches a confirmation step."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        react_executor: ReactExecutor,
        confirmation_service: ConfirmationService,
    ):
        self._tool_registry = tool_registry
        self._react_executor = react_executor
        self._confirmation_service = confirmation_service

    async def execute(
        self,
        *,
        plan: AgentPlan,
        tenant_id: str,
        user_id: str,
        question: str,
        context: ToolExecutionContext,
    ) -> PlanExecutionResult:
        """Execute plan steps with side-effect confirmation safety."""

        observations: list[dict] = []

        seen_tool_keys: set[tuple[str, str]] = set()

        for step in plan.steps[: plan.max_steps]:
            if step.action_type == "confirm" and step.tool_name:
                tool = self._tool_registry.require(step.tool_name)

                action = await self._confirmation_service.create_for_tool(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=plan.conversation_id,
                    tool=tool,
                    arguments=step.arguments,

                    langgraph_thread_id=plan.plan_id,
                )

                step.status = "success"

                return PlanExecutionResult(
                    plan=plan,
                    observations=observations,
                    waiting_confirmation_id=action.id,
                    final_answer=action.confirmation_prompt,
                )

            if step.action_type == "tool" and step.tool_name:
                key = (step.tool_name, str(sorted(step.arguments.items())))
                if key in seen_tool_keys:
                    step.status = "skipped"
                    observations.append({"step_id": step.step_id, "reason": "duplicate_tool_call"})
                    continue
                seen_tool_keys.add(key)

            step.status = "running"
            observation = await self._react_executor.execute_step(
                tenant_id=tenant_id,
                question=question,
                step=step,
                context=context,
            )

            observations.append(observation)

            step.status = "success"

        return PlanExecutionResult(plan=plan, observations=observations)
