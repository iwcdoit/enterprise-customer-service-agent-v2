from __future__ import annotations

import json

from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult, PlanStep
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry


class PlanExecutor:
    """Execute a bounded plan with explicit loop and dependency protection."""

    def __init__(self, *, tool_registry: ToolRegistry, max_step_attempts: int = 1):
        self._tool_registry = tool_registry
        self._max_step_attempts = max_step_attempts

    async def execute(
        self,
        *,
        plan: AgentPlan,
        context: ToolExecutionContext,
    ) -> PlanExecutionResult:
        """Execute plan steps once in order.

        This executor intentionally does not run an open-ended loop. Later LangGraph
        nodes can call it safely because every step has a bounded number of attempts.
        """

        result = PlanExecutionResult(plan=plan)
        completed: set[str] = set()

        for index, step in enumerate(plan.steps):
            if index >= plan.max_steps:
                self._mark_skipped(step, result, reason="max_steps_exceeded")
                continue

            missing_dependencies = [item for item in step.depends_on if item not in completed]
            if missing_dependencies:
                self._mark_skipped(
                    step,
                    result,
                    reason="dependency_not_completed",
                    missing_dependencies=missing_dependencies,
                )
                continue

            await self._execute_step(step, context=context, result=result)
            if step.status == "completed":
                completed.add(step.id)

        return result

    async def _execute_step(
        self,
        step: PlanStep,
        *,
        context: ToolExecutionContext,
        result: PlanExecutionResult,
    ) -> None:
        if step.attempts >= self._max_step_attempts:
            self._mark_failed(step, result, reason="max_attempts_exceeded")
            return

        step.status = "running"
        step.attempts += 1

        if step.action_type != "tool":
            step.status = "completed"
            step.observation = {"message": f"{step.action_type} step is deferred to graph nodes"}
            result.completed_step_ids.append(step.id)
            result.observations[step.id] = step.observation
            return

        if not step.tool_name:
            self._mark_failed(step, result, reason="tool_name_missing")
            return

        try:
            payload = await self._tool_registry.execute(
                name=step.tool_name,
                arguments_json=json.dumps(step.arguments, ensure_ascii=False),
                context=context,
            )
        except Exception as exc:
            self._mark_failed(step, result, reason="tool_failed", error=str(exc))
            return

        step.observation = payload
        result.observations[step.id] = payload
        if payload.get("requires_confirmation") is True:
            step.status = "blocked"
            result.blocked_step_ids.append(step.id)
            return

        step.status = "completed"
        result.completed_step_ids.append(step.id)

    @staticmethod
    def _mark_skipped(
        step: PlanStep,
        result: PlanExecutionResult,
        *,
        reason: str,
        **metadata,
    ) -> None:
        step.status = "skipped"
        step.observation = {"reason": reason, **metadata}
        result.skipped_step_ids.append(step.id)
        result.observations[step.id] = step.observation

    @staticmethod
    def _mark_failed(
        step: PlanStep,
        result: PlanExecutionResult,
        *,
        reason: str,
        **metadata,
    ) -> None:
        step.status = "failed"
        step.observation = {"reason": reason, **metadata}
        result.failed_step_ids.append(step.id)
        result.observations[step.id] = step.observation
