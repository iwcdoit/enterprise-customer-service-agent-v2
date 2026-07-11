from __future__ import annotations

import json
from asyncio import TimeoutError, wait_for
from typing import Any

from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult, PlanStep
from customer_service_app.services.rag_service import RagService
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry


class ReactExecutor:
    """执行有界计划，并收集结构化观察结果。

    这里刻意不实现无限自主循环：每个计划都有最大步骤数，每个步骤最多执行一次，
    每次外部调用也都有超时时间。
    """

    def __init__(
        self,
        *,
        rag_service: RagService,
        tool_registry: ToolRegistry,
        max_steps: int = 6,
        step_timeout_seconds: float = 20.0,
    ) -> None:
        self._rag_service = rag_service
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._step_timeout_seconds = step_timeout_seconds

    async def execute(
        self,
        *,
        plan: AgentPlan,
        tenant_id: str,
        question: str,
        context: ToolExecutionContext,
    ) -> PlanExecutionResult:
        """按依赖顺序执行可运行步骤，并严格限制最大执行步数。"""

        result = PlanExecutionResult(plan=plan)
        completed: set[str] = set()
        execution_limit = min(plan.max_steps, self._max_steps)

        for index, step in enumerate(plan.steps):
            if index >= execution_limit:
                self._mark_skipped(step, result, reason="max_steps_exceeded")
                continue

            missing = [dependency for dependency in step.depends_on if dependency not in completed]
            if missing:
                self._mark_skipped(
                    step,
                    result,
                    reason="dependency_not_completed",
                    missing_dependencies=missing,
                )
                continue

            await self._execute_step(
                step,
                tenant_id=tenant_id,
                question=question,
                context=context,
                result=result,
            )
            if step.status == "completed":
                completed.add(step.id)

        return result

    async def _execute_step(
        self,
        step: PlanStep,
        *,
        tenant_id: str,
        question: str,
        context: ToolExecutionContext,
        result: PlanExecutionResult,
    ) -> None:
        step.status = "running"
        step.attempts += 1

        try:
            observation = await wait_for(
                self._dispatch(
                    step,
                    tenant_id=tenant_id,
                    question=question,
                    context=context,
                ),
                timeout=self._step_timeout_seconds,
            )
        except TimeoutError:
            self._mark_failed(step, result, reason="step_timeout")
            return
        except Exception as exc:
            self._mark_failed(step, result, reason="step_failed", error=str(exc))
            return

        step.observation = observation
        result.observations[step.id] = observation
        if observation.get("requires_confirmation") is True:
            step.status = "blocked"
            result.blocked_step_ids.append(step.id)
            return

        step.status = "completed"
        result.completed_step_ids.append(step.id)

    async def _dispatch(
        self,
        step: PlanStep,
        *,
        tenant_id: str,
        question: str,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if step.action_type == "rag":
            chunks = await self._rag_service.retrieve(
                tenant_id=tenant_id,
                question=question,
            )
            return {
                "action_type": "rag",
                "chunk_count": len(chunks),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }

        if step.action_type == "tool":
            if not step.tool_name:
                raise ValueError("tool step is missing tool_name")
            payload = await self._tool_registry.execute(
                name=step.tool_name,
                arguments_json=json.dumps(step.arguments, ensure_ascii=False),
                context=context,
            )
            return {
                "action_type": "tool",
                "tool_name": step.tool_name,
                "arguments": step.arguments,
                **payload,
            }

        # LLM、最终回答和转人工步骤由后续 Graph 节点处理。
        # 这里只记录 observation，避免执行器内部偷偷递归调用模型。
        return {
            "action_type": step.action_type,
            "deferred": True,
            "message": "该步骤由后续回答节点处理",
        }

    @staticmethod
    def _mark_skipped(
        step: PlanStep,
        result: PlanExecutionResult,
        *,
        reason: str,
        **metadata: Any,
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
        **metadata: Any,
    ) -> None:
        step.status = "failed"
        step.observation = {"reason": reason, **metadata}
        result.failed_step_ids.append(step.id)
        result.observations[step.id] = step.observation
