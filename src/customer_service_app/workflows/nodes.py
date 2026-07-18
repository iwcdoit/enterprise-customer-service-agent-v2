from __future__ import annotations

import json
from typing import Awaitable, Callable, Literal

from langgraph.types import Command, interrupt

from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult, PlanStep
from customer_service_app.domain.schemas import ChatRequest, ChatResponse, ChatTraceStep
from customer_service_app.infrastructure.llm.base import LLMClient
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.conversation_service import ConversationService
from customer_service_app.services.cost_governance_service import CostGovernanceService
from customer_service_app.services.planner_service import PlannerService
from customer_service_app.services.react_executor import ReactExecutor
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry
from customer_service_app.workflows.state import CustomerServiceGraphState


AnswerHandler = Callable[
    [ChatRequest, AgentPlan | None, PlanExecutionResult | None],
    Awaitable[ChatResponse],
]
GraphNodeName = Literal[
    "plan",
    "execute_plan",
    "await_confirmation",
    "apply_confirmation",
    "answer",
]


class CustomerServiceGraphNodes:
    """客服 Graph 的请求级业务节点。

    拓扑由 ``customer_service_graph.py`` 声明；这里负责节点内部的业务状态变更。
    每个跳转都使用 ``Command(goto=...)``，因此路由结果和状态更新会一起写入 checkpoint。
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        planner_service: PlannerService,
        react_executor: ReactExecutor,
        cost_service: CostGovernanceService,
        conversation_service: ConversationService,
        confirmation_service: ConfirmationService,
        tool_registry: ToolRegistry,
        tool_context: ToolExecutionContext,
        answer_handler: AnswerHandler,
        planning_enabled: bool = True,
    ) -> None:
        self._llm_client = llm_client
        self._planner_service = planner_service
        self._react_executor = react_executor
        self._cost_service = cost_service
        self._conversation_service = conversation_service
        self._confirmation_service = confirmation_service
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._answer_handler = answer_handler
        self._planning_enabled = planning_enabled

    async def prepare(
        self,
        state: CustomerServiceGraphState,
    ) -> Command[Literal["plan", "answer"]]:
        """创建或恢复会话，并用零 token 规则决定是否进入 Planner。"""

        request = ChatRequest.model_validate(state["request"])
        conversation = await self._conversation_service.ensure_conversation(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            first_question=request.question,
        )
        request = request.model_copy(
            update={"conversation_id": conversation.id, "thread_id": state["thread_id"]}
        )
        self._tool_context.tenant_id = request.tenant_id
        self._tool_context.user_id = request.user_id
        self._tool_context.conversation_id = conversation.id

        needs_plan = self._planning_enabled and self._planner_service.needs_plan(request)
        route: Literal["plan", "answer"] = "plan" if needs_plan else "answer"
        trace = [
            ChatTraceStep(
                stage="graph_prepare",
                detail="Graph 已完成会话初始化和请求分流",
                metadata={"route": route, "conversation_id": conversation.id},
            ).model_dump(mode="json")
        ]
        return Command(
            update={
                "request": request.model_dump(mode="json"),
                "conversation_id": conversation.id,
                "route": route,
                "status": "running",
                "plan": None,
                "plan_execution": None,
                "pending_confirmations": [],
                "confirmation_cursor": 0,
                "confirmation_decision": None,
                "trace": trace,
                "error": None,
            },
            goto=route,
        )

    async def plan(
        self,
        state: CustomerServiceGraphState,
    ) -> Command[Literal["execute_plan"]]:
        """生成步骤数受限的结构化计划，解析失败时由 Planner 规则兜底。"""

        request = ChatRequest.model_validate(state["request"])
        strategy = await self._cost_service.choose_strategy(tenant_id=request.tenant_id)
        plan, planner_response = await self._planner_service.build_plan(
            request=request,
            llm_client=self._llm_client,
            model=strategy.model,
        )
        if planner_response is not None:
            await self._cost_service.record_llm_usage(
                tenant_id=request.tenant_id,
                usage=planner_response.usage,
            )

        trace = self._append_trace(
            state,
            ChatTraceStep(
                stage="plan",
                detail="复杂请求已生成有界执行计划",
                metadata={
                    "plan_id": plan.id if plan else None,
                    "step_count": len(plan.steps) if plan else 0,
                    "source": plan.source if plan else None,
                },
            ),
        )
        return Command(
            update={
                "plan": plan.model_dump(mode="json") if plan else None,
                "trace": trace,
            },
            goto="execute_plan",
        )

    async def execute_plan(
        self,
        state: CustomerServiceGraphState,
    ) -> Command[Literal["await_confirmation", "answer"]]:
        """执行有界计划；高风险步骤先落库，再进入 Graph HIL 暂停点。"""

        request = ChatRequest.model_validate(state["request"])
        plan_payload = state.get("plan")
        if not plan_payload:
            return Command(update={"plan_execution": None}, goto="answer")

        plan = AgentPlan.model_validate(plan_payload)
        result = await self._react_executor.execute(
            plan=plan,
            tenant_id=request.tenant_id,
            question=request.question,
            context=self._tool_context,
        )
        pending = await self._create_pending_actions(
            request=request,
            thread_id=state["thread_id"],
            result=result,
        )
        trace = self._append_trace(
            state,
            ChatTraceStep(
                stage="plan_execution",
                detail="有界计划执行结束",
                metadata={
                    "completed": len(result.completed_step_ids),
                    "blocked": len(result.blocked_step_ids),
                    "failed": len(result.failed_step_ids),
                    "confirmation_count": len(pending),
                },
            ),
        )
        next_node: Literal["await_confirmation", "answer"] = (
            "await_confirmation" if pending else "answer"
        )
        return Command(
            update={
                "plan": result.plan.model_dump(mode="json"),
                "plan_execution": result.model_dump(mode="json"),
                "pending_confirmations": pending,
                "confirmation_cursor": 0,
                "status": "waiting_confirmation" if pending else "running",
                "trace": trace,
            },
            goto=next_node,
        )

    async def await_confirmation(
        self,
        state: CustomerServiceGraphState,
    ) -> Command[Literal["apply_confirmation"]]:
        """暂停当前 Graph thread，等待前端确认或拒绝高风险动作。"""

        pending = self._current_confirmation(state)
        decision = interrupt(
            {
                "type": "tool_confirmation",
                "thread_id": state["thread_id"],
                "pending_confirmation": pending,
            }
        )
        if not isinstance(decision, dict):
            raise AppError("Invalid confirmation decision", code="confirmation_invalid")
        return Command(
            update={"confirmation_decision": decision, "status": "running"},
            goto="apply_confirmation",
        )

    async def apply_confirmation(
        self,
        state: CustomerServiceGraphState,
    ) -> Command[Literal["await_confirmation", "answer"]]:
        """校验恢复参数，并在批准后只执行当前确认单绑定的那个工具。"""

        pending = self._current_confirmation(state)
        decision = state.get("confirmation_decision") or {}
        action_id = str(pending["id"])
        if str(decision.get("confirmation_id") or "") != action_id:
            raise AppError(
                "Confirmation does not match the interrupted action",
                code="confirmation_mismatch",
                status_code=409,
            )

        request = ChatRequest.model_validate(state["request"])
        execution = PlanExecutionResult.model_validate(state["plan_execution"])
        step = self._find_step(execution, str(pending["step_id"]))
        decision_name = str(decision.get("decision") or "").lower()
        if decision_name == "approve":
            await self._confirmation_service.approve(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                action_id=action_id,
                comment=str(decision.get("reason") or "") or None,
            )
            try:
                result = await self._execute_confirmed_tool(pending)
            except Exception as exc:
                await self._confirmation_service.mark_failed(
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    action_id=action_id,
                    error_message=str(exc),
                )
                raise
            else:
                await self._confirmation_service.mark_executed(
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    action_id=action_id,
                    result=result,
                )
            self._complete_step(execution, step, result)
            detail = "确认动作已执行，Graph 继续运行"
        elif decision_name == "reject":
            await self._confirmation_service.reject(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                action_id=action_id,
                comment=str(decision.get("reason") or "") or None,
            )
            result = {"status": "rejected", "message": "用户已取消该操作"}
            self._skip_step(execution, step, result)
            detail = "确认动作已拒绝，Graph 继续生成说明"
        else:
            raise AppError(
                "Decision must be approve or reject",
                code="confirmation_invalid",
                status_code=400,
            )

        cursor = int(state.get("confirmation_cursor", 0)) + 1
        pending_items = state.get("pending_confirmations", [])
        has_next = cursor < len(pending_items)
        trace = self._append_trace(
            state,
            ChatTraceStep(
                stage="hil_resume",
                detail=detail,
                metadata={
                    "confirmation_id": action_id,
                    "decision": decision_name,
                    "remaining": max(len(pending_items) - cursor, 0),
                },
            ),
        )
        next_node: Literal["await_confirmation", "answer"] = (
            "await_confirmation" if has_next else "answer"
        )
        return Command(
            update={
                "plan": execution.plan.model_dump(mode="json"),
                "plan_execution": execution.model_dump(mode="json"),
                "confirmation_cursor": cursor,
                "confirmation_decision": None,
                "status": "waiting_confirmation" if has_next else "running",
                "trace": trace,
            },
            goto=next_node,
        )

    async def answer(self, state: CustomerServiceGraphState) -> CustomerServiceGraphState:
        """调用现有回答链路生成回复，并把最终响应写回 Graph State。"""

        request = ChatRequest.model_validate(state["request"])
        plan = AgentPlan.model_validate(state["plan"]) if state.get("plan") else None
        execution = (
            PlanExecutionResult.model_validate(state["plan_execution"])
            if state.get("plan_execution")
            else None
        )
        response = await self._answer_handler(request, plan, execution)
        response.thread_id = state["thread_id"]
        response.status = "completed"
        response.trace = [
            *[ChatTraceStep.model_validate(item) for item in state.get("trace", [])],
            *response.trace,
        ]
        return {
            "response": response.model_dump(mode="json"),
            "status": "completed",
            "trace": [item.model_dump(mode="json") for item in response.trace],
        }

    def to_response(self, state: dict[str, object]) -> ChatResponse:
        """把完成态或中断态 State 转成稳定的 API 响应。"""

        response_payload = state.get("response")
        if isinstance(response_payload, dict):
            return ChatResponse.model_validate(response_payload)

        pending_items = state.get("pending_confirmations")
        cursor = int(state.get("confirmation_cursor") or 0)
        pending = None
        if isinstance(pending_items, list) and cursor < len(pending_items):
            pending = pending_items[cursor]
        trace = [
            ChatTraceStep.model_validate(item)
            for item in state.get("trace", [])
            if isinstance(item, dict)
        ]
        plan = AgentPlan.model_validate(state["plan"]) if state.get("plan") else None
        execution = (
            PlanExecutionResult.model_validate(state["plan_execution"])
            if state.get("plan_execution")
            else None
        )
        return ChatResponse(
            conversation_id=str(state.get("conversation_id") or ""),
            thread_id=str(state.get("thread_id") or ""),
            status=str(state.get("status") or "waiting_confirmation"),
            answer="该操作需要确认后才能继续。",
            pending_confirmation=pending,
            plan=plan,
            plan_execution=execution,
            trace=trace,
        )

    async def _create_pending_actions(
        self,
        *,
        request: ChatRequest,
        thread_id: str,
        result: PlanExecutionResult,
    ) -> list[dict[str, object]]:
        pending: list[dict[str, object]] = []
        for step in result.plan.steps:
            if step.status != "blocked" or not step.tool_name:
                continue
            action = await self._confirmation_service.create_pending_action(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                thread_id=thread_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
            )
            item = action.model_dump(mode="json")
            item["step_id"] = step.id
            pending.append(item)
            step.observation.update(
                {
                    "action_id": action.id,
                    "confirmation_id": action.confirmation_id,
                    "status": action.status,
                }
            )
            result.observations[step.id] = step.observation
        return pending

    async def _execute_confirmed_tool(self, pending: dict[str, object]) -> dict:
        tool_name = str(pending["tool_name"])
        arguments = pending.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise AppError("Invalid pending tool arguments", code="confirmation_invalid")

        self._tool_context.confirmed_tools.add(tool_name)
        try:
            return await self._tool_registry.execute(
                name=tool_name,
                arguments_json=json.dumps(arguments, ensure_ascii=False),
                context=self._tool_context,
            )
        finally:
            self._tool_context.confirmed_tools.discard(tool_name)

    @staticmethod
    def _current_confirmation(state: CustomerServiceGraphState) -> dict[str, object]:
        pending = state.get("pending_confirmations", [])
        cursor = int(state.get("confirmation_cursor", 0))
        if cursor >= len(pending):
            raise AppError("No pending confirmation", code="confirmation_missing", status_code=409)
        return pending[cursor]

    @staticmethod
    def _find_step(execution: PlanExecutionResult, step_id: str) -> PlanStep:
        for step in execution.plan.steps:
            if step.id == step_id:
                return step
        raise AppError("Plan step not found", code="plan_step_missing", status_code=409)

    @staticmethod
    def _complete_step(
        execution: PlanExecutionResult,
        step: PlanStep,
        result: dict,
    ) -> None:
        step.status = "completed"
        step.observation = result
        execution.observations[step.id] = result
        execution.blocked_step_ids = [item for item in execution.blocked_step_ids if item != step.id]
        if step.id not in execution.completed_step_ids:
            execution.completed_step_ids.append(step.id)

    @staticmethod
    def _skip_step(
        execution: PlanExecutionResult,
        step: PlanStep,
        result: dict,
    ) -> None:
        step.status = "skipped"
        step.observation = result
        execution.observations[step.id] = result
        execution.blocked_step_ids = [item for item in execution.blocked_step_ids if item != step.id]
        if step.id not in execution.skipped_step_ids:
            execution.skipped_step_ids.append(step.id)

    @staticmethod
    def _append_trace(
        state: CustomerServiceGraphState,
        step: ChatTraceStep,
    ) -> list[dict]:
        return [*state.get("trace", []), step.model_dump(mode="json")]
