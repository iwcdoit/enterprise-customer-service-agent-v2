from __future__ import annotations

from typing import Awaitable, Callable

from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult
from customer_service_app.domain.schemas import ChatRequest, ChatResponse, ChatTraceStep
from customer_service_app.infrastructure.llm.base import LLMClient
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.conversation_service import ConversationService
from customer_service_app.services.cost_governance_service import CostGovernanceService
from customer_service_app.services.planner_service import PlannerService
from customer_service_app.services.react_executor import ReactExecutor
from customer_service_app.services.tool_registry import ToolExecutionContext
from customer_service_app.workflows.state import CustomerServiceGraphState


AnswerHandler = Callable[
    [ChatRequest, AgentPlan | None, PlanExecutionResult | None],
    Awaitable[ChatResponse],
]


class CustomerServiceGraphNodes:
    """LangGraph 拓扑使用的请求级业务节点集合。"""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        planner_service: PlannerService,
        react_executor: ReactExecutor,
        cost_service: CostGovernanceService,
        conversation_service: ConversationService,
        confirmation_service: ConfirmationService,
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
        self._tool_context = tool_context
        self._answer_handler = answer_handler
        self._planning_enabled = planning_enabled

    async def prepare(self, state: CustomerServiceGraphState) -> CustomerServiceGraphState:
        """校验请求、创建或加载会话，并选择简单回答或计划执行分支。"""

        request = ChatRequest.model_validate(state["request"])
        conversation = await self._conversation_service.ensure_conversation(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            first_question=request.question,
        )
        request = request.model_copy(update={"conversation_id": conversation.id})
        self._tool_context.tenant_id = request.tenant_id
        self._tool_context.user_id = request.user_id
        self._tool_context.conversation_id = conversation.id
        needs_plan = self._planning_enabled and self._planner_service.needs_plan(request)
        trace = [
            ChatTraceStep(
                stage="graph_prepare",
                detail="LangGraph 已完成请求路由",
                metadata={
                    "route": "plan" if needs_plan else "answer",
                    "conversation_id": conversation.id,
                },
            ).model_dump(mode="json")
        ]
        return {
            "request": request.model_dump(mode="json"),
            "route": "plan" if needs_plan else "answer",
            "plan": None,
            "plan_execution": None,
            "trace": trace,
            "error": None,
        }

    async def plan(self, state: CustomerServiceGraphState) -> CustomerServiceGraphState:
        """为复杂、多意图请求生成步骤数受限的执行计划。"""

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
        return {
            "plan": plan.model_dump(mode="json") if plan else None,
            "trace": trace,
        }

    async def execute_plan(self, state: CustomerServiceGraphState) -> CustomerServiceGraphState:
        """执行一次计划，并为被安全规则阻断的动作创建待确认记录。"""

        request = ChatRequest.model_validate(state["request"])
        plan_payload = state.get("plan")
        if not plan_payload:
            return {"plan_execution": None}

        plan = AgentPlan.model_validate(plan_payload)
        result = await self._react_executor.execute(
            plan=plan,
            tenant_id=request.tenant_id,
            question=request.question,
            context=self._tool_context,
        )
        await self._create_pending_actions(request=request, result=result)
        trace = self._append_trace(
            state,
            ChatTraceStep(
                stage="plan_execution",
                detail="有界 ReAct 执行已结束",
                metadata={
                    "completed": len(result.completed_step_ids),
                    "blocked": len(result.blocked_step_ids),
                    "failed": len(result.failed_step_ids),
                    "skipped": len(result.skipped_step_ids),
                },
            ),
        )
        return {
            "plan": result.plan.model_dump(mode="json"),
            "plan_execution": result.model_dump(mode="json"),
            "trace": trace,
        }

    async def answer(self, state: CustomerServiceGraphState) -> CustomerServiceGraphState:
        """调用现有客服单轮处理逻辑生成最终回答并完成落库。"""

        request = ChatRequest.model_validate(state["request"])
        plan = AgentPlan.model_validate(state["plan"]) if state.get("plan") else None
        execution = (
            PlanExecutionResult.model_validate(state["plan_execution"])
            if state.get("plan_execution")
            else None
        )
        response = await self._answer_handler(request, plan, execution)
        response.trace = [
            *[ChatTraceStep.model_validate(item) for item in state.get("trace", [])],
            *response.trace,
        ]
        return {"response": response.model_dump(mode="json")}

    async def _create_pending_actions(
        self,
        *,
        request: ChatRequest,
        result: PlanExecutionResult,
    ) -> None:
        for step in result.plan.steps:
            if step.status != "blocked" or not step.tool_name:
                continue
            pending = await self._confirmation_service.create_pending_action(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
            )
            step.observation.update(
                {
                    "action_id": pending.id,
                    "confirmation_id": pending.confirmation_id,
                    "status": pending.status,
                }
            )
            result.observations[step.id] = step.observation

    @staticmethod
    def _append_trace(
        state: CustomerServiceGraphState,
        step: ChatTraceStep,
    ) -> list[dict]:
        return [*state.get("trace", []), step.model_dump(mode="json")]
