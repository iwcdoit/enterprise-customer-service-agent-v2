from __future__ import annotations

import json
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult
from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatTraceStep,
    GraphStateView,
    KnowledgeChunk,
    PendingActionView,
    ToolCallView,
    ToolResultView,
)
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.llm.base import LLMClient, LLMToolCall
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.prompts.customer_service import (
    CUSTOMER_SERVICE_SYSTEM_PROMPT,
    format_knowledge_context,
)
from customer_service_app.services.business_gateway import BusinessGateway
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.conversation_service import ConversationService
from customer_service_app.services.conversation_memory import ConversationMemoryCompactor
from customer_service_app.services.cost_governance_service import CostGovernanceService
from customer_service_app.services.human_support_service import HumanSupportService
from customer_service_app.services.planner_service import PlannerService
from customer_service_app.services.question_preprocessor import QuestionPreprocessor
from customer_service_app.services.rag_service import RagService
from customer_service_app.services.react_executor import ReactExecutor
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry
from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.nodes import CustomerServiceGraphNodes
from customer_service_app.workflows.runtime import CustomerServiceGraphRuntime


class CustomerServiceAgent:
    """一轮客服请求的业务编排入口。

    Agent 负责协调 LLM、RAG、工具、语义缓存、数据库和提示词。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        session: AsyncSession,
        graph_runtime: CustomerServiceGraphRuntime,
        llm_client: LLMClient,
        rag_service: RagService,
        tool_registry: ToolRegistry,
        search_client: SerpApiSearchClient,
        business_gateway: BusinessGateway,
        semantic_cache: RedisSemanticCache | None,
        cost_service: CostGovernanceService,
        human_support_service: HumanSupportService,
    ):
        """注入处理一轮客服请求所需的依赖。"""

        self._settings = settings
        self._session = session
        self._llm_client = llm_client
        self._rag_service = rag_service
        self._tool_registry = tool_registry
        self._search_client = search_client
        self._business_gateway = business_gateway
        self._semantic_cache = semantic_cache
        self._cost_service = cost_service
        self._human_support_service = human_support_service
        self._graph_runtime = graph_runtime
        self._conversation_service = ConversationService(session)
        self._confirmation_service = ConfirmationService(session, settings)
        self._memory_compactor = ConversationMemoryCompactor()
        self._question_preprocessor = QuestionPreprocessor()

        tool_context = ToolExecutionContext(
            tenant_id="",
            user_id="",
            conversation_id=None,
            session=session,
            search_client=search_client,
            business_gateway=business_gateway,
        )
        planner_service = PlannerService(
            tool_registry=tool_registry,
            max_steps=settings.plan_max_steps,
        )
        react_executor = ReactExecutor(
            rag_service=rag_service,
            tool_registry=tool_registry,
            max_steps=settings.plan_max_steps,
            step_timeout_seconds=settings.react_step_timeout_seconds,
        )
        graph_nodes = CustomerServiceGraphNodes(
            llm_client=llm_client,
            planner_service=planner_service,
            react_executor=react_executor,
            cost_service=cost_service,
            conversation_service=self._conversation_service,
            confirmation_service=self._confirmation_service,
            tool_registry=tool_registry,
            tool_context=tool_context,
            answer_handler=self._answer_turn,
            planning_enabled=settings.planning_enabled,
        )
        self._graph_nodes = graph_nodes
        self._graph_context = CustomerServiceGraphContext(nodes=graph_nodes)

    async def answer(self, request: ChatRequest) -> ChatResponse:
        """统一从 LangGraph 编排入口处理一轮客服请求。"""

        try:
            # 人工接管后，会话入口只转发并保存消息，不再同时触发 Bot。
            if request.conversation_id:
                handoff = await self._human_support_service.get_active_for_customer(
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                )
                if handoff is not None:
                    await self._human_support_service.accept_customer_message(
                        handoff=handoff,
                        content=request.question,
                        metadata=request.metadata,
                    )
                    answer = (
                        "消息已发送给人工客服，当前会话由模拟坐席处理。"
                        if handoff.assigned_agent_id
                        else "已进入人工客服队列，坐席接入后会继续处理。"
                    )
                    await self._human_support_service.add_service_notice(
                        handoff=handoff,
                        content=answer,
                    )
                    await self._session.commit()
                    return ChatResponse(
                        conversation_id=request.conversation_id,
                        thread_id=request.thread_id,
                        answer=answer,
                        status="human_active" if handoff.assigned_agent_id else "waiting_human",
                        service_mode="human",
                        human_handoff=self._human_support_service.to_view(handoff),
                        trace=[
                            ChatTraceStep(
                                stage="human_support",
                                detail="会话已由人工客服接管，本轮未调用模型",
                            )
                        ],
                    )
            state = await self._graph_runtime.invoke(
                request_payload=request.model_dump(mode="json"),
                context=self._graph_context,
                thread_id=request.thread_id,
            )
            await self._session.commit()
            return self._graph_nodes.to_response(state)
        except Exception:
            await self._session.rollback()
            raise

    async def resume_confirmation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        confirmation_id: str,
        decision: str,
        reason: str | None,
    ) -> ChatResponse:
        """恢复确认单绑定的原 LangGraph thread，而不是重新发起聊天。"""

        try:
            action = await self._confirmation_service.get_owned(
                tenant_id=tenant_id,
                user_id=user_id,
                action_id=confirmation_id,
            )
            if not action.thread_id:
                raise AppError(
                    "Confirmation is not bound to a graph thread",
                    code="confirmation_thread_missing",
                    status_code=409,
                )
            state = await self._graph_runtime.resume(
                thread_id=action.thread_id,
                decision={
                    "confirmation_id": confirmation_id,
                    "decision": decision,
                    "reason": reason,
                },
                context=self._graph_context,
            )
            await self._session.commit()
            return self._graph_nodes.to_response(state)
        except Exception:
            await self._session.rollback()
            raise

    async def graph_state(self, *, thread_id: str) -> GraphStateView:
        """读取指定 Graph thread 的脱敏 checkpoint 快照。"""

        return await self._graph_runtime.get_state(thread_id=thread_id)

    async def get_confirmation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        confirmation_id: str,
    ) -> PendingActionView:
        """读取当前用户自己的确认单。"""

        action = await self._confirmation_service.get_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            action_id=confirmation_id,
        )
        return self._confirmation_service.to_view(action)

    async def _answer_turn(
        self,
        request: ChatRequest,
        plan: AgentPlan | None,
        plan_execution: PlanExecutionResult | None,
    ) -> ChatResponse:
        """处理一轮完整的客服问答。

        这是项目最核心的主流程：会话 -> 缓存 -> RAG -> LLM 决策 -> 工具执行
        -> 二次 LLM -> 保存消息 -> 返回响应。
        """

        # Trace 记录本轮请求的主要处理阶段，并返回给运营验证台展示。
        trace: list[ChatTraceStep] = []
        question_profile = self._question_preprocessor.analyze(request.question)
        if question_profile.normalized_question != request.question:
            request = request.model_copy(update={"question": question_profile.normalized_question})
        trace.append(
            ChatTraceStep(
                stage="preprocess",
                detail="完成用户问题标准化和轻量意图识别",
                metadata=question_profile.as_trace_metadata(),
            )
        )

        conversation = await self._conversation_service.ensure_conversation(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            first_question=request.question,
        )
        trace.append(ChatTraceStep(stage="conversation", detail="会话已加载或创建"))

        cost_strategy = await self._cost_service.choose_strategy(tenant_id=request.tenant_id)
        trace.append(
            ChatTraceStep(
                stage="cost",
                detail="已选择本轮模型和上下文策略",
                metadata=cost_strategy.model_dump(),
            )
        )

        # 复杂计划可能已经执行了查询或业务工具，不能复用旧业务状态下的语义缓存答案。
        skip_cache = bool(request.metadata.get("skip_cache")) or plan is not None
        if self._semantic_cache is not None and skip_cache:
            trace.append(ChatTraceStep(stage="cache", detail="本次请求已跳过语义缓存"))

        if self._semantic_cache is not None and not skip_cache:
            cached = await self._semantic_cache.lookup(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                question=request.question,
            )
            if cached is not None:
                trace.append(
                    ChatTraceStep(
                        stage="cache",
                        detail="命中 Redis 语义缓存",
                        metadata={"similarity": cached.similarity},
                    )
                )
                await self._conversation_service.save_turn(
                    conversation_id=conversation.id,
                    question=request.question,
                    answer=cached.answer,
                    metadata={"assistant": {"cache_hit": True}},
                )
                await self._session.commit()
                return ChatResponse(
                    conversation_id=conversation.id,
                    answer=cached.answer,
                    cache_hit=True,
                    plan=plan,
                    plan_execution=plan_execution,
                    trace=trace,
                )
            trace.append(ChatTraceStep(stage="cache", detail="语义缓存未命中"))

        knowledge = await self._rag_service.retrieve(
            tenant_id=request.tenant_id,
            question=request.question,
            top_k=cost_strategy.rag_top_k,
        )
        trace.append(
            ChatTraceStep(
                stage="rag",
                detail="完成知识库检索",
                metadata={"chunk_count": len(knowledge)},
            )
        )

        messages = await self._build_llm_messages(
            request,
            conversation.id,
            knowledge,
            trace,
            history_turns=cost_strategy.history_turns,
            plan_execution=plan_execution,
        )

        # 计划请求已经在 Graph 中执行过有界步骤。
        # 最终模型只总结 observation，不能再次调用同一批工具造成重复业务操作。
        if plan_execution is not None:
            first_response = await self._llm_client.chat(
                messages,
                model=cost_strategy.model,
            )
        else:
            # 工具定义通过 LLM API 的结构化 tools 参数传入，不拼进用户问题。
            # 模型可以根据问题决定不调用工具，或返回一个或多个 tool_calls。
            first_response = await self._llm_client.chat(
                messages,
                tools=self._tool_registry.definitions(),
                tool_choice="auto",
                model=cost_strategy.model,
            )
        await self._cost_service.record_llm_usage(
            tenant_id=request.tenant_id,
            usage=first_response.usage,
        )
        trace.append(
            ChatTraceStep(
                stage="llm_decision",
                detail="模型完成首轮回答或工具调用决策",
                metadata={"finish_reason": first_response.finish_reason},
            )
        )

        if plan_execution is not None:
            tool_calls, tool_results = self._planned_tool_views(plan_execution)
        else:
            tool_calls, tool_results = await self._execute_tool_calls(
                request=request,
                conversation_id=conversation.id,
                calls=first_response.tool_calls,
                messages=messages,
            )

        if tool_results and plan_execution is None:
            pending_count = sum(
                1 for item in tool_results if item.payload.get("requires_confirmation") is True
            )
            trace.append(
                ChatTraceStep(
                    stage="tools",
                    detail="已执行模型请求的工具",
                    metadata={"tool_count": len(tool_results), "pending_actions": pending_count},
                )
            )
            final_response = await self._llm_client.chat(messages, model=cost_strategy.model)
            await self._cost_service.record_llm_usage(
                tenant_id=request.tenant_id,
                usage=final_response.usage,
            )
            answer = final_response.content
        else:
            answer = first_response.content

        await self._conversation_service.save_turn(
            conversation_id=conversation.id,
            question=request.question,
            answer=answer,
            metadata={
                "assistant": {
                    "knowledge_count": len(knowledge),
                    "tool_calls": [item.model_dump() for item in tool_calls],
                    "tool_results": [item.model_dump() for item in tool_results],
                }
            },
        )

        if self._semantic_cache is not None and answer and not skip_cache:
            await self._semantic_cache.update(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                question=request.question,
                answer=answer,
                metadata={"conversation_id": conversation.id},
            )
            trace.append(ChatTraceStep(stage="cache", detail="已写入语义缓存"))

        await self._session.commit()
        trace.append(ChatTraceStep(stage="done", detail="回答已生成并落库"))
        return ChatResponse(
            conversation_id=conversation.id,
            answer=answer,
            cache_hit=False,
            knowledge=knowledge,
            tool_calls=tool_calls,
            tool_results=tool_results,
            plan=plan,
            plan_execution=plan_execution,
            trace=trace,
        )

    async def stream_answer(self, request: ChatRequest) -> AsyncIterator[str]:
        """SSE 流式接口。

        当前实现是“事件级流式”：先把 trace、knowledge、tool_result、answer
        分事件发给前端。以后如果要做 token 级流式，可以在工具决策完成后扩展。
        """

        response = await self.answer(request)
        for step in response.trace:
            yield self._sse("trace", step.model_dump())
        for chunk in response.knowledge:
            yield self._sse("knowledge", chunk.model_dump())
        for tool_result in response.tool_results:
            yield self._sse("tool_result", tool_result.model_dump())
        if response.pending_confirmation is not None:
            yield self._sse(
                "confirmation_required",
                response.pending_confirmation.model_dump(mode="json"),
            )
        yield self._sse("answer", {"content": response.answer})
        yield self._sse("done", response.model_dump())

    async def _build_llm_messages(
        self,
        request: ChatRequest,
        conversation_id: str,
        knowledge: list[KnowledgeChunk],
        trace: list[ChatTraceStep],
        history_turns: int,
        plan_execution: PlanExecutionResult | None = None,
    ) -> list[dict[str, Any]]:
        """用系统规则、历史消息、知识块和当前问题组装模型 messages。"""

        history = request.history
        if not history:
            history = await self._conversation_service.recent_history(
                conversation_id,
                limit=history_turns,
            )
        elif len(history) > history_turns:
            history = history[-history_turns:]
        memory_window = self._memory_compactor.compact(history)
        trace.append(
            ChatTraceStep(
                stage="memory",
                detail="已整理短期历史消息",
                metadata={
                    "original_count": memory_window.original_count,
                    "compressed_count": memory_window.compressed_count,
                    "final_count": len(memory_window.messages),
                    "compressed": memory_window.compressed,
                },
            )
        )
        knowledge_context = format_knowledge_context([item.model_dump() for item in knowledge])
        system_prompt = CUSTOMER_SERVICE_SYSTEM_PROMPT.format(
            tenant_id=request.tenant_id,
            knowledge_context=knowledge_context,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._to_llm_messages(memory_window.messages))
        if plan_execution is not None:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是本轮有界执行计划的真实后端观察结果。请基于结果回答，"
                        "不要声称执行了结果中未完成的动作：\n"
                        + json.dumps(
                            plan_execution.model_dump(mode="json"),
                            ensure_ascii=False,
                        )
                    ),
                }
            )
        messages.append({"role": "user", "content": request.question})
        return messages

    @staticmethod
    def _planned_tool_views(
        execution: PlanExecutionResult,
    ) -> tuple[list[ToolCallView], list[ToolResultView]]:
        """把计划中的工具 observation 转为普通工具调用共用的 API 返回结构。"""

        calls: list[ToolCallView] = []
        results: list[ToolResultView] = []
        for step in execution.plan.steps:
            if step.action_type != "tool" or not step.tool_name:
                continue
            call_id = f"plan:{step.id}"
            calls.append(
                ToolCallView(
                    id=call_id,
                    name=step.tool_name,
                    arguments=step.arguments,
                )
            )
            results.append(
                ToolResultView(
                    tool_call_id=call_id,
                    name=step.tool_name,
                    ok=step.status not in {"failed", "skipped"},
                    payload=step.observation,
                )
            )
        return calls, results

    async def _execute_tool_calls(
        self,
        *,
        request: ChatRequest,
        conversation_id: str,
        calls: list[LLMToolCall],
        messages: list[dict[str, Any]],
    ) -> tuple[list[ToolCallView], list[ToolResultView]]:
        """执行模型请求的工具，并把工具结果追加到 messages。"""

        if not calls:
            return [], []

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call.as_openai_tool_call() for call in calls],
            }
        )
        raw_confirmed_tools = request.metadata.get("confirmed_tools", [])
        if isinstance(raw_confirmed_tools, str):
            raw_confirmed_tools = [raw_confirmed_tools]
        confirmed_tools = {str(item) for item in raw_confirmed_tools if str(item).strip()}

        context = ToolExecutionContext(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            conversation_id=conversation_id,
            session=self._session,
            search_client=self._search_client,
            business_gateway=self._business_gateway,
            confirmed_tools=confirmed_tools,
        )

        tool_call_views: list[ToolCallView] = []
        tool_result_views: list[ToolResultView] = []
        for call in calls:
            arguments = json.loads(call.arguments or "{}")
            tool_call_views.append(ToolCallView(id=call.id, name=call.name, arguments=arguments))
            try:
                payload = await self._tool_registry.execute(
                    name=call.name,
                    arguments_json=call.arguments,
                    context=context,
                )
                if payload.get("requires_confirmation") is True:
                    pending_action = await self._confirmation_service.create_pending_action(
                        tenant_id=request.tenant_id,
                        user_id=request.user_id,
                        conversation_id=conversation_id,
                        tool_name=call.name,
                        arguments=arguments,
                    )
                    payload = {
                        **payload,
                        "action_id": pending_action.id,
                        "status": pending_action.status,
                    }
                ok = True
            except Exception as exc:
                payload = {"error": str(exc)}
                ok = False

            tool_result_views.append(
                ToolResultView(
                    tool_call_id=call.id,
                    name=call.name,
                    ok=ok,
                    payload=payload,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )
        return tool_call_views, tool_result_views

    @staticmethod
    def _to_llm_messages(history: list[ChatMessage]) -> list[dict[str, str]]:
        """把内部消息对象转换为 OpenAI-compatible 消息字典。"""

        return [{"role": item.role, "content": item.content} for item in history]

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> str:
        """把一个事件序列化为 Server-Sent Events 文本。"""

        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
