from __future__ import annotations

import json
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatTraceStep,
    KnowledgeChunk,
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
from customer_service_app.services.question_preprocessor import QuestionPreprocessor
from customer_service_app.services.rag_service import RagService
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry


class CustomerServiceAgent:
    """Production orchestration for one customer service turn.

    The agent coordinates LLM, RAG, tools, semantic cache, database, and prompts.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        session: AsyncSession,
        llm_client: LLMClient,
        rag_service: RagService,
        tool_registry: ToolRegistry,
        search_client: SerpApiSearchClient,
        business_gateway: BusinessGateway,
        semantic_cache: RedisSemanticCache | None,
        cost_service: CostGovernanceService,
    ):
        """Inject the dependencies needed to handle a customer-service turn."""

        self._settings = settings
        self._session = session
        self._llm_client = llm_client
        self._rag_service = rag_service
        self._tool_registry = tool_registry
        self._search_client = search_client
        self._business_gateway = business_gateway
        self._semantic_cache = semantic_cache
        self._cost_service = cost_service
        self._conversation_service = ConversationService(session)
        self._confirmation_service = ConfirmationService(session)
        self._memory_compactor = ConversationMemoryCompactor()
        self._question_preprocessor = QuestionPreprocessor()

    async def answer(self, request: ChatRequest) -> ChatResponse:
        """处理一轮完整的客服问答。

        这是项目最核心的主流程：会话 -> 缓存 -> RAG -> LLM 决策 -> 工具执行
        -> 二次 LLM -> 保存消息 -> 返回响应。
        """

        # Trace records the main processing stages returned to the ops console.
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

        skip_cache = bool(request.metadata.get("skip_cache"))
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
        )

        # Tool definitions are passed as a structured API parameter.
        # They are not embedded in the user question.
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

        tool_calls, tool_results = await self._execute_tool_calls(
            request=request,
            conversation_id=conversation.id,
            calls=first_response.tool_calls,
            messages=messages,
        )

        if tool_results:
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
        yield self._sse("answer", {"content": response.answer})
        yield self._sse("done", response.model_dump())

    async def _build_llm_messages(
        self,
        request: ChatRequest,
        conversation_id: str,
        knowledge: list[KnowledgeChunk],
        trace: list[ChatTraceStep],
        history_turns: int,
    ) -> list[dict[str, Any]]:
        """Build model messages from policy prompt, history, knowledge, and the current question."""

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
        messages.append({"role": "user", "content": request.question})
        return messages

    async def _execute_tool_calls(
        self,
        *,
        request: ChatRequest,
        conversation_id: str,
        calls: list[LLMToolCall],
        messages: list[dict[str, Any]],
    ) -> tuple[list[ToolCallView], list[ToolResultView]]:
        """Execute tool calls requested by the model and append tool results to messages."""

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
        """Convert internal chat messages to OpenAI-compatible message dicts."""

        return [{"role": item.role, "content": item.content} for item in history]

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> str:
        """Serialize one Server-Sent Events message."""

        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
