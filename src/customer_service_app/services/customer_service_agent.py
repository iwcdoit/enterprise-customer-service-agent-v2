from __future__ import annotations

import json
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.schemas import ChatRequest, ChatResponse, GraphStateView
from customer_service_app.domain.confirmations import PendingActionView
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.human_support_service import HumanSupportService
from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.nodes import CustomerServiceGraphNodes
from customer_service_app.workflows.runtime import CustomerServiceGraphRuntime


class CustomerServiceAgent:
    """客服 Agent 的 API 门面。

    这一层可以理解为 Java 里的 Application Service / Facade：
    - API 路由只调用 Agent，不直接碰 LLM、RAG、Tool、LangGraph 节点。
    - Agent 负责事务边界、Graph 启动、Graph 恢复、响应对象转换。
    - 真正的业务步骤在 CustomerServiceGraphNodes 中，不写在这里。
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        graph_runtime: CustomerServiceGraphRuntime,
        graph_nodes: CustomerServiceGraphNodes,
        confirmation_service: ConfirmationService,
        human_support_service: HumanSupportService,
    ):
        self._session = session

        self._graph_runtime = graph_runtime

        self._graph_nodes = graph_nodes

        self._confirmation_service = confirmation_service
        self._human_support_service = human_support_service

        self._graph_context = CustomerServiceGraphContext(nodes=graph_nodes)

    async def answer(self, request: ChatRequest) -> ChatResponse:
        """启动一轮普通聊天请求。

        这里的“启动”不是直接调用大模型，而是把请求交给 LangGraph。
        Graph 可能完整跑完，也可能在 HIL 节点 interrupt 暂停。
        """
        try:
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
                    await self._session.commit()
                    return ChatResponse(
                        conversation_id=request.conversation_id,
                        thread_id=request.thread_id,
                        answer=(
                            "消息已发送给人工客服，当前会话由人工坐席处理。"
                            if handoff.assigned_agent_id
                            else "已进入人工客服队列，消息已保存，坐席接入后会继续处理。"
                        ),
                        status=(
                            "human_active" if handoff.assigned_agent_id else "waiting_human"
                        ),
                        service_mode="human",
                        human_handoff=self._human_support_service.to_view(handoff),
                    )

            request_payload = request.model_dump(mode="json")

            state = await self._graph_runtime.invoke(
                request_payload=request_payload,
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
        """根据用户确认/拒绝结果恢复被 interrupt 暂停的 Graph。

        注意：恢复必须使用确认单里保存的 langgraph_thread_id。
        这能确保恢复的是原来那条暂停的执行链路，而不是重新发起一次聊天。
        """
        try:
            action = await self._confirmation_service.get_owned(
                tenant_id=tenant_id,
                user_id=user_id,
                action_id=confirmation_id,
            )
            if not action.langgraph_thread_id:
                raise AppError(
                    "Confirmation is not bound to a graph thread",
                    code="confirmation_thread_missing",
                    status_code=409,
                )

            decision_payload = {
                "confirmation_id": confirmation_id,
                "decision": decision,
                "reason": reason,
            }
            state = await self._graph_runtime.resume(
                thread_id=action.langgraph_thread_id,
                decision=decision_payload,
                context=self._graph_context,
            )
            await self._session.commit()
            return self._graph_nodes.to_response(state)
        except Exception:
            await self._session.rollback()
            raise

    async def graph_state(self, *, thread_id: str) -> GraphStateView:
        """读取指定 LangGraph thread 的 checkpoint 快照。"""
        return await self._graph_runtime.get_state(thread_id=thread_id)

    async def get_confirmation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        confirmation_id: str,
    ) -> PendingActionView:
        """查询当前用户自己的待确认动作。"""
        action = await self._confirmation_service.get_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            action_id=confirmation_id,
        )
        return self._confirmation_service.to_view(action)

    async def stream_answer(self, request: ChatRequest) -> AsyncIterator[str]:
        """以 SSE 事件流形式返回聊天结果。"""
        response = await self.answer(request)

        for item in response.trace:
            yield self._sse("trace", item.model_dump(mode="json"))

        if response.plan is not None:
            yield self._sse("plan", response.plan.model_dump(mode="json"))

        for item in response.knowledge:
            yield self._sse("knowledge", item.model_dump(mode="json"))

        for item in response.tool_calls:
            yield self._sse("tool_call", item.model_dump(mode="json"))

        for item in response.tool_results:
            yield self._sse("tool_result", item.model_dump(mode="json"))

        if response.pending_confirmation is not None:
            yield self._sse(
                "confirmation_required",
                response.pending_confirmation.model_dump(mode="json"),
            )

        yield self._sse(
            "answer",
            {
                "conversation_id": response.conversation_id,
                "thread_id": response.thread_id,
                "run_id": response.run_id,
                "status": response.status,
                "answer": response.answer,
            },
        )
        yield self._sse("done", {"status": response.status})

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> str:
        """把一个事件序列化成浏览器可识别的 SSE 文本帧。"""
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"
