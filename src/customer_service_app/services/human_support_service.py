from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.human_support import HumanHandoffView
from customer_service_app.infrastructure.db.models import Conversation, HumanHandoffSession
from customer_service_app.infrastructure.db.repositories import (
    ConversationRepository,
    HumanHandoffRepository,
)


class HumanSupportService:
    """模拟人工坐席的接管、回复和结束流程。

    这里不实现排班、在线状态和即时消息网关，只保留 Agent 项目需要验证的会话所有权切换。
    """

    def __init__(self, session: AsyncSession):
        self._handoffs = HumanHandoffRepository(session)
        self._conversations = ConversationRepository(session)

    async def start_handoff(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        support_ticket_id: str | None,
        origin_thread_id: str | None,
        reason: str,
        priority: str,
        queue_name: str,
        idempotency_key: str,
    ) -> HumanHandoffSession:
        """用户确认转人工后，把会话切换到等待坐席状态。"""
        conversation = await self._require_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        existing = await self._handoffs.get_active_for_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if existing is not None:
            return existing

        item = await self._handoffs.create_or_get(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            support_ticket_id=support_ticket_id,
            origin_thread_id=origin_thread_id,
            reason=reason,
            priority=priority,
            queue_name=queue_name,
            idempotency_key=idempotency_key,
        )
        conversation.service_mode = "waiting_human"
        await self._handoffs.flush()
        return item

    async def get_active_for_customer(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> HumanHandoffSession | None:
        return await self._handoffs.get_active_for_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def accept_customer_message(
        self,
        *,
        handoff: HumanHandoffSession,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """人工接管期间保存客户消息，但不触发 Bot。"""
        await self._conversations.append_message(
            conversation_id=handoff.conversation_id,
            role="user",
            content=content,
            metadata={
                **metadata,
                "channel": "human_support",
                "handoff_id": handoff.id,
            },
        )

    async def add_service_notice(
        self,
        *,
        handoff: HumanHandoffSession,
        content: str,
    ) -> None:
        """保存排队提示，使历史消息与前端看到的响应保持一致。"""
        await self._conversations.append_message(
            conversation_id=handoff.conversation_id,
            role="assistant",
            content=content,
            metadata={"channel": "human_support", "handoff_id": handoff.id},
        )

    async def list_queue(
        self,
        *,
        tenant_id: str,
        statuses: list[str] | None,
        limit: int,
    ) -> list[HumanHandoffSession]:
        return await self._handoffs.list_queue(
            tenant_id=tenant_id,
            statuses=statuses,
            limit=limit,
        )

    async def assign(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
        agent_id: str,
        expected_version: int | None,
    ) -> HumanHandoffSession:
        item = await self._require_handoff(tenant_id=tenant_id, handoff_id=handoff_id)
        self._check_version(item, expected_version)
        if item.status != "waiting_assignment":
            if item.assigned_agent_id == agent_id and item.status in {"assigned", "in_service"}:
                return item
            raise self._transition_error(item, "接入")

        item.status = "assigned"
        item.assigned_agent_id = agent_id
        item.assigned_at = datetime.now(timezone.utc)
        conversation = await self._require_conversation(
            tenant_id=item.tenant_id,
            user_id=item.user_id,
            conversation_id=item.conversation_id,
        )
        conversation.service_mode = "human"
        await self._handoffs.flush()
        return item

    async def send_agent_message(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
        agent_id: str,
        content: str,
    ) -> HumanHandoffSession:
        item = await self._require_assignee(
            tenant_id=tenant_id,
            handoff_id=handoff_id,
            agent_id=agent_id,
        )
        if item.status not in {"assigned", "in_service"}:
            raise self._transition_error(item, "回复")

        item.status = "in_service"
        await self._conversations.append_message(
            conversation_id=item.conversation_id,
            role="assistant",
            content=content,
            metadata={
                "channel": "human_support",
                "handoff_id": item.id,
                "agent_id": agent_id,
            },
        )
        await self._handoffs.flush()
        return item

    async def submit_resolution(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
        agent_id: str,
        resolution_code: str,
        summary: str,
        next_mode: str,
        metadata: dict[str, Any],
    ) -> HumanHandoffSession:
        item = await self._require_assignee(
            tenant_id=tenant_id,
            handoff_id=handoff_id,
            agent_id=agent_id,
        )
        if item.status not in {"assigned", "in_service"}:
            raise self._transition_error(item, "提交处理结果")

        item.status = "resolution_submitted"
        item.resolution_code = resolution_code
        item.resolution_summary = summary
        item.resolution_metadata_json = metadata
        item.next_mode = next_mode
        item.resolution_submitted_at = datetime.now(timezone.utc)
        conversation = await self._require_conversation(
            tenant_id=item.tenant_id,
            user_id=item.user_id,
            conversation_id=item.conversation_id,
        )
        conversation.service_mode = "resolution_review"
        await self._handoffs.flush()
        return item

    async def confirm_resolution(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
        operator_id: str,
        expected_version: int | None,
    ) -> HumanHandoffSession:
        """模拟系统确认坐席结论，并恢复 Bot 或关闭会话。"""
        item = await self._require_handoff(tenant_id=tenant_id, handoff_id=handoff_id)
        self._check_version(item, expected_version)
        if item.status != "resolution_submitted":
            raise self._transition_error(item, "确认处理结果")
        if not item.resolution_code or not item.resolution_summary:
            raise AppError(
                "人工处理结果不完整",
                code="handoff_resolution_incomplete",
                status_code=409,
            )

        item.status = "resolved"
        item.resolved_at = datetime.now(timezone.utc)
        conversation = await self._require_conversation(
            tenant_id=item.tenant_id,
            user_id=item.user_id,
            conversation_id=item.conversation_id,
        )
        if item.next_mode == "close_conversation":
            conversation.status = "closed"
            conversation.service_mode = "bot"
        else:
            conversation.status = "active"
            conversation.service_mode = "bot"

        await self._conversations.append_message(
            conversation_id=item.conversation_id,
            role="assistant",
            content=f"人工处理已完成：{item.resolution_summary}",
            metadata={
                "channel": "human_support",
                "handoff_id": item.id,
                "operator_id": operator_id,
                "resolution_code": item.resolution_code,
            },
        )
        await self._handoffs.flush()
        return item

    @staticmethod
    def to_view(item: HumanHandoffSession) -> HumanHandoffView:
        return HumanHandoffView(
            id=item.id,
            tenant_id=item.tenant_id,
            user_id=item.user_id,
            conversation_id=item.conversation_id,
            support_ticket_id=item.support_ticket_id,
            origin_thread_id=item.origin_thread_id,
            status=item.status,
            queue_name=item.queue_name,
            priority=item.priority,
            reason=item.reason,
            assigned_agent_id=item.assigned_agent_id,
            resolution_code=item.resolution_code,
            resolution_summary=item.resolution_summary,
            next_mode=item.next_mode,
            version=item.version,
            requested_at=item.requested_at.isoformat(),
            updated_at=item.updated_at.isoformat(),
        )

    async def _require_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> Conversation:
        item = await self._conversations.get_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if item is None:
            raise AppError("会话不存在", code="conversation_not_found", status_code=404)
        return item

    async def _require_handoff(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
    ) -> HumanHandoffSession:
        item = await self._handoffs.get_owned(tenant_id=tenant_id, handoff_id=handoff_id)
        if item is None:
            raise AppError("人工接管记录不存在", code="handoff_not_found", status_code=404)
        return item

    async def _require_assignee(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
        agent_id: str,
    ) -> HumanHandoffSession:
        item = await self._require_handoff(tenant_id=tenant_id, handoff_id=handoff_id)
        if item.assigned_agent_id != agent_id:
            raise AppError(
                "只有当前接入坐席可以操作该会话",
                code="handoff_agent_mismatch",
                status_code=403,
            )
        return item

    @staticmethod
    def _check_version(item: HumanHandoffSession, expected_version: int | None) -> None:
        if expected_version is not None and item.version != expected_version:
            raise AppError(
                "人工服务状态已更新，请刷新后重试",
                code="handoff_version_conflict",
                status_code=409,
            )

    @staticmethod
    def _transition_error(item: HumanHandoffSession, action: str) -> AppError:
        return AppError(
            f"当前状态 {item.status} 不能执行{action}",
            code="handoff_invalid_transition",
            status_code=409,
        )
