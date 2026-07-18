from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from customer_service_app.core.exceptions import AppError
from customer_service_app.infrastructure.db.models import (
    Conversation,
    HumanHandoffSession,
    Message,
    Order,
    PendingAction,
    SupportTicket,
    TenantUsageDaily,
)


class ConversationRepository:
    """Data access object for conversations and messages."""

    def __init__(self, session: AsyncSession):
        """Store the request-scoped database session."""
        self._session = session

    async def create(self, *, tenant_id: str, user_id: str, title: str = "新会话") -> Conversation:
        """创建一条会话记录。"""
        conversation = Conversation(tenant_id=tenant_id, user_id=user_id, title=title)
        self._session.add(conversation)
        await self._session.flush()
        # flush 会把 SQL 发送到数据库，但不等于 commit；事务最终由外层统一提交。
        return conversation

    async def get_owned(self, *, tenant_id: str, user_id: str, conversation_id: str) -> Conversation | None:
        """按租户、用户、会话 id 查询会话，防止越权访问别人的会话。"""
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == tenant_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[Conversation]:
        """查询某个用户最近的会话列表。"""
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.tenant_id == tenant_id, Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def append_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        """Append one message to a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_json=metadata or {},
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def recent_messages(self, *, conversation_id: str, limit: int = 12) -> list[Message]:
        """按时间倒序查最近消息，再反转成正常阅读顺序。"""
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))


class OrderRepository:
    """订单表的数据访问对象，供工具函数查询订单状态。"""

    def __init__(self, session: AsyncSession):
        """保存数据库会话。"""
        self._session = session

    async def get_by_order_id(self, *, tenant_id: str, user_id: str, order_id: str) -> Order | None:
        """按租户、用户、订单号查询订单，避免查到别人的订单。"""
        result = await self._session.execute(
            select(Order).where(
                Order.tenant_id == tenant_id,
                Order.user_id == user_id,
                Order.order_id == order_id,
            )
        )
        return result.scalar_one_or_none()


class TicketRepository:
    """售后工单的数据访问对象。"""

    def __init__(self, session: AsyncSession):
        """保存数据库会话。"""
        self._session = session

    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        category: str,
        title: str,
        detail: str,
        priority: str = "normal",
        metadata: dict | None = None,
    ) -> SupportTicket:
        """创建售后工单。

        在真实生产系统里，这里可能不是直接插数据库，
        而是调用工单中心、CRM、ERP 或消息队列。
        """
        ticket = SupportTicket(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            category=category,
            title=title,
            detail=detail,
            priority=priority,
            metadata_json=metadata or {},
        )
        self._session.add(ticket)
        await self._session.flush()
        return ticket


class HumanHandoffRepository:
    """人工接管状态的数据访问边界。"""

    ACTIVE_STATUSES = {
        "waiting_assignment",
        "assigned",
        "in_service",
        "resolution_submitted",
    }

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_or_get(
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
        """按幂等键创建人工接管记录，避免重试生成重复队列任务。"""
        existing = await self.get_by_idempotency_key(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            return existing

        item = HumanHandoffSession(
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
        self._session.add(item)
        await self._session.flush()
        return item

    async def get_by_idempotency_key(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
    ) -> HumanHandoffSession | None:
        result = await self._session.execute(
            select(HumanHandoffSession).where(
                HumanHandoffSession.tenant_id == tenant_id,
                HumanHandoffSession.idempotency_key == idempotency_key,
            )
        )
        return result.scalar_one_or_none()

    async def get_owned(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
    ) -> HumanHandoffSession | None:
        """按租户读取接管记录，避免坐席跨租户访问。"""
        result = await self._session.execute(
            select(HumanHandoffSession).where(
                HumanHandoffSession.id == handoff_id,
                HumanHandoffSession.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_active_for_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> HumanHandoffSession | None:
        result = await self._session.execute(
            select(HumanHandoffSession)
            .where(
                HumanHandoffSession.tenant_id == tenant_id,
                HumanHandoffSession.user_id == user_id,
                HumanHandoffSession.conversation_id == conversation_id,
                HumanHandoffSession.status.in_(self.ACTIVE_STATUSES),
            )
            .order_by(HumanHandoffSession.requested_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_queue(
        self,
        *,
        tenant_id: str,
        statuses: list[str] | None,
        limit: int,
    ) -> list[HumanHandoffSession]:
        statement = select(HumanHandoffSession).where(
            HumanHandoffSession.tenant_id == tenant_id
        )
        if statuses:
            statement = statement.where(HumanHandoffSession.status.in_(statuses))
        statement = statement.order_by(
            case(
                (HumanHandoffSession.priority == "urgent", 0),
                (HumanHandoffSession.priority == "high", 1),
                (HumanHandoffSession.priority == "normal", 2),
                else_=3,
            ),
            HumanHandoffSession.requested_at.asc(),
        ).limit(limit)
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def flush(self) -> None:
        """提交本轮 ORM 变更，并把并发更新转换成稳定的业务错误。"""
        try:
            await self._session.flush()
        except StaleDataError as exc:
            raise AppError(
                "人工服务状态已被其他坐席更新，请刷新后重试",
                code="handoff_version_conflict",
                status_code=409,
            ) from exc


class UsageRepository:
    """Data access object for daily tenant usage."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_today_usage(self, *, tenant_id: str) -> TenantUsageDaily | None:
        today = date.today()
        result = await self._session.execute(
            select(TenantUsageDaily).where(
                TenantUsageDaily.tenant_id == tenant_id,
                TenantUsageDaily.usage_date == today,
            )
        )
        return result.scalar_one_or_none()

    async def add_llm_usage(
        self,
        *,
        tenant_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> TenantUsageDaily:
        today = date.today()
        result = await self._session.execute(
            select(TenantUsageDaily).where(
                TenantUsageDaily.tenant_id == tenant_id,
                TenantUsageDaily.usage_date == today,
            )
        )
        usage = result.scalar_one_or_none()
        if usage is None:
            usage = TenantUsageDaily(tenant_id=tenant_id, usage_date=today)
            self._session.add(usage)
        usage.llm_calls += 1
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.total_tokens += total_tokens
        await self._session.flush()
        return usage


class PendingActionRepository:
    """Data access object for high-risk tool confirmations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        thread_id: str | None,
        confirmation_id: str | None,
        tool_name: str,
        arguments: dict,
    ) -> PendingAction:
        if confirmation_id:
            result = await self._session.execute(
                select(PendingAction).where(
                    PendingAction.tenant_id == tenant_id,
                    PendingAction.user_id == user_id,
                    PendingAction.confirmation_id == confirmation_id,
                    PendingAction.status == "pending",
                )
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return existing

        action = PendingAction(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            confirmation_id=confirmation_id,
            tool_name=tool_name,
            arguments_json=arguments,
            status="pending",
        )
        self._session.add(action)
        await self._session.flush()
        return action

    async def get_owned(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
    ) -> PendingAction | None:
        result = await self._session.execute(
            select(PendingAction).where(
                PendingAction.id == action_id,
                PendingAction.tenant_id == tenant_id,
                PendingAction.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_pending(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 50,
    ) -> list[PendingAction]:
        result = await self._session.execute(
            select(PendingAction)
            .where(
                PendingAction.tenant_id == tenant_id,
                PendingAction.user_id == user_id,
                PendingAction.status == "pending",
            )
            .order_by(PendingAction.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_approved(self, action: PendingAction, *, comment: str | None) -> PendingAction:
        action.status = "approved"
        action.comment = comment
        action.decided_at = datetime.now(timezone.utc)
        await self._session.flush()
        return action

    async def mark_rejected(self, action: PendingAction, *, comment: str | None) -> PendingAction:
        action.status = "rejected"
        action.comment = comment
        action.decided_at = datetime.now(timezone.utc)
        await self._session.flush()
        return action

    async def mark_executed(
        self,
        action: PendingAction,
        *,
        result: dict,
    ) -> PendingAction:
        """保存 HIL 恢复后真实工具的执行结果。"""

        action.status = "executed"
        action.result_json = result
        action.error_message = None
        await self._session.flush()
        return action

    async def mark_failed(
        self,
        action: PendingAction,
        *,
        error_message: str,
    ) -> PendingAction:
        """记录用户已批准、但真实工具执行失败的状态。"""

        action.status = "failed"
        action.error_message = error_message
        await self._session.flush()
        return action
