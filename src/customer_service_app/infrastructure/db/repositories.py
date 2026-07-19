from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import case, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError

from customer_service_app.core.exceptions import AppError

from customer_service_app.infrastructure.db.models import (
    AgentRun,
    AgentRunStep,
    Conversation,
    ConversationSummary,
    CustomerMemory,
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
        conversation = Conversation(tenant_id=tenant_id, user_id=user_id, title=title, )
        self._session.add(conversation)
        await self._session.flush()
        return conversation


    async def get_owned(self, *, tenant_id: str, user_id: str, conversation_id: str) -> Conversation | None:
        """按租户、用户、会话 id 查询会话，防止越权访问别人的会话。"""
        statement = select(Conversation).where(Conversation.id == conversation_id).where(Conversation.tenant_id == tenant_id).where(Conversation.user_id == user_id)
        result = await self._session.execute(statement)
        return result.scalars().one_or_none()

    async def list_by_user(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[Conversation]:
        """查询某个用户最近的会话列表。"""
        statement = select(Conversation).where(Conversation.tenant_id == tenant_id).where(
            Conversation.user_id == user_id).limit(limit)
        result = await self._session.execute(statement)
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
        statement = select(Message).where(Message.conversation_id == conversation_id).limit(limit).order_by(Message.created_at.desc())
        result = await self._session.execute(statement)
        rows = list(result.scalars().all())
        return list(reversed(rows))

    async def list_messages(self, *, conversation_id: str, limit: int = 200) -> list[Message]:
        """Return conversation messages in chronological order for page restoration."""
        statement = select(Message).where(Message.conversation_id == conversation_id).order_by(
            Message.created_at.asc()).limit(limit)
        result = await self._session.execute(statement)
        return list(result.scalars().all())


class OrderRepository:
    """订单表的数据访问对象，供工具函数查询订单状态。"""

    def __init__(self, session: AsyncSession):
        """保存数据库会话。"""
        self._session = session

    async def get_by_order_id(self, *, tenant_id: str, user_id: str, order_id: str) -> Order | None:
        """按租户、用户、订单号查询订单，避免查到别人的订单。"""
        statement = select(Order).where(Order.tenant_id == tenant_id).where(Order.user_id == user_id).where(
            Order.order_id == order_id)
        result = await self._session.execute(statement)
        return result.scalars().one_or_none()


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
        idempotency_key: str | None = None,
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
            idempotency_key=idempotency_key,
        )
        self._session.add(ticket)
        await self._session.flush()
        return ticket


class HumanHandoffRepository:
    """Persistence boundary for the durable human-support state machine."""

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
        existing = await self.get_by_idempotency_key(idempotency_key=idempotency_key)
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
        self, *, idempotency_key: str
    ) -> HumanHandoffSession | None:
        result = await self._session.execute(
            select(HumanHandoffSession).where(
                HumanHandoffSession.idempotency_key == idempotency_key
            )
        )
        return result.scalar_one_or_none()

    async def get_owned(
        self, *, tenant_id: str, handoff_id: str
    ) -> HumanHandoffSession | None:
        result = await self._session.execute(
            select(HumanHandoffSession).where(
                HumanHandoffSession.id == handoff_id,
                HumanHandoffSession.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_active_for_conversation(
        self, *, tenant_id: str, user_id: str, conversation_id: str
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
        self, *, tenant_id: str, statuses: list[str] | None, limit: int
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
        try:
            await self._session.flush()
        except StaleDataError as exc:
            raise AppError(
                "Handoff has been updated by another operator",
                code="handoff_version_conflict",
                status_code=409,
            ) from exc


class PendingActionRepository:
    """Data access object for user-confirmed side-effect operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        action_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        confirmation_prompt: str,
        idempotency_key: str,
        risk_level: str = "low",
        expire_at: datetime | None = None,
        langgraph_thread_id: str | None = None,
    ) -> PendingAction:
        """Create one pending action waiting for confirmation."""

        pending_action = PendingAction(tenant_id=tenant_id, user_id=user_id, conversation_id=conversation_id,
                               action_type=action_type, tool_name=tool_name, arguments_json=arguments,
                               confirmation_prompt=confirmation_prompt, idempotency_key=idempotency_key,
                               risk_level=risk_level, expire_at=expire_at, langgraph_thread_id=langgraph_thread_id,
                               status = 'pending')
        self._session.add(pending_action)
        await self._session.flush()
        return pending_action

    async def get_owned(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
    ) -> PendingAction | None:
        """Load a pending action by owner to prevent cross-user confirmation."""
        statement = select(PendingAction).where(PendingAction.tenant_id == tenant_id).where(
            PendingAction.user_id == user_id).where(PendingAction.id == action_id)
        result = await self._session.execute(statement)
        return result.scalars().one_or_none()

    async def get_by_idempotency_key(
        self,
        *,
        idempotency_key: str,
    ) -> PendingAction | None:
        """Return an existing action created by a retried graph node."""
        statement = select(PendingAction).where(PendingAction.idempotency_key == idempotency_key)
        result = await self._session.execute(statement)
        return result.scalars().one_or_none()

    async def list_pending_for_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> list[PendingAction]:
        """Return pending actions for a conversation."""
        statement = select(PendingAction).where(PendingAction.tenant_id == tenant_id).where(
            PendingAction.user_id == user_id).where(PendingAction.conversation_id == conversation_id).where(
            PendingAction.status == "pending").order_by(PendingAction.created_at.desc())
        result = await  self._session.execute(statement)
        return list(result.scalars().all())

    async def list_pending(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 50,
    ) -> list[PendingAction]:
        """按用户查询待确认记录，包括尚未清理的过期记录。"""
        statement = (
            select(PendingAction)
            .where(
                PendingAction.tenant_id == tenant_id,
                PendingAction.user_id == user_id,
                PendingAction.status == "pending",
            )
            .order_by(PendingAction.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def mark_approved(self, action: PendingAction) -> None:
        """Mark action as approved before executing the real tool."""
        action.status = 'approved'
        action.approved_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def mark_expired(self, action: PendingAction) -> None:
        """Mark an expired confirmation so it cannot be resumed later."""
        action.status = 'expired'
        await self._session.flush()

    async def mark_rejected(self, action: PendingAction) -> None:
        """Mark action as rejected by the user."""
        action.status = 'rejected'
        action.rejected_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def mark_executed(self, action: PendingAction, result: dict[str, Any]) -> None:
        """Mark action as executed and persist tool result."""
        action.status = 'executed'
        action.executed_at = datetime.now(timezone.utc)
        action.execution_result_json = result
        await self._session.flush()

    async def mark_failed(self, action: PendingAction, result: dict[str, Any]) -> None:
        """Mark action as failed and persist error payload."""
        action.status = 'failed'
        action.execution_result_json = result
        await self._session.flush()


class MemoryRepository:
    """Data access for short-term summaries and long-term memories."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_latest_summary(self, *, conversation_id: str) -> ConversationSummary | None:
        """Return the latest summary for a conversation."""
        statement = select(ConversationSummary).where(ConversationSummary.conversation_id == conversation_id).order_by(
            ConversationSummary.updated_at.desc()).limit(1)
        result = await self._session.execute(statement)
        return result.scalars().one_or_none()

    async def save_summary(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        summary: str,
        message_start_id: str | None,
        message_end_id: str | None,
    ) -> ConversationSummary:
        """Persist a conversation summary."""
        conversation_summary = ConversationSummary(conversation_id=conversation_id, tenant_id=tenant_id,
                                                   user_id=user_id, summary=summary, message_start_id=message_start_id,
                                                   message_end_id=message_end_id)
        self._session.add(conversation_summary)
        await self._session.flush()
        return conversation_summary

    async def upsert_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_type: str,
        memory_key: str,
        memory_value: dict[str, Any],
        confidence: float,
        source: str,
        verification_status: str,
        evidence_ids: list[str],
        sensitivity: str,
        expires_at: datetime | None = None,
    ) -> CustomerMemory:
        """Create or update a stable long-term memory item."""
        old_statement = (select(CustomerMemory).where(CustomerMemory.tenant_id == tenant_id).where(
            CustomerMemory.user_id == user_id).where(CustomerMemory.memory_type == memory_type).where(
            CustomerMemory.memory_key == memory_key))
        result = await self._session.execute(old_statement)
        history_memory = result.scalars().one_or_none()
        if history_memory is None:
            memory = CustomerMemory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type=memory_type,
                memory_key=memory_key,
                memory_value_json=memory_value,
                confidence=confidence,
                source=source,
                verification_status=verification_status,
                evidence_json=evidence_ids,
                sensitivity=sensitivity,
                expires_at=expires_at,
            )
            self._session.add(memory)
        else:
            memory = history_memory
            memory.memory_value_json = memory_value
            memory.confidence = confidence
            memory.source = source
            memory.verification_status = verification_status
            memory.evidence_json = evidence_ids
            memory.sensitivity = sensitivity
            memory.expires_at = expires_at

        await self._session.flush()
        return memory


    async def list_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 10,
    ) -> list[CustomerMemory]:
        """Load recent long-term memories for a user."""
        state = select(CustomerMemory).where(CustomerMemory.tenant_id == tenant_id).where(
            CustomerMemory.user_id == user_id).where(
            CustomerMemory.verification_status.in_({
                "explicit_user", "verified_tool", "business_system", "human_confirmed", "risk_engine"
            })
        ).where(
            or_(CustomerMemory.expires_at.is_(None), CustomerMemory.expires_at > datetime.now(timezone.utc))
        ).limit(limit).order_by(CustomerMemory.updated_at.desc())
        result = await self._session.execute(state)
        return list(result.scalars().all())


class UsageRepository:
    """Data access object for tenant daily usage statistics."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_today_usage(self, *, tenant_id: str) -> TenantUsageDaily | None:
        """Load today's usage row for a tenant."""
        today = date.today()
        state = select(TenantUsageDaily).where(TenantUsageDaily.tenant_id == tenant_id).where(
            TenantUsageDaily.usage_date == today)
        result = await self._session.execute(state)
        return result.scalars().one_or_none()

    async def add_llm_usage(
        self,
        *,
        tenant_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> TenantUsageDaily:
        """Increment daily LLM usage for one tenant."""
        today = date.today()
        state = select(TenantUsageDaily).where(TenantUsageDaily.tenant_id == tenant_id).where(
            TenantUsageDaily.usage_date == today)
        result = await self._session.execute(state)
        usage_daily = result.scalars().one_or_none()
        if usage_daily is None:
            usage_daily = TenantUsageDaily(
                tenant_id=tenant_id,
                usage_date=today,
                llm_calls=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            self._session.add(usage_daily)
        usage_daily.llm_calls += 1
        usage_daily.prompt_tokens += prompt_tokens
        usage_daily.completion_tokens += completion_tokens
        usage_daily.total_tokens += total_tokens
        await self._session.flush()
        return usage_daily

class AgentRunRepository:
    """Data access object for request-level trace persistence."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        request_id: str | None,
    ) -> AgentRun:
        """Create one Agent run record."""
        run = AgentRun(tenant_id=tenant_id, user_id=user_id, conversation_id=conversation_id, request_id=request_id,status = 'running')
        self._session.add(run)
        await self._session.flush()
        return run

    async def add_step(
        self,
        *,
        run_id: str,
        stage: str,
        name: str,
        status: str,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        latency_ms: float | None = None,
    ) -> AgentRunStep:
        """Append a trace step to an Agent run."""
        step = AgentRunStep(
            run_id=run_id,
            stage=stage,
            name=name,
            status=status,
            input_json=input_payload or {},
            output_json=output_payload or {},
            latency_ms=latency_ms,
        )
        self._session.add(step)
        await self._session.flush()
        return step


    async def mark_success(
        self,
        *,
        run: AgentRun,
        model: str | None = None,
        total_tokens: int = 0,
        latency_ms: float | None = None,
    ) -> None:
        """Mark a run as successful."""
        run.status = 'success'
        run.model = model
        run.total_tokens = total_tokens
        run.latency_ms = latency_ms
        run.ended_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def mark_waiting(self, *, run: AgentRun) -> None:
        """Mark a run as suspended at a LangGraph interrupt."""
        run.status = 'waiting_confirmation'
        await self._session.flush()


    async def mark_running(self, *, run: AgentRun) -> None:
        """Mark a suspended run as active after Command(resume=...)."""
        run.status = 'running'
        await self._session.flush()

    async def mark_failed(
        self,
        *,
        run: AgentRun,
        error_code: str,
        error_message: str,
        latency_ms: float | None = None,
    ) -> None:
        """Mark a run as failed."""
        run.status = 'failed'
        run.error_code = error_code
        run.error_message = error_message
        run.ended_at = datetime.now(timezone.utc)
        run.latency_ms = latency_ms
        await self._session.flush()

    async def get_run(self, *, run_id: str) -> AgentRun | None:
        """Load one Agent run by id."""
        state = select(AgentRun).where(AgentRun.id == run_id)
        result = await self._session.execute(state)
        return result.scalars().one_or_none()

    async def list_steps(self, *, run_id: str) -> list[AgentRunStep]:
        """Load all trace steps for one Agent run."""
        state = select(AgentRunStep).where(AgentRunStep.run_id == run_id).order_by(AgentRunStep.created_at.asc())
        result = await self._session.execute(state)
        return list(result.scalars().all())
