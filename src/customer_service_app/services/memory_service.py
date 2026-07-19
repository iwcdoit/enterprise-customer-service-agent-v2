from __future__ import annotations

from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.domain.memory import CustomerMemoryView, MemoryType, ShortTermContext
from customer_service_app.domain.schemas import ChatMessage
from customer_service_app.infrastructure.db.repositories import (
    ConversationRepository,
    MemoryRepository,
    PendingActionRepository,
)


class MemoryService:
    """组装 LLM 本轮需要的短期上下文。"""

    def __init__(self, *, settings: Settings, session: AsyncSession):
        self._settings = settings
        self._conversation_repo = ConversationRepository(session)
        self._memory_repo = MemoryRepository(session)
        self._pending_repo = PendingActionRepository(session)

    async def build_short_term_context(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        history_turns: int | None = None,
    ) -> ShortTermContext:
        """读取最近消息、会话摘要、待确认动作和长期记忆。"""
        # 一轮对话包含 user + assistant 两条消息，所以 turns * 2。
        limit = max(2, (history_turns or self._settings.short_term_history_turns) * 2)
        recent_messages = await self._conversation_repo.recent_messages(
            conversation_id=conversation_id,
            limit=limit,
        )
        latest_summary = await self._memory_repo.get_latest_summary(
            conversation_id=conversation_id,
        )
        pending_actions = await self._pending_repo.list_pending_for_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        memories = await self._memory_repo.list_memories(
            tenant_id=tenant_id,
            user_id=user_id,
        )

        chat_messages: list[ChatMessage] = []
        for message in recent_messages:
            if message.role in {"user", "assistant"}:
                chat_messages.append(
                    ChatMessage(
                        role=message.role,  # type: ignore[arg-type]
                        content=message.content,
                        metadata=message.metadata_json or {},
                    )
                )

        pending_views: list[dict[str, Any]] = []
        for action in pending_actions:
            pending_views.append(
                {
                    "id": action.id,
                    "tool_name": action.tool_name,
                    "risk_level": action.risk_level,
                    "status": action.status,
                }
            )

        memory_views: list[CustomerMemoryView] = []
        for memory in memories:
            memory_views.append(
                CustomerMemoryView(
                    id=memory.id,
                    memory_type=cast(MemoryType, memory.memory_type),
                    memory_key=memory.memory_key,
                    memory_value=memory.memory_value_json,
                    confidence=memory.confidence,
                    source=memory.source,
                    verification_status=memory.verification_status,
                    evidence_ids=list(memory.evidence_json or []),
                )
            )

        return ShortTermContext(
            recent_messages=chat_messages,
            summary=latest_summary.summary if latest_summary else None,
            pending_actions=pending_views,
            memories=memory_views,
        )

    async def maybe_update_summary(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> None:
        """历史消息超过阈值时写一条轻量摘要占位。"""
        threshold = self._settings.conversation_summary_threshold_messages or 20
        recent_messages = await self._conversation_repo.recent_messages(
            conversation_id=conversation_id,
            limit=threshold + 1,
        )
        if len(recent_messages) <= threshold:
            return

        # 只压缩即将滑出短期窗口的旧消息；最近几轮仍保留原文。
        keep_recent = max(4, self._settings.short_term_history_turns * 2)
        covered = recent_messages[: max(1, len(recent_messages) - keep_recent)]
        first = covered[0]
        last = covered[-1]
        latest_summary = await self._memory_repo.get_latest_summary(
            conversation_id=conversation_id
        )
        if latest_summary and latest_summary.message_end_id == last.id:
            return
        excerpts = [
            f"{message.role}: {' '.join(message.content.split())[:240]}"
            for message in covered
            if message.role in {"user", "assistant"}
        ]
        if not excerpts:
            return
        summary = "历史对话摘录（只用于保持上下文，不代表已核实事实）：\n" + "\n".join(excerpts)
        await self._memory_repo.save_summary(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            summary=summary,
            message_start_id=first.id,
            message_end_id=last.id,
        )
