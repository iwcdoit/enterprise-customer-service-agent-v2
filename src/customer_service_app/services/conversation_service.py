from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError, NotFoundError
from customer_service_app.domain.schemas import ChatMessage
from customer_service_app.infrastructure.db.models import Conversation, Message
from customer_service_app.infrastructure.db.repositories import ConversationRepository


class ConversationService:
    """会话服务，封装会话创建、历史读取和消息保存。"""

    def __init__(self, session: AsyncSession):
        self._repo = ConversationRepository(session)

    async def ensure_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        first_question: str,
        allow_closed: bool = False,
    ) -> Conversation:
        """获取已有会话，或在首轮请求时创建新会话。"""
        if conversation_id:
            conversation = await self._repo.get_owned(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if conversation is None:
                raise NotFoundError("Conversation not found or not owned by user")
            if conversation.status == "closed" and not allow_closed:
                raise AppError(
                    "Conversation is closed; create a new conversation to continue",
                    code="conversation_closed",
                    status_code=409,
                )
            return conversation

        title = self._build_title(question=first_question, max_length=24)
        return await self._repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
        )

    async def recent_history(self, conversation_id: str, limit: int = 12) -> list[ChatMessage]:
        """读取最近 user/assistant 消息，作为 LLM 短期上下文。"""
        messages = await self._repo.recent_messages(
            conversation_id=conversation_id,
            limit=limit,
        )
        result: list[ChatMessage] = []
        for message in messages:
            if message.role in {"user", "assistant"}:
                result.append(
                    ChatMessage(
                        role=message.role,  # type: ignore[arg-type]
                        content=message.content,
                        metadata=message.metadata_json or {},
                    )
                )
        return result

    async def save_turn(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        metadata: dict | None = None,
    ) -> None:
        """保存一轮问答：用户消息一条，助手消息一条。"""
        metadata = metadata or {}
        await self._repo.append_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
            metadata=metadata.get("user", {}),
        )
        await self._repo.append_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            metadata=metadata.get("assistant", {}),
        )

    async def append_assistant_message(
        self,
        *,
        conversation_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """追加一条助手消息，常用于 HIL 恢复后的最终答复。"""
        await self._repo.append_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            metadata=metadata or {},
        )

    async def list_conversations(self, *, tenant_id: str, user_id: str) -> list[Conversation]:
        """列出某个用户自己的会话。"""
        return await self._repo.list_by_user(tenant_id=tenant_id, user_id=user_id)

    async def get_conversation_detail(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message_limit: int = 50,
    ) -> tuple[Conversation, list[Message]]:
        """校验会话归属后读取会话和最近消息。"""
        conversation = await self._repo.get_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise NotFoundError("Conversation not found or not owned by user")

        messages = await self._repo.recent_messages(
            conversation_id=conversation.id,
            limit=message_limit,
        )
        return conversation, messages

    async def list_message_records(self, *, conversation_id: str, limit: int = 200):
        """读取会话消息，供前端恢复历史。"""
        return await self._repo.list_messages(conversation_id=conversation_id, limit=limit)

    @staticmethod
    def _build_title(question: str, max_length: int = 24) -> str:
        """用用户首问生成一个短标题。"""
        title = " ".join(question.split())
        if len(title) <= max_length:
            return title
        return title[:max_length]
