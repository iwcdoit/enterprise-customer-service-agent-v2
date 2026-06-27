from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import NotFoundError
from customer_service_app.domain.schemas import ChatMessage
from customer_service_app.infrastructure.db.models import Conversation
from customer_service_app.infrastructure.db.repositories import ConversationRepository


class ConversationService:
    """会话服务。

    它把“会话创建、历史消息读取、消息保存”封装起来。
    CustomerServiceAgent 不直接操作数据库表，而是通过这个服务处理会话记忆。
    """

    def __init__(self, session: AsyncSession):
        """把当前请求里的数据库 session 传给 Repository。

        一个请求通常共用一个 session，这样创建会话、执行工具、保存消息可以在同一事务里提交。
        """

        self._session = session
        self._repo = ConversationRepository(session)

    async def ensure_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        first_question: str,
    ) -> Conversation:
        """获取已有会话，或创建新会话。

        如果前端传了 conversation_id，就校验这个会话是否属于当前 tenant/user。
        如果没传 conversation_id，就用用户首问生成标题并创建新会话。
        """

        if conversation_id:
            conversation = await self._repo.get_owned(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if conversation is None:
                raise NotFoundError("Conversation not found or not owned by user")
            return conversation

        title = self._build_title(first_question)
        return await self._repo.create(tenant_id=tenant_id, user_id=user_id, title=title)

    async def recent_history(self, conversation_id: str, limit: int = 12) -> list[ChatMessage]:
        """读取最近几条历史消息，作为下一次 LLM 调用的上下文。

        这里只返回 user/assistant 消息，不把 system/tool 消息混进普通历史。
        """

        messages = await self._repo.recent_messages(conversation_id=conversation_id, limit=limit)
        return [
            ChatMessage(role=message.role, content=message.content, metadata=message.metadata_json)
            for message in messages
            if message.role in {"user", "assistant"}
        ]

    async def save_turn(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        metadata: dict,
    ) -> None:
        """保存一轮问答。

        一轮问答会写两条消息：用户 question 和助手 answer。
        metadata 用来保存工具调用、知识库命中数量等调试信息。
        """

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

    async def list_conversations(self, *, tenant_id: str, user_id: str) -> list[Conversation]:
        """列出某个用户自己的会话。"""

        return await self._repo.list_by_user(tenant_id=tenant_id, user_id=user_id)

    @staticmethod
    def _build_title(question: str, max_length: int = 24) -> str:
        """用用户首问生成会话标题。

        这里只做简单截断，真实产品里可以让模型生成更自然的标题。
        """

        title = " ".join(question.split())
        return title if len(title) <= max_length else title[:max_length] + "..."
