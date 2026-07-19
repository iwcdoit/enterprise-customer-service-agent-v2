from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import NotFoundError
from customer_service_app.domain.schemas import (
    ConversationCreateRequest,
    ConversationMessageView,
    ConversationView,
)
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.conversation_service import ConversationService


router = APIRouter(prefix="/conversations", tags=["conversations"])
"""会话管理路由组，最终路径是 `/api/v1/conversations`。"""


@router.post("", response_model=ConversationView)
async def create_conversation(
    request: ConversationCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ConversationView:
    """创建新会话。

    这里没有直接操作数据库，而是交给 `ConversationService`，
    保持 API 层薄、业务层清晰。
    """
    service = ConversationService(session)

    conversation = await service.ensure_conversation(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        conversation_id=None,
        first_question=request.title,
    )

    await session.commit()

    return ConversationView(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        user_id=conversation.user_id,
        title=conversation.title,
        status=conversation.status,
        service_mode=conversation.service_mode,
    )


@router.get("", response_model=list[ConversationView])
async def list_conversations(
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationView]:
    """查询某个用户的会话列表。"""
    conversations = await ConversationService(session).list_conversations(
        tenant_id=tenant_id,
        user_id=user_id,
    )

    return [
        ConversationView(
            id=item.id,
            tenant_id=item.tenant_id,
            user_id=item.user_id,
            title=item.title,
            status=item.status,
            service_mode=item.service_mode,
        )
        for item in conversations
    ]


@router.get("/{conversation_id}/messages", response_model=list[ConversationMessageView])
async def list_conversation_messages(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationMessageView]:
    """Restore one owned conversation in the customer-service workspace."""

    service = ConversationService(session)
    conversation = await service.ensure_conversation(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        first_question="",
        allow_closed=True,
    )

    if conversation is None:
        raise NotFoundError("Conversation not found")

    messages = await service.list_message_records(conversation_id=conversation_id)

    return [
        ConversationMessageView(
            id=item.id,
            role=item.role,
            content=item.content,
            metadata=item.metadata_json,
            created_at=item.created_at.isoformat(),
        )
        for item in messages
        if item.role in {"system", "user", "assistant", "tool"}
    ]
