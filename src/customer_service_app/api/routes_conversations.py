from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.domain.schemas import (
    ConversationCreateRequest,
    ConversationDetailView,
    ConversationMessageView,
    ConversationView,
)
from customer_service_app.infrastructure.db.models import Conversation, Message
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
    return _to_conversation_view(conversation)


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
        _to_conversation_view(item)
        for item in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationDetailView)
async def get_conversation_detail(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    message_limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationDetailView:
    """查询一个会话详情和最近消息。

    这个接口适合前端进入某个会话窗口时加载历史，也适合运营验证台排查
    一轮 Agent 是否正确保存了用户消息、助手回答和工具元数据。
    """
    conversation, messages = await ConversationService(session).get_conversation_detail(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        message_limit=message_limit,
    )
    return ConversationDetailView(
        conversation=_to_conversation_view(conversation),
        messages=[_to_message_view(item) for item in messages],
    )


@router.get("/{conversation_id}/messages", response_model=list[ConversationMessageView])
async def list_conversation_messages(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationMessageView]:
    """只查询一个会话的最近消息。

    与详情接口一样，这里也会先校验会话是否属于当前 tenant/user。
    """
    _, messages = await ConversationService(session).get_conversation_detail(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        message_limit=limit,
    )
    return [_to_message_view(item) for item in messages]


def _to_conversation_view(conversation: Conversation) -> ConversationView:
    return ConversationView(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        user_id=conversation.user_id,
        title=conversation.title,
        status=conversation.status,
    )


def _to_message_view(message: Message) -> ConversationMessageView:
    return ConversationMessageView(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        metadata=dict(message.metadata_json or {}),
        created_at=message.created_at,
    )
