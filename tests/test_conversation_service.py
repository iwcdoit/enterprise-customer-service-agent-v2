from __future__ import annotations

from types import SimpleNamespace

import pytest

from customer_service_app.core.exceptions import NotFoundError
from customer_service_app.services.conversation_service import ConversationService


class FakeConversationRepository:
    def __init__(self, conversation):
        self.conversation = conversation
        self.messages_requested = False

    async def get_owned(self, *, tenant_id: str, user_id: str, conversation_id: str):
        if self.conversation is None:
            return None
        if (
            self.conversation.tenant_id == tenant_id
            and self.conversation.user_id == user_id
            and self.conversation.id == conversation_id
        ):
            return self.conversation
        return None

    async def recent_messages(self, *, conversation_id: str, limit: int):
        self.messages_requested = True
        return [
            SimpleNamespace(
                id="m1",
                conversation_id=conversation_id,
                role="user",
                content="订单什么时候到？",
                metadata_json={},
            )
        ]


@pytest.mark.asyncio
async def test_get_conversation_detail_checks_owner_before_reading_messages() -> None:
    service = ConversationService(session=object())  # type: ignore[arg-type]
    fake_repo = FakeConversationRepository(
        SimpleNamespace(id="c1", tenant_id="tenant-a", user_id="u1")
    )
    service._repo = fake_repo  # type: ignore[assignment]

    conversation, messages = await service.get_conversation_detail(
        tenant_id="tenant-a",
        user_id="u1",
        conversation_id="c1",
        message_limit=20,
    )

    assert conversation.id == "c1"
    assert messages[0].content == "订单什么时候到？"
    assert fake_repo.messages_requested is True


@pytest.mark.asyncio
async def test_get_conversation_detail_rejects_unowned_conversation() -> None:
    service = ConversationService(session=object())  # type: ignore[arg-type]
    fake_repo = FakeConversationRepository(
        SimpleNamespace(id="c1", tenant_id="tenant-a", user_id="u1")
    )
    service._repo = fake_repo  # type: ignore[assignment]

    with pytest.raises(NotFoundError):
        await service.get_conversation_detail(
            tenant_id="tenant-a",
            user_id="other-user",
            conversation_id="c1",
        )

    assert fake_repo.messages_requested is False

