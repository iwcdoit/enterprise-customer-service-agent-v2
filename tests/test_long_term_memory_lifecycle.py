from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from customer_service_app.core.exceptions import NotFoundError
from customer_service_app.core.config import Settings
from customer_service_app.domain.memory import MemoryWriteCommand
from customer_service_app.services.long_term_memory_service import LongTermMemoryService


def _service_with_repo(repo: SimpleNamespace) -> LongTermMemoryService:
    service = object.__new__(LongTermMemoryService)
    service._repo = repo
    service._settings = Settings(long_term_memory_enabled=True)
    return service


@pytest.mark.asyncio
async def test_opted_out_user_does_not_receive_new_memory() -> None:
    repo = SimpleNamespace(
        is_long_term_memory_enabled=AsyncMock(return_value=False),
        upsert_memory=AsyncMock(),
    )
    service = _service_with_repo(repo)

    stored = await service.remember(
        MemoryWriteCommand(
            tenant_id="tenant-a",
            user_id="user-a",
            memory_type="profile",
            memory_key="preferred_contact",
            memory_value={"channel": "email"},
            source="user",
            verification_status="explicit_user",
        )
    )

    assert stored is False
    repo.upsert_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_correction_becomes_explicitly_verified_memory() -> None:
    current = SimpleNamespace(
        id="memory-1",
        tenant_id="tenant-a",
        user_id="user-a",
        memory_type="profile",
        memory_key="preferred_contact",
        memory_value_json={"channel": "phone"},
        confidence=0.9,
        source="crm",
        verification_status="business_system",
        evidence_json=["crm-1"],
        sensitivity="internal",
        expires_at=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    updated = SimpleNamespace(
        **{
            **current.__dict__,
            "memory_value_json": {"channel": "email"},
            "confidence": 1.0,
            "source": "user",
            "verification_status": "explicit_user",
            "evidence_json": [],
        }
    )
    repo = SimpleNamespace(
        get_owned_memory=AsyncMock(return_value=current),
        upsert_memory=AsyncMock(return_value=updated),
    )
    service = _service_with_repo(repo)

    view = await service.update_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        memory_id="memory-1",
        memory_value={"channel": "email"},
        expires_at=None,
    )

    assert view.memory_value == {"channel": "email"}
    assert view.verification_status == "explicit_user"
    assert repo.upsert_memory.await_args.kwargs["source"] == "user"


@pytest.mark.asyncio
async def test_memory_ownership_is_enforced_before_delete() -> None:
    repo = SimpleNamespace(
        get_owned_memory=AsyncMock(return_value=None),
        delete_memory=AsyncMock(),
    )
    service = _service_with_repo(repo)

    with pytest.raises(NotFoundError):
        await service.delete_memory(
            tenant_id="tenant-a",
            user_id="user-a",
            memory_id="another-users-memory",
        )

    repo.delete_memory.assert_not_awaited()
