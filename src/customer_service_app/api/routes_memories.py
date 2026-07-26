from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.api.dependencies import require_roles
from customer_service_app.core.security import CurrentPrincipal, authorize_identity
from customer_service_app.domain.memory import (
    CustomerMemoryView,
    MemoryCleanupView,
    MemoryPreferenceUpdateRequest,
    MemoryPreferenceView,
    MemoryUpdateRequest,
)
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.long_term_memory_service import LongTermMemoryService


router = APIRouter(prefix="/memories", tags=["memories"])
_memory_roles = require_roles("customer", "agent", "operator", "admin")


@router.get("", response_model=list[CustomerMemoryView])
async def list_memories(
    tenant_id: str,
    user_id: str,
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
    principal: CurrentPrincipal = Depends(_memory_roles),
) -> list[CustomerMemoryView]:
    """List active long-term memories retained for one user."""

    tenant_id, user_id = authorize_identity(
        principal,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    return await LongTermMemoryService(session).list_memories(
        tenant_id=tenant_id,
        user_id=user_id,
        limit=limit,
    )


@router.patch("/{memory_id}", response_model=CustomerMemoryView)
async def update_memory(
    memory_id: str,
    tenant_id: str,
    user_id: str,
    request: MemoryUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
    principal: CurrentPrincipal = Depends(_memory_roles),
) -> CustomerMemoryView:
    """Correct one owned memory and mark the value as explicitly user-verified."""

    tenant_id, user_id = authorize_identity(
        principal,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    value = await LongTermMemoryService(session).update_memory(
        tenant_id=tenant_id,
        user_id=user_id,
        memory_id=memory_id,
        memory_value=request.memory_value,
        expires_at=request.expires_at,
    )
    await session.commit()
    return value


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
    principal: CurrentPrincipal = Depends(_memory_roles),
) -> Response:
    """Delete one owned long-term memory."""

    tenant_id, user_id = authorize_identity(
        principal,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    await LongTermMemoryService(session).delete_memory(
        tenant_id=tenant_id,
        user_id=user_id,
        memory_id=memory_id,
    )
    await session.commit()
    return Response(status_code=204)


@router.get("/preference/current", response_model=MemoryPreferenceView)
async def get_memory_preference(
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
    principal: CurrentPrincipal = Depends(_memory_roles),
) -> MemoryPreferenceView:
    """Return the effective long-term memory consent."""

    tenant_id, user_id = authorize_identity(
        principal,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    enabled, preference = await LongTermMemoryService(session).get_preference(
        tenant_id=tenant_id,
        user_id=user_id,
    )
    return MemoryPreferenceView(
        tenant_id=tenant_id,
        user_id=user_id,
        enabled=enabled,
        updated_at=preference.updated_at if preference else None,
    )


@router.put("/preference/current", response_model=MemoryPreferenceView)
async def update_memory_preference(
    tenant_id: str,
    user_id: str,
    request: MemoryPreferenceUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
    principal: CurrentPrincipal = Depends(_memory_roles),
) -> MemoryPreferenceView:
    """Enable or disable future long-term memory reads and writes."""

    tenant_id, user_id = authorize_identity(
        principal,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    preference = await LongTermMemoryService(session).set_preference(
        tenant_id=tenant_id,
        user_id=user_id,
        enabled=request.enabled,
    )
    await session.commit()
    return MemoryPreferenceView(
        tenant_id=tenant_id,
        user_id=user_id,
        enabled=preference.long_term_memory_enabled,
        updated_at=preference.updated_at,
    )


@router.delete("/expired/cleanup", response_model=MemoryCleanupView)
async def cleanup_expired_memories(
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
    principal: CurrentPrincipal = Depends(_memory_roles),
) -> MemoryCleanupView:
    """Physically remove expired memories owned by the current user."""

    tenant_id, user_id = authorize_identity(
        principal,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    deleted = await LongTermMemoryService(session).cleanup_expired(
        tenant_id=tenant_id,
        user_id=user_id,
    )
    await session.commit()
    return MemoryCleanupView(deleted_count=deleted)
