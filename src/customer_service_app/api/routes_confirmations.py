from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.domain.schemas import ConfirmationDecisionRequest, PendingActionView
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.confirmation_service import ConfirmationService


router = APIRouter(prefix="/confirmations", tags=["confirmations"])
"""Confirmation APIs for high-risk tool actions."""


@router.get("", response_model=list[PendingActionView])
async def list_pending_actions(
    tenant_id: str,
    user_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> list[PendingActionView]:
    """List pending tool actions for the current user."""
    return await ConfirmationService(session).list_pending_actions(
        tenant_id=tenant_id,
        user_id=user_id,
    )


@router.post("/{action_id}/approve", response_model=PendingActionView)
async def approve_pending_action(
    action_id: str,
    request: ConfirmationDecisionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> PendingActionView:
    """Approve one pending high-risk tool action."""
    view = await ConfirmationService(session).approve(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        action_id=action_id,
        comment=request.comment,
    )
    await session.commit()
    return view


@router.post("/{action_id}/reject", response_model=PendingActionView)
async def reject_pending_action(
    action_id: str,
    request: ConfirmationDecisionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> PendingActionView:
    """Reject one pending high-risk tool action."""
    view = await ConfirmationService(session).reject(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        action_id=action_id,
        comment=request.comment,
    )
    await session.commit()
    return view
