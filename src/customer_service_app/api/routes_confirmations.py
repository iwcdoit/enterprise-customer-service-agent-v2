from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.api.dependencies import get_customer_service_agent
from customer_service_app.domain.schemas import (
    ConfirmationDecisionRequest,
    ConfirmationDecisionResponse,
    PendingActionView,
)
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


router = APIRouter(prefix="/confirmations", tags=["confirmations"])


@router.get("", response_model=list[PendingActionView])
async def list_pending_actions(
    tenant_id: str,
    user_id: str,
    include_expired: bool = False,
    session: AsyncSession = Depends(get_db_session),
) -> list[PendingActionView]:
    """列出当前用户等待确认的高风险动作。"""

    return await ConfirmationService(session).list_pending_actions(
        tenant_id=tenant_id,
        user_id=user_id,
        include_expired=include_expired,
    )


@router.get("/{confirmation_id}", response_model=PendingActionView)
async def get_confirmation(
    confirmation_id: str,
    tenant_id: str,
    user_id: str,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> PendingActionView:
    """读取确认详情，并校验租户和用户归属。"""

    return await agent.get_confirmation(
        tenant_id=tenant_id,
        user_id=user_id,
        confirmation_id=confirmation_id,
    )


@router.post("/{confirmation_id}/approve", response_model=ConfirmationDecisionResponse)
async def approve_confirmation(
    confirmation_id: str,
    request: ConfirmationDecisionRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> ConfirmationDecisionResponse:
    """批准动作并恢复原来被 interrupt 暂停的 Graph thread。"""

    response = await agent.resume_confirmation(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        confirmation_id=confirmation_id,
        decision="approve",
        reason=request.comment,
    )
    return ConfirmationDecisionResponse(
        confirmation_id=confirmation_id,
        decision="approve",
        graph_status=response.status,
        conversation_id=response.conversation_id,
        thread_id=response.thread_id,
        answer=response.answer,
        next_confirmation=response.pending_confirmation,
        tool_results=response.tool_results,
        trace=response.trace,
    )


@router.post("/{confirmation_id}/reject", response_model=ConfirmationDecisionResponse)
async def reject_confirmation(
    confirmation_id: str,
    request: ConfirmationDecisionRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> ConfirmationDecisionResponse:
    """拒绝动作并恢复原 Graph thread 生成取消说明。"""

    response = await agent.resume_confirmation(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        confirmation_id=confirmation_id,
        decision="reject",
        reason=request.comment,
    )
    return ConfirmationDecisionResponse(
        confirmation_id=confirmation_id,
        decision="reject",
        graph_status=response.status,
        conversation_id=response.conversation_id,
        thread_id=response.thread_id,
        answer=response.answer,
        next_confirmation=response.pending_confirmation,
        tool_results=response.tool_results,
        trace=response.trace,
    )
