from __future__ import annotations

from fastapi import APIRouter, Depends

from customer_service_app.api.dependencies import get_customer_service_agent
from customer_service_app.domain.confirmations import (
    ConfirmationDecisionRequest,
    ConfirmationDecisionResponse,
    PendingActionView,
)
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


router = APIRouter(prefix="/confirmations", tags=["confirmations"])
"""人工确认接口路由组。

最终路径会叠加 main.py 的 `/api/v1` 前缀：
`/api/v1/confirmations/{confirmation_id}/approve`
"""


@router.get("/{confirmation_id}", response_model=PendingActionView)
async def get_confirmation(
    confirmation_id: str,
    tenant_id: str,
    user_id: str,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> PendingActionView:
    """Load the pending action that owns the interrupted graph thread."""

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
    """Resume LangGraph with an approval decision."""

    response = await agent.resume_confirmation(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        confirmation_id=confirmation_id,
        decision="approve",
        reason=request.reason,
    )
    return ConfirmationDecisionResponse(
        confirmation_id=confirmation_id,
        status="executed",
        message="操作已确认，原 LangGraph 线程已恢复并执行完成",
        result={
            "tool_results": [item.model_dump() for item in response.tool_results],
            "plan": response.plan.model_dump() if response.plan else None,
        },
        conversation_id=response.conversation_id,
        thread_id=response.thread_id,
        run_id=response.run_id,
        answer=response.answer,
        graph_status=response.status,
        next_confirmation=response.pending_confirmation,
        plan=response.plan.model_dump() if response.plan else None,
        tool_results=[item.model_dump() for item in response.tool_results],
        trace=[item.model_dump() for item in response.trace],
    )


@router.post("/{confirmation_id}/reject", response_model=ConfirmationDecisionResponse)
async def reject_confirmation(
    confirmation_id: str,
    request: ConfirmationDecisionRequest,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> ConfirmationDecisionResponse:
    """Resume LangGraph with a rejection decision."""

    response = await agent.resume_confirmation(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        confirmation_id=confirmation_id,
        decision="reject",
        reason=request.reason,
    )
    return ConfirmationDecisionResponse(
        confirmation_id=confirmation_id,
        status="rejected",
        message="操作已取消，原 LangGraph 线程已恢复并结束",
        conversation_id=response.conversation_id,
        thread_id=response.thread_id,
        run_id=response.run_id,
        answer=response.answer,
        graph_status=response.status,
        next_confirmation=response.pending_confirmation,
        plan=response.plan.model_dump() if response.plan else None,
        tool_results=[item.model_dump() for item in response.tool_results],
        trace=[item.model_dump() for item in response.trace],
    )
