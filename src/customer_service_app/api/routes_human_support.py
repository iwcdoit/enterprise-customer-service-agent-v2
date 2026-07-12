from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.domain.human_support import (
    HumanAssignmentRequest,
    HumanHandoffView,
    HumanMessageRequest,
    HumanResolutionConfirmationRequest,
    HumanResolutionRequest,
)
from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.human_support_service import HumanSupportService


router = APIRouter(prefix="/human-support", tags=["human-support"])


@router.get("/queue", response_model=list[HumanHandoffView])
async def list_handoff_queue(
    tenant_id: str,
    status: list[str] | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
) -> list[HumanHandoffView]:
    service = HumanSupportService(session)
    items = await service.list_queue(tenant_id=tenant_id, statuses=status, limit=limit)
    return [service.to_view(item) for item in items]


@router.post("/{handoff_id}/assign", response_model=HumanHandoffView)
async def assign_handoff(
    handoff_id: str,
    request: HumanAssignmentRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HumanHandoffView:
    service = HumanSupportService(session)
    item = await service.assign(
        tenant_id=request.tenant_id,
        handoff_id=handoff_id,
        agent_id=request.agent_id,
        expected_version=request.expected_version,
    )
    await session.commit()
    return service.to_view(item)


@router.post("/{handoff_id}/messages", response_model=HumanHandoffView)
async def send_human_message(
    handoff_id: str,
    request: HumanMessageRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HumanHandoffView:
    service = HumanSupportService(session)
    item = await service.send_agent_message(
        tenant_id=request.tenant_id,
        handoff_id=handoff_id,
        agent_id=request.agent_id,
        content=request.content,
    )
    await session.commit()
    return service.to_view(item)


@router.post("/{handoff_id}/resolution", response_model=HumanHandoffView)
async def submit_human_resolution(
    handoff_id: str,
    request: HumanResolutionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HumanHandoffView:
    service = HumanSupportService(session)
    item = await service.submit_resolution(
        tenant_id=request.tenant_id,
        handoff_id=handoff_id,
        agent_id=request.agent_id,
        resolution_code=request.resolution_code,
        summary=request.summary,
        next_mode=request.next_mode,
        metadata=request.metadata,
    )
    await session.commit()
    return service.to_view(item)


@router.post("/{handoff_id}/resolution/confirm", response_model=HumanHandoffView)
async def confirm_human_resolution(
    handoff_id: str,
    request: HumanResolutionConfirmationRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HumanHandoffView:
    service = HumanSupportService(session)
    item = await service.confirm_resolution(
        tenant_id=request.tenant_id,
        handoff_id=handoff_id,
        operator_id=request.operator_id,
        expected_version=request.expected_version,
    )
    await session.commit()
    return service.to_view(item)
