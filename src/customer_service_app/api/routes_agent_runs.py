from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.schemas import AgentRunStepView, AgentRunView
from customer_service_app.infrastructure.db.repositories import AgentRunRepository
from customer_service_app.infrastructure.db.session import get_db_session


router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])
"""Agent 运行记录查询路由组。

用于查看一次请求的 trace、模型、token、错误等落库信息。
"""


@router.get("/{run_id}", response_model=AgentRunView)
async def get_agent_run(
    run_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> AgentRunView:
    """Return persisted trace information for one Agent request."""

    repo = AgentRunRepository(session)
    run = await repo.get_run(run_id=run_id)

    if run is None:
        raise AppError("Agent run not found", code="agent_run_not_found", status_code=404)

    steps = await repo.list_steps(run_id=run_id)

    return AgentRunView(
        id=run.id,
        tenant_id=run.tenant_id,
        user_id=run.user_id,
        conversation_id=run.conversation_id,
        status=run.status,
        model=run.model,
        total_tokens=run.total_tokens,
        error_code=run.error_code,
        error_message=run.error_message,
        steps=[
            AgentRunStepView(
                id=step.id,
                stage=step.stage,
                name=step.name,
                status=step.status,
                input=step.input_json,
                output=step.output_json,
                latency_ms=step.latency_ms,
            )
            for step in steps
        ],
    )
