from __future__ import annotations

from fastapi import APIRouter, Depends

from customer_service_app.api.dependencies import get_customer_service_agent
from customer_service_app.domain.schemas import GraphStateView
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/threads/{thread_id}", response_model=GraphStateView)
async def get_graph_state(
    thread_id: str,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> GraphStateView:
    """读取 Graph 当前节点、interrupt 和脱敏状态，不会继续执行线程。"""

    return await agent.graph_state(thread_id=thread_id)
