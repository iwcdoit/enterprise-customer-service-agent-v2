from __future__ import annotations

from fastapi import APIRouter, Depends

from customer_service_app.api.dependencies import get_customer_service_agent
from customer_service_app.domain.schemas import GraphStateView
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


router = APIRouter(prefix="/graph", tags=["graph"])
"""Graph 调试接口路由组。

最终路径是 `/api/v1/graph/...`，主要给运营验证台和开发排查使用。
"""


@router.get("/threads/{thread_id}", response_model=GraphStateView)
async def get_graph_thread(
    thread_id: str,
    agent: CustomerServiceAgent = Depends(get_customer_service_agent),
) -> GraphStateView:
    """Inspect a sanitized LangGraph checkpoint for operations and debugging."""

    return await agent.graph_state(thread_id=thread_id)
