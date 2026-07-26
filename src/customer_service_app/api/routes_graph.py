from __future__ import annotations

from fastapi import APIRouter, Depends

from customer_service_app.api.dependencies import (
    get_customer_service_agent,
    require_roles,
)
from customer_service_app.core.security import CurrentPrincipal, authorize_identity
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
    principal: CurrentPrincipal = Depends(require_roles("operator", "admin")),
) -> GraphStateView:
    """Inspect a sanitized LangGraph checkpoint for operations and debugging."""

    state = await agent.graph_state(thread_id=thread_id)
    request_payload = state.values.get("request", {})
    if isinstance(request_payload, dict) and request_payload.get("tenant_id"):
        authorize_identity(
            principal,
            tenant_id=str(request_payload["tenant_id"]),
            user_id=(
                str(request_payload["user_id"])
                if request_payload.get("user_id")
                else None
            ),
        )
    return state
