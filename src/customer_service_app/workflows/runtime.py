from __future__ import annotations

from customer_service_app.domain.schemas import ChatRequest, ChatResponse
from customer_service_app.workflows.customer_service_graph import build_customer_service_graph
from customer_service_app.workflows.nodes import CustomerServiceGraphNodes


class CustomerServiceGraphRuntime:
    """负责编译并调用请求级客服 Graph。"""

    def __init__(self, nodes: CustomerServiceGraphNodes) -> None:
        self._graph = build_customer_service_graph(nodes)

    async def invoke(self, request: ChatRequest) -> ChatResponse:
        """从 START 执行到 END，并把最终 State 校验为 ChatResponse。"""

        state = await self._graph.ainvoke(
            {
                "request": request.model_dump(mode="json"),
                "trace": [],
                "error": None,
            }
        )
        return ChatResponse.model_validate(state["response"])
