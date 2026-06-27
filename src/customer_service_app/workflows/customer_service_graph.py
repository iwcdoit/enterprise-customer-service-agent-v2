from __future__ import annotations

from typing import TypedDict

from customer_service_app.domain.schemas import ChatRequest, ChatResponse
from customer_service_app.services.customer_service_agent import CustomerServiceAgent


class CustomerServiceGraphState(TypedDict, total=False):
    """Minimal LangGraph state for wrapping the current agent flow."""

    request: ChatRequest
    response: ChatResponse
    error: str


def build_customer_service_graph(agent: CustomerServiceAgent):
    """Wrap the current service agent with a minimal LangGraph entrypoint."""

    from langgraph.graph import END, StateGraph

    async def answer_node(state: CustomerServiceGraphState) -> CustomerServiceGraphState:
        """图里的一个节点：读取 state.request，调用 Agent，写回 state.response。"""
        response = await agent.answer(state["request"])
        return {"response": response}

    graph = StateGraph(CustomerServiceGraphState)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()
