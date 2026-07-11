from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from customer_service_app.workflows.nodes import CustomerServiceGraphNodes
from customer_service_app.workflows.state import CustomerServiceGraphState


def build_customer_service_graph(nodes: CustomerServiceGraphNodes):
    """构建第一版业务的 V2 编排图。

    简单请求跳过 Planner；复杂请求先生成有界计划并经过 ReAct 执行，
    最后统一进入原有回答链路生成回复并保存结果。
    """

    graph = StateGraph(CustomerServiceGraphState)
    graph.add_node("prepare", nodes.prepare)
    graph.add_node("plan", nodes.plan)
    graph.add_node("execute_plan", nodes.execute_plan)
    graph.add_node("answer", nodes.answer)

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare",
        lambda state: state["route"],
        {"plan": "plan", "answer": "answer"},
    )
    graph.add_edge("plan", "execute_plan")
    graph.add_edge("execute_plan", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
