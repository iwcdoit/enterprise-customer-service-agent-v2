from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.state import CustomerServiceGraphState


def build_customer_service_graph(checkpointer):
    """构建支持 Checkpoint 和 HIL 恢复的客服业务图。

    图文件只声明拓扑。每个 wrapper 从 Runtime Context 取得当前请求的 Nodes，真正的
    会话、Planner、工具执行和确认逻辑都在 ``nodes.py`` 中。
    """

    graph = StateGraph(
        CustomerServiceGraphState,
        context_schema=CustomerServiceGraphContext,
    )

    async def prepare(state, runtime: Runtime[CustomerServiceGraphContext]):
        return await runtime.context.nodes.prepare(state)

    async def plan(state, runtime: Runtime[CustomerServiceGraphContext]):
        return await runtime.context.nodes.plan(state)

    async def execute_plan(state, runtime: Runtime[CustomerServiceGraphContext]):
        return await runtime.context.nodes.execute_plan(state)

    async def await_confirmation(state, runtime: Runtime[CustomerServiceGraphContext]):
        return await runtime.context.nodes.await_confirmation(state)

    async def apply_confirmation(state, runtime: Runtime[CustomerServiceGraphContext]):
        return await runtime.context.nodes.apply_confirmation(state)

    async def answer(state, runtime: Runtime[CustomerServiceGraphContext]):
        return await runtime.context.nodes.answer(state)

    external_retry = RetryPolicy(
        max_attempts=2,
        initial_interval=0.5,
        backoff_factor=2,
        retry_on=ExternalServiceError,
    )
    graph.add_node("prepare", prepare, destinations=("plan", "answer"))
    graph.add_node("plan", plan, retry_policy=external_retry, destinations=("execute_plan",))
    graph.add_node(
        "execute_plan",
        execute_plan,
        destinations=("await_confirmation", "answer"),
    )
    graph.add_node(
        "await_confirmation",
        await_confirmation,
        destinations=("apply_confirmation",),
    )
    graph.add_node(
        "apply_confirmation",
        apply_confirmation,
        destinations=("await_confirmation", "answer"),
    )
    graph.add_node("answer", answer)
    graph.add_edge(START, "prepare")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)
