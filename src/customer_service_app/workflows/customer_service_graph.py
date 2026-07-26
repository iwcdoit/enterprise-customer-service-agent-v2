from __future__ import annotations

from functools import wraps

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.observability.metrics import observe_graph_node
from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.state import CustomerServiceGraphState


def build_customer_service_graph(checkpointer):
    """构建并编译单轮客服 LangGraph。"""

    graph = StateGraph(
        CustomerServiceGraphState,
        context_schema=CustomerServiceGraphContext,
    )

    def observed(node_name: str):
        """为节点统一记录耗时，不采集 state 中的用户或业务数据。"""

        def decorate(handler):
            @wraps(handler)
            async def wrapped(*args, **kwargs):
                with observe_graph_node(node_name):
                    return await handler(*args, **kwargs)

            return wrapped

        return decorate

    @observed("prepare")
    async def prepare(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.prepare(state)

    @observed("retrieve")
    async def retrieve(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.retrieve(state)

    @observed("rewrite")
    async def rewrite(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.rewrite(state)

    @observed("clarify")
    async def clarify(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.clarify(state)

    @observed("evaluate_retrieval")
    async def evaluate_retrieval(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.evaluate_retrieval(state)

    @observed("decide")
    async def decide(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.decide(state)

    @observed("execute_plan_step")
    async def execute_plan_step(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.execute_plan_step(state)

    @observed("execute_tool_call")
    async def execute_tool_call(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.execute_tool_call(state)

    @observed("create_pending_action")
    async def create_pending_action(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.create_pending_action(state)

    @observed("await_confirmation")
    async def await_confirmation(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.await_confirmation(state)

    @observed("apply_confirmation")
    async def apply_confirmation(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.apply_confirmation(state)

    @observed("finalize")
    async def finalize(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.finalize(state)

    @observed("persist")
    async def persist(
        state: CustomerServiceGraphState,
        runtime: Runtime[CustomerServiceGraphContext],
    ):
        return await runtime.context.nodes.persist(state)

    external_retry = RetryPolicy(
        max_attempts=2,
        initial_interval=0.5,
        backoff_factor=2,
        retry_on=ExternalServiceError,
    )

    graph.add_node("prepare", prepare, destinations=("rewrite",))

    graph.add_node(
        "rewrite",
        rewrite,
        retry_policy=external_retry,
        destinations=("clarify", "retrieve", "persist"),
    )

    graph.add_node("clarify", clarify, destinations=("persist",))

    graph.add_node(
        "retrieve",
        retrieve,
        retry_policy=external_retry,
        destinations=("evaluate_retrieval",),
    )

    graph.add_node(
        "evaluate_retrieval",
        evaluate_retrieval,
        destinations=("retrieve", "clarify", "decide"),
    )

    graph.add_node(
        "decide",
        decide,
        retry_policy=external_retry,
        destinations=("execute_plan_step", "execute_tool_call", "finalize"),
    )

    graph.add_node(
        "execute_plan_step",
        execute_plan_step,
        destinations=("execute_plan_step", "create_pending_action", "finalize"),
    )

    graph.add_node(
        "execute_tool_call",
        execute_tool_call,
        destinations=("execute_tool_call", "create_pending_action", "finalize"),
    )

    graph.add_node(
        "create_pending_action",
        create_pending_action,
        destinations=("await_confirmation",),
    )

    graph.add_node(
        "await_confirmation",
        await_confirmation,
        destinations=("apply_confirmation",),
    )

    graph.add_node(
        "apply_confirmation",
        apply_confirmation,
        destinations=("execute_plan_step", "execute_tool_call", "persist"),
    )

    graph.add_node(
        "finalize",
        finalize,
        retry_policy=external_retry,
        destinations=("persist",),
    )

    graph.add_node("persist", persist)

    graph.add_edge(START, "prepare")

    graph.add_edge("persist", END)

    return graph.compile(checkpointer=checkpointer)
