from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.customer_service_graph import build_customer_service_graph


class RoutingNodes:
    def __init__(self, *, planned: bool) -> None:
        self.planned = planned
        self.calls: list[str] = []

    async def prepare(self, state):
        self.calls.append("prepare")
        return Command(goto="plan" if self.planned else "answer")

    async def plan(self, state):
        self.calls.append("plan")
        return Command(update={"plan": {"steps": []}}, goto="execute_plan")

    async def execute_plan(self, state):
        self.calls.append("execute_plan")
        return Command(update={"plan_execution": {"completed_step_ids": []}}, goto="answer")

    async def answer(self, state):
        self.calls.append("answer")
        return {"response": {"conversation_id": "conversation-1", "answer": "ok"}}

    async def await_confirmation(self, state):
        raise AssertionError("unexpected node")

    async def apply_confirmation(self, state):
        raise AssertionError("unexpected node")


async def test_graph_skips_planner_for_simple_request() -> None:
    nodes = RoutingNodes(planned=False)
    graph = build_customer_service_graph(InMemorySaver())

    state = await graph.ainvoke(
        {"request": {}, "thread_id": "simple-thread"},
        config={"configurable": {"thread_id": "simple-thread"}},
        context=CustomerServiceGraphContext(nodes=nodes),  # type: ignore[arg-type]
    )

    assert state["response"]["answer"] == "ok"
    assert nodes.calls == ["prepare", "answer"]


async def test_graph_runs_bounded_plan_branch_for_complex_request() -> None:
    nodes = RoutingNodes(planned=True)
    graph = build_customer_service_graph(InMemorySaver())

    state = await graph.ainvoke(
        {"request": {}, "thread_id": "plan-thread"},
        config={"configurable": {"thread_id": "plan-thread"}},
        context=CustomerServiceGraphContext(nodes=nodes),  # type: ignore[arg-type]
    )

    assert state["response"]["answer"] == "ok"
    assert nodes.calls == ["prepare", "plan", "execute_plan", "answer"]


class HilNodes:
    """用最小节点验证真实拓扑的 interrupt、checkpoint 和 resume。"""

    async def prepare(self, state):
        return Command(
            update={
                "status": "running",
                "pending_confirmations": [{"id": "confirmation-1"}],
                "confirmation_cursor": 0,
            },
            goto="await_confirmation",
        )

    async def await_confirmation(self, state):
        decision = interrupt({"pending_confirmation": state["pending_confirmations"][0]})
        return Command(update={"confirmation_decision": decision}, goto="apply_confirmation")

    async def apply_confirmation(self, state):
        assert state["confirmation_decision"]["decision"] == "approve"
        return Command(update={"status": "running"}, goto="answer")

    async def answer(self, state):
        return {
            "status": "completed",
            "response": {"conversation_id": "conversation-1", "answer": "已执行"},
        }

    async def plan(self, state):
        raise AssertionError("unexpected node")

    async def execute_plan(self, state):
        raise AssertionError("unexpected node")


async def test_graph_interrupt_checkpoint_and_resume() -> None:
    graph = build_customer_service_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "hil-thread"}}
    context = CustomerServiceGraphContext(nodes=HilNodes())  # type: ignore[arg-type]

    interrupted = await graph.ainvoke(
        {"request": {}, "thread_id": "hil-thread"},
        config=config,
        context=context,
    )

    assert interrupted["pending_confirmations"][0]["id"] == "confirmation-1"
    assert interrupted["__interrupt__"]
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("await_confirmation",)

    completed = await graph.ainvoke(
        Command(resume={"confirmation_id": "confirmation-1", "decision": "approve"}),
        config=config,
        context=context,
    )

    assert completed["status"] == "completed"
    assert completed["response"]["answer"] == "已执行"
    assert (await graph.aget_state(config)).next == ()
