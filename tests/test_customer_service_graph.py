from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.customer_service_graph import build_customer_service_graph


class DirectAnswerNodes:
    """用最小节点实现验证完整图的直接回答分支。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def prepare(self, state):
        self.calls.append("prepare")
        return Command(goto="rewrite")

    async def rewrite(self, state):
        self.calls.append("rewrite")
        return Command(goto="retrieve")

    async def retrieve(self, state):
        self.calls.append("retrieve")
        return Command(goto="evaluate_retrieval")

    async def evaluate_retrieval(self, state):
        self.calls.append("evaluate_retrieval")
        return Command(goto="decide")

    async def decide(self, state):
        self.calls.append("decide")
        return Command(update={"final_answer": "ok"}, goto="finalize")

    async def finalize(self, state):
        self.calls.append("finalize")
        return Command(goto="persist")

    async def persist(self, state):
        self.calls.append("persist")
        return {"status": "completed"}


async def test_graph_runs_complete_direct_answer_path() -> None:
    nodes = DirectAnswerNodes()
    graph = build_customer_service_graph(InMemorySaver())

    state = await graph.ainvoke(
        {"request": {}, "thread_id": "simple-thread"},
        config={"configurable": {"thread_id": "simple-thread"}},
        context=CustomerServiceGraphContext(nodes=nodes),  # type: ignore[arg-type]
    )

    assert state["final_answer"] == "ok"
    assert state["status"] == "completed"
    assert nodes.calls == [
        "prepare",
        "rewrite",
        "retrieve",
        "evaluate_retrieval",
        "decide",
        "finalize",
        "persist",
    ]


class HilNodes:
    """验证完整拓扑中的 interrupt、checkpoint 和 resume。"""

    async def prepare(self, state):
        return Command(
            update={"status": "running", "conversation_id": "conversation-1"},
            goto="create_pending_action",
        )

    async def create_pending_action(self, state):
        return Command(
            update={
                "pending_confirmation": {
                    "id": "confirmation-1",
                    "tool_name": "create_refund_case",
                },
                "status": "waiting_confirmation",
                "final_answer": "请确认",
            },
            goto="await_confirmation",
        )

    async def await_confirmation(self, state):
        decision = interrupt({"pending_confirmation": state["pending_confirmation"]})
        return Command(
            update={"confirmation_decision": decision, "status": "running"},
            goto="apply_confirmation",
        )

    async def apply_confirmation(self, state):
        assert state["confirmation_decision"]["decision"] == "approve"
        return Command(
            update={"pending_confirmation": None, "final_answer": "已执行"},
            goto="persist",
        )

    async def persist(self, state):
        return {"status": "completed"}


async def test_graph_interrupt_checkpoint_and_resume() -> None:
    graph = build_customer_service_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "hil-thread"}}
    context = CustomerServiceGraphContext(nodes=HilNodes())  # type: ignore[arg-type]

    interrupted = await graph.ainvoke(
        {"request": {}, "thread_id": "hil-thread"},
        config=config,
        context=context,
    )

    assert interrupted["status"] == "waiting_confirmation"
    assert interrupted["__interrupt__"]
    assert (await graph.aget_state(config)).next == ("await_confirmation",)

    completed = await graph.ainvoke(
        Command(resume={"confirmation_id": "confirmation-1", "decision": "approve"}),
        config=config,
        context=context,
    )

    assert completed["status"] == "completed"
    assert completed["final_answer"] == "已执行"
    assert (await graph.aget_state(config)).next == ()
