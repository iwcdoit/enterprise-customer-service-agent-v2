from __future__ import annotations

from customer_service_app.workflows.customer_service_graph import build_customer_service_graph


class FakeNodes:
    def __init__(self, *, planned: bool) -> None:
        self.planned = planned
        self.calls: list[str] = []

    async def prepare(self, state):
        self.calls.append("prepare")
        return {"route": "plan" if self.planned else "answer"}

    async def plan(self, state):
        self.calls.append("plan")
        return {"plan": {"steps": []}}

    async def execute_plan(self, state):
        self.calls.append("execute_plan")
        return {"plan_execution": {"completed_step_ids": []}}

    async def answer(self, state):
        self.calls.append("answer")
        return {
            "response": {
                "conversation_id": "conversation-1",
                "answer": "ok",
            }
        }


async def test_graph_skips_planner_for_simple_request() -> None:
    nodes = FakeNodes(planned=False)
    graph = build_customer_service_graph(nodes)  # type: ignore[arg-type]

    state = await graph.ainvoke({"request": {}, "trace": []})

    assert state["response"]["answer"] == "ok"
    assert nodes.calls == ["prepare", "answer"]


async def test_graph_runs_bounded_plan_branch_for_complex_request() -> None:
    nodes = FakeNodes(planned=True)
    graph = build_customer_service_graph(nodes)  # type: ignore[arg-type]

    state = await graph.ainvoke({"request": {}, "trace": []})

    assert state["response"]["answer"] == "ok"
    assert nodes.calls == ["prepare", "plan", "execute_plan", "answer"]
