from __future__ import annotations

from customer_service_app.domain.schemas import ChatRequest
from customer_service_app.infrastructure.llm.base import LLMResponse
from customer_service_app.services.planner_service import PlannerService
from customer_service_app.tools.default_registry import build_default_tool_registry


class FakeLLMClient:
    def __init__(self, content: str):
        self.content = content
        self.called = False

    async def chat(self, messages, **kwargs):
        self.called = True
        return LLMResponse(content=self.content, tool_calls=[], model=kwargs.get("model"))


def test_simple_question_does_not_need_plan() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry())
    request = ChatRequest(tenant_id="t1", user_id="u1", question="你好，退货政策是什么？")

    assert planner.needs_plan(request) is False


def test_multi_intent_question_needs_plan() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry())
    request = ChatRequest(
        tenant_id="t1",
        user_id="u1",
        question="帮我查订单，如果已经签收，再帮我申请退款。",
    )

    assert planner.needs_plan(request) is True


async def test_build_plan_normalizes_llm_json() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry(), max_steps=3)
    llm = FakeLLMClient(
        '{"steps":[{"id":"s1","title":"查订单","action_type":"tool",'
        '"tool_name":"query_order_status","arguments":{"order_id":"o1"}},'
        '{"id":"s2","title":"申请退款","action_type":"tool",'
        '"tool_name":"create_refund_ticket","depends_on":["s1"]}]}'
    )
    request = ChatRequest(
        tenant_id="t1",
        user_id="u1",
        question="帮我查订单，如果已经签收，再帮我申请退款。",
    )

    plan, response = await planner.build_plan(request=request, llm_client=llm, model="planner")

    assert response is not None
    assert plan is not None
    assert plan.source == "llm"
    assert [step.tool_name for step in plan.steps] == ["query_order_status", "create_refund_ticket"]
    assert plan.steps[1].requires_confirmation is True


async def test_build_plan_falls_back_to_rules_on_invalid_json() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry(), max_steps=3)
    llm = FakeLLMClient("not json")
    request = ChatRequest(
        tenant_id="t1",
        user_id="u1",
        question="帮我查订单，如果已经签收，再帮我申请退款。",
    )

    plan, response = await planner.build_plan(request=request, llm_client=llm, model="planner")

    assert response is not None
    assert plan is not None
    assert plan.source == "rule"
    assert len(plan.steps) >= 2
