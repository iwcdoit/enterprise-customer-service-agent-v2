from __future__ import annotations

import json

import pytest

from customer_service_app.domain.query_rewrite import QueryRewriteResult
from customer_service_app.domain.schemas import ChatRequest
from customer_service_app.infrastructure.llm.base import LLMResponse, LLMToolCall
from customer_service_app.services.planner_service import PlannerService
from customer_service_app.tools.default_registry import build_default_tool_registry


class StructuredPlannerLLM:
    """返回仍需后端治理和标准化的规划结果。"""

    async def chat(self, messages, **kwargs):
        return LLMResponse(
            content=json.dumps(
                {
                    "steps": [
                        {
                            "title": "查询订单",
                            "goal": "查询订单状态",
                            "action_type": "tool",
                            "tool_name": "query_order_status",
                            "arguments": {"order_id": "202606040001"},
                        },
                        {
                            "title": "提交退款",
                            "goal": "创建退款申请",
                            "action_type": "tool",
                            "tool_name": "create_refund_case",
                            "arguments": {
                                "order_id": "202606040001",
                                "reason": "商品损坏",
                                "refund_type": "return_refund",
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            tool_calls=[],
            model="planner-model",
            total_tokens=120,
        )


async def test_complex_request_uses_validated_llm_plan() -> None:
    service = PlannerService(tool_registry=build_default_tool_registry(), max_steps=6)
    plan, response = await service.build_plan(
        request=ChatRequest(
            tenant_id="default",
            user_id="u001",
            question="查订单 202606040001，然后帮我退款",
        ),
        conversation_id="conversation-1",
        llm_client=StructuredPlannerLLM(),
        model="planner-model",
    )

    assert response is not None
    assert plan is not None
    assert plan.steps[0].action_type == "tool"
    assert plan.steps[1].action_type == "confirm"
    assert plan.steps[1].requires_confirmation is True


class ForcedPlannerLLM:
    def __init__(self):
        self.kwargs = None
        self.messages = None

    async def chat(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return LLMResponse(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="plan-call-1",
                    name="submit_plan",
                    arguments=json.dumps(
                        {
                            "steps": [
                                {
                                    "step_id": "lookup-a",
                                    "title": "查询订单",
                                    "goal": "查询第一笔订单",
                                    "action_type": "tool",
                                    "tool_name": "query_order_status",
                                    "depends_on": [],
                                    "arguments": {"order_id": "202606040001"},
                                },
                                {
                                    "step_id": "lookup-b",
                                    "title": "查询物流",
                                    "goal": "查询第二笔订单物流",
                                    "action_type": "tool",
                                    "tool_name": "query_logistics_status",
                                    "depends_on": [],
                                    "arguments": {"order_id": "202606040002"},
                                },
                            ]
                        },
                        ensure_ascii=False,
                    ),
                )
            ],
            model="planner-model",
            prompt_tokens=30,
            completion_tokens=20,
            total_tokens=50,
        )


async def test_planner_uses_resolved_goal_and_forced_structured_tool() -> None:
    llm = ForcedPlannerLLM()
    service = PlannerService(tool_registry=build_default_tool_registry(), max_steps=6)
    request = ChatRequest(tenant_id="default", user_id="u001", question="查这两笔订单")

    plan, response = await service.build_plan(
        request=request,
        conversation_id="conversation-1",
        llm_client=llm,
        model="planner-model",
        resolved_question="查询订单 202606040001 和订单 202606040002 的状态与物流",
    )

    assert plan is not None
    assert response is not None
    assert plan.user_goal.startswith("查询订单 202606040001")
    assert plan.steps[0].depends_on == []
    assert plan.steps[1].depends_on == []
    assert llm.kwargs["tool_choice"]["function"]["name"] == "submit_plan"


async def test_planner_receives_verified_rewrite_fields_only() -> None:
    llm = ForcedPlannerLLM()
    service = PlannerService(tool_registry=build_default_tool_registry(), max_steps=6)
    request = ChatRequest(
        tenant_id="default",
        user_id="u001",
        question="查这两笔订单",
    )

    await service.build_plan(
        request=request,
        conversation_id="conversation-1",
        llm_client=llm,
        model="planner-model",
        resolved_question="查询订单 202606040001 和订单 202606040002",
        rewrite_result=QueryRewriteResult(
            original_question=request.question,
            standalone_question="查询订单 202606040001 和订单 202606040002",
            dense_query="查询两个订单状态",
            sparse_query="202606040001 202606040002 订单 状态",
            entities={"order_ids": ["202606040001", "202606040002"]},
            missing_fields=[],
            needs_rewrite=True,
        ),
        memory_context={
            "recent_messages": [{"role": "user", "content": "不应重复传给 Planner"}],
            "memories": [
                {
                    "memory_key": "preferred_contact",
                    "verification_status": "explicit_user",
                },
                {
                    "memory_key": "model_guess",
                    "verification_status": "unverified",
                },
            ],
        },
    )

    payload = json.loads(llm.messages[1]["content"])
    assert payload["original_question"] == request.question
    assert payload["verified_entities"]["order_ids"] == [
        "202606040001",
        "202606040002",
    ]
    assert "recent_messages" not in payload["controlled_context"]
    assert len(payload["controlled_context"]["verified_memories"]) == 1


def test_planner_topologically_sorts_forward_dependencies() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry())
    request = ChatRequest(
        tenant_id="t1",
        user_id="u1",
        question="查询订单后申请退款",
    )

    plan = planner._normalize_llm_plan(
        payload={
            "steps": [
                {
                    "step_id": "refund",
                    "title": "创建退款",
                    "goal": "申请退款",
                    "action_type": "tool",
                    "tool_name": "create_refund_case",
                    "depends_on": ["lookup"],
                    "arguments": {
                        "order_id": "202606040001",
                        "reason": "商品损坏",
                        "refund_type": "return_refund",
                    },
                },
                {
                    "step_id": "lookup",
                    "title": "查询订单",
                    "goal": "确认订单状态",
                    "action_type": "tool",
                    "tool_name": "query_order_status",
                    "depends_on": [],
                    "arguments": {"order_id": "202606040001"},
                },
            ]
        },
        request=request,
        conversation_id="c1",
    )

    assert [step.tool_name for step in plan.steps] == [
        "query_order_status",
        "create_refund_case",
    ]
    assert plan.steps[1].depends_on == ["s1"]


def test_planner_rejects_dependency_cycle() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry())
    request = ChatRequest(tenant_id="t1", user_id="u1", question="查询两个订单")

    with pytest.raises(ValueError, match="cyclic"):
        planner._normalize_llm_plan(
            payload={
                "steps": [
                    {
                        "step_id": "a",
                        "title": "A",
                        "goal": "A",
                        "action_type": "rag",
                        "depends_on": ["b"],
                        "arguments": {},
                    },
                    {
                        "step_id": "b",
                        "title": "B",
                        "goal": "B",
                        "action_type": "llm",
                        "depends_on": ["a"],
                        "arguments": {},
                    },
                ]
            },
            request=request,
            conversation_id="c1",
        )


def test_rule_plan_extracts_order_id_and_requires_confirmation() -> None:
    planner = PlannerService(tool_registry=build_default_tool_registry())
    request = ChatRequest(
        tenant_id="t1",
        user_id="u1",
        question="查询订单 202607110001，如果已经签收就申请退款。",
    )

    plan = planner.build_rule_based_plan(
        request=request,
        conversation_id="conversation-1",
    )

    assert plan is not None
    assert plan.steps[0].arguments == {"order_id": "202607110001"}
    refund_step = plan.steps[-1]
    assert refund_step.action_type == "confirm"
    assert refund_step.arguments["order_id"] == "202607110001"
