from __future__ import annotations

import json
from typing import Any

from customer_service_app.domain.planning import AgentPlan, PlanActionType, PlanStep
from customer_service_app.domain.schemas import ChatRequest
from customer_service_app.infrastructure.llm.base import LLMClient, LLMResponse
from customer_service_app.services.tool_registry import ToolRegistry, ToolSpec


class PlannerService:
    """Build a bounded plan for complex multi-intent requests."""

    _condition_words = ("如果", "不行", "不能", "然后", "同时", "再", "并且", "否则")
    _order_words = ("订单", "物流", "快递", "签收", "发货")
    _refund_words = ("退款", "退货", "退换", "换货", "售后")
    _compensation_words = ("补偿", "赔偿", "价保", "差价")
    _handoff_words = ("人工", "客服", "投诉", "升级")

    def __init__(self, *, tool_registry: ToolRegistry, max_steps: int = 6):
        self._tool_registry = tool_registry
        self._max_steps = max_steps

    def needs_plan(self, request: ChatRequest) -> bool:
        """Use a zero-token gate so simple questions never pay planner cost."""

        question = request.question
        if any(word in question for word in self._condition_words):
            return True

        intents = 0
        if any(word in question for word in self._order_words):
            intents += 1
        if any(word in question for word in self._refund_words):
            intents += 1
        if any(word in question for word in self._compensation_words):
            intents += 1
        if any(word in question for word in self._handoff_words):
            intents += 1
        return intents >= 2

    async def build_plan(
        self,
        *,
        request: ChatRequest,
        llm_client: LLMClient,
        model: str,
    ) -> tuple[AgentPlan | None, LLMResponse | None]:
        """Generate and validate a plan, falling back to deterministic rules."""

        if not self.needs_plan(request):
            return None, None

        tools = [
            self._tool_summary(self._tool_registry.require(name))
            for name in sorted(self._tool_registry.names())
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是客服 Agent 的任务规划器。只输出 JSON，不回答用户问题。"
                    "计划步骤必须有限，不要生成循环。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_goal": request.question,
                        "max_steps": self._max_steps,
                        "available_tools": tools,
                        "output_schema": {
                            "steps": [
                                {
                                    "id": "step_1",
                                    "title": "步骤说明",
                                    "action_type": "tool|rag|llm|handoff|final",
                                    "tool_name": "工具名，可为空",
                                    "arguments": {},
                                    "depends_on": [],
                                    "requires_confirmation": False,
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        response = await llm_client.chat(messages, model=model, temperature=0)
        try:
            payload = json.loads(_extract_json(response.content))
            plan = self._normalize_llm_plan(payload, request=request)
        except (TypeError, ValueError, json.JSONDecodeError):
            plan = self.build_rule_based_plan(request)
        return plan, response

    def build_rule_based_plan(self, request: ChatRequest) -> AgentPlan:
        """Build a conservative plan without spending LLM tokens."""

        question = request.question
        steps: list[PlanStep] = []

        if any(word in question for word in self._order_words):
            steps.append(
                self._tool_step(
                    step_id="step_1",
                    title="查询订单或物流状态",
                    tool_name="query_order_status",
                )
            )

        if any(word in question for word in self._refund_words):
            steps.append(
                self._tool_step(
                    step_id=f"step_{len(steps) + 1}",
                    title="根据售后诉求创建退款或退货申请",
                    tool_name="create_refund_ticket",
                    depends_on=[steps[-1].id] if steps else [],
                )
            )

        if any(word in question for word in self._handoff_words):
            steps.append(
                self._tool_step(
                    step_id=f"step_{len(steps) + 1}",
                    title="创建转人工处理请求",
                    tool_name="transfer_to_human",
                    depends_on=[steps[-1].id] if steps else [],
                )
            )

        if not steps:
            steps.append(
                PlanStep(
                    id="step_1",
                    title="直接组织最终回复",
                    action_type="final",
                )
            )

        return AgentPlan(
            user_goal=question,
            max_steps=self._max_steps,
            steps=steps[: self._max_steps],
            source="rule",
        )

    def _normalize_llm_plan(self, payload: dict[str, Any], *, request: ChatRequest) -> AgentPlan:
        if not isinstance(payload, dict):
            raise ValueError("Planner output must be a JSON object")

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("Planner output must contain steps")

        steps: list[PlanStep] = []
        for index, item in enumerate(raw_steps[: self._max_steps], start=1):
            if not isinstance(item, dict):
                continue
            tool_name = _optional_str(item.get("tool_name"))
            action_type = self._normalize_action_type(item.get("action_type"), tool_name)
            if action_type == "tool" and tool_name and tool_name not in self._tool_registry.names():
                continue
            depends_on = [
                str(value)
                for value in item.get("depends_on", [])
                if isinstance(value, str)
            ]
            requires_confirmation = bool(item.get("requires_confirmation"))
            if tool_name and tool_name in self._tool_registry.names():
                requires_confirmation = requires_confirmation or self._tool_registry.require(
                    tool_name
                ).requires_confirmation
            steps.append(
                PlanStep(
                    id=str(item.get("id") or f"step_{index}"),
                    title=str(item.get("title") or f"执行步骤 {index}"),
                    action_type=action_type,
                    tool_name=tool_name,
                    arguments=(
                        item.get("arguments")
                        if isinstance(item.get("arguments"), dict)
                        else {}
                    ),
                    depends_on=depends_on,
                    requires_confirmation=requires_confirmation,
                )
            )

        if not steps:
            return self.build_rule_based_plan(request)
        return AgentPlan(
            user_goal=str(payload.get("user_goal") or request.question),
            max_steps=self._max_steps,
            steps=steps,
            source="llm",
        )

    def _tool_step(
        self,
        *,
        step_id: str,
        title: str,
        tool_name: str,
        depends_on: list[str] | None = None,
    ) -> PlanStep:
        tool = self._tool_registry.require(tool_name)
        return PlanStep(
            id=step_id,
            title=title,
            action_type="tool",
            tool_name=tool.name,
            depends_on=depends_on or [],
            requires_confirmation=tool.requires_confirmation,
        )

    @staticmethod
    def _tool_summary(tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "requires_confirmation": tool.requires_confirmation,
        }

    @staticmethod
    def _normalize_action_type(value: Any, tool_name: str | None) -> PlanActionType:
        allowed = {"tool", "rag", "llm", "handoff", "final"}
        if isinstance(value, str) and value in allowed:
            return value  # type: ignore[return-value]
        return "tool" if tool_name else "final"


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return stripped


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
