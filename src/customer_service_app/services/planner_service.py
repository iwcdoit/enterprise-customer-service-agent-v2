from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel, Field, ValidationError

from customer_service_app.domain.planning import AgentPlan, PlanStep
from customer_service_app.domain.schemas import ChatRequest
from customer_service_app.infrastructure.llm.base import LLMClient, LLMResponse
from customer_service_app.services.tool_registry import ToolRegistry, ToolSpec


class PlannerStepDraft(BaseModel):
    """Untrusted planner output before backend governance is applied."""

    step_id: str = ""
    title: str
    goal: str
    action_type: Literal["rag", "tool", "llm", "confirm", "handoff"]
    tool_name: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)


class PlannerDraft(BaseModel):
    steps: list[PlannerStepDraft] = Field(min_length=1)


class PlannerService:
    """Generate bounded structured plans only for genuinely complex requests."""

    _condition_words = ("如果", "不行", "不能", "然后", "同时", "再", "并且", "否则")

    def __init__(self, *, tool_registry: ToolRegistry, max_steps: int = 6):
        self._tool_registry = tool_registry
        self._max_steps = max_steps

    def needs_plan(self, request: ChatRequest, *, resolved_question: str | None = None) -> bool:
        """Use a zero-token gate so simple questions never pay planner cost."""
        question = (resolved_question or request.question).strip()

        if not question:
            return False

        for condition in self._condition_words:
            if condition in question:
                return True

        intents = 0

        if self._contains_any(question, ("订单", "物流", "快递", "运单", "单号", "发货", "签收")):
            intents += 1

        if self._contains_any(question, ("退款", "退货", "退掉", "换货", "补发", "重新发")):
            intents += 1

        if self._contains_any(question, ("补偿", "赔偿", "价保", "保价", "差价", "优惠券")):
            intents += 1

        if self._contains_any(question, ("人工", "客服", "投诉", "举报", "升级处理", "12315")):
            intents += 1

        action_verbs = sum(
            question.count(word)
            for word in ("查询", "查看", "申请", "创建", "取消", "修改", "转", "处理")
        )
        order_ids = set(re.findall(r"\b(?:[A-Za-z]{2,8}-?)?\d{8,24}\b", question))

        return intents >= 2 or action_verbs >= 2 or len(order_ids) >= 2




    async def build_plan(
        self,
        *,
        request: ChatRequest,
        conversation_id: str,
        llm_client: LLMClient,
        model: str,
        resolved_question: str | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> tuple[AgentPlan | None, LLMResponse | None]:
        """Generate and validate a plan, falling back deterministically on malformed output."""
        goal = (resolved_question or request.question).strip()
        if not self.needs_plan(request, resolved_question=goal):
            return None, None

        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_confirmation": tool.requires_confirmation,
                "parameters": tool.parameters,
            }
            for tool in self._registered_tools()
        ]

        submit_plan_tool = {
            "type": "function",
            "function": {
                "name": "submit_plan",
                "description": "提交经过依赖排序的有界执行计划",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": self._max_steps,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "goal": {"type": "string"},
                                    "action_type": {
                                        "type": "string",
                                        "enum": ["rag", "tool", "llm", "confirm", "handoff"],
                                    },
                                    "tool_name": {"type": ["string", "null"]},
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "arguments": {"type": "object"},
                                },
                                "required": [
                                    "step_id",
                                    "title",
                                    "goal",
                                    "action_type",
                                    "depends_on",
                                    "arguments",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["steps"],
                    "additionalProperties": False,
                },
            },
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是客户服务任务规划器。不要回答用户，只调用 submit_plan。"
                    "计划必须有界、可执行，禁止重复工具；查询先于写操作；"
                    "退款、补偿、换货、转人工等写操作使用 confirm 或 handoff；"
                    "只有存在真实数据依赖时才填写 depends_on，无依赖步骤可由后端并行执行。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_question": request.question,
                        "resolved_goal": goal,
                        "context": memory_context or {},
                        "max_steps": self._max_steps,
                        "available_tools": tools,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            response = await llm_client.chat(
                messages,
                model=model,
                temperature=0,
                tools=[submit_plan_tool],
                tool_choice={"type": "function", "function": {"name": "submit_plan"}},
            )
        except Exception:
            return (
                self.build_rule_based_plan(
                    request=request,
                    conversation_id=conversation_id,
                    resolved_question=goal,
                ),
                None,
            )

        validation_error: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for attempt in range(2):
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            total_tokens += response.total_tokens
            try:
                payload = self._planner_payload(response)
                plan = self._normalize_llm_plan(
                    payload=payload,
                    request=request,
                    conversation_id=conversation_id,
                    resolved_question=goal,
                )
                response.prompt_tokens = prompt_tokens
                response.completion_tokens = completion_tokens
                response.total_tokens = total_tokens
                return plan, response
            except (
                json.JSONDecodeError,
                JsonSchemaValidationError,
                ValidationError,
                TypeError,
                ValueError,
            ) as exc:
                validation_error = str(exc)
                if attempt == 1:
                    break
                repair_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "上一次计划未通过后端校验。请修正后重新调用 submit_plan。"
                            f"校验错误：{validation_error[:800]}"
                        ),
                    },
                ]
                try:
                    response = await llm_client.chat(
                        repair_messages,
                        model=model,
                        temperature=0,
                        tools=[submit_plan_tool],
                        tool_choice={"type": "function", "function": {"name": "submit_plan"}},
                    )
                except Exception:
                    break

        plan = self.build_rule_based_plan(
            request=request,
            conversation_id=conversation_id,
            resolved_question=goal,
        )
        response.prompt_tokens = prompt_tokens
        response.completion_tokens = completion_tokens
        response.total_tokens = total_tokens
        return plan, response


    def _normalize_llm_plan(
        self,
        *,
        payload: dict[str, Any],
        request: ChatRequest,
        conversation_id: str,
        resolved_question: str | None = None,
    ) -> AgentPlan:
        draft = PlannerDraft.model_validate(payload)
        raw_steps = draft.steps

        steps: list[PlanStep] = []
        seen_tools: set[tuple[str, str]] = set()

        id_mapping: dict[str, str] = {}
        for index, item in enumerate(raw_steps[: self._max_steps], start=1):
            if item.step_id:
                if item.step_id in id_mapping:
                    raise ValueError("Planner step_id must be unique")
                id_mapping[item.step_id] = f"s{index}"

        for index, item in enumerate(raw_steps[: self._max_steps], start=1):
            tool_name = item.tool_name
            action_type = item.action_type
            arguments = item.arguments
            requires_confirmation = False

            if tool_name:
                tool_name = str(tool_name)
                if tool_name not in self._tool_registry.names():
                    raise ValueError(f"Planner referenced unknown tool: {tool_name}")
                tool = self._tool_registry.require(tool_name)
                validate_json_schema(instance=arguments, schema=tool.parameters)
                requires_confirmation = tool.requires_confirmation
                action_type = "confirm" if requires_confirmation else "tool"
                duplicate_key = (
                    tool_name,
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                )
                if duplicate_key in seen_tools:
                    raise ValueError(f"Planner duplicated tool call: {tool_name}")
                seen_tools.add(duplicate_key)
            elif action_type not in {"rag", "llm"}:
                raise ValueError(f"Planner action {action_type} requires tool_name")

            unknown_dependencies = [
                value for value in item.depends_on if value not in id_mapping
            ]
            if unknown_dependencies:
                raise ValueError(f"Unknown plan dependencies: {unknown_dependencies}")
            dependencies = [id_mapping[value] for value in item.depends_on]
            current_step_id = f"s{len(steps) + 1}"
            if any(int(value[1:]) >= int(current_step_id[1:]) for value in dependencies):
                raise ValueError("Plan dependency must point to an earlier step")

            steps.append(
                PlanStep(
                    step_id=f"s{len(steps) + 1}",
                    title=item.title or f"步骤 {index}",
                    goal=item.goal or resolved_question or request.question,
                    action_type=action_type,
                    tool_name=tool_name,
                    depends_on=dependencies,
                    requires_confirmation=requires_confirmation,
                    arguments=arguments,
                )
            )

        if not steps:
            raise ValueError("Planner output contains no executable step")

        return AgentPlan(
            plan_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_goal=resolved_question or request.question,
            steps=steps,
            max_steps=self._max_steps,
        )

    def build_rule_based_plan(
        self,
        *,
        request: ChatRequest,
        conversation_id: str,
        resolved_question: str | None = None,
    ) -> AgentPlan | None:
        """Deterministic fallback used when planner output cannot pass validation."""
        if not self.needs_plan(request, resolved_question=resolved_question):
            return None

        question = resolved_question or request.question
        order_id = self._extract_order_id(question)
        steps: list[PlanStep] = []

        if order_id and "query_order_status" in self._tool_registry.names():
            steps.append(
                PlanStep(
                    step_id="s1",
                    title="查询订单状态",
                    goal="确认订单归属和履约状态",
                    action_type="tool",
                    tool_name="query_order_status",
                    arguments={"order_id": order_id},
                )
            )

        steps.append(
            PlanStep(
                step_id=f"s{len(steps) + 1}",
                title="检索售后政策",
                goal="检索与当前诉求匹配的售后规则",
                action_type="rag",
                depends_on=["s1"] if order_id else [],
            )
        )

        requested_actions: list[str] = []
        if self._contains_any(question, ("退款", "退货", "退掉")):
            requested_actions.append("refund")
        if self._contains_any(question, ("补偿", "赔偿", "价保", "保价", "差价", "优惠券")):
            requested_actions.append("compensation")
        if self._contains_any(question, ("换货", "补发", "重新发")):
            requested_actions.append("exchange")
        if self._contains_any(question, ("人工", "客服", "投诉", "举报", "12315")):
            requested_actions.append("handoff")

        if len(requested_actions) > 1:
            steps.append(
                PlanStep(
                    step_id=f"s{len(steps) + 1}",
                    title="确认后续处理分支",
                    goal="基于查询结果向用户说明可选方案，并在执行副作用前再次确认",
                    action_type="llm",
                    depends_on=[step.step_id for step in steps],
                )
            )
        elif requested_actions and requested_actions[0] in {"refund", "compensation", "exchange"} and not order_id:
            steps.append(
                PlanStep(
                    step_id=f"s{len(steps) + 1}",
                    title="补充业务参数",
                    goal="向用户询问订单号，未获得订单归属信息前不创建售后工单",
                    action_type="llm",
                    depends_on=[step.step_id for step in steps],
                )
            )
        elif requested_actions == ["refund"]:
            steps.append(
                self._confirmation_step(
                    steps=steps,
                    title="准备退款确认",
                    tool_name="create_refund_case",
                    arguments={
                        "order_id": order_id or "",
                        "reason": question,
                        "refund_type": "return_refund",
                    },
                    order_id=order_id,
                )
            )
        elif requested_actions == ["compensation"]:
            steps.append(
                self._confirmation_step(
                    steps=steps,
                    title="准备补偿确认",
                    tool_name="create_compensation_case",
                    arguments={
                        "order_id": order_id or "",
                        "reason": question,
                        "compensation_type": "partial_refund",
                    },
                    order_id=order_id,
                )
            )
        elif requested_actions == ["exchange"]:
            steps.append(
                self._confirmation_step(
                    steps=steps,
                    title="准备换货确认",
                    tool_name="create_exchange_case",
                    arguments={
                        "order_id": order_id or "",
                        "reason": question,
                    },
                    order_id=order_id,
                )
            )
        elif requested_actions == ["handoff"]:
            steps.append(
                self._confirmation_step(
                    steps=steps,
                    title="准备转人工确认",
                    tool_name="transfer_to_human",
                    arguments={"reason": question, "priority": "high"},
                    order_id=None,
                )
            )

        return AgentPlan(
            plan_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_goal=question,
            steps=steps[: self._max_steps],
            max_steps=self._max_steps,
        )

    @staticmethod
    def _extract_json(content: str) -> str:
        value = content.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value)
            value = re.sub(r"\s*```$", "", value)
        return value

    def _planner_payload(self, response: LLMResponse) -> dict[str, Any]:
        """Read forced tool arguments first, with content JSON as provider fallback."""
        for call in response.tool_calls:
            if call.name == "submit_plan":
                value = json.loads(call.arguments or "{}")
                if not isinstance(value, dict):
                    raise ValueError("submit_plan arguments must be an object")
                return value
        raw = self._extract_json(response.content)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Planner output must be a JSON object")
        return value

    @staticmethod
    def _extract_order_id(question: str) -> str | None:
        match = re.search(r"\b\d{8,24}\b", question)
        return match.group(0) if match else None

    @staticmethod
    def _contains_any(text: str, words: tuple[str, ...]) -> bool:
        return any(word in text for word in words)

    @staticmethod
    def _confirmation_step(
        *,
        steps: list[PlanStep],
        title: str,
        tool_name: str,
        arguments: dict[str, Any],
        order_id: str | None,
    ) -> PlanStep:
        return PlanStep(
            step_id=f"s{len(steps) + 1}",
            title=title,
            goal="准备副作用操作并等待用户确认",
            action_type="confirm",
            tool_name=tool_name,

            depends_on=["s1"] if order_id else [],
            requires_confirmation=True,
            arguments=arguments,
        )

    def _registered_tools(self) -> list[ToolSpec]:
        return [
            self._tool_registry.require(name)
            for name in sorted(self._tool_registry.names())
        ]
