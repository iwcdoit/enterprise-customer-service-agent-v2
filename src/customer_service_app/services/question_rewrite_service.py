from __future__ import annotations

import json
import re
from typing import Any

from customer_service_app.core.config import Settings
from customer_service_app.domain.memory import ShortTermContext
from customer_service_app.domain.query_rewrite import QueryRewriteResult
from customer_service_app.infrastructure.llm.base import LLMClient, LLMResponse


class QuestionRewriteService:
    """将依赖上下文的用户追问改写为可独立检索的问题。"""

    _vague_words = ("这个", "那个", "它", "刚才", "上面", "还能", "怎么弄", "那笔")
    _business_id_pattern = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]{2,8}-?)?\d{8,24}(?![A-Za-z0-9])")

    def __init__(self, *, settings: Settings, llm_client: LLMClient):
        self._settings = settings
        self._llm_client = llm_client

    def should_rewrite(self, question: str) -> bool:
        """使用零 token 门禁判断是否值得调用改写模型。"""
        compact = question.strip()
        if not compact:
            return False
        return len(compact) <= 20 or any(word in compact for word in self._vague_words)

    async def resolve(
        self,
        *,
        question: str,
        context: ShortTermContext,
        model: str,
        retrieval_feedback: dict[str, Any] | None = None,
    ) -> tuple[QueryRewriteResult, LLMResponse | None]:
        """生成结构化改写；模型失败时使用确定性规则降级。"""
        original = question.strip()
        needs_rewrite = self.should_rewrite(original) or retrieval_feedback is not None
        if not self._settings.semantic_rewrite_enabled or not needs_rewrite:
            return self._original_result(original), None

        context_payload = self._context_payload(context)
        if not self._has_trusted_context(context) and self._contains_vague_reference(original):
            return (
                QueryRewriteResult(
                    original_question=original,
                    standalone_question=original,
                    dense_query=original,
                    sparse_query=original,
                    needs_clarification=True,
                    clarification_question="请说明您指的订单、商品或刚才的哪个处理步骤。",
                    confidence=0.0,
                    source="rule",
                    reason="vague_reference_without_context",
                ),
                None,
            )

        tool = self._rewrite_tool_schema()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是客服检索问题改写器，只调用 submit_query_rewrite。"
                    "不得创造上下文中没有的订单号、日期、金额和业务事实。"
                    "standalone_question 用于规划和回答，dense_query 面向向量语义检索，"
                    "sparse_query 保留产品名、规则词和业务 ID 供 BM25 检索。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": original,
                        "trusted_context": context_payload,
                        "retrieval_feedback": retrieval_feedback or {},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = await self._llm_client.chat(
                messages,
                model=model,
                temperature=0,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "submit_query_rewrite"}},
            )
            result = self._parse_response(response=response, original=original)
            self._validate_no_invented_business_ids(
                result=result,
                original=original,
                context_payload=context_payload,
            )
            if result.confidence < self._settings.semantic_rewrite_min_confidence:
                return self._rule_fallback(original, context), response
            return result, response
        except Exception:
            return self._rule_fallback(original, context), None

    def _rule_fallback(
        self, question: str, context: ShortTermContext
    ) -> QueryRewriteResult:
        hints: list[str] = []
        if context.summary and context.summary.strip():
            hints.append(context.summary.strip())
        for message in context.recent_messages[-2:]:
            if message.content.strip():
                hints.append(message.content.strip())
        if not hints:
            return self._original_result(question, source="fallback")
        standalone = f"上下文：{' 。'.join(hints[-2:])}。当前问题：{question}"
        return QueryRewriteResult(
            original_question=question,
            standalone_question=standalone,
            dense_query=standalone,
            sparse_query=question,
            confidence=0.55,
            source="fallback",
            reason="llm_rewrite_unavailable",
        )

    def _parse_response(self, *, response: LLMResponse, original: str) -> QueryRewriteResult:
        payload: dict[str, Any] | None = None
        for call in response.tool_calls:
            if call.name == "submit_query_rewrite":
                value = json.loads(call.arguments or "{}")
                if isinstance(value, dict):
                    payload = value
                    break
        if payload is None:
            value = json.loads(response.content or "{}")
            if not isinstance(value, dict):
                raise ValueError("query rewrite response must be an object")
            payload = value
        return QueryRewriteResult(original_question=original, source="llm", **payload)

    def _validate_no_invented_business_ids(
        self,
        *,
        result: QueryRewriteResult,
        original: str,
        context_payload: dict[str, Any],
    ) -> None:
        trusted_text = original + json.dumps(context_payload, ensure_ascii=False)
        trusted_ids = set(self._business_id_pattern.findall(trusted_text))
        generated_text = " ".join(
            (result.standalone_question, result.dense_query, result.sparse_query)
        )
        generated_ids = set(self._business_id_pattern.findall(generated_text))
        if not generated_ids.issubset(trusted_ids):
            raise ValueError("query rewrite invented an untrusted business id")

    @staticmethod
    def _context_payload(context: ShortTermContext) -> dict[str, Any]:
        return {
            "summary": context.summary,
            "recent_messages": [
                {"role": item.role, "content": item.content}
                for item in context.recent_messages[-4:]
            ],
            "verified_memories": [item.model_dump(mode="json") for item in context.memories[:5]],
            "pending_actions": context.pending_actions[:3],
        }

    def _contains_vague_reference(self, question: str) -> bool:
        return any(word in question for word in self._vague_words)

    @staticmethod
    def _has_trusted_context(context: ShortTermContext) -> bool:
        return bool(
            (context.summary and context.summary.strip())
            or context.recent_messages
            or context.memories
            or context.pending_actions
        )

    @staticmethod
    def _original_result(
        question: str, *, source: str = "original"
    ) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_question=question,
            standalone_question=question,
            dense_query=question,
            sparse_query=question,
            source=source,  # type: ignore[arg-type]
            reason="rewrite_not_required",
        )

    @staticmethod
    def _rewrite_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit_query_rewrite",
                "description": "提交受控的检索问题改写结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "standalone_question": {"type": "string"},
                        "dense_query": {"type": "string"},
                        "sparse_query": {"type": "string"},
                        "intent": {"type": "string"},
                        "entities": {"type": "object"},
                        "needs_clarification": {"type": "boolean"},
                        "clarification_question": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "standalone_question",
                        "dense_query",
                        "sparse_query",
                        "intent",
                        "entities",
                        "needs_clarification",
                        "confidence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        }
