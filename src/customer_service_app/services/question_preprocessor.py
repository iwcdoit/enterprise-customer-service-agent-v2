from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from customer_service_app.core.exceptions import AppError


QuestionIntent = Literal["order", "refund", "handoff", "policy", "general"]
"""轻量意图标签。

这里不是让后端替代大模型做复杂理解，只是给 trace、缓存策略、运营分析一个低成本信号。
"""

_ORDER_ID_PATTERN = re.compile(r"\b\d{8,}\b")


@dataclass(slots=True)
class QuestionProfile:
    """用户问题预处理后的结构化结果。"""

    original_question: str
    normalized_question: str
    intent: QuestionIntent
    contains_order_id: bool

    def as_trace_metadata(self) -> dict[str, object]:
        """转成 trace metadata，避免在日志里重复输出完整用户问题。"""

        return {
            "intent": self.intent,
            "contains_order_id": self.contains_order_id,
            "original_length": len(self.original_question),
            "normalized_length": len(self.normalized_question),
            "changed": self.original_question != self.normalized_question,
        }


class QuestionPreprocessor:
    """用户问题预处理器。

    它只做确定性、低成本的处理：
    - 合并多余空白，避免缓存和 RAG 因格式差异失效。
    - 识别粗粒度业务意图，方便 trace 和后续扩展。
    - 拒绝全空白问题，让错误在进入 LLM 前暴露。
    """

    def analyze(self, question: str) -> QuestionProfile:
        """分析并标准化用户问题。"""

        normalized = " ".join(question.split())
        if not normalized:
            raise AppError("Question cannot be blank", code="invalid_question", status_code=400)

        return QuestionProfile(
            original_question=question,
            normalized_question=normalized,
            intent=self._detect_intent(normalized),
            contains_order_id=bool(_ORDER_ID_PATTERN.search(normalized)),
        )

    def _detect_intent(self, question: str) -> QuestionIntent:
        """用关键词做轻量意图识别。

        复杂语义仍交给 LLM，这里只覆盖客服系统里最常见的显性表达。
        """

        lowered = question.lower()
        if any(keyword in question for keyword in ("退款", "退货", "售后", "赔偿", "补偿", "保价")):
            return "refund"
        order_keywords = ("订单", "物流", "快递", "运单", "签收")
        has_order_keyword = any(keyword in question for keyword in order_keywords)
        if has_order_keyword or _ORDER_ID_PATTERN.search(question):
            return "order"
        if any(keyword in question for keyword in ("人工", "客服", "投诉", "升级处理")):
            return "handoff"
        if any(keyword in question for keyword in ("政策", "规则", "要求", "条件")) or "policy" in lowered:
            return "policy"
        return "general"
