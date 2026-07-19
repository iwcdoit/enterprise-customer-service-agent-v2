from __future__ import annotations

import json

from customer_service_app.core.config import Settings
from customer_service_app.domain.memory import ShortTermContext
from customer_service_app.domain.schemas import ChatMessage
from customer_service_app.infrastructure.llm.base import LLMResponse, LLMToolCall
from customer_service_app.services.question_rewrite_service import QuestionRewriteService


class RewriteLLM:
    def __init__(self, *, standalone_question: str):
        self._standalone_question = standalone_question

    async def chat(self, messages, **kwargs):
        payload = {
            "standalone_question": self._standalone_question,
            "dense_query": self._standalone_question,
            "sparse_query": "202607180001 退货 签收 政策",
            "intent": "return_policy",
            "entities": {"order_id": "202607180001"},
            "needs_clarification": False,
            "clarification_question": None,
            "confidence": 0.92,
            "reason": "resolved_from_recent_message",
        }
        return LLMResponse(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="rewrite-1",
                    name="submit_query_rewrite",
                    arguments=json.dumps(payload, ensure_ascii=False),
                )
            ],
            model="rewrite-model",
        )


async def test_vague_question_without_context_requests_clarification() -> None:
    service = QuestionRewriteService(
        settings=Settings(),
        llm_client=RewriteLLM(standalone_question="unused"),
    )

    result, response = await service.resolve(
        question="那个还能退吗？",
        context=ShortTermContext(),
        model="rewrite-model",
    )

    assert response is None
    assert result.needs_clarification is True
    assert result.clarification_question


async def test_rewrite_uses_trusted_context_and_preserves_business_id() -> None:
    standalone = "订单 202607180001 已签收，是否符合退货政策？"
    service = QuestionRewriteService(
        settings=Settings(),
        llm_client=RewriteLLM(standalone_question=standalone),
    )
    context = ShortTermContext(
        recent_messages=[
            ChatMessage(role="user", content="我的订单是 202607180001"),
            ChatMessage(role="assistant", content="订单显示已签收"),
        ]
    )

    result, response = await service.resolve(
        question="那个还能退吗？",
        context=context,
        model="rewrite-model",
    )

    assert response is not None
    assert result.source == "llm"
    assert result.standalone_question == standalone


async def test_rewrite_rejects_invented_business_id() -> None:
    service = QuestionRewriteService(
        settings=Settings(),
        llm_client=RewriteLLM(
            standalone_question="订单 999999999999 是否可以退货？"
        ),
    )
    context = ShortTermContext(
        recent_messages=[ChatMessage(role="user", content="订单 202607180001")]
    )

    result, _ = await service.resolve(
        question="那个还能退吗？",
        context=context,
        model="rewrite-model",
    )

    assert result.source == "fallback"
    assert "999999999999" not in result.standalone_question
