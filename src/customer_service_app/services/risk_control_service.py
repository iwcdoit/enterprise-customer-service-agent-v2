from __future__ import annotations

from typing import Any


class RiskControlService:
    """Small rule engine for high-risk after-sales actions."""

    _medium_risk_tools = {
        "transfer_to_human",
        "create_exchange_case",
        "create_refund_ticket",
    }
    _write_tools = {
        "create_refund_case",
        "create_compensation_case",
        "create_exchange_case",
        "create_refund_ticket",
        "transfer_to_human",
    }
    _high_risk_words = (
        "投诉",
        "起诉",
        "法律",
        "律师",
        "曝光",
        "媒体",
        "监管",
        "消协",
        "12315",
        "报警",
        "差评",
    )

    def evaluate_tool_risk(self, *, tool_name: str, arguments: dict[str, Any]) -> str:
        """Return low, medium, or high risk for a planned tool call."""
        if tool_name not in self._write_tools:
            return "low"

        risk_text = self._join_argument_text(arguments)

        if any(word in risk_text for word in self._high_risk_words):
            return "high"

        if tool_name in {"create_refund_case", "create_compensation_case"}:
            return "high"

        if tool_name in self._medium_risk_tools:
            return "medium"

        return "medium"

    @staticmethod
    def _join_argument_text(arguments: dict[str, Any]) -> str:
        """Extract human-readable text fields from tool arguments."""
        text_fields = ("reason", "detail", "title", "refund_type", "compensation_type")
        parts: list[str] = []
        for field in text_fields:
            value = arguments.get(field)
            if value is not None:
                parts.append(str(value))
        return " ".join(parts)
