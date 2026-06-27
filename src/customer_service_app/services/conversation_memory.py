from __future__ import annotations

from dataclasses import dataclass

from customer_service_app.domain.schemas import ChatMessage


@dataclass(slots=True)
class MemoryWindow:
    """整理后的短期记忆窗口。"""

    messages: list[ChatMessage]
    original_count: int
    compressed_count: int

    @property
    def compressed(self) -> bool:
        """是否发生了历史压缩。"""

        return self.compressed_count > 0


class ConversationMemoryCompactor:
    """简单的短期记忆压缩器。

    这里采用确定性压缩策略：
    - 最近几条消息保留原文。
    - 更早的消息压成一条 system 摘要。
    """

    def __init__(
        self,
        *,
        max_history_messages: int = 12,
        keep_recent_messages: int = 8,
        max_summary_chars_per_message: int = 80,
    ) -> None:
        self._max_history_messages = max_history_messages
        self._keep_recent_messages = keep_recent_messages
        self._max_summary_chars_per_message = max_summary_chars_per_message

    def compact(self, history: list[ChatMessage]) -> MemoryWindow:
        """压缩历史消息，返回最终要放进 LLM messages 的窗口。"""

        if len(history) <= self._max_history_messages:
            return MemoryWindow(
                messages=history,
                original_count=len(history),
                compressed_count=0,
            )

        earlier = history[: -self._keep_recent_messages]
        recent = history[-self._keep_recent_messages :]
        summary = ChatMessage(
            role="system",
            content=self._build_summary(earlier),
            metadata={
                "memory_type": "history_summary",
                "source_message_count": len(earlier),
            },
        )
        return MemoryWindow(
            messages=[summary, *recent],
            original_count=len(history),
            compressed_count=len(earlier),
        )

    def _build_summary(self, messages: list[ChatMessage]) -> str:
        """把较早历史整理成简短摘要文本。"""

        lines = ["以下是较早历史对话的简要摘要，用于帮助理解当前上下文："]
        for message in messages:
            role = self._role_label(message.role)
            content = self._shorten(message.content)
            if content:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    def _shorten(self, content: str) -> str:
        """裁剪单条消息，避免摘要本身过长。"""

        normalized = " ".join(content.split())
        limit = self._max_summary_chars_per_message
        return normalized if len(normalized) <= limit else normalized[:limit] + "..."

    @staticmethod
    def _role_label(role: str) -> str:
        """把 role 转成更适合摘要阅读的中文标签。"""

        labels = {
            "user": "用户",
            "assistant": "客服",
            "system": "系统",
            "tool": "工具",
        }
        return labels.get(role, role)
