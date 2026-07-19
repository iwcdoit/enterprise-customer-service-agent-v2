from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass(slots=True)
class LLMToolCall:
    """大模型返回的工具调用意图。"""

    id: str
    name: str
    arguments: str

    def as_openai_tool_call(self) -> dict[str, Any]:
        """转回 OpenAI tool call 兼容格式，方便二次 LLM 调用时带上工具结果。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(slots=True)
class LLMResponse:
    """一次非流式 LLM 调用的统一返回对象。"""

    content: str
    tool_calls: list[LLMToolCall]
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient(Protocol):
    """聊天模型客户端协议。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """普通聊天调用，可把工具列表绑定给模型。"""
        ...

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """流式返回文本片段。"""
        ...
