from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass(slots=True)
class LLMToolCall:
    """大模型返回的工具调用意图。

    `@dataclass` 会自动生成 `__init__`、`__repr__` 等方法，
    类似 Java 里 Lombok 的 `@Data` 或 Java record 的简化写法。
    `slots=True` 表示固定字段集合，节省内存，也防止随手添加不存在的属性。
    """

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


class LLMClient(Protocol):
    """聊天模型客户端接口。

    `Protocol` 类似 Java `interface`，本项目只要求实现：
    - `chat`：一次性返回完整结果，支持 tools。
    - `stream_chat`：流式返回文本片段，用于前端打字机效果。
    """

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
        """流式聊天调用。

        `AsyncIterator[str]` 表示它不是一次 return 字符串，
        而是像异步版迭代器一样不断 `yield` 字符串片段。
        """
        ...
