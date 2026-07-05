from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient

if TYPE_CHECKING:
    from customer_service_app.services.business_gateway import BusinessGateway


@dataclass(slots=True)
class ToolExecutionContext:
    """工具执行时需要的后端上下文。

    模型只会给出工具名和参数，但真正执行工具时还需要当前用户、租户、
    会话、数据库 session、搜索客户端等后端对象。
    """

    tenant_id: str
    user_id: str
    conversation_id: str | None
    session: AsyncSession
    search_client: SerpApiSearchClient
    business_gateway: BusinessGateway | None = None
    confirmed_tools: set[str] = field(default_factory=set)


ToolHandler = Callable[[dict[str, Any], ToolExecutionContext], Awaitable[dict[str, Any]]]
"""Tool handler signature: async function receiving arguments and execution context."""


@dataclass(slots=True)
class ToolSpec:
    """一个工具的完整定义。

    name/description/parameters 是给模型看的结构化说明；
    handler 是后端真正执行的 Python 函数。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    requires_confirmation: bool = False

    def as_openai_tool(self) -> dict[str, Any]:
        """把内部 ToolSpec 转成 OpenAI-compatible tools 格式。

        这个返回值会作为 chat.completions.create(..., tools=[...]) 的一部分传给模型。
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Function Calling registry.

    `definitions()` is passed to the LLM API as the structured `tools` parameter.
    """

    def __init__(self) -> None:
        """初始化一个空工具表。"""

        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """注册一个工具。

        只有注册进来的工具，后续才会通过 definitions() 暴露给大模型。
        """

        if tool.name in self._tools:
            raise AppError(f"Tool already registered: {tool.name}", code="duplicate_tool")
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        """导出所有工具定义，供 LLM 首轮决策使用。"""

        return [tool.as_openai_tool() for tool in self._tools.values()]

    def names(self) -> list[str]:
        """Return registered tool names."""

        return list(self._tools.keys())

    def require(self, name: str) -> ToolSpec:
        """Return a registered tool or raise a business error."""

        tool = self._tools.get(name)
        if tool is None:
            raise AppError(f"Unknown tool: {name}", code="unknown_tool", status_code=400)
        return tool

    async def execute(
        self,
        *,
        name: str,
        arguments_json: str,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """执行模型指定的某个工具。

        模型返回 tool_calls 后，后端会根据 name 找到 ToolSpec，
        解析 JSON 参数，然后调用对应 handler。
        """

        tool = self._tools.get(name)
        if tool is None:
            raise AppError(f"Unknown tool: {name}", code="unknown_tool", status_code=400)
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise AppError(
                f"Invalid tool arguments for {name}: {arguments_json}",
                code="invalid_tool_arguments",
                status_code=400,
            ) from exc
        if not isinstance(arguments, dict):
            raise AppError(
                "Tool arguments must be a JSON object",
                code="invalid_tool_arguments",
                status_code=400,
            )

        if tool.requires_confirmation:
            confirmed_tools = context.confirmed_tools if context is not None else set()
            if name not in confirmed_tools:
                return {
                    "requires_confirmation": True,
                    "tool_name": name,
                    "arguments": arguments,
                    "message": "该操作需要用户或人工客服确认后再执行。",
                }

        return await tool.handler(arguments, context)
