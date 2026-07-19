from __future__ import annotations

import json
from asyncio import wait_for
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient


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
    business_gateway: Any | None = None
    approval_token: str | None = None
    langgraph_thread_id: str | None = None
    confirmation_id: str | None = None

ToolHandler = Callable[[dict[str, Any], ToolExecutionContext], Awaitable[dict[str, Any]]]
"""Tool handler signature: async function receiving arguments and execution context."""

RiskLevel = Literal["low", "medium", "high"]


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
    read_only: bool = True
    requires_confirmation: bool = False
    risk_level: RiskLevel = "low"
    timeout_seconds: int = 8
    idempotent: bool = True
    action_type: str = "tool"
    parallel_safe: bool = False

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

    def get(self, name: str) -> ToolSpec | None:
        """Return tool metadata by name."""

        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec:
        """Return tool metadata or raise a business error."""

        tool = self.get(name)
        if tool is None:
            raise AppError(f"Unknown tool: {name}", code="unknown_tool", status_code=400)
        return tool

    def names(self) -> set[str]:
        """Return all registered tool names."""

        return set(self._tools)

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

        tool = self.require(name)
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise AppError(f"Invalid tool arguments for {name}: {arguments_json}") from exc
        self._validate_arguments(tool, arguments)
        return await wait_for(tool.handler(arguments, context), timeout=tool.timeout_seconds)

    async def execute_dict(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Execute a tool with already parsed arguments."""

        tool = self.require(name)
        self._validate_arguments(tool, arguments)
        return await wait_for(tool.handler(arguments, context), timeout=tool.timeout_seconds)

    @staticmethod
    def _validate_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> None:
        """Treat model/MCP arguments as untrusted input before executing a handler."""
        try:
            validate_json_schema(instance=arguments, schema=tool.parameters)
        except JsonSchemaValidationError as exc:
            raise AppError(
                f"Invalid arguments for tool {tool.name}: {exc.message}",
                code="tool_arguments_invalid",
                status_code=400,
            ) from exc
