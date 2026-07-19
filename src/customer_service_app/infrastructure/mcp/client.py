from __future__ import annotations

import json
import logging
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, ReadResourceResult, TextResourceContents, Tool

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.mcp.base import MCPBusinessClient, MCPToolContract


def build_after_sales_mcp_client(settings: Settings) -> MCPBusinessClient | None:
    """Build the configured after-sales MCP client."""

    if not settings.mcp_after_sales_enabled:
        return None
    if settings.mcp_after_sales_transport == "stdio":
        return StdioMCPBusinessClient(
            command=settings.mcp_after_sales_command,
            args=shlex.split(settings.mcp_after_sales_args),
            timeout_seconds=settings.mcp_timeout_seconds,
        )
    return StreamableHttpMCPBusinessClient(
        url=settings.mcp_after_sales_url,
        timeout_seconds=settings.mcp_timeout_seconds,
    )


class NotConfiguredMCPClient(MCPBusinessClient):
    """Fail explicitly when a mandatory MCP capability has not been configured."""

    async def list_tools(self) -> list[MCPToolContract]:
        return []

    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ExternalServiceError(f"MCP client is not configured for tool: {name}")



class StreamableHttpMCPBusinessClient(MCPBusinessClient):
    """MCP client for an independently deployed Streamable HTTP server."""


    def __init__(self, *, url: str, timeout_seconds: int):
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._http_client = httpx.AsyncClient(timeout=timeout_seconds)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise ExternalServiceError(
                "Python package `mcp` is required for MCP integration"
            ) from exc

        async with streamable_http_client(
                self._url,
                http_client=self._http_client,
        ) as (read, write, _):
            async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
            ) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[MCPToolContract]:
        """从 Streamable HTTP MCP 服务发现工具及其治理契约。"""

        async with self._session() as session:
            response: ListToolsResult = await session.list_tools()
            contracts = await _read_contracts(session)
            return _to_contracts(response.tools, contracts)


    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用远程 MCP 业务工具并标准化返回值。"""

        async with self._session() as session:
            tool_call_result: CallToolResult = await session.call_tool(name=name, arguments=arguments,
                                               read_timeout_seconds=timedelta(seconds=self._timeout_seconds), )
            normalized = _normalize_tool_result(tool_call_result)
            return normalized



    async def close(self) -> None:
        """Close the shared HTTP connection pool."""

        await self._http_client.aclose()


class StdioMCPBusinessClient(MCPBusinessClient):
    """Development-only client that starts a local MCP subprocess per operation."""

    def __init__(self, *, command: str, args: list[str], timeout_seconds: int):
        self._command = command
        self._args = args
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ExternalServiceError(
                "Python package `mcp` is required for MCP integration"
            ) from exc

        params = StdioServerParameters(
            command=self._command,
            args=self._args,
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
            ) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[MCPToolContract]:
        """从本地 stdio MCP 子进程发现工具。"""

        async with self._session() as session:
            response = await session.list_tools()
            contracts = await _read_contracts(session)
            return _to_contracts(response.tools, contracts)


    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """通过 stdio 传输调用 MCP 工具。"""

        async with self._session() as session:
            result = await session.call_tool(name, arguments)
            normalized = _normalize_tool_result(result)
            return normalized



async def _read_contracts(session: ClientSession) -> dict[str, dict[str, Any]]:
    try:
        response: ReadResourceResult = await session.read_resource("after-sales://tool-contracts")
    except Exception as exc:
        logging.info("session.read_resource error: %s", exc)
        return {}

    text_parts = [
        item.text
        for item in response.contents
        if isinstance(item, TextResourceContents) and item.text
    ]
    if not text_parts:
        return {}

    try:
        items = json.loads("\n".join(text_parts))
    except json.JSONDecodeError:
        return {}

    return {
        item["name"]: item
        for item in items
        if isinstance(item, dict) and "name" in item
    }




def _to_contracts(tools: list[Tool], contracts_by_name: dict[str, dict[str, Any]]) -> list[MCPToolContract]:
    result: list[MCPToolContract] = []
    for tool in tools:
        contract = contracts_by_name.get(tool.name, {})
        result.append(MCPToolContract(
            name=tool.name,
            description=tool.description or "",
            parameters=getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {},
            read_only=bool(contract.get("read_only", True)),
            requires_confirmation=bool(contract.get("requires_confirmation", False)),
            risk_level=str(contract.get("risk_level", "low")),
            action_type=str(contract.get("action_type", "tool")),
        ))
    return result


def _normalize_tool_result(result: CallToolResult) -> dict[str, Any]:
    structured_content = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if isinstance(structured_content, dict):
        return structured_content

    texts = [
        item.text
        for item in getattr(result, "content", [])
        if getattr(item, "text", None)
    ]
    if not texts:
        return {}

    raw = "\n".join(texts)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}
