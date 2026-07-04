from __future__ import annotations

import json
import shlex
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.infrastructure.mcp.base import MCPBusinessClient, MCPToolContract


def build_after_sales_mcp_client(settings: Settings) -> MCPBusinessClient | None:
    """Create the after-sales MCP client when MCP is enabled."""

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


class StreamableHttpMCPBusinessClient(MCPBusinessClient):
    """MCP client for an independently deployed Streamable HTTP server."""

    def __init__(self, *, url: str, timeout_seconds: int):
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._http_client = httpx.AsyncClient(timeout=timeout_seconds)

    @asynccontextmanager
    async def _session(self):
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise ExternalServiceError("Python package `mcp` is required") from exc

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
        try:
            async with self._session() as session:
                response = await session.list_tools()
                contracts = await _read_contracts(session)
                return _to_contracts(response.tools, contracts)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError("Failed to list MCP tools") from exc

    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._session() as session:
                result = await session.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
                )
                return _normalize_tool_result(result)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(f"Failed to call MCP tool: {name}") from exc

    async def close(self) -> None:
        await self._http_client.aclose()


class StdioMCPBusinessClient(MCPBusinessClient):
    """Local-development client that starts an MCP subprocess per operation."""

    def __init__(self, *, command: str, args: list[str], timeout_seconds: int):
        self._command = command
        self._args = args
        self._timeout_seconds = timeout_seconds

    @asynccontextmanager
    async def _session(self):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ExternalServiceError("Python package `mcp` is required") from exc

        params = StdioServerParameters(command=self._command, args=self._args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
            ) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[MCPToolContract]:
        try:
            async with self._session() as session:
                response = await session.list_tools()
                contracts = await _read_contracts(session)
                return _to_contracts(response.tools, contracts)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError("Failed to list MCP tools") from exc

    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._session() as session:
                result = await session.call_tool(name, arguments)
                return _normalize_tool_result(result)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(f"Failed to call MCP tool: {name}") from exc


async def _read_contracts(session: Any) -> dict[str, dict[str, Any]]:
    try:
        resource = await session.read_resource("after-sales://tool-contracts")
    except Exception:
        return {}
    text_parts = [
        item.text
        for item in getattr(resource, "contents", [])
        if getattr(item, "text", None)
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


def _to_contracts(
    tools: list[Any],
    contracts_by_name: dict[str, dict[str, Any]],
) -> list[MCPToolContract]:
    result: list[MCPToolContract] = []
    for tool in tools:
        contract = contracts_by_name.get(tool.name, {})
        result.append(
            MCPToolContract(
                name=tool.name,
                description=tool.description or "",
                parameters=getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
                or {},
                read_only=bool(contract.get("read_only", True)),
                requires_confirmation=bool(contract.get("requires_confirmation", False)),
                risk_level=str(contract.get("risk_level", "low")),
                action_type=str(contract.get("action_type", "tool")),
            )
        )
    return result


def _normalize_tool_result(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if isinstance(structured, dict):
        return structured
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
    return parsed if isinstance(parsed, dict) else {"result": parsed}
