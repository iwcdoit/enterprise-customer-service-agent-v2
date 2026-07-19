from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class MCPToolContract(BaseModel):
    """Business tool metadata that can be imported from an MCP server."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True
    requires_confirmation: bool = False
    risk_level: str = "low"
    action_type: str = "tool"


class MCPBusinessClient(Protocol):
    """Protocol for a future MCP client implementation."""

    async def list_tools(self) -> list[MCPToolContract]:
        """List tools exposed by a business MCP server."""
        ...

    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool by name."""
        ...
