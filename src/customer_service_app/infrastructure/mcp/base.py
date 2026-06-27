from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class MCPToolContract(BaseModel):
    """Business tool metadata discovered from an MCP server."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True
    requires_confirmation: bool = False
    risk_level: str = "low"
    action_type: str = "tool"


class MCPBusinessClient(Protocol):
    """Minimal protocol used by the application to call business MCP services."""

    async def list_tools(self) -> list[MCPToolContract]:
        """List tools exposed by the MCP service."""
        ...

    async def call_tool(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool by name."""
        ...

