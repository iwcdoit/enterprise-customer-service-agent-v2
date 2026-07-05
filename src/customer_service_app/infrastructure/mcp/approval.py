from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt

from customer_service_app.core.config import Settings


def build_confirmation_id(
    *,
    tool_name: str,
    tenant_id: str,
    user_id: str,
    arguments: dict[str, Any],
) -> str:
    """Build a stable id for one exact approved MCP operation."""

    raw = json.dumps(
        {
            "tool_name": tool_name,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_approval_token(
    *,
    settings: Settings,
    tool_name: str,
    tenant_id: str,
    user_id: str,
    arguments: dict[str, Any],
    confirmation_id: str | None = None,
) -> str:
    """Create a signed token that authorizes one high-risk MCP operation.

    The MCP server still receives tenant_id, user_id, and tool arguments as normal
    tool parameters. The signed token binds those parameters to the approval
    decision so a caller cannot reuse the token for another tenant, user, or tool.
    """

    secret = settings.require("MCP_APPROVAL_SIGNING_SECRET", settings.mcp_approval_signing_secret)
    now = datetime.now(timezone.utc)
    resolved_confirmation_id = confirmation_id or build_confirmation_id(
        tool_name=tool_name,
        tenant_id=tenant_id,
        user_id=user_id,
        arguments=arguments,
    )
    claims = {
        "iss": settings.mcp_approval_issuer,
        "sub": user_id,
        "tenant_id": tenant_id,
        "tool_name": tool_name,
        "confirmation_id": resolved_confirmation_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.mcp_approval_token_ttl_seconds)).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm="HS256")
