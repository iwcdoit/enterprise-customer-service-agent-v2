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
    """为一次确定的工具动作生成稳定确认 ID。"""

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


def create_approval_token(
    *,
    settings: Settings,
    confirmation_id: str,
    tenant_id: str,
    user_id: str,
    tool_name: str,
) -> str:
    """Create a short-lived token proving that LangGraph HIL approved one exact action."""

    secret = settings.require(
        "MCP_APPROVAL_SIGNING_SECRET",
        settings.mcp_approval_signing_secret,
    )
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "iss": settings.mcp_approval_issuer,
            "sub": user_id,
            "tenant_id": tenant_id,
            "tool_name": tool_name,
            "confirmation_id": confirmation_id,
            "iat": int(now.timestamp()),
            "exp": int(
                (
                    now
                    + timedelta(seconds=settings.mcp_approval_token_ttl_seconds)
                ).timestamp()
            ),
        },
        secret,
        algorithm="HS256",
    )


def build_approval_token(
    *,
    settings: Settings,
    tool_name: str,
    tenant_id: str,
    user_id: str,
    arguments: dict[str, Any],
    confirmation_id: str | None = None,
) -> str:
    """兼容 Public 调用，并把授权绑定到同一租户、用户、工具和确认动作。"""

    resolved_confirmation_id = confirmation_id or build_confirmation_id(
        tool_name=tool_name,
        tenant_id=tenant_id,
        user_id=user_id,
        arguments=arguments,
    )
    return create_approval_token(
        settings=settings,
        confirmation_id=resolved_confirmation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        tool_name=tool_name,
    )
