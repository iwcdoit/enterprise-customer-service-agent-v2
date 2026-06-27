from __future__ import annotations

from typing import Any

from jose import JWTError, jwt

from after_sales_mcp.config import get_settings


def verify_approval_token(
    *,
    token: str,
    tenant_id: str,
    user_id: str,
    tool_name: str,
) -> dict[str, Any]:
    """Verify that one exact HIL-approved action may execute."""

    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.approval_signing_secret,
            algorithms=["HS256"],
            issuer=settings.approval_issuer,
        )
    except JWTError as exc:
        raise ValueError("Invalid or expired approval token") from exc
    expected = {
        "tenant_id": tenant_id,
        "sub": user_id,
        "tool_name": tool_name,
    }
    for key, value in expected.items():
        if claims.get(key) != value:
            raise ValueError(f"Approval token does not match {key}")
    if not claims.get("confirmation_id"):
        raise ValueError("Approval token does not contain confirmation_id")
    return claims
