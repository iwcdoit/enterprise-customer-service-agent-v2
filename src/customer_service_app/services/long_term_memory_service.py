from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.domain.memory import MemoryWriteCommand
from customer_service_app.infrastructure.db.repositories import MemoryRepository


class LongTermMemoryService:
    """Write only attributable, verified and policy-allowed long-term memories."""

    _sensitive_keys = {
        "password",
        "token",
        "secret",
        "id_card",
        "bank_card",
        "payment_account",
        "身份证",
        "银行卡",
        "密码",
    }
    _allowed_sources = {
        "explicit_user": {"user"},
        "verified_tool": {"customer_service_graph", "mcp", "business_gateway"},
        "business_system": {"crm", "oms", "member_center"},
        "human_confirmed": {"human_support"},
        "risk_engine": {"risk_engine"},
    }

    def __init__(self, session: AsyncSession):
        self._repo = MemoryRepository(session)

    async def remember(self, command: MemoryWriteCommand) -> bool:
        """Validate provenance before persisting; model inference alone is never sufficient."""
        if not self._is_allowed(command):
            return False
        expires_at = datetime.fromisoformat(command.expires_at) if command.expires_at else None
        await self._repo.upsert_memory(
            tenant_id=command.tenant_id,
            user_id=command.user_id,
            memory_type=command.memory_type,
            memory_key=command.memory_key,
            memory_value=command.memory_value,
            confidence=command.confidence,
            source=command.source,
            verification_status=command.verification_status,
            evidence_ids=command.evidence_ids,
            sensitivity=command.sensitivity,
            expires_at=expires_at,
        )
        return True

    def _is_allowed(self, command: MemoryWriteCommand) -> bool:
        if not command.memory_key or not command.memory_value or command.confidence < 0.8:
            return False
        allowed_sources = self._allowed_sources.get(command.verification_status, set())
        if command.source not in allowed_sources:
            return False
        if command.memory_type == "risk" and command.verification_status not in {
            "risk_engine",
            "human_confirmed",
        }:
            return False
        if self._contains_sensitive_key(command.memory_value):
            return False
        return True

    def _contains_sensitive_key(self, value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in self._sensitive_keys or self._contains_sensitive_key(item):
                    return True
        elif isinstance(value, list):
            return any(self._contains_sensitive_key(item) for item in value)
        elif isinstance(value, str):
            compact = re.sub(r"[\s-]", "", value)
            # 拒绝自动保存疑似中国身份证号、银行卡号及明文凭据表达。
            if re.search(r"(?<!\d)\d{17}[\dXx](?!\d)", compact):
                return True
            if re.search(r"(?<!\d)\d{16,19}(?!\d)", compact):
                return True
            if any(word in value.lower() for word in ("password=", "token=", "密码是", "验证码是")):
                return True
        return False
