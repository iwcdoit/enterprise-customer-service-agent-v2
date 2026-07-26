from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings, get_settings
from customer_service_app.core.exceptions import AppError, NotFoundError
from customer_service_app.domain.memory import CustomerMemoryView, MemoryWriteCommand
from customer_service_app.infrastructure.db.models import CustomerMemory, MemoryPreference
from customer_service_app.infrastructure.db.repositories import MemoryRepository


class LongTermMemoryService:
    """Write only attributable, verified and policy-allowed long-term memories."""

    _sensitive_keys: ClassVar[set[str]] = {
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
    _allowed_sources: ClassVar[dict[str, set[str]]] = {
        "explicit_user": {"user"},
        "verified_tool": {"customer_service_graph", "mcp", "business_gateway"},
        "business_system": {"crm", "oms", "member_center"},
        "human_confirmed": {"human_support"},
        "risk_engine": {"risk_engine"},
    }

    def __init__(self, session: AsyncSession, settings: Settings | None = None):
        self._repo = MemoryRepository(session)
        self._settings = settings or get_settings()

    async def remember(self, command: MemoryWriteCommand) -> bool:
        """Validate provenance before persisting; model inference alone is never sufficient."""
        if not self._settings.long_term_memory_enabled:
            return False
        if not await self._repo.is_long_term_memory_enabled(
            tenant_id=command.tenant_id,
            user_id=command.user_id,
        ):
            return False
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

    async def list_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 50,
    ) -> list[CustomerMemoryView]:
        """Return active memories so users can inspect what the system retained."""

        memories = await self._repo.list_memories(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=min(max(limit, 1), 200),
        )
        return [self._to_view(memory) for memory in memories]

    async def update_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        memory_value: dict[str, Any],
        expires_at: datetime | None,
    ) -> CustomerMemoryView:
        """Let a user correct one owned memory and mark it as explicitly verified."""

        memory = await self._require_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
        )
        command = MemoryWriteCommand(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=memory.memory_type,
            memory_key=memory.memory_key,
            memory_value=memory_value,
            confidence=1.0,
            source="user",
            verification_status="explicit_user",
            evidence_ids=[],
            sensitivity=memory.sensitivity,
            expires_at=expires_at.isoformat() if expires_at else None,
        )
        if not self._is_allowed(command):
            raise AppError(
                "Memory value is empty or contains sensitive credentials",
                code="invalid_memory",
                status_code=400,
            )
        updated = await self._repo.upsert_memory(
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
        return self._to_view(updated)

    async def delete_memory(self, *, tenant_id: str, user_id: str, memory_id: str) -> None:
        """Delete one owned memory without exposing cross-user existence."""

        memory = await self._require_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
        )
        await self._repo.delete_memory(memory)

    async def get_preference(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> tuple[bool, MemoryPreference | None]:
        """Return explicit preference and the default-enabled effective value."""

        preference = await self._repo.get_preference(tenant_id=tenant_id, user_id=user_id)
        return (
            preference is None or preference.long_term_memory_enabled,
            preference,
        )

    async def set_preference(
        self,
        *,
        tenant_id: str,
        user_id: str,
        enabled: bool,
    ) -> MemoryPreference:
        """Persist explicit user consent for future memory reads and writes."""

        return await self._repo.set_long_term_memory_enabled(
            tenant_id=tenant_id,
            user_id=user_id,
            enabled=enabled,
        )

    async def cleanup_expired(self, *, tenant_id: str, user_id: str | None = None) -> int:
        """Physically remove expired records; intended for API or scheduled jobs."""

        return await self._repo.delete_expired_memories(
            tenant_id=tenant_id,
            user_id=user_id,
        )

    async def _require_owned(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_id: str,
    ) -> CustomerMemory:
        memory = await self._repo.get_owned_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
        )
        if memory is None:
            raise NotFoundError("Memory not found")
        return memory

    @staticmethod
    def _to_view(memory: CustomerMemory) -> CustomerMemoryView:
        return CustomerMemoryView(
            id=memory.id,
            memory_type=memory.memory_type,
            memory_key=memory.memory_key,
            memory_value=memory.memory_value_json,
            confidence=memory.confidence,
            source=memory.source,
            verification_status=memory.verification_status,
            evidence_ids=list(memory.evidence_json or []),
            sensitivity=memory.sensitivity,
            expires_at=memory.expires_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

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
