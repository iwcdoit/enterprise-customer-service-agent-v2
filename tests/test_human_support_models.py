from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_service_app.domain.human_support import HumanResolutionRequest
from customer_service_app.infrastructure.db.models import HumanHandoffSession


def test_handoff_uses_version_for_concurrent_updates() -> None:
    """并发坐席更新同一接管记录时，ORM 应使用 version 做乐观锁。"""

    assert HumanHandoffSession.__mapper__.version_id_col is HumanHandoffSession.version.property.columns[0]


def test_resolution_only_accepts_supported_next_mode() -> None:
    """人工结论只能恢复机器人或关闭会话，不能写入未知状态。"""

    with pytest.raises(ValidationError):
        HumanResolutionRequest(
            tenant_id="tenant-a",
            agent_id="agent-1",
            resolution_code="resolved",
            summary="问题已处理",
            next_mode="unknown",
        )
