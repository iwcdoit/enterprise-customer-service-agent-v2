from __future__ import annotations

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.services.ops_service import OpsService


def test_runtime_config_returns_safe_summary_and_warnings() -> None:
    service = OpsService(
        settings=Settings(
            app_name="Customer Service",
            runtime_env="production",
            llm_api_key="",
            database_url="",
            rag_enabled=True,
            embedding_model="",
            vector_store_provider="qdrant",
            qdrant_url="",
        )
    )

    view = service.runtime_config()

    assert view.app_name == "Customer Service"
    assert view.rag_enabled is True
    assert "LLM_API_KEY is not configured" in view.warnings
    assert "DATABASE_URL is not configured" in view.warnings
    assert not hasattr(view, "llm_api_key")


@pytest.mark.asyncio
async def test_tenant_strategy_preview_supports_degrade_simulation() -> None:
    service = OpsService(
        settings=Settings(
            llm_model="standard-model",
            llm_model_degraded="cheap-model",
            standard_daily_token_budget=100,
            cost_warning_ratio=0.8,
        )
    )

    view = await service.tenant_strategy(tenant_id="tenant-a", used_tokens=100)

    assert view.strategy.degraded is True
    assert view.strategy.model == "cheap-model"
    assert view.strategy.degradation_reason == "daily_budget_exceeded"
    assert view.strategy.remaining_tokens == 0
    assert view.strategy.budget_exceeded is True
    assert view.notes


@pytest.mark.asyncio
async def test_pending_action_summary_without_session_returns_empty_summary() -> None:
    service = OpsService(settings=Settings())

    view = await service.pending_action_summary(tenant_id="tenant-a", user_id="u1")

    assert view.tenant_id == "tenant-a"
    assert view.user_id == "u1"
    assert view.total_pending == 0
    assert view.actions == []
