from __future__ import annotations

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.services.cost_governance_service import CostGovernanceService


@pytest.mark.asyncio
async def test_basic_tenant_strategy_tracks_budget_state() -> None:
    service = CostGovernanceService(
        settings=Settings(
            cost_governance_enabled=True,
            tenant_tier_map='{"tenant-basic":"basic"}',
            llm_model="fallback-model",
            llm_model_basic="basic-model",
            basic_daily_token_budget=1_000,
            basic_tenant_rag_top_k=2,
            cost_warning_ratio=0.8,
        )
    )

    strategy = await service.choose_strategy(tenant_id="tenant-basic", used_tokens=250)

    assert strategy.tier == "basic"
    assert strategy.model == "basic-model"
    assert strategy.rag_top_k == 2
    assert strategy.history_turns == 4
    assert strategy.cache_first is True
    assert strategy.remaining_tokens == 750
    assert strategy.usage_ratio == 0.25
    assert strategy.usage_percent == 25.0
    assert strategy.budget_warning is False
    assert strategy.budget_exceeded is False
    assert strategy.degraded is False


@pytest.mark.asyncio
async def test_standard_tenant_soft_degrades_after_warning_ratio() -> None:
    service = CostGovernanceService(
        settings=Settings(
            cost_governance_enabled=True,
            tenant_default_tier="standard",
            llm_model="fallback-model",
            llm_model_basic="basic-model",
            llm_model_standard="standard-model",
            standard_daily_token_budget=100,
            standard_tenant_rag_top_k=5,
            basic_tenant_rag_top_k=2,
            cost_warning_ratio=0.75,
        )
    )

    strategy = await service.choose_strategy(tenant_id="tenant-standard", used_tokens=80)

    assert strategy.tier == "standard"
    assert strategy.model == "basic-model"
    assert strategy.rag_top_k == 2
    assert strategy.history_turns == 4
    assert strategy.remaining_tokens == 20
    assert strategy.budget_warning is True
    assert strategy.budget_exceeded is False
    assert strategy.degraded is True
    assert strategy.degradation_reason == "daily_budget_warning"
    assert strategy.cache_first is True
    assert strategy.use_rerank is False


@pytest.mark.asyncio
async def test_premium_tenant_hard_degrades_after_budget_exceeded() -> None:
    service = CostGovernanceService(
        settings=Settings(
            cost_governance_enabled=True,
            tenant_tier_map='{"tenant-premium":"premium"}',
            llm_model="fallback-model",
            llm_model_premium="premium-model",
            llm_model_degraded="cheap-model",
            premium_daily_token_budget=100,
            premium_tenant_rag_top_k=8,
            degraded_rag_top_k=1,
            degraded_history_turns=2,
            cost_warning_ratio=0.8,
        )
    )

    strategy = await service.choose_strategy(tenant_id="tenant-premium", used_tokens=150)

    assert strategy.tier == "premium"
    assert strategy.model == "cheap-model"
    assert strategy.rag_top_k == 1
    assert strategy.history_turns == 2
    assert strategy.remaining_tokens == 0
    assert strategy.usage_ratio == 1.5
    assert strategy.usage_percent == 150.0
    assert strategy.budget_warning is True
    assert strategy.budget_exceeded is True
    assert strategy.degraded is True
    assert strategy.degradation_reason == "daily_budget_exceeded"
    assert strategy.cache_first is True
    assert strategy.use_rerank is False


@pytest.mark.asyncio
async def test_invalid_tier_and_negative_usage_fall_back_safely() -> None:
    service = CostGovernanceService(
        settings=Settings(
            cost_governance_enabled=True,
            tenant_tier_map='{"tenant-bad":"unknown"}',
            llm_model="fallback-model",
            llm_model_standard="standard-model",
            standard_daily_token_budget=500,
        )
    )

    strategy = await service.choose_strategy(tenant_id="tenant-bad", used_tokens=-20)

    assert strategy.tier == "standard"
    assert strategy.model == "standard-model"
    assert strategy.used_tokens == 0
    assert strategy.remaining_tokens == 500
    assert strategy.usage_ratio == 0.0
    assert strategy.budget_warning is False
    assert strategy.degraded is False


@pytest.mark.asyncio
async def test_disabled_governance_keeps_strategy_but_still_exposes_budget_state() -> None:
    service = CostGovernanceService(
        settings=Settings(
            cost_governance_enabled=False,
            llm_model="fallback-model",
            llm_model_standard="standard-model",
            standard_daily_token_budget=100,
            cost_warning_ratio=0.8,
        )
    )

    strategy = await service.choose_strategy(tenant_id="tenant-standard", used_tokens=100)

    assert strategy.model == "standard-model"
    assert strategy.budget_warning is True
    assert strategy.budget_exceeded is True
    assert strategy.degraded is False


@pytest.mark.asyncio
async def test_explain_strategy_returns_operator_readable_notes() -> None:
    service = CostGovernanceService(
        settings=Settings(
            cost_governance_enabled=True,
            llm_model="fallback-model",
            llm_model_degraded="cheap-model",
            standard_daily_token_budget=100,
        )
    )
    strategy = await service.choose_strategy(tenant_id="tenant-standard", used_tokens=100)

    notes = service.explain_strategy(strategy)

    assert any("token 用量" in item for item in notes)
    assert any("强降级" in item for item in notes)
    assert any("daily_budget_exceeded" in item for item in notes)
