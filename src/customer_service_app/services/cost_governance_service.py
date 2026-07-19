from __future__ import annotations

import json
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.domain.cost import CostStrategy, TenantTier, TokenUsage
from customer_service_app.infrastructure.db.repositories import UsageRepository


class CostGovernanceService:
    """Tenant-tier and usage-aware cost governance.

    The service does not reject normal paid traffic when a budget is exceeded. Instead it degrades
    model choice, RAG breadth, and memory window according to tenant tier and daily token usage.
    """

    def __init__(self, *, settings: Settings, session: AsyncSession | None = None):
        self._settings = settings
        self._repo = UsageRepository(session) if session is not None else None

    async def choose_strategy(
        self,
        *,
        tenant_id: str,
        used_tokens: int | None = None,
    ) -> CostStrategy:
        """Choose a tenant-aware runtime strategy."""
        tier = self._resolve_tenant_tier(tenant_id)

        resolved_used_tokens = max(int(used_tokens or 0), 0)

        budget = self._budget_for_tier(tier)

        if used_tokens is None and self._settings.cost_governance_enabled and self._repo is not None:
            usage = await self._repo.get_today_usage(tenant_id=tenant_id)
            if usage is not None:
                resolved_used_tokens = usage.total_tokens

        usage_ratio = self._usage_ratio(
            used_tokens=resolved_used_tokens,
            budget_tokens=budget,
        )

        base = self._base_strategy(
            tenant_id=tenant_id,
            tier=tier,
            budget_tokens=budget,
            used_tokens=resolved_used_tokens,
            usage_ratio=usage_ratio,
        )

        if not self._settings.cost_governance_enabled:
            return base
        if usage_ratio >= 1.0:
            return self._hard_degrade(base, reason="daily_budget_exceeded")
        if usage_ratio >= self._settings.cost_warning_ratio:
            return self._soft_degrade(base, reason="daily_budget_warning")
        return base

    def explain_strategy(self, strategy: CostStrategy) -> list[str]:
        """生成给运营人员阅读的策略说明，不暴露密钥等配置。"""
        notes = [
            (
                f"租户等级 {strategy.tier}，模型 {strategy.model}，"
                f"RAG top_k={strategy.rag_top_k}，历史窗口={strategy.history_turns}"
            ),
            (
                f"今日 token 用量 {strategy.used_tokens}/{strategy.budget_tokens} "
                f"({strategy.usage_percent}%)，剩余 {strategy.remaining_tokens}"
            ),
        ]
        if strategy.budget_exceeded:
            notes.append("已超过日预算，启用强降级但不拒绝服务")
        elif strategy.budget_warning:
            notes.append("已接近日预算，优先使用低成本策略")
        if strategy.degraded:
            notes.append(f"降级原因：{strategy.degradation_reason}")
        if strategy.cache_first:
            notes.append("当前策略优先查询语义缓存")
        if strategy.use_rerank:
            notes.append("当前策略允许使用 rerank")
        return notes


    async def record_llm_usage(self, *, tenant_id: str, usage: TokenUsage) -> None:
        """Persist token usage for daily cost accounting."""
        if not self._settings.cost_governance_enabled or self._repo is None:
            return
        await self._repo.add_llm_usage(
            tenant_id=tenant_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


    def _resolve_tenant_tier(self, tenant_id: str) -> TenantTier:
        """Resolve tenant tier from env mapping, falling back to default tier."""
        tenant_map = self._parse_tier_map()
        tenant_level = tenant_map.get(tenant_id)
        if tenant_level is None:
            tenant_level = self._settings.tenant_default_tier
        return self._normalize_tier(tenant_level)

    def _parse_tier_map(self) -> dict[str, str]:
        """Parse TENANT_TIER_MAP JSON, ignoring invalid configuration safely."""
        raw = self._settings.tenant_tier_map
        if not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    def _base_strategy(
        self,
        *,
        tenant_id: str,
        tier: TenantTier,
        budget_tokens: int,
        used_tokens: int,
        usage_ratio: float,
    ) -> CostStrategy:
        """Build normal strategy for a tenant tier."""
        budget_state = self._budget_state(
            budget_tokens=budget_tokens,
            used_tokens=used_tokens,
            usage_ratio=usage_ratio,
        )
        if tier == "basic":
            return CostStrategy(
                tenant_id=tenant_id,
                tier=tier,
                model=self._model_for("basic"),
                rag_top_k=self._settings.basic_tenant_rag_top_k,
                history_turns=max(self._settings.degraded_history_turns,4),
                use_rerank=False,
                cache_first=True,
                budget_tokens=budget_tokens,
                used_tokens=used_tokens,
                usage_ratio=usage_ratio,
                **budget_state,
            )

        if tier == "premium":
            return CostStrategy(
                tenant_id=tenant_id,
                tier=tier,
                model=self._model_for("premium"),
                rag_top_k=self._settings.premium_tenant_rag_top_k,
                history_turns=max(self._settings.short_term_history_turns, 12),
                use_rerank=True,
                cache_first=False,
                budget_tokens=budget_tokens,
                used_tokens=used_tokens,
                usage_ratio=usage_ratio,
                **budget_state,
            )

        return CostStrategy(
            tenant_id=tenant_id,
            tier=tier,
            model=self._model_for("standard"),
            rag_top_k=self._settings.standard_tenant_rag_top_k,
            history_turns=self._settings.short_term_history_turns,
            use_rerank=False,
            cache_first=True,
            budget_tokens=budget_tokens,
            used_tokens=used_tokens,
            usage_ratio=usage_ratio,
            **budget_state,
        )

    def _soft_degrade(self, strategy: CostStrategy, *, reason: str) -> CostStrategy:
        """Degrade one level when tenant approaches daily budget."""
        if strategy.tier == "premium":
            strategy.model = self._model_for("standard")
            strategy.rag_top_k = self._settings.standard_tenant_rag_top_k
            strategy.history_turns = self._settings.short_term_history_turns

        elif strategy.tier == "standard":
            strategy.model = self._model_for("basic")
            strategy.rag_top_k = self._settings.basic_tenant_rag_top_k
            strategy.history_turns = 4

        else:
            strategy.model = self._model_for("degraded")
            strategy.rag_top_k = self._settings.degraded_rag_top_k
            strategy.history_turns = self._settings.degraded_history_turns

        strategy.use_rerank = False
        strategy.cache_first = True
        strategy.degraded = True
        strategy.degradation_reason = reason
        return strategy

    def _hard_degrade(self, strategy: CostStrategy, *, reason: str) -> CostStrategy:
        """Apply strong degradation after budget is exceeded without hard-blocking service."""
        strategy.model = self._model_for("degraded")
        strategy.rag_top_k = self._settings.degraded_rag_top_k
        strategy.history_turns = self._settings.degraded_history_turns
        strategy.use_rerank = False
        strategy.cache_first = True
        strategy.degraded = True
        strategy.degradation_reason = reason
        return strategy

    def _model_for(self, tier: str) -> str:
        """Return configured model for a tier, falling back to LLM_MODEL."""
        model_by_tier = {
            "basic": self._settings.llm_model_basic,
            "standard": self._settings.llm_model_standard,
            "premium": self._settings.llm_model_premium,
            "degraded": self._settings.llm_model_degraded,
        }
        return model_by_tier.get(tier, "") or self._settings.llm_model

    def _budget_for_tier(self, tier: TenantTier) -> int:
        """Return daily token budget for a tenant tier."""
        if tier == "basic":
            return self._settings.basic_daily_token_budget
        if tier == "premium":
            return self._settings.premium_daily_token_budget

        return self._settings.standard_daily_token_budget

    def _budget_state(
        self,
        *,
        budget_tokens: int,
        used_tokens: int,
        usage_ratio: float,
    ) -> dict[str, int | float | bool]:
        return {
            "remaining_tokens": max(budget_tokens - used_tokens, 0),
            "usage_percent": round(usage_ratio * 100, 2),
            "budget_warning": usage_ratio >= self._warning_ratio(),
            "budget_exceeded": usage_ratio >= 1.0,
        }

    @staticmethod
    def _usage_ratio(*, used_tokens: int, budget_tokens: int) -> float:
        if budget_tokens <= 0:
            return 1.0 if used_tokens > 0 else 0.0
        return round(used_tokens / budget_tokens, 4)

    def _warning_ratio(self) -> float:
        return min(max(self._settings.cost_warning_ratio, 0.0), 1.0)


    @staticmethod
    def _normalize_tier(value: str) -> TenantTier:
        """Normalize untrusted config into a supported tier."""
        supported_tiers = {"basic", "standard", "premium"}
        if value not in supported_tiers:
            return cast(TenantTier, "standard")
        return cast(TenantTier, value)
