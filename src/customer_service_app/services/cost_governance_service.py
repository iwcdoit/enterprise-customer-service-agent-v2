from __future__ import annotations

import json
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.domain.cost import CostStrategy, TenantTier, TokenUsage
from customer_service_app.infrastructure.db.repositories import UsageRepository


class CostGovernanceService:
    """Choose tenant-aware runtime strategy without hard-blocking traffic."""

    def __init__(self, *, settings: Settings, session: AsyncSession | None = None):
        self._settings = settings
        self._repo = UsageRepository(session) if session is not None else None

    async def choose_strategy(self, *, tenant_id: str, used_tokens: int | None = None) -> CostStrategy:
        """Choose model, RAG breadth, and memory window for one tenant request."""
        tier = self._resolve_tenant_tier(tenant_id)
        if used_tokens is None:
            used_tokens = 0
            if self._settings.cost_governance_enabled and self._repo is not None:
                usage = await self._repo.get_today_usage(tenant_id=tenant_id)
                used_tokens = usage.total_tokens if usage else 0
        used_tokens = max(int(used_tokens), 0)
        budget = self._budget_for_tier(tier)
        usage_ratio = self._usage_ratio(used_tokens=used_tokens, budget_tokens=budget)
        strategy = self._base_strategy(
            tenant_id=tenant_id,
            tier=tier,
            budget_tokens=budget,
            used_tokens=used_tokens,
            usage_ratio=usage_ratio,
        )

        if not self._settings.cost_governance_enabled:
            return strategy
        if usage_ratio >= 1.0:
            return self._hard_degrade(strategy, reason="daily_budget_exceeded")
        if usage_ratio >= self._warning_ratio():
            return self._soft_degrade(strategy, reason="daily_budget_warning")
        return strategy

    def explain_strategy(self, strategy: CostStrategy) -> list[str]:
        """Return operator-facing notes for the selected cost strategy."""
        notes = [
            (
                f"租户等级 {strategy.tier}，本次使用模型 {strategy.model or 'default'}，"
                f"RAG top_k={strategy.rag_top_k}，历史窗口={strategy.history_turns}"
            ),
            (
                f"今日 token 用量 {strategy.used_tokens}/{strategy.budget_tokens} "
                f"({strategy.usage_percent}%)，剩余额度约 {strategy.remaining_tokens}"
            ),
        ]
        if strategy.budget_exceeded:
            notes.append("当前租户已超过日预算，系统会启用强降级策略但不直接拒绝服务")
        elif strategy.budget_warning:
            notes.append("当前租户已接近日预算，系统会优先选择更低成本的运行策略")
        if strategy.degraded:
            notes.append(f"已触发降级：{strategy.degradation_reason}")
        if strategy.cache_first:
            notes.append("当前策略优先尝试语义缓存，减少重复问题的模型调用")
        if strategy.use_rerank:
            notes.append("当前策略允许使用 rerank 提升知识召回质量")
        return notes

    async def record_llm_usage(self, *, tenant_id: str, usage: TokenUsage) -> None:
        """Persist LLM token usage for daily accounting."""
        if not self._settings.cost_governance_enabled or self._repo is None:
            return
        if usage.total_tokens <= 0:
            return
        await self._repo.add_llm_usage(
            tenant_id=tenant_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _resolve_tenant_tier(self, tenant_id: str) -> TenantTier:
        tier_map = self._parse_tier_map()
        return self._normalize_tier(tier_map.get(tenant_id, self._settings.tenant_default_tier))

    def _parse_tier_map(self) -> dict[str, str]:
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
                history_turns=4,
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
                history_turns=10,
                use_rerank=True,
                budget_tokens=budget_tokens,
                used_tokens=used_tokens,
                usage_ratio=usage_ratio,
                **budget_state,
            )
        return CostStrategy(
            tenant_id=tenant_id,
            tier="standard",
            model=self._model_for("standard"),
            rag_top_k=self._settings.standard_tenant_rag_top_k,
            history_turns=8,
            budget_tokens=budget_tokens,
            used_tokens=used_tokens,
            usage_ratio=usage_ratio,
            **budget_state,
        )

    def _soft_degrade(self, strategy: CostStrategy, *, reason: str) -> CostStrategy:
        if strategy.tier == "premium":
            strategy.model = self._model_for("standard")
            strategy.rag_top_k = self._settings.standard_tenant_rag_top_k
            strategy.history_turns = 8
            strategy.use_rerank = False
        elif strategy.tier == "standard":
            strategy.model = self._model_for("basic")
            strategy.rag_top_k = self._settings.basic_tenant_rag_top_k
            strategy.history_turns = 4
            strategy.cache_first = True
        else:
            strategy.model = self._model_for("degraded")
            strategy.rag_top_k = self._settings.degraded_rag_top_k
            strategy.history_turns = self._settings.degraded_history_turns
        strategy.cache_first = True
        strategy.use_rerank = False
        strategy.degraded = True
        strategy.degradation_reason = reason
        return strategy

    def _hard_degrade(self, strategy: CostStrategy, *, reason: str) -> CostStrategy:
        strategy.model = self._model_for("degraded")
        strategy.rag_top_k = self._settings.degraded_rag_top_k
        strategy.history_turns = self._settings.degraded_history_turns
        strategy.use_rerank = False
        strategy.cache_first = True
        strategy.degraded = True
        strategy.degradation_reason = reason
        return strategy

    def _model_for(self, tier: str) -> str:
        model_by_tier = {
            "basic": self._settings.llm_model_basic,
            "standard": self._settings.llm_model_standard,
            "premium": self._settings.llm_model_premium,
            "degraded": self._settings.llm_model_degraded,
        }
        return model_by_tier.get(tier, "") or self._settings.llm_model

    def _budget_for_tier(self, tier: TenantTier) -> int:
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
        warning_ratio = self._warning_ratio()
        return {
            "remaining_tokens": max(budget_tokens - used_tokens, 0),
            "usage_percent": round(usage_ratio * 100, 2),
            "budget_warning": usage_ratio >= warning_ratio,
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
        if value in {"basic", "standard", "premium"}:
            return cast(TenantTier, value)
        return "standard"
