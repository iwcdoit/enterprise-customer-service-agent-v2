from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import (
    PendingActionSummaryView,
    RuntimeConfigView,
    TenantStrategyView,
)
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.cost_governance_service import CostGovernanceService


class OpsService:
    """Read-only operational queries for the local ops console and deployment checks."""

    def __init__(self, *, settings: Settings, session: AsyncSession | None = None):
        self._settings = settings
        self._session = session

    def runtime_config(self) -> RuntimeConfigView:
        """Return a safe configuration snapshot without exposing secrets."""

        return RuntimeConfigView(
            app_name=self._settings.app_name,
            runtime_env=self._settings.runtime_env,
            api_prefix=self._settings.api_prefix,
            llm_provider=self._settings.llm_provider,
            embedding_provider=self._settings.embedding_provider,
            vector_store_provider=self._settings.vector_store_provider,
            rag_enabled=self._settings.rag_enabled,
            semantic_cache_enabled=self._settings.semantic_cache_enabled,
            mcp_after_sales_enabled=self._settings.mcp_after_sales_enabled,
            cost_governance_enabled=self._settings.cost_governance_enabled,
            search_enabled=bool(self._settings.serpapi_key),
            warnings=self._runtime_warnings(),
        )

    async def tenant_strategy(
        self,
        *,
        tenant_id: str,
        used_tokens: int | None = None,
    ) -> TenantStrategyView:
        """Preview the model/RAG/history strategy selected for one tenant."""

        cost_service = CostGovernanceService(
            settings=self._settings,
            session=self._session,
        )
        strategy = await cost_service.choose_strategy(tenant_id=tenant_id, used_tokens=used_tokens)
        return TenantStrategyView(
            tenant_id=tenant_id,
            strategy=strategy,
            notes=cost_service.explain_strategy(strategy),
        )

    async def pending_action_summary(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> PendingActionSummaryView:
        """Summarize pending confirmations, including expired records."""

        if self._session is None:
            return PendingActionSummaryView(
                tenant_id=tenant_id,
                user_id=user_id,
                total_pending=0,
                active_pending=0,
                expired_pending=0,
                actions=[],
            )
        actions = await ConfirmationService(
            self._session,
            settings=self._settings,
        ).list_pending_actions(
            tenant_id=tenant_id,
            user_id=user_id,
            include_expired=True,
        )
        expired_count = sum(1 for item in actions if item.expired)
        return PendingActionSummaryView(
            tenant_id=tenant_id,
            user_id=user_id,
            total_pending=len(actions),
            active_pending=len(actions) - expired_count,
            expired_pending=expired_count,
            actions=actions,
        )

    def _runtime_warnings(self) -> list[str]:
        """Collect deployment warnings that do not contain secret values."""

        warnings: list[str] = []
        if not self._settings.llm_api_key:
            warnings.append("LLM_API_KEY is not configured")
        if not self._settings.llm_base_url:
            warnings.append("LLM_BASE_URL is not configured")
        if not self._settings.database_url:
            warnings.append("DATABASE_URL is not configured")
        if self._settings.rag_enabled and not self._settings.embedding_model:
            warnings.append("EMBEDDING_MODEL is not configured while RAG is enabled")
        if self._settings.vector_store_provider == "qdrant" and not self._settings.qdrant_url:
            warnings.append("QDRANT_URL is not configured while Qdrant is selected")
        if self._settings.vector_store_provider == "milvus" and not self._settings.milvus_uri:
            warnings.append("MILVUS_URI is not configured while Milvus is selected")
        if self._settings.semantic_cache_enabled and not self._settings.redis_url:
            warnings.append("REDIS_URL is not configured while semantic cache is enabled")
        if self._settings.mcp_after_sales_enabled:
            if not self._settings.mcp_after_sales_url:
                warnings.append("MCP_AFTER_SALES_URL is not configured while MCP is enabled")
            if not self._settings.mcp_approval_signing_secret:
                warnings.append("MCP_APPROVAL_SIGNING_SECRET is not configured while MCP is enabled")
        return warnings
