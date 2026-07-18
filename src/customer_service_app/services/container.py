from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.embeddings.factory import build_embedding_client
from customer_service_app.infrastructure.llm.openai_compatible import OpenAICompatibleLLMClient
from customer_service_app.infrastructure.mcp.client import build_after_sales_mcp_client
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.infrastructure.vector_store.factory import build_vector_store
from customer_service_app.services.business_gateway import BusinessGateway
from customer_service_app.services.cost_governance_service import CostGovernanceService
from customer_service_app.services.customer_service_agent import CustomerServiceAgent
from customer_service_app.services.human_support_service import HumanSupportService
from customer_service_app.services.rag_service import RagService
from customer_service_app.tools.default_registry import build_default_tool_registry
from customer_service_app.workflows.runtime import CustomerServiceGraphRuntime


class ApplicationContainer:
    """保存应用级 Graph Runtime，并为每个请求组装事务级 Agent。"""

    def __init__(self, *, settings: Settings, graph_runtime: CustomerServiceGraphRuntime) -> None:
        self.settings = settings
        self.graph_runtime = graph_runtime

    @classmethod
    async def create(cls, settings: Settings) -> "ApplicationContainer":
        """在 FastAPI 启动阶段只编译一次 Graph 和 Checkpointer。"""

        graph_runtime = await CustomerServiceGraphRuntime.create(settings)
        return cls(settings=settings, graph_runtime=graph_runtime)

    def build_agent(self, session: AsyncSession) -> CustomerServiceAgent:
        """围绕当前请求的数据库 Session 组装业务依赖。"""

        embedding_client = build_embedding_client(self.settings)
        vector_store = build_vector_store(self.settings)
        rag_service = RagService(
            settings=self.settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        semantic_cache = (
            RedisSemanticCache(self.settings, embedding_client)
            if self.settings.semantic_cache_enabled
            else None
        )
        human_support_service = HumanSupportService(session)
        business_gateway = BusinessGateway(
            settings=self.settings,
            session=session,
            mcp_client=build_after_sales_mcp_client(self.settings),
            human_support_service=human_support_service,
        )
        return CustomerServiceAgent(
            settings=self.settings,
            session=session,
            graph_runtime=self.graph_runtime,
            llm_client=OpenAICompatibleLLMClient(self.settings),
            rag_service=rag_service,
            tool_registry=build_default_tool_registry(),
            search_client=SerpApiSearchClient(self.settings),
            business_gateway=business_gateway,
            semantic_cache=semantic_cache,
            cost_service=CostGovernanceService(settings=self.settings, session=session),
            human_support_service=human_support_service,
        )

    async def close(self) -> None:
        """释放应用级 Checkpointer。请求级资源由请求生命周期关闭。"""

        await self.graph_runtime.close()
