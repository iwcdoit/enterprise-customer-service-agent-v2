from __future__ import annotations

import inspect
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient
from customer_service_app.infrastructure.embeddings.factory import build_embedding_client
from customer_service_app.infrastructure.lexical_search import build_lexical_retriever
from customer_service_app.infrastructure.lexical_search.base import LexicalKnowledgeRetriever
from customer_service_app.infrastructure.llm.openai_compatible import OpenAICompatibleLLMClient
from customer_service_app.infrastructure.mcp.base import MCPBusinessClient
from customer_service_app.infrastructure.mcp.client import build_after_sales_mcp_client
from customer_service_app.infrastructure.rerank import build_reranker
from customer_service_app.infrastructure.rerank.base import KnowledgeReranker
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.infrastructure.vector_store.base import KnowledgeVectorStore
from customer_service_app.infrastructure.vector_store.factory import build_vector_store
from customer_service_app.services.business_gateway import BusinessGateway
from customer_service_app.services.cost_governance_service import CostGovernanceService
from customer_service_app.services.customer_service_agent import CustomerServiceAgent
from customer_service_app.services.human_support_service import HumanSupportService
from customer_service_app.services.rag_service import RagService
from customer_service_app.services.tool_registry import ToolRegistry
from customer_service_app.tools.default_registry import build_default_tool_registry
from customer_service_app.workflows.runtime import CustomerServiceGraphRuntime


class ApplicationContainer:
    """持有应用级网络客户端，并为每个请求组装事务级 Agent。"""

    def __init__(
        self,
        *,
        settings: Settings,
        graph_runtime: CustomerServiceGraphRuntime,
        llm_client: OpenAICompatibleLLMClient,
        embedding_client: EmbeddingClient,
        vector_store: KnowledgeVectorStore,
        lexical_retriever: LexicalKnowledgeRetriever | None,
        reranker: KnowledgeReranker | None,
        rag_service: RagService,
        semantic_cache: RedisSemanticCache | None,
        search_client: SerpApiSearchClient,
        mcp_client: MCPBusinessClient | None,
        tool_registry: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.graph_runtime = graph_runtime
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.lexical_retriever = lexical_retriever
        self.reranker = reranker
        self.rag_service = rag_service
        self.semantic_cache = semantic_cache
        self.search_client = search_client
        self.mcp_client = mcp_client
        self.tool_registry = tool_registry

    @classmethod
    async def create(cls, settings: Settings) -> "ApplicationContainer":
        """进程启动时创建一次 Graph、连接池和无状态服务。"""

        graph_runtime = await CustomerServiceGraphRuntime.create(settings)
        llm_client = OpenAICompatibleLLMClient(settings)
        embedding_client = build_embedding_client(settings)
        vector_store = build_vector_store(settings)
        lexical_retriever = build_lexical_retriever(settings)
        reranker = build_reranker(settings)
        rag_service = RagService(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
            lexical_retriever=lexical_retriever,
            reranker=reranker,
        )
        semantic_cache = (
            RedisSemanticCache(settings, embedding_client)
            if settings.semantic_cache_enabled
            else None
        )
        return cls(
            settings=settings,
            graph_runtime=graph_runtime,
            llm_client=llm_client,
            embedding_client=embedding_client,
            vector_store=vector_store,
            lexical_retriever=lexical_retriever,
            reranker=reranker,
            rag_service=rag_service,
            semantic_cache=semantic_cache,
            search_client=SerpApiSearchClient(settings),
            mcp_client=build_after_sales_mcp_client(settings),
            tool_registry=build_default_tool_registry(),
        )

    def build_agent(self, session: AsyncSession) -> CustomerServiceAgent:
        """用当前 HTTP 请求的 Session 组装 Agent。

        网络客户端可以跨请求复用，数据库 Session 绑定当前事务，不能共享。
        """

        human_support_service = HumanSupportService(session)
        business_gateway = BusinessGateway(
            settings=self.settings,
            session=session,
            mcp_client=self.mcp_client,
            human_support_service=human_support_service,
        )
        return CustomerServiceAgent(
            settings=self.settings,
            session=session,
            graph_runtime=self.graph_runtime,
            llm_client=self.llm_client,
            rag_service=self.rag_service,
            tool_registry=self.tool_registry,
            search_client=self.search_client,
            business_gateway=business_gateway,
            semantic_cache=self.semantic_cache,
            cost_service=CostGovernanceService(settings=self.settings, session=session),
            human_support_service=human_support_service,
        )

    async def close(self) -> None:
        """应用停止时统一关闭 Checkpointer 和连接池。"""

        resources = (
            self.graph_runtime,
            self.mcp_client,
            self.lexical_retriever,
            self.reranker,
            self.semantic_cache,
            self.vector_store,
            self.embedding_client,
            self.llm_client,
        )
        closed_ids: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in closed_ids:
                continue
            closed_ids.add(id(resource))
            await _close_resource(resource)


async def _close_resource(resource: Any) -> None:
    """兼容各个 SDK 的 `close` / `aclose` 命名差异。"""

    close = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result
