from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import get_settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.embeddings.factory import build_embedding_client
from customer_service_app.infrastructure.llm.openai_compatible import OpenAICompatibleLLMClient
from customer_service_app.infrastructure.mcp.client import build_after_sales_mcp_client
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.infrastructure.vector_store.factory import build_vector_store
from customer_service_app.services.business_gateway import BusinessGateway
from customer_service_app.services.customer_service_agent import CustomerServiceAgent
from customer_service_app.services.rag_service import RagService
from customer_service_app.tools.default_registry import build_default_tool_registry


def build_customer_service_agent(session: AsyncSession) -> CustomerServiceAgent:
    """组装 CustomerServiceAgent 需要的所有依赖。

    这个文件可以理解为轻量 DI 容器：
    - 创建 LLM 客户端
    - 创建 Embedding 客户端
    - 创建 Qdrant/RAG/Redis/工具注册表
    - 最后把它们注入到 Agent

    Centralizing dependency construction keeps route handlers thin and makes providers replaceable.
    """
    settings = get_settings()
    embedding_client = build_embedding_client(settings)
    vector_store = build_vector_store(settings)
    rag_service = RagService(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    semantic_cache = (
        RedisSemanticCache(settings, embedding_client) if settings.semantic_cache_enabled else None
    )
    business_gateway = BusinessGateway(
        settings=settings,
        session=session,
        mcp_client=build_after_sales_mcp_client(settings),
    )
    return CustomerServiceAgent(
        settings=settings,
        session=session,
        llm_client=OpenAICompatibleLLMClient(settings),
        rag_service=rag_service,
        tool_registry=build_default_tool_registry(),
        search_client=SerpApiSearchClient(settings),
        business_gateway=business_gateway,
        semantic_cache=semantic_cache,
    )
