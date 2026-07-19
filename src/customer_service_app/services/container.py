from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.embeddings.factory import build_embedding_client
from customer_service_app.infrastructure.llm.openai_compatible import OpenAICompatibleLLMClient
from customer_service_app.infrastructure.lexical_search.factory import build_lexical_retriever
from customer_service_app.infrastructure.mcp.client import build_after_sales_mcp_client
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.infrastructure.vector_store.factory import build_vector_store
from customer_service_app.infrastructure.rerank.factory import build_reranker
from customer_service_app.services.business_gateway import BusinessGateway
from customer_service_app.services.confirmation_service import ConfirmationService
from customer_service_app.services.conversation_service import ConversationService
from customer_service_app.services.cost_governance_service import CostGovernanceService
from customer_service_app.services.customer_service_agent import CustomerServiceAgent
from customer_service_app.services.memory_service import MemoryService
from customer_service_app.services.long_term_memory_service import LongTermMemoryService
from customer_service_app.services.human_support_service import HumanSupportService
from customer_service_app.services.planner_service import PlannerService
from customer_service_app.services.question_rewrite_service import QuestionRewriteService
from customer_service_app.services.rag_service import RagService
from customer_service_app.services.risk_control_service import RiskControlService
from customer_service_app.services.tool_registry import ToolExecutionContext
from customer_service_app.services.trace_service import TraceService
from customer_service_app.tools.default_registry import build_default_tool_registry
from customer_service_app.workflows.nodes import CustomerServiceGraphNodes
from customer_service_app.workflows.runtime import CustomerServiceGraphRuntime


class ApplicationContainer:
    """Application-scoped infrastructure clients and request-scoped service composition."""

    def __init__(
        self,
        *,
        settings: Settings,
        graph_runtime: CustomerServiceGraphRuntime,
    ):
        self.settings = settings
        self.graph_runtime = graph_runtime

        self.embedding_client = build_embedding_client(settings)
        self.vector_store = build_vector_store(settings)
        self.lexical_retriever = build_lexical_retriever(settings)
        self.reranker = build_reranker(settings)

        self.rag_service = RagService(
            settings=settings,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            lexical_retriever=self.lexical_retriever,
            reranker=self.reranker,
        )

        self.semantic_cache = (
            RedisSemanticCache(settings, self.embedding_client)
            if settings.semantic_cache_enabled
            else None
        )

        self.llm_client = OpenAICompatibleLLMClient(settings)
        self.search_client = SerpApiSearchClient(settings)
        self.tool_registry = build_default_tool_registry()
        self.mcp_client = build_after_sales_mcp_client(settings)

    @classmethod
    async def create(cls, settings: Settings) -> "ApplicationContainer":
        graph_runtime = await CustomerServiceGraphRuntime.create(settings)

        return cls(settings=settings, graph_runtime=graph_runtime)

    def build_agent(self, session: AsyncSession) -> CustomerServiceAgent:
        """Create transaction-bound services around shared network clients."""

        human_support_service = HumanSupportService(session)
        business_gateway = BusinessGateway(
            session,
            mcp_client=self.mcp_client,
            human_support_service=human_support_service,
        )

        confirmation_service = ConfirmationService(
            settings=self.settings,
            session=session,
            tool_registry=self.tool_registry,
            search_client=self.search_client,
            business_gateway=business_gateway,
        )

        tool_context = ToolExecutionContext(
            tenant_id="",
            user_id="",
            conversation_id=None,
            session=session,
            search_client=self.search_client,
            business_gateway=business_gateway,
        )

        nodes = CustomerServiceGraphNodes(
            settings=self.settings,
            session=session,
            llm_client=self.llm_client,
            rag_service=self.rag_service,
            tool_registry=self.tool_registry,
            semantic_cache=self.semantic_cache,
            conversation_service=ConversationService(session),
            confirmation_service=confirmation_service,
            memory_service=MemoryService(settings=self.settings, session=session),
            long_term_memory_service=LongTermMemoryService(session),
            cost_service=CostGovernanceService(settings=self.settings, session=session),
            trace_service=TraceService(session),
            planner_service=PlannerService(
                tool_registry=self.tool_registry,
                max_steps=self.settings.plan_max_steps,
            ),
            question_rewrite_service=QuestionRewriteService(
                settings=self.settings,
                llm_client=self.llm_client,
            ),
            risk_control_service=RiskControlService(),
            tool_context=tool_context,
        )

        return CustomerServiceAgent(
            session=session,
            graph_runtime=self.graph_runtime,
            graph_nodes=nodes,
            confirmation_service=confirmation_service,
            human_support_service=human_support_service,
        )

    async def close(self) -> None:
        """Close shared pools in reverse dependency order."""

        await self.graph_runtime.close()

        for resource in (
            self.mcp_client,
            self.semantic_cache,
            self.vector_store,
            self.lexical_retriever,
            self.reranker,
            self.embedding_client,
            self.llm_client,
        ):
            closer = getattr(resource, "close", None)
            if closer is not None:
                result = closer()
                if hasattr(result, "__await__"):
                    await result
