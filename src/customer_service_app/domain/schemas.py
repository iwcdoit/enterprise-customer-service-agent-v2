from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_service_app.domain.cost import CostStrategy
from customer_service_app.domain.confirmations import PendingActionView
from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult
from customer_service_app.domain.human_support import HumanHandoffView


MessageRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """对话消息。"""

    role: MessageRole
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """聊天请求。"""

    tenant_id: str = Field(default="default", description="租户或业务线 ID")
    user_id: str = Field(description="当前用户 ID")
    conversation_id: str | None = Field(default=None, description="为空时自动创建会话")
    thread_id: str | None = Field(default=None, description="LangGraph checkpoint 线程 ID")
    question: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    """RAG 检索命中的知识块。"""

    id: str
    source: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallView(BaseModel):
    """模型生成的工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultView(BaseModel):
    """后端工具执行结果。"""

    tool_call_id: str
    name: str
    ok: bool
    payload: dict[str, Any]


class ChatTraceStep(BaseModel):
    """一次请求中的执行轨迹。

    运营测试台右侧的 Trace 面板就是展示这个结构。
    它用于解释请求走到了 cache、rag、tools 等哪个阶段。
    """

    stage: str
    detail: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """聊天接口的响应体。

    answer 是最终给用户看的回复；knowledge、tool_calls、tool_results、trace
    是给开发者和运营验证链路用的证据。
    """

    conversation_id: str
    thread_id: str | None = None
    answer: str
    status: str = "completed"
    run_id: str | None = None
    cache_hit: bool = False
    query_rewrite: dict[str, Any] = Field(default_factory=dict)
    retrieval_quality: dict[str, Any] = Field(default_factory=dict)
    knowledge: list[KnowledgeChunk] = Field(default_factory=list)
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    tool_results: list[ToolResultView] = Field(default_factory=list)
    pending_confirmation: PendingActionView | None = None
    plan: AgentPlan | None = None
    plan_execution: PlanExecutionResult | None = None
    trace: list[ChatTraceStep] = Field(default_factory=list)
    service_mode: str = "bot"
    human_handoff: HumanHandoffView | None = None


class ConversationMessageView(BaseModel):
    """One persisted conversation message returned to the frontend."""

    id: str
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    created_at: datetime | str | None = None


class ConversationDetailView(BaseModel):
    """一个会话及其最近消息。"""

    conversation: "ConversationView"
    messages: list[ConversationMessageView] = Field(default_factory=list)


class GraphTaskView(BaseModel):
    """One pending LangGraph task in a checkpoint snapshot."""

    id: str
    name: str
    interrupts: list[dict[str, Any]] = Field(default_factory=list)


class GraphStateView(BaseModel):
    """Serializable view of a LangGraph checkpoint."""

    thread_id: str
    status: str
    next_nodes: list[str] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    tasks: list[GraphTaskView] = Field(default_factory=list)


class AgentRunStepView(BaseModel):
    """One persisted trace step for an Agent run."""

    id: str
    stage: str
    name: str
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None


class AgentRunView(BaseModel):
    """Trace view for one Agent request."""

    id: str
    tenant_id: str
    user_id: str
    conversation_id: str | None = None
    status: str
    model: str | None = None
    total_tokens: int = 0
    error_code: str | None = None
    error_message: str | None = None
    steps: list[AgentRunStepView] = Field(default_factory=list)


class RuntimeConfigView(BaseModel):
    """不暴露密钥的运行配置摘要。"""

    app_name: str
    runtime_env: str
    api_prefix: str
    llm_provider: str
    embedding_provider: str
    vector_store_provider: str
    rag_enabled: bool
    semantic_cache_enabled: bool
    mcp_after_sales_enabled: bool
    cost_governance_enabled: bool
    search_enabled: bool
    warnings: list[str] = Field(default_factory=list)


class TenantStrategyView(BaseModel):
    """租户当前成本策略。"""

    tenant_id: str
    strategy: CostStrategy
    notes: list[str] = Field(default_factory=list)


class PendingActionSummaryView(BaseModel):
    """用户待确认动作概览。"""

    tenant_id: str
    user_id: str
    total_pending: int
    active_pending: int
    expired_pending: int
    actions: list[PendingActionView] = Field(default_factory=list)


class ConversationCreateRequest(BaseModel):
    """创建会话接口的请求体。"""

    tenant_id: str = "default"
    user_id: str
    title: str = "新会话"


class ConversationView(BaseModel):
    """会话列表或创建会话后返回给前端的简要信息。"""

    id: str
    tenant_id: str
    user_id: str
    title: str
    status: str
    service_mode: str = "bot"


class HealthResponse(BaseModel):
    """健康检查接口返回值。

    只说明服务进程是否正常，不代表 LLM、数据库、Qdrant 都已经可用。
    """

    status: str
    app: str
    runtime_env: str


ConversationDetailView.model_rebuild()
