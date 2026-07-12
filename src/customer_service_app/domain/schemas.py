from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_service_app.domain.cost import CostStrategy
from customer_service_app.domain.planning import AgentPlan, PlanExecutionResult


MessageRole = Literal["system", "user", "assistant", "tool"]
"""消息角色枚举。

`Literal[...]` 表示变量只能取这些固定字符串之一，类似 Java enum 的轻量写法。
"""


class ChatMessage(BaseModel):
    """一条对话消息。

    role 表示消息是谁说的：用户、助手、系统提示词，或工具返回结果。
    这个模型主要用于前端传入历史消息，以及从数据库恢复历史上下文。

    `BaseModel` 是 Pydantic 的数据模型基类，类似 Java DTO + 参数校验注解的组合。
    """

    role: MessageRole
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # `Field(default_factory=dict)` 表示每个对象都创建自己的空 dict；
    # 不要写成 `metadata: dict = {}`，否则多个对象可能共享同一个默认字典。


class ChatRequest(BaseModel):
    """聊天接口的请求体。

    运营测试台或前端每发一次消息，都会提交这个结构。
    tenant_id / user_id 用来做数据隔离，conversation_id 用来延续同一轮会话。
    metadata 可以放调试开关，例如 skip_cache=true 表示本次请求跳过语义缓存。
    """

    tenant_id: str = Field(default="default", description="租户或业务线 ID")
    user_id: str = Field(description="当前用户 ID")
    conversation_id: str | None = Field(default=None, description="为空时自动创建会话")
    question: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    """RAG 从向量库里命中的一段知识。

    content 会被拼进 system prompt，让大模型基于企业知识库回答。
    score 是向量相似度，越高代表越接近用户问题。
    """

    id: str
    source: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallView(BaseModel):
    """模型决定调用工具时，对外展示的工具调用信息。

    这里只记录模型“想调用什么工具、传了什么参数”。
    真正执行工具的是后端 ToolRegistry，不是模型本身。
    """

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultView(BaseModel):
    """后端执行工具后的结果。

    ok 表示工具是否执行成功，payload 是工具真实返回的数据，例如订单状态或工单号。
    """

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
    answer: str
    cache_hit: bool = False
    knowledge: list[KnowledgeChunk] = Field(default_factory=list)
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    tool_results: list[ToolResultView] = Field(default_factory=list)
    plan: AgentPlan | None = None
    plan_execution: PlanExecutionResult | None = None
    trace: list[ChatTraceStep] = Field(default_factory=list)


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


class ConversationMessageView(BaseModel):
    """会话详情页展示的一条历史消息。"""

    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ConversationDetailView(BaseModel):
    """一个会话及其最近消息。"""

    conversation: ConversationView
    messages: list[ConversationMessageView] = Field(default_factory=list)


class ConfirmationDecisionRequest(BaseModel):
    """确认或拒绝高风险工具动作的请求体。"""

    tenant_id: str
    user_id: str
    comment: str | None = None


class PendingActionView(BaseModel):
    """等待用户或人工客服确认的工具动作。"""

    id: str
    tenant_id: str
    user_id: str
    conversation_id: str | None
    thread_id: str | None = None
    confirmation_id: str | None = None
    tool_name: str
    arguments: dict[str, Any]
    status: str
    comment: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    expired: bool = False


class RuntimeConfigView(BaseModel):
    """运营诊断页展示的安全运行配置摘要。

    这里不能返回 API key、数据库密码、Redis 密码等敏感信息，只展示功能开关、
    provider 和必要的配置完整性提示。
    """

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
    """运营侧查看某个租户当前会采用的成本策略。"""

    tenant_id: str
    strategy: CostStrategy
    notes: list[str] = Field(default_factory=list)


class PendingActionSummaryView(BaseModel):
    """运营侧查看某个用户的待确认动作概览。"""

    tenant_id: str
    user_id: str
    total_pending: int
    active_pending: int
    expired_pending: int
    actions: list[PendingActionView] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """健康检查接口返回值。

    只说明服务进程是否正常，不代表 LLM、数据库、Qdrant 都已经可用。
    """

    status: str
    app: str
    runtime_env: str
