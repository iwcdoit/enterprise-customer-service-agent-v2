from __future__ import annotations

from typing import Any, TypedDict


class CustomerServiceGraphState(TypedDict, total=False):
    """Checkpointed state for one customer-service graph thread.

    Only JSON-like values are stored here. Database sessions, clients, repositories, and other
    request-scoped objects live in the LangGraph runtime context and are never checkpointed.
    """

    request: dict[str, Any]

    tenant_id: str
    user_id: str
    conversation_id: str

    thread_id: str

    run_id: str

    timer_start: float

    status: str

    cost_strategy: dict[str, Any]

    memory_context: dict[str, Any]

    rewritten_question: str

    query_rewrite: dict[str, Any]
    rewrite_attempts: int

    knowledge: list[dict[str, Any]]

    retrieval_quality: dict[str, Any]

    messages: list[dict[str, Any]]

    plan: dict[str, Any] | None
    plan_cursor: int
    plan_observations: list[dict[str, Any]]
    executed_plan_steps: list[str]

    first_response: dict[str, Any]

    tool_calls: list[dict[str, Any]]
    tool_cursor: int
    tool_results: list[dict[str, Any]]

    active_tool_name: str
    active_arguments: dict[str, Any]
    active_call_id: str
    active_step_id: str
    flow_origin: str

    pending_confirmation: dict[str, Any] | None
    confirmation_decision: dict[str, Any] | None
    confirmation_resumed: bool

    final_answer: str

    model: str | None
    total_tokens: int

    cache_hit: bool
    cache_eligible: bool
    skip_cache: bool

    turn_saved: bool

    trace: list[dict[str, Any]]

    error: str | None
