from __future__ import annotations

from typing import Any, TypedDict


class CustomerServiceGraphState(TypedDict, total=False):
    """一个客服 Graph 线程的可持久化状态。

    State 会写入 checkpoint，因此只保存 JSON-like 数据。数据库 Session、LLM 客户端、
    Repository 等运行时对象由 ``CustomerServiceGraphContext`` 传递，不能放进这里。
    """

    request: dict[str, Any]
    thread_id: str
    conversation_id: str
    status: str
    route: str
    plan: dict[str, Any] | None
    plan_execution: dict[str, Any] | None
    pending_confirmations: list[dict[str, Any]]
    confirmation_cursor: int
    confirmation_decision: dict[str, Any] | None
    response: dict[str, Any]
    trace: list[dict[str, Any]]
    error: str | None
