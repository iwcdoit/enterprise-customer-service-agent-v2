from __future__ import annotations

from typing import Any, Literal, TypedDict


GraphRoute = Literal["plan", "answer"]


class CustomerServiceGraphState(TypedDict, total=False):
    """客服 Graph 各节点之间传递的可序列化状态。

    网络客户端、数据库 Session 和 Service 对象不能放进 State。
    这些运行时依赖保留在请求级 Nodes 中，后续接入 Checkpointer 时才能安全持久化 State。
    """

    request: dict[str, Any]
    route: GraphRoute
    plan: dict[str, Any] | None
    plan_execution: dict[str, Any] | None
    response: dict[str, Any]
    trace: list[dict[str, Any]]
    error: str | None
