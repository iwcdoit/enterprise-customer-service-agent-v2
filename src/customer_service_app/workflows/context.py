from __future__ import annotations

from dataclasses import dataclass

from customer_service_app.workflows.nodes import CustomerServiceGraphNodes


@dataclass(slots=True)
class CustomerServiceGraphContext:
    """请求级 Graph 依赖。

    Context 只在本次执行中使用，不会写入 checkpoint。数据库 Session、网络客户端和
    Service 对象都应放在这里；Graph State 只保存可序列化的业务状态。
    """

    nodes: CustomerServiceGraphNodes
