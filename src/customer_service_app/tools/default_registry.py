from __future__ import annotations

from customer_service_app.services.tool_registry import ToolRegistry
from customer_service_app.tools.after_sales_tools import (
    COMPENSATION_CASE_TOOL,
    EXCHANGE_CASE_TOOL,
    REFUND_CASE_TOOL,
)
from customer_service_app.tools.handoff_tools import HUMAN_HANDOFF_CASE_TOOL
from customer_service_app.tools.order_tools import (
    CUSTOMER_PROFILE_TOOL,
    LOGISTICS_STATUS_TOOL,
    ORDER_STATUS_TOOL,
    PRICE_PROTECTION_TOOL,
)
from customer_service_app.tools.search_tools import WEB_SEARCH_TOOL


def build_default_tool_registry() -> ToolRegistry:
    """注册项目默认工具。

    这里注册的工具才会出现在 LLM API 的 `tools` 参数里。
    如果你后续新增业务工具，一般就是：
    1. 写一个 async 工具函数。
    2. 定义一个 ToolSpec。
    3. 在这里 `registry.register(你的工具)`。
    """
    registry = ToolRegistry()
    registry.register(ORDER_STATUS_TOOL)
    registry.register(LOGISTICS_STATUS_TOOL)
    registry.register(PRICE_PROTECTION_TOOL)
    registry.register(CUSTOMER_PROFILE_TOOL)
    registry.register(REFUND_CASE_TOOL)
    registry.register(COMPENSATION_CASE_TOOL)
    registry.register(EXCHANGE_CASE_TOOL)
    registry.register(HUMAN_HANDOFF_CASE_TOOL)
    registry.register(WEB_SEARCH_TOOL)
    return registry
