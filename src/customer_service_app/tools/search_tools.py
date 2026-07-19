from __future__ import annotations

from typing import Any

from customer_service_app.services.tool_registry import ToolExecutionContext, ToolSpec


async def search_public_web(arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    """公开网页搜索工具。

    这个工具适合需要最新外部信息的问题。客服业务里不一定频繁使用，
    但它演示了“工具不只可以查数据库，也可以调用第三方 API”。
    """
    query = str(arguments["query"])
    results = await context.search_client.search(query)
    return {"query": query, "results": results}


WEB_SEARCH_TOOL = ToolSpec(
    name="search_public_web",
    description="搜索公开互联网信息，适合实时新闻、外部公告、需要最新信息的问题。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "适合搜索引擎检索的关键词"}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=search_public_web,
    read_only=True,
    requires_confirmation=False,
    risk_level="low",
    action_type="search",
    parallel_safe=True,
)
