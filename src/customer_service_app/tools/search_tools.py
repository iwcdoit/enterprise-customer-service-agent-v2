from __future__ import annotations

from typing import Any

from customer_service_app.services.tool_registry import ToolExecutionContext, ToolSpec


async def search_public_web(arguments: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
    """公开网页搜索工具。

    这个工具适合需要最新外部信息的问题，可作为知识库和订单系统之外的补充来源。
    """
    query = str(arguments["query"])
    results = await context.search_client.search(query)
    return {"query": query, "results": results}


WEB_SEARCH_TOOL = ToolSpec(
    # 给模型看的搜索工具定义。真正执行搜索的是上面的 search_public_web 函数。
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
)
