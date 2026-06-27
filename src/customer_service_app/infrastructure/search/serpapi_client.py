from __future__ import annotations

from typing import Any

import httpx

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError


class SerpApiSearchClient:
    """SerpAPI 搜索客户端。

    这是外部搜索能力的适配层。业务工具 `search_public_web` 不直接写 HTTP 细节，
    而是调用这个客户端。
    """

    def __init__(self, settings: Settings):
        """保存配置，搜索时读取 SERPAPI_KEY 和结果数量。"""
        self._settings = settings

    async def search(self, query: str) -> list[dict[str, Any]]:
        """执行一次搜索并返回统一结构。

        返回值只保留 title/url/snippet，避免把第三方 API 的复杂原始结构扩散到业务层。
        """
        api_key = self._settings.require("SERPAPI_KEY", self._settings.serpapi_key)
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": self._settings.search_result_count,
            "hl": "zh-cn",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get("https://serpapi.com/search.json", params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise ExternalServiceError(f"SerpAPI search failed: {exc}") from exc

        results = []
        for item in payload.get("organic_results", [])[: self._settings.search_result_count]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results
