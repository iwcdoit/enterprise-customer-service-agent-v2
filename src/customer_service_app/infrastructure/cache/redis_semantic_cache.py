from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis.asyncio as redis

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.embeddings.base import EmbeddingClient


@dataclass(slots=True)
class SemanticCacheEntry:
    """语义缓存命中的结果。"""

    answer: str
    similarity: float
    metadata: dict[str, Any]


class RedisSemanticCache:
    """基于 Redis + Embedding 的语义缓存。

    重点理解：
    - 不是大模型判断两个问题相似。
    - 是先把问题转成 embedding 向量，再用余弦相似度比较。
    - 相似度超过 `SEMANTIC_CACHE_THRESHOLD` 才复用历史答案。

    这个实现把向量 JSON 存进 Redis。高并发生产环境可以升级为
    Redis Stack 向量检索或独立向量数据库。
    """

    def __init__(self, settings: Settings, embedding_client: EmbeddingClient):
        """注入配置和 embedding 客户端。"""
        self._settings = settings
        self._embedding_client = embedding_client
        self._redis: redis.Redis | None = None

    @property
    def redis(self) -> redis.Redis:
        """懒加载 Redis 客户端。"""
        if self._redis is None:
            redis_url = self._settings.require("REDIS_URL", self._settings.redis_url)
            self._redis = redis.from_url(redis_url, decode_responses=True)
        return self._redis

    async def lookup(self, *, tenant_id: str, user_id: str, question: str) -> SemanticCacheEntry | None:
        """查找是否有语义相近的问题答案可复用。

        流程：
        1. 当前问题 -> embedding 向量。
        2. 扫描当前租户、当前用户的历史问题向量。
        3. 计算余弦相似度。
        4. 超过阈值时返回相似度最高的缓存答案。
        """
        query_vector = await self._embedding_client.embed_query(question)
        prefix = self._prefix(tenant_id, user_id)
        best: SemanticCacheEntry | None = None

        async for key in self.redis.scan_iter(match=f"{prefix}:vec:*", count=100):
            raw_vector = await self.redis.get(key)
            if not raw_vector:
                continue
            cached_vector = json.loads(raw_vector)
            similarity = self._cosine(query_vector, cached_vector)
            if similarity < self._settings.semantic_cache_threshold:
                continue

            cache_id = key.split(":")[-1]
            answer = await self.redis.get(f"{prefix}:answer:{cache_id}")
            metadata_raw = await self.redis.get(f"{prefix}:meta:{cache_id}")
            if not answer:
                continue
            entry = SemanticCacheEntry(
                answer=answer,
                similarity=similarity,
                metadata=json.loads(metadata_raw or "{}"),
            )
            if best is None or entry.similarity > best.similarity:
                best = entry
        return best

    async def update(
        self,
        *,
        tenant_id: str,
        user_id: str,
        question: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """把本次问题、答案、向量写入 Redis，供下次相似问题复用。"""
        vector = await self._embedding_client.embed_query(question)
        prefix = self._prefix(tenant_id, user_id)
        cache_id = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
        ttl = self._settings.semantic_cache_ttl_seconds
        await self.redis.set(f"{prefix}:vec:{cache_id}", json.dumps(vector), ex=ttl)
        await self.redis.set(f"{prefix}:answer:{cache_id}", answer, ex=ttl)
        await self.redis.set(f"{prefix}:meta:{cache_id}", json.dumps(metadata or {}), ex=ttl)

    def _prefix(self, tenant_id: str, user_id: str) -> str:
        """生成 Redis key 前缀。

        这里对 tenant_id/user_id 做 hash，避免 Redis key 里直接暴露原始业务 id。
        """
        tenant_hash = hashlib.sha1(tenant_id.encode("utf-8")).hexdigest()[:12]
        user_hash = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:12]
        return f"semantic_cache:{tenant_hash}:{user_hash}"

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度。

        结果越接近 1，表示语义越接近；越接近 0，表示关系越弱。
        """
        a = np.array(left, dtype=np.float32)
        b = np.array(right, dtype=np.float32)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0:
            return 0.0
        return float(np.dot(a, b) / denominator)
