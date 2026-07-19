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

        理解点：
        tenantA:user001:vec:abc123 - 存的是问题的 embedding 向量：[0.012, -0.087, 0.334, ...]
        tenantA:user001:answer:abc123 - 当时大模型生成的答案
        tenantA:user001:meta:abc123 - 排查、trace、运营分析，不是回答正文：{"conversation_id": "xxx","model": "qwen-plus","source": "semantic_cache"}

        """
        query_vector = await self._embedding_client.embed_query(question)

        prefix = self._prefix(tenant_id, user_id)

        best: SemanticCacheEntry | None = None

        redis_client = self.redis

        async for key in redis_client.scan_iter(match=f"{prefix}:vec:*", count=100):
            raw_vector = await redis_client.get(key)

            if not raw_vector:
                continue

            try:
                cached_vector = json.loads(raw_vector)
            except json.JSONDecodeError:
                continue

            if not isinstance(cached_vector, list):
                continue

            similarity = self._cosine(query_vector, cached_vector)

            if similarity < self._settings.semantic_cache_threshold:
                continue

            cache_id = key.split(":")[-1]

            answer = await redis_client.get(f"{prefix}:answer:{cache_id}")

            if not answer:
                continue

            raw_metadata = await redis_client.get(f"{prefix}:meta:{cache_id}")

            metadata: dict[str, Any] = {}

            if raw_metadata:
                try:
                    parsed_metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    parsed_metadata = {}
                if isinstance(parsed_metadata, dict):
                    metadata = parsed_metadata

            entry = SemanticCacheEntry(
                answer=answer,
                similarity=similarity,
                metadata=metadata,
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
        normalized_question = question.strip().lower()

        if not normalized_question or not answer:
            return

        vector = await self._embedding_client.embed_query(question)

        prefix = self._prefix(tenant_id, user_id)

        cache_id = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()

        ttl = self._settings.semantic_cache_ttl_seconds

        redis_client = self.redis

        vector_key = f"{prefix}:vec:{cache_id}"

        answer_key = f"{prefix}:answer:{cache_id}"

        metadata_key = f"{prefix}:meta:{cache_id}"

        await redis_client.set(vector_key, json.dumps(vector), ex=ttl)

        await redis_client.set(answer_key, answer, ex=ttl)

        await redis_client.set(
            metadata_key,
            json.dumps(metadata or {}, ensure_ascii=False),
            ex=ttl,
        )

    def _prefix(self, tenant_id: str, user_id: str) -> str:
        """生成 Redis key 前缀。

        这里对 tenant_id/user_id 做 hash，避免 Redis key 里直接暴露原始业务 id。
        """
        tenant_hash = tenant_id[:12]
        user_hash = user_id[:12]
        return f"{tenant_hash}:{user_hash}"

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度。

        结果越接近 1，表示语义越接近；越接近 0，表示关系越弱。
        两个向量必须来自相同 embedding 模型且维度一致。
        """
        a = np.asarray(left, dtype=np.float32)

        b = np.asarray(right, dtype=np.float32)

        if a.shape != b.shape:
            return 0.0

        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

        if denominator == 0.0:
            return 0.0

        return float(np.dot(a, b) / denominator)

    async def close(self) -> None:
        """Close the shared Redis connection pool."""
        if self.redis is not None:
            await self.redis.close()
