from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

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


@dataclass(frozen=True, slots=True)
class SemanticCacheContext:
    """决定答案是否允许复用的硬条件，不依赖向量相似度。"""

    intent: str
    corpus_version: str
    entities: dict[str, Any] = field(default_factory=dict)
    audience: str = "customer"
    answer_type: str = "knowledge"
    realtime: bool = False
    tool_dependency: bool = False

    _safe_answer_types: ClassVar[frozenset[str]] = frozenset(
        {"faq", "policy", "guide", "troubleshooting", "knowledge"}
    )
    _business_entity_keys: ClassVar[frozenset[str]] = frozenset(
        {
            "order_id",
            "refund_id",
            "ticket_id",
            "tracking_number",
            "payment_id",
            "amount",
        }
    )

    def is_reusable(self) -> bool:
        """实时状态、业务单据和依赖工具的答案不进入语义答案缓存。"""

        entity_keys = {str(key).lower() for key in self.entities}
        return (
            bool(self.intent.strip())
            and bool(self.corpus_version.strip())
            and self.answer_type in self._safe_answer_types
            and not self.realtime
            and not self.tool_dependency
            and not entity_keys.intersection(self._business_entity_keys)
        )


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

    async def lookup(
        self,
        *,
        tenant_id: str,
        user_id: str,
        question: str,
        context: SemanticCacheContext,
    ) -> SemanticCacheEntry | None:
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
        if not context.is_reusable():
            return None

        prefix = self._prefix(tenant_id, user_id)
        redis_client = self.redis
        exact_id = self._cache_id(question=question, context=context)
        exact = await self._read_entry(
            redis_client=redis_client,
            prefix=prefix,
            cache_id=exact_id,
            context=context,
            similarity=1.0,
        )
        if exact is not None:
            return exact

        query_vector = await self._embedding_client.embed_query(question)

        best: SemanticCacheEntry | None = None

        async for key in redis_client.scan_iter(match=f"{prefix}:vec:*", count=100):
            cache_id = key.split(":")[-1]
            if cache_id == exact_id:
                continue
            metadata = await self._read_metadata(
                redis_client=redis_client,
                prefix=prefix,
                cache_id=cache_id,
            )
            if not self._metadata_matches(metadata=metadata, expected=context):
                continue

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

            answer = await redis_client.get(f"{prefix}:answer:{cache_id}")

            if not answer:
                continue

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
        context: SemanticCacheContext,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """把本次问题、答案、向量写入 Redis，供下次相似问题复用。"""
        normalized_question = question.strip().lower()

        if not normalized_question or not answer or not context.is_reusable():
            return

        vector = await self._embedding_client.embed_query(question)

        prefix = self._prefix(tenant_id, user_id)

        cache_id = self._cache_id(question=question, context=context)

        ttl = self._settings.semantic_cache_ttl_seconds

        redis_client = self.redis

        vector_key = f"{prefix}:vec:{cache_id}"

        answer_key = f"{prefix}:answer:{cache_id}"

        metadata_key = f"{prefix}:meta:{cache_id}"

        await redis_client.set(vector_key, json.dumps(vector), ex=ttl)

        await redis_client.set(answer_key, answer, ex=ttl)

        await redis_client.set(
            metadata_key,
            json.dumps(
                {
                    **(metadata or {}),
                    "cache_context": asdict(context),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            ex=ttl,
        )

    def _prefix(self, tenant_id: str, user_id: str) -> str:
        """生成 Redis key 前缀。

        这里对 tenant_id/user_id 做 hash，避免 Redis key 里直接暴露原始业务 id。
        """
        tenant_hash = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:12]
        user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
        return f"{tenant_hash}:{user_hash}"

    @staticmethod
    def _cache_id(*, question: str, context: SemanticCacheContext) -> str:
        """把问题和复用硬条件一起纳入精确缓存指纹。"""

        fingerprint = {
            "question": " ".join(question.strip().lower().split()),
            "intent": context.intent,
            "entities": context.entities,
            "corpus_version": context.corpus_version,
            "audience": context.audience,
            "answer_type": context.answer_type,
        }
        raw = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _read_entry(
        self,
        *,
        redis_client: redis.Redis,
        prefix: str,
        cache_id: str,
        context: SemanticCacheContext,
        similarity: float,
    ) -> SemanticCacheEntry | None:
        metadata = await self._read_metadata(
            redis_client=redis_client,
            prefix=prefix,
            cache_id=cache_id,
        )
        if not self._metadata_matches(metadata=metadata, expected=context):
            return None
        answer = await redis_client.get(f"{prefix}:answer:{cache_id}")
        if not answer:
            return None
        return SemanticCacheEntry(
            answer=answer,
            similarity=similarity,
            metadata=metadata,
        )

    @staticmethod
    async def _read_metadata(
        *,
        redis_client: redis.Redis,
        prefix: str,
        cache_id: str,
    ) -> dict[str, Any]:
        raw_metadata = await redis_client.get(f"{prefix}:meta:{cache_id}")
        if not raw_metadata:
            return {}
        try:
            value = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _metadata_matches(
        *,
        metadata: dict[str, Any],
        expected: SemanticCacheContext,
    ) -> bool:
        """意图、实体、受众和知识版本必须完全一致，相似度不能覆盖这些条件。"""

        raw_context = metadata.get("cache_context")
        if not isinstance(raw_context, dict):
            return False
        try:
            cached = SemanticCacheContext(**raw_context)
        except TypeError:
            return False
        return cached.is_reusable() and cached == expected

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
        if self._redis is not None:
            await self._redis.aclose()
