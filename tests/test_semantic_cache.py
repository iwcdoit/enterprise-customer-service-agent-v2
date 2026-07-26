from __future__ import annotations

import json
from fnmatch import fnmatch
from dataclasses import asdict

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import (
    RedisSemanticCache,
    SemanticCacheContext,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]


class FakeRedis:
    def __init__(self, values: dict[str, str]):
        self.values = values

    async def scan_iter(self, *, match: str, count: int):
        for key in list(self.values):
            if fnmatch(key, match):
                yield key

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.values[key] = value

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.values:
                del self.values[key]
                deleted += 1
        return deleted


@pytest.mark.asyncio
async def test_semantic_cache_ignores_invalid_vectors_and_returns_best_match() -> None:
    settings = Settings(
        redis_url="redis://cache.example.com/0",
        semantic_cache_threshold=0.8,
    )
    context = SemanticCacheContext(
        intent="refund_policy",
        corpus_version="v1",
        answer_type="policy",
    )
    cache = RedisSemanticCache(settings, FakeEmbeddingClient())
    prefix = cache._prefix("tenant-a", "user-a")
    metadata = json.dumps({"cache_context": asdict(context)}, ensure_ascii=False)
    cache._redis = FakeRedis(
        {
            f"{prefix}:vec:bad-json": "not-json",
            f"{prefix}:meta:bad-json": metadata,
            f"{prefix}:vec:wrong-dimension": json.dumps([1.0, 0.0, 0.0]),
            f"{prefix}:meta:wrong-dimension": metadata,
            f"{prefix}:vec:low-score": json.dumps([0.0, 1.0]),
            f"{prefix}:meta:low-score": metadata,
            f"{prefix}:vec:good": json.dumps([1.0, 0.0]),
            f"{prefix}:answer:good": "cached answer",
            f"{prefix}:meta:good": metadata,
        }
    )

    entry = await cache.lookup(
        tenant_id="tenant-a",
        user_id="user-a",
        question="退款要求",
        context=context,
    )

    assert entry is not None
    assert entry.answer == "cached answer"
    assert entry.similarity == pytest.approx(1.0)
    assert entry.metadata["cache_context"]["corpus_version"] == "v1"


@pytest.mark.asyncio
async def test_semantic_cache_exact_hit_requires_same_business_context() -> None:
    settings = Settings(
        redis_url="redis://cache.example.com/0",
        semantic_cache_threshold=0.8,
    )
    embedding = FakeEmbeddingClient()
    cache = RedisSemanticCache(settings, embedding)
    cache._redis = FakeRedis({})
    context = SemanticCacheContext(
        intent="refund_policy",
        corpus_version="v1",
        answer_type="policy",
    )

    await cache.update(
        tenant_id="tenant-a",
        user_id="user-a",
        question="退款多久到账",
        answer="通常一到三个工作日。",
        context=context,
    )
    entry = await cache.lookup(
        tenant_id="tenant-a",
        user_id="user-a",
        question="退款多久到账",
        context=context,
    )
    changed_version = await cache.lookup(
        tenant_id="tenant-a",
        user_id="user-a",
        question="退款多久到账",
        context=SemanticCacheContext(
            intent="refund_policy",
            corpus_version="v2",
            answer_type="policy",
        ),
    )

    assert entry is not None
    assert entry.similarity == 1.0
    assert changed_version is None


@pytest.mark.asyncio
async def test_semantic_cache_rejects_realtime_and_business_entity_answers() -> None:
    settings = Settings(redis_url="redis://cache.example.com/0")
    embedding = FakeEmbeddingClient()
    cache = RedisSemanticCache(settings, embedding)
    cache._redis = FakeRedis({})

    realtime = await cache.lookup(
        tenant_id="tenant-a",
        user_id="user-a",
        question="订单到哪里了",
        context=SemanticCacheContext(
            intent="logistics_status",
            corpus_version="v1",
            realtime=True,
        ),
    )
    order_specific = await cache.lookup(
        tenant_id="tenant-a",
        user_id="user-a",
        question="这个订单为什么退款失败",
        context=SemanticCacheContext(
            intent="refund_status",
            corpus_version="v1",
            entities={"order_id": "ORDER-001"},
        ),
    )

    assert realtime is None
    assert order_specific is None
    assert embedding.calls == 0


def test_cosine_returns_zero_for_mismatched_dimensions() -> None:
    assert RedisSemanticCache._cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_semantic_cache_invalidates_answers_using_changed_chunks() -> None:
    cache = RedisSemanticCache(
        Settings(redis_url="redis://cache.example.com/0"),
        FakeEmbeddingClient(),
    )
    prefix = cache._prefix("tenant-a", "user-a")
    cache._redis = FakeRedis(
        {
            f"{prefix}:vec:affected": "[1.0, 0.0]",
            f"{prefix}:answer:affected": "old answer",
            f"{prefix}:meta:affected": json.dumps({"source_chunk_ids": ["chunk-old"]}),
            f"{prefix}:vec:unrelated": "[1.0, 0.0]",
            f"{prefix}:answer:unrelated": "other answer",
            f"{prefix}:meta:unrelated": json.dumps({"source_chunk_ids": ["chunk-other"]}),
        }
    )

    invalidated = await cache.invalidate_by_chunk_ids(
        tenant_id="tenant-a",
        chunk_ids=["chunk-old"],
    )

    assert invalidated == 1
    assert f"{prefix}:answer:affected" not in cache._redis.values
    assert f"{prefix}:answer:unrelated" in cache._redis.values
