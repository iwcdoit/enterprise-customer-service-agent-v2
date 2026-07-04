from __future__ import annotations

import json

import pytest

from customer_service_app.core.config import Settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache


class FakeEmbeddingClient:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeRedis:
    def __init__(self, values: dict[str, str]):
        self.values = values

    async def scan_iter(self, *, match: str, count: int):
        for key in self.values:
            if ":vec:" in key:
                yield key

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.values[key] = value


@pytest.mark.asyncio
async def test_semantic_cache_ignores_invalid_vectors_and_returns_best_match() -> None:
    settings = Settings(
        redis_url="redis://cache.example.com/0",
        semantic_cache_threshold=0.8,
    )
    cache = RedisSemanticCache(settings, FakeEmbeddingClient())
    prefix = cache._prefix("tenant-a", "user-a")
    cache._redis = FakeRedis(
        {
            f"{prefix}:vec:bad-json": "not-json",
            f"{prefix}:vec:wrong-dimension": json.dumps([1.0, 0.0, 0.0]),
            f"{prefix}:vec:low-score": json.dumps([0.0, 1.0]),
            f"{prefix}:vec:good": json.dumps([1.0, 0.0]),
            f"{prefix}:answer:good": "cached answer",
            f"{prefix}:meta:good": "not-json",
        }
    )

    entry = await cache.lookup(tenant_id="tenant-a", user_id="user-a", question="退款要求")

    assert entry is not None
    assert entry.answer == "cached answer"
    assert entry.similarity == pytest.approx(1.0)
    assert entry.metadata == {}


def test_cosine_returns_zero_for_mismatched_dimensions() -> None:
    assert RedisSemanticCache._cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
