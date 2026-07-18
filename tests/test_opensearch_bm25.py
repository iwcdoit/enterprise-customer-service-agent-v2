from __future__ import annotations

import json

import httpx

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.lexical_search.opensearch_bm25 import (
    OpenSearchBM25Retriever,
)


async def test_bm25_search_always_filters_tenant() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200)
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk-1",
                            "_score": 3.2,
                            "_source": {
                                "source": "faq/refund.md",
                                "title": "退款到账",
                                "content": "支付宝通常一到三个工作日到账。",
                                "metadata": {"document_type": "faq"},
                            },
                        }
                    ]
                }
            },
        )

    retriever = OpenSearchBM25Retriever(
        Settings(opensearch_enabled=True, opensearch_url="https://search.example.com")
    )
    await retriever._client.aclose()
    retriever._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await retriever.search(tenant_id="tenant-a", query="退款多久到账", top_k=5)

    filters = captured["query"]["bool"]["filter"]
    assert filters == [{"term": {"tenant_id": "tenant-a"}}]
    assert result[0].id == "chunk-1"
    assert result[0].metadata["retriever"] == "bm25"
    await retriever.close()


async def test_bm25_upsert_preserves_structured_chunk_metadata() -> None:
    bulk_lines: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200)
        for line in request.content.decode().strip().splitlines():
            bulk_lines.append(json.loads(line))
        return httpx.Response(200, json={"errors": False})

    retriever = OpenSearchBM25Retriever(
        Settings(opensearch_enabled=True, opensearch_url="https://search.example.com")
    )
    await retriever._client.aclose()
    retriever._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunk = KnowledgeChunk(
        id="chunk-1",
        source="policy/refund.md",
        title="售后 / 退款",
        content="退款政策正文",
        score=1.0,
        metadata={"document_type": "policy", "heading_path": ["售后", "退款"]},
    )

    await retriever.upsert_chunks(tenant_id="tenant-a", chunks=[chunk])

    assert bulk_lines[0]["index"]["_id"] == "chunk-1"
    assert bulk_lines[1]["tenant_id"] == "tenant-a"
    assert bulk_lines[1]["document_type"] == "policy"
    assert bulk_lines[1]["heading_path"] == ["售后", "退款"]
    await retriever.close()
