from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_service_app.core.config import get_settings  # noqa: E402
from customer_service_app.infrastructure.embeddings.factory import (  # noqa: E402
    build_embedding_client,
)
from customer_service_app.infrastructure.lexical_search import (  # noqa: E402
    build_lexical_retriever,
)
from customer_service_app.infrastructure.rerank import build_reranker  # noqa: E402
from customer_service_app.infrastructure.vector_store.factory import (  # noqa: E402
    build_vector_store,
)
from customer_service_app.services.rag_service import RagService  # noqa: E402


async def evaluate(
    *,
    query_file: Path,
    tenant_id: str,
    top_k: int,
    use_rerank: bool,
) -> None:
    """使用真实检索链路计算 Hit@K 和 MRR，不调用聊天模型。"""

    settings = get_settings()
    embedding_client = build_embedding_client(settings)
    vector_store = build_vector_store(settings)
    lexical_retriever = build_lexical_retriever(settings)
    reranker = build_reranker(settings)
    service = RagService(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
        lexical_retriever=lexical_retriever,
        reranker=reranker,
    )
    queries = json.loads(query_file.read_text(encoding="utf-8"))
    hits = 0
    reciprocal_rank = 0.0
    try:
        for item in queries:
            chunks = await service.retrieve(
                tenant_id=tenant_id,
                question=str(item["query"]),
                top_k=top_k,
                use_rerank=use_rerank,
            )
            sources = [chunk.source for chunk in chunks]
            expected = str(item["expected_source"])
            rank = sources.index(expected) + 1 if expected in sources else None
            if rank is not None:
                hits += 1
                reciprocal_rank += 1.0 / rank
            print(
                json.dumps(
                    {
                        "id": item["id"],
                        "hit": rank is not None,
                        "rank": rank,
                        "expected": expected,
                        "sources": sources,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        for resource in (reranker, lexical_retriever, vector_store, embedding_client):
            await _close_resource(resource)

    count = len(queries)
    print(
        json.dumps(
            {
                "queries": count,
                f"hit@{top_k}": round(hits / count, 4) if count else 0.0,
                "mrr": round(reciprocal_rank / count, 4) if count else 0.0,
            },
            ensure_ascii=False,
        )
    )


async def _close_resource(resource: Any) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate knowledge retrieval quality")
    parser.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "knowledge_base" / "evaluation_queries.json",
    )
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        evaluate(
            query_file=args.queries,
            tenant_id=args.tenant_id,
            top_k=max(args.top_k, 1),
            use_rerank=args.rerank,
        )
    )
