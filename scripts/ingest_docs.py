from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_service_app.core.config import get_settings  # noqa: E402
from customer_service_app.domain.knowledge_lifecycle import (  # noqa: E402
    KnowledgeSourceDocument,
)
from customer_service_app.infrastructure.cache.redis_semantic_cache import (  # noqa: E402
    RedisSemanticCache,
)
from customer_service_app.infrastructure.db.repositories import (  # noqa: E402
    KnowledgeDocumentRepository,
)
from customer_service_app.infrastructure.db.session import session_context  # noqa: E402
from customer_service_app.infrastructure.embeddings.factory import (  # noqa: E402
    build_embedding_client,
)
from customer_service_app.infrastructure.knowledge_ingestion import (  # noqa: E402
    ChunkingConfig,
    MarkdownKnowledgeChunker,
)
from customer_service_app.infrastructure.lexical_search import (  # noqa: E402
    build_lexical_retriever,
)
from customer_service_app.infrastructure.vector_store.factory import (  # noqa: E402
    build_vector_store,
)
from customer_service_app.services.knowledge_lifecycle_service import (  # noqa: E402
    KnowledgeLifecycleService,
)


async def ingest(directory: Path, tenant_id: str, *, prune_missing: bool = False) -> None:
    """Synchronize Markdown sources with vector, BM25, manifest, and cache state."""

    settings = get_settings()
    embedding_client = build_embedding_client(settings)
    vector_store = build_vector_store(settings)
    lexical_retriever = build_lexical_retriever(settings)
    semantic_cache = (
        RedisSemanticCache(settings, embedding_client)
        if settings.semantic_cache_enabled
        else None
    )
    chunker = MarkdownKnowledgeChunker(
        ChunkingConfig(
            max_chars=settings.knowledge_chunk_max_chars,
            min_chars=settings.knowledge_chunk_min_chars,
            overlap_chars=settings.knowledge_chunk_overlap_chars,
        )
    )

    # 目录 README 只是维护说明，不是面向用户的业务知识。
    paths = sorted(
        path for path in directory.rglob("*.md") if path.name.lower() != "readme.md"
    )
    documents: list[KnowledgeSourceDocument] = []
    for path in paths:
        source = path.relative_to(directory).as_posix()
        documents.append(
            KnowledgeSourceDocument(
                source=source,
                text=path.read_text(encoding="utf-8"),
                metadata={
                    "file_name": path.name,
                    "format": "markdown",
                    "corpus_version": settings.knowledge_corpus_version,
                },
            )
        )

    if not documents and not prune_missing:
        await _close_resource(semantic_cache)
        await _close_resource(lexical_retriever)
        await _close_resource(vector_store)
        await _close_resource(embedding_client)
        print("No markdown documents found; pass --prune-missing to reconcile an empty corpus.")
        return

    try:
        async with session_context() as session:
            service = KnowledgeLifecycleService(
                settings=settings,
                repository=KnowledgeDocumentRepository(session),
                chunker=chunker,
                embedding_client=embedding_client,
                vector_store=vector_store,
                lexical_retriever=lexical_retriever,
                semantic_cache=semantic_cache,
            )
            report = await service.sync(
                tenant_id=tenant_id,
                documents=documents,
                prune_missing=prune_missing,
            )
            await session.commit()
    finally:
        await _close_resource(semantic_cache)
        await _close_resource(lexical_retriever)
        await _close_resource(vector_store)
        await _close_resource(embedding_client)

    destinations = [settings.vector_store_provider]
    if lexical_retriever is not None:
        destinations.append("opensearch-bm25")
    print(
        f"Knowledge sync completed for {len(paths)} source documents in {', '.join(destinations)}: "
        f"indexed={len(report.indexed)}, unchanged={len(report.unchanged)}, "
        f"scheduled={len(report.scheduled)}, expired={len(report.expired)}, "
        f"deleted={len(report.deleted)}, cache_invalidated={report.invalidated_cache_entries}."
    )


async def _close_resource(resource: Any) -> None:
    """脚本退出前关闭已创建的 SDK 客户端。"""

    if resource is None:
        return
    close = getattr(resource, "close", None) or getattr(resource, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Markdown knowledge documents")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Delete indexed documents that are absent from this directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        ingest(
            args.directory,
            args.tenant_id,
            prune_missing=args.prune_missing,
        )
    )
