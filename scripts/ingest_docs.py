from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_service_app.core.config import get_settings  # noqa: E402
from customer_service_app.infrastructure.embeddings.factory import (  # noqa: E402
    build_embedding_client,
)
from customer_service_app.infrastructure.knowledge_ingestion import (  # noqa: E402
    ChunkingConfig,
    MarkdownKnowledgeChunker,
)
from customer_service_app.infrastructure.vector_store.factory import (  # noqa: E402
    build_vector_store,
)


async def ingest(directory: Path, tenant_id: str) -> None:
    """结构化切分 Markdown，批量向量化并写入租户隔离的向量库。"""

    settings = get_settings()
    embedding_client = build_embedding_client(settings)
    vector_store = build_vector_store(settings)
    chunker = MarkdownKnowledgeChunker(
        ChunkingConfig(
            max_chars=settings.knowledge_chunk_max_chars,
            min_chars=settings.knowledge_chunk_min_chars,
            overlap_chars=settings.knowledge_chunk_overlap_chars,
        )
    )

    chunks = []
    for path in sorted(directory.rglob("*.md")):
        source = path.relative_to(directory).as_posix()
        chunks.extend(
            chunker.chunk(
                text=path.read_text(encoding="utf-8"),
                source=source,
                document_metadata={"file_name": path.name, "format": "markdown"},
            )
        )

    if not chunks:
        print("No markdown documents found.")
        return

    batch_size = max(settings.knowledge_ingest_batch_size, 1)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = await embedding_client.embed_documents([item.content for item in batch])
        await vector_store.upsert_chunks(
            tenant_id=tenant_id,
            chunks=batch,
            vectors=vectors,
        )

    print(
        f"Ingested {len(chunks)} structured chunks from "
        f"{len(list(directory.rglob('*.md')))} documents into "
        f"{settings.vector_store_provider}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Markdown knowledge documents")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tenant-id", default="default")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest(args.directory, args.tenant_id))
