from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# 让脚本可以直接导入 src/customer_service_app 里的项目代码。

from customer_service_app.core.config import get_settings  # noqa: E402
from customer_service_app.domain.schemas import KnowledgeChunk  # noqa: E402
from customer_service_app.infrastructure.embeddings.factory import build_embedding_client  # noqa: E402
from customer_service_app.infrastructure.vector_store.factory import build_vector_store  # noqa: E402


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """把长文档切成适合 embedding 的小块。

    `overlap` 表示相邻 chunk 保留一部分重叠内容，避免答案刚好被切断。
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, 0)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


async def ingest(directory: Path, tenant_id: str) -> None:
    """导入 Markdown 知识库到配置的向量库。

    流程：读取 md -> 切 chunk -> 调 embedding -> 写入向量库。
    这就是 RAG 项目里“离线知识入库”的最小生产流程。
    """
    settings = get_settings()
    embedding_client = build_embedding_client(settings)
    vector_store = build_vector_store(settings)

    chunks: list[KnowledgeChunk] = []
    for path in directory.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for index, chunk_text in enumerate(split_text(text)):
            raw_id = f"{tenant_id}:{path}:{index}"
            chunks.append(
                KnowledgeChunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id)),
                    source=str(path),
                    title=path.stem,
                    content=chunk_text,
                    score=1.0,
                    metadata={"chunk_index": index},
                )
            )

    if not chunks:
        print("No markdown documents found.")
        return

    vectors = await embedding_client.embed_documents([chunk.content for chunk in chunks])
    await vector_store.upsert_chunks(tenant_id=tenant_id, chunks=chunks, vectors=vectors)
    print(
        "Ingested "
        f"{len(chunks)} chunks into {settings.vector_store_provider} vector store."
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tenant-id", default="default")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest(args.directory, args.tenant_id))
