"""Knowledge document parsing and chunking utilities."""

from customer_service_app.infrastructure.knowledge_ingestion.markdown_chunker import (
    ChunkingConfig,
    MarkdownKnowledgeChunker,
)

__all__ = ["ChunkingConfig", "MarkdownKnowledgeChunker"]
