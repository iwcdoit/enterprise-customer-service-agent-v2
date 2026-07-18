from __future__ import annotations

from collections import Counter
from pathlib import Path

from customer_service_app.infrastructure.knowledge_ingestion import MarkdownKnowledgeChunker


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = ROOT / "knowledge_base"
EXPECTED_TYPES = {
    "faq",
    "policy",
    "sop",
    "manual",
    "troubleshooting",
    "benefit",
    "announcement",
    "matrix",
    "risk",
    "merchant_rule",
    "sla",
}


def test_knowledge_corpus_has_two_documents_for_each_supported_type() -> None:
    chunker = MarkdownKnowledgeChunker()
    type_counts: Counter[str] = Counter()
    all_chunk_ids: list[str] = []
    documents = [
        path
        for path in KNOWLEDGE_ROOT.rglob("*.md")
        if path.name.lower() != "readme.md"
    ]

    for path in documents:
        source = path.relative_to(KNOWLEDGE_ROOT).as_posix()
        chunks = chunker.chunk(text=path.read_text(encoding="utf-8"), source=source)
        assert chunks, f"knowledge document produced no chunks: {source}"
        document_types = {str(item.metadata["document_type"]) for item in chunks}
        assert len(document_types) == 1
        type_counts.update(document_types)
        all_chunk_ids.extend(item.id for item in chunks)

    assert len(documents) == 22
    assert set(type_counts) == EXPECTED_TYPES
    assert all(type_counts[item] == 2 for item in EXPECTED_TYPES)
    assert len(all_chunk_ids) == len(set(all_chunk_ids))


def test_retrieval_evaluation_set_references_existing_documents() -> None:
    import json

    queries = json.loads(
        (KNOWLEDGE_ROOT / "evaluation_queries.json").read_text(encoding="utf-8")
    )
    assert len(queries) >= 20
    for item in queries:
        assert (KNOWLEDGE_ROOT / item["expected_source"]).is_file()
        assert item["kind"] in {"exact", "semantic"}
