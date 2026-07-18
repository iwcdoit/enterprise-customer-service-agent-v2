from __future__ import annotations

from customer_service_app.infrastructure.knowledge_ingestion import (
    ChunkingConfig,
    MarkdownKnowledgeChunker,
)


def test_markdown_chunker_preserves_heading_hierarchy() -> None:
    text = """# 售后服务

## 退款政策

用户签收后七天内可以申请退货。商品需要保持完整，不影响二次销售。

退款审核通过后原路退回。
"""

    chunks = MarkdownKnowledgeChunker().chunk(text=text, source="policy/refund.md")

    assert len(chunks) == 1
    assert chunks[0].title == "售后服务 / 退款政策"
    assert chunks[0].content.startswith("# 售后服务\n## 退款政策")
    assert chunks[0].metadata["heading_path"] == ["售后服务", "退款政策"]
    assert chunks[0].metadata["document_type"] == "policy"


def test_markdown_chunker_keeps_faq_question_and_answer_together() -> None:
    text = """---
type: faq
audience: customer
---
# 常见问题

Q：退款多久到账？

A：微信和支付宝通常一到三个工作日到账。

Q：可以修改退款账户吗？

A：退款默认原路退回，不能由客服直接修改收款账户。
"""

    chunks = MarkdownKnowledgeChunker(
        ChunkingConfig(max_chars=260, min_chars=20, overlap_chars=30)
    ).chunk(text=text, source="faq/refund.md")

    assert chunks
    joined = "\n".join(item.content for item in chunks)
    assert "Q：退款多久到账？\n\nA：微信和支付宝" in joined
    assert all(item.metadata["document_type"] == "faq" for item in chunks)
    assert all(item.metadata["audience"] == "customer" for item in chunks)


def test_markdown_chunker_splits_long_paragraph_on_sentences_with_overlap() -> None:
    sentences = [f"第{index}条规则用于说明退款处理边界。" for index in range(1, 25)]
    text = "# 退款规则\n\n" + "".join(sentences)
    chunker = MarkdownKnowledgeChunker(
        ChunkingConfig(max_chars=240, min_chars=20, overlap_chars=45)
    )

    chunks = chunker.chunk(text=text, source="policy/long.md")

    assert len(chunks) >= 2
    assert all(len(item.content) <= 300 for item in chunks)
    assert any(item.metadata["overlap_chars"] > 0 for item in chunks[1:])
    assert all(item.content.endswith(("。", "规则")) for item in chunks)


def test_markdown_chunk_ids_are_stable_for_same_source_and_content() -> None:
    chunker = MarkdownKnowledgeChunker()
    text = "# 物流\n\n订单发货后可通过运单号查询物流轨迹。"

    first = chunker.chunk(text=text, source="guide/logistics.md")
    second = chunker.chunk(text=text, source="guide/logistics.md")

    assert [item.id for item in first] == [item.id for item in second]
    assert first[0].metadata["document_id"] == second[0].metadata["document_id"]
