from __future__ import annotations

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.services.rag_service import RagService


def build_service(*, retrieval_mode: str = "hybrid") -> RagService:
    return RagService(
        settings=Settings(retrieval_mode=retrieval_mode),
        embedding_client=object(),  # type: ignore[arg-type]
        vector_store=object(),  # type: ignore[arg-type]
    )


def test_policy_query_without_evidence_recommends_one_rewrite() -> None:
    quality = build_service().evaluate_quality(
        question="七天无理由退货政策是什么？",
        chunks=[],
    )

    assert quality.sufficient is False
    assert quality.retry_recommended is True


def test_realtime_order_query_does_not_require_rag_evidence() -> None:
    quality = build_service().evaluate_quality(
        question="查询订单 202607180001 现在的物流状态",
        chunks=[],
    )

    assert quality.sufficient is True
    assert quality.reason == "business_query_routed_to_tools"


def test_dense_retrieval_respects_score_gate() -> None:
    quality = build_service(retrieval_mode="dense").evaluate_quality(
        question="退货政策",
        chunks=[
            KnowledgeChunk(
                id="k1",
                source="policy.md",
                title="退货",
                content="退货政策",
                score=0.2,
            )
        ],
    )

    assert quality.sufficient is False
