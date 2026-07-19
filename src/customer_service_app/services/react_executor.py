from __future__ import annotations

from typing import Any

from customer_service_app.domain.planning import PlanStep
from customer_service_app.services.rag_service import RagService
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry


class ReactExecutor:
    """Execute one plan step and return an observation."""

    def __init__(self, *, rag_service: RagService, tool_registry: ToolRegistry):
        self._rag_service = rag_service
        self._tool_registry = tool_registry

    async def execute_step(
        self,
        *,
        tenant_id: str,
        question: str,
        step: PlanStep,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Run one RAG/tool/LLM-like step and return a structured observation."""

        if step.action_type == "rag":
            chunks = await self._rag_service.retrieve(tenant_id=tenant_id, question=question)
            return {
                "step_id": step.step_id,
                "action_type": "rag",
                "chunk_count": len(chunks),
                "chunks": [item.model_dump() for item in chunks],
            }
        if step.action_type == "tool" and step.tool_name:
            result = await self._tool_registry.execute_dict(
                name=step.tool_name,
                arguments=step.arguments,
                context=context,
            )
            return {
                "step_id": step.step_id,
                "action_type": "tool",
                "tool_name": step.tool_name,
                "result": result,
            }
        return {
            "step_id": step.step_id,
            "action_type": step.action_type,
            "result": "step_deferred_to_agent",
        }
