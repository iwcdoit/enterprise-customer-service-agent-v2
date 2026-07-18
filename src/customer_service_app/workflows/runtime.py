from __future__ import annotations

import uuid
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import GraphStateView, GraphTaskView
from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.customer_service_graph import build_customer_service_graph


class CustomerServiceGraphRuntime:
    """持有应用级编译图和 Checkpointer 生命周期。"""

    def __init__(
        self,
        *,
        settings: Settings,
        graph: Any,
        checkpointer_context: AbstractAsyncContextManager | None = None,
    ) -> None:
        self._settings = settings
        self._graph = graph
        self._checkpointer_context = checkpointer_context

    @classmethod
    async def create(cls, settings: Settings) -> "CustomerServiceGraphRuntime":
        """创建内存 Checkpointer，或连接可跨进程恢复的 PostgreSQL Checkpointer。"""

        checkpointer_context: AbstractAsyncContextManager | None = None
        if settings.graph_checkpointer == "postgres":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            connection_string = settings.require(
                "GRAPH_CHECKPOINT_POSTGRES_URL",
                settings.graph_checkpoint_postgres_url,
            )
            checkpointer_context = AsyncPostgresSaver.from_conn_string(connection_string)
            checkpointer = await checkpointer_context.__aenter__()
            if settings.graph_checkpoint_setup:
                await checkpointer.setup()
        else:
            checkpointer = InMemorySaver()

        return cls(
            settings=settings,
            graph=build_customer_service_graph(checkpointer),
            checkpointer_context=checkpointer_context,
        )

    async def close(self) -> None:
        """应用关闭时释放 PostgreSQL Checkpointer 连接。"""

        if self._checkpointer_context is not None:
            await self._checkpointer_context.__aexit__(None, None, None)

    async def invoke(
        self,
        *,
        request_payload: dict[str, Any],
        context: CustomerServiceGraphContext,
        thread_id: str | None,
    ) -> dict[str, Any]:
        """在一个持久化 Graph thread 中启动新的客服请求。"""

        resolved_thread_id = thread_id or str(uuid.uuid4())
        return await self._graph.ainvoke(
            {
                "request": request_payload,
                "thread_id": resolved_thread_id,
                "status": "running",
                "trace": [],
                "error": None,
            },
            config=self._config(resolved_thread_id, request_payload),
            context=context,
            durability=self._durability(),
        )

    async def resume(
        self,
        *,
        thread_id: str,
        decision: dict[str, Any],
        context: CustomerServiceGraphContext,
    ) -> dict[str, Any]:
        """用 ``Command(resume=...)`` 恢复指定的 HIL 中断线程。"""

        return await self._graph.ainvoke(
            Command(resume=decision),
            config=self._config(thread_id),
            context=context,
            durability=self._durability(),
        )

    async def get_state(self, *, thread_id: str) -> GraphStateView:
        """返回脱敏后的 checkpoint 快照，供前端和运营排查。"""

        snapshot = await self._graph.aget_state(self._config(thread_id))
        values = dict(snapshot.values)
        visible_keys = (
            "conversation_id",
            "status",
            "route",
            "plan",
            "plan_execution",
            "pending_confirmations",
            "confirmation_cursor",
            "response",
            "trace",
            "error",
        )
        safe_values = {key: values.get(key) for key in visible_keys if key in values}
        tasks = [
            GraphTaskView(
                id=task.id,
                name=task.name,
                interrupts=[{"id": item.id, "value": item.value} for item in task.interrupts],
            )
            for task in snapshot.tasks
        ]
        status = str(values.get("status") or ("completed" if not snapshot.next else "running"))
        return GraphStateView(
            thread_id=thread_id,
            status=status,
            next_nodes=list(snapshot.next),
            values=safe_values,
            tasks=tasks,
        )

    def _config(
        self,
        thread_id: str,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = request_payload or {}
        tenant_id = str(payload.get("tenant_id") or "")
        user_id = str(payload.get("user_id") or "")
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.graph_recursion_limit,
            "run_name": "customer-service-turn",
            "tags": [
                "customer-service",
                f"tenant:{tenant_id}" if tenant_id else "tenant:resume",
            ],
            "metadata": {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
            },
        }

    def _durability(self) -> Any:
        return cast(Any, self._settings.graph_durability)
