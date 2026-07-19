from __future__ import annotations

import uuid
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import GraphStateView, GraphTaskView
from customer_service_app.workflows.context import CustomerServiceGraphContext
from customer_service_app.workflows.customer_service_graph import build_customer_service_graph


class CompiledCustomerServiceGraph(Protocol):
    """The small part of LangGraph's compiled graph API used by this runtime.

    LangGraph 的 compiled graph 是动态对象，PyCharm 不一定能自动推断出
    `ainvoke` 和 `aget_state`。这里用 Protocol 显式告诉 IDE：
    self._graph 至少支持这两个方法。
    """

    async def ainvoke(self, input: Any, **kwargs: Any) -> dict[str, Any]: ...

    async def aget_state(self, config: dict[str, Any]) -> Any: ...


class CustomerServiceGraphRuntime:
    """Own the compiled graph and the lifecycle of its checkpointer.

    Runtime 这一层不写客服业务逻辑，它只负责管理 LangGraph 的运行时能力：

    1. 应用启动时创建 checkpointer 和 compiled graph。
    2. 普通聊天时调用 `ainvoke` 启动 graph。
    3. HIL 确认后调用 `Command(resume=...)` 恢复同一个 graph。
    4. 运营台排查时读取 checkpoint 快照。
    5. 应用关闭时释放 Postgres checkpointer 连接。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        graph: CompiledCustomerServiceGraph,
        checkpointer_context: AbstractAsyncContextManager[Any] | None = None,
    ):
        self._settings: Settings = settings

        self._graph: CompiledCustomerServiceGraph = graph

        self._checkpointer_context: AbstractAsyncContextManager[Any] | None = checkpointer_context

    @classmethod
    async def create(cls, settings: Settings) -> "CustomerServiceGraphRuntime":
        """Create a runtime with memory or PostgreSQL checkpoint persistence."""

        checkpointer_context: AbstractAsyncContextManager[Any] | None = None

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

        graph = build_customer_service_graph(checkpointer)
        return cls(
            settings=settings,
            graph=graph,
            checkpointer_context=checkpointer_context,
        )

    async def close(self) -> None:
        """Release the checkpointer connection when the FastAPI app shuts down."""

        if self._checkpointer_context is None:
            return

        await self._checkpointer_context.__aexit__(None, None, None)

        self._checkpointer_context = None

    async def invoke(
        self,
        *,
        request_payload: dict[str, Any],
        context: CustomerServiceGraphContext,
        thread_id: str | None,
    ) -> dict[str, Any]:
        """Start a new graph turn or continue a normal non-interrupted thread."""

        resolved_thread_id = thread_id or str(uuid.uuid4())

        config = self._config(resolved_thread_id, request_payload)

        initial_state = {
            "request": request_payload,
            "thread_id": resolved_thread_id,
        }

        return await self._graph.ainvoke(
            initial_state,
            config=config,
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
        """Resume the exact graph thread that previously stopped at interrupt."""

        resume_command = Command(resume=decision)

        return await self._graph.ainvoke(
            resume_command,
            config=self._config(thread_id),
            context=context,
            durability=self._durability(),
        )

    async def get_state(self, *, thread_id: str) -> GraphStateView:
        """Return a sanitized checkpoint snapshot for the operations console."""

        snapshot = await self._graph.aget_state(self._config(thread_id))
        values = dict(snapshot.values or {})

        safe_keys = {
            "conversation_id",
            "run_id",
            "status",
            "plan",
            "plan_cursor",
            "tool_cursor",
            "tool_results",
            "pending_confirmation",
            "confirmation_decision",
            "final_answer",
            "cache_hit",
            "query_rewrite",
            "retrieval_quality",
            "rewrite_attempts",
            "trace",
        }
        safe_values = {key: values[key] for key in safe_keys if key in values}

        tasks: list[GraphTaskView] = []
        for task in snapshot.tasks or ():
            interrupts: list[dict[str, Any]] = []
            for item in getattr(task, "interrupts", ()) or ():
                interrupts.append(
                    {
                        "id": str(getattr(item, "id", "")),
                        "value": getattr(item, "value", None),
                    }
                )
            tasks.append(
                GraphTaskView(
                    id=str(getattr(task, "id", "")),
                    name=str(getattr(task, "name", "")),
                    interrupts=interrupts,
                )
            )

        next_nodes = [str(item) for item in (snapshot.next or ())]
        status = str(values.get("status") or ("running" if next_nodes else "completed"))

        return GraphStateView(
            thread_id=thread_id,
            status=status,
            next_nodes=next_nodes,
            values=safe_values,
            tasks=tasks,
        )

    def _config(
        self,
        thread_id: str,
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the LangGraph RunnableConfig used by invoke/resume/get_state."""

        payload = request_payload or {}
        tenant_id = str(payload.get("tenant_id") or "unknown")
        user_id = str(payload.get("user_id") or "unknown")

        return {
            "configurable": {"thread_id": thread_id},

            "recursion_limit": self._settings.graph_recursion_limit,

            "run_name": "customer-service-turn",
            "tags": ["customer-service", f"tenant:{tenant_id}"],
            "metadata": {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
            },
        }

    def _durability(self) -> Any:
        """Return LangGraph durability mode from configuration."""

        return cast(Any, self._settings.graph_durability)
