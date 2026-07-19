from __future__ import annotations

from time import perf_counter
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.infrastructure.db.models import AgentRun
from customer_service_app.infrastructure.db.repositories import AgentRunRepository


class TraceService:
    """保存一次 Agent 请求的运行轨迹。

    run 记录一轮请求的总体状态；step 记录每个 Graph 节点发生了什么。
    这不是给大模型看的数据，而是给调试、运营台、监控排查看的数据。
    """

    def __init__(self, session: AsyncSession):
        self._repo = AgentRunRepository(session)

    async def start_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        request_id: str | None,
    ) -> tuple[AgentRun, float]:
        """创建 run 记录，并返回一个本地计时起点。"""
        run = await self._repo.create_run(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=request_id,
        )
        return run, perf_counter()

    async def step(
        self,
        *,
        run_id: str,
        stage: str,
        name: str,
        status: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        """追加一条节点执行记录。"""
        await self._repo.add_step(
            run_id=run_id,
            stage=stage,
            name=name,
            status=status,
            output_payload=output or {},
        )

    async def get_run(self, *, run_id: str) -> AgentRun | None:
        """按 id 读取 run，供 Graph 节点恢复或结束时使用。"""
        return await self._repo.get_run(run_id=run_id)

    async def finish_success(
        self,
        *,
        run: AgentRun,
        timer_start: float,
        model: str | None,
        total_tokens: int,
    ) -> None:
        """把 run 标记为成功完成。"""
        latency_ms = self._elapsed_ms(timer_start)
        await self._repo.mark_success(
            run=run,
            model=model,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    async def finish_failed(
        self,
        *,
        run: AgentRun,
        timer_start: float,
        error_code: str,
        error_message: str,
    ) -> None:
        """把 run 标记为失败。"""
        latency_ms = self._elapsed_ms(timer_start)
        await self._repo.mark_failed(
            run=run,
            error_code=error_code,
            error_message=error_message,
            latency_ms=latency_ms,
        )

    async def mark_waiting(self, *, run: AgentRun) -> None:
        """Graph interrupt 前把 run 标成等待确认。"""
        await self._repo.mark_waiting(run=run)

    async def mark_running(self, *, run: AgentRun) -> None:
        """Graph resume 后把 run 从等待恢复为运行中。"""
        await self._repo.mark_running(run=run)

    @staticmethod
    def _elapsed_ms(timer_start: float) -> float | None:
        """根据 perf_counter 起点计算毫秒耗时。"""
        if timer_start <= 0:
            return None
        return round((perf_counter() - timer_start) * 1000, 2)
