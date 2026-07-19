from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.confirmations import PendingActionView
from customer_service_app.infrastructure.db.models import PendingAction
from customer_service_app.infrastructure.db.repositories import PendingActionRepository
from customer_service_app.infrastructure.mcp.approval import create_approval_token
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.services.business_gateway import BusinessGateway
from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry, ToolSpec


class ConfirmationService:
    """Create, approve, reject, and execute pending side-effect actions."""

    def __init__(
        self,
        *,
        settings: Settings,
        session: AsyncSession,
        tool_registry: ToolRegistry | None = None,
        search_client: SerpApiSearchClient | None = None,
        business_gateway: BusinessGateway | None = None,
    ):
        self._settings = settings
        self._session = session
        self._repo = PendingActionRepository(session)
        self._tool_registry = tool_registry
        self._search_client = search_client
        self._business_gateway = business_gateway

    async def list_pending_actions(
        self,
        *,
        tenant_id: str,
        user_id: str,
        include_expired: bool = False,
        limit: int = 50,
    ) -> list[PendingActionView]:
        """查询用户待确认动作，供运营台和会话恢复使用。"""
        actions = await self._repo.list_pending(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
        )
        views: list[PendingActionView] = []
        for action in actions:
            expired = self._is_expired(action)
            if expired and not include_expired:
                continue
            views.append(self.to_view(action))
        return views

    async def create_for_tool(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        tool: ToolSpec,
        arguments: dict[str, Any],
        langgraph_thread_id: str | None = None,
    ) -> PendingAction:
        """Persist a side-effect tool call that must wait for user approval."""
        expire_at = datetime.now(timezone.utc) + timedelta(
            seconds=self._settings.pending_action_ttl_seconds
        )

        idempotency_key = self._build_idempotency_key(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            tool_name=tool.name,
            arguments=arguments,
        )

        existing = await self._repo.get_by_idempotency_key(idempotency_key=idempotency_key)
        if existing is not None:
            return existing

        confirmation_prompt = self._build_confirmation_prompt(tool=tool, arguments=arguments)

        return await self._repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            langgraph_thread_id=langgraph_thread_id,
            action_type=tool.action_type,
            tool_name=tool.name,
            arguments=arguments,
            risk_level=tool.risk_level,
            confirmation_prompt=confirmation_prompt,
            expire_at=expire_at,
            idempotency_key=idempotency_key,
        )

    async def get_owned(self, *, tenant_id: str, user_id: str, action_id: str) -> PendingAction:
        """Load a pending action or raise a domain error."""
        action = await self._repo.get_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            action_id=action_id,
        )
        if action is None:
            raise AppError("Confirmation not found", code="confirmation_not_found", status_code=404)
        return action

    async def approve(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        """Approve and execute one pending action."""
        action = await self.get_owned(tenant_id=tenant_id, user_id=user_id, action_id=action_id)

        self._ensure_pending_and_not_expired(action)

        await self._repo.mark_approved(action)

        approval_token: str | None = None
        if self._settings.mcp_after_sales_enabled:
            approval_token = create_approval_token(
                settings=self._settings,
                confirmation_id=action.id,
                tenant_id=tenant_id,
                user_id=user_id,
                tool_name=action.tool_name,
            )

        if self._tool_registry is None or self._business_gateway is None:
            raise AppError(
                "Confirmation executor is not configured",
                code="confirmation_executor_unavailable",
                status_code=503,
            )
        context = ToolExecutionContext(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=action.conversation_id,
            session=self._session,
            search_client=self._search_client,
            business_gateway=self._business_gateway,
            approval_token=approval_token,
            langgraph_thread_id=action.langgraph_thread_id,
            confirmation_id=action.id,
        )

        try:
            result = await self._tool_registry.execute_dict(
                name=action.tool_name,
                arguments=action.arguments_json,
                context=context,
            )
        except Exception as exc:
            await self._repo.mark_failed(action, {"error": str(exc)})
            raise

        await self._repo.mark_executed(action, result)
        return result

    async def reject(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        """Reject one pending action without executing the side effect."""
        action = await self.get_owned(tenant_id=tenant_id, user_id=user_id, action_id=action_id)
        self._ensure_pending_and_not_expired(action)

        await self._repo.mark_rejected(action)
        return {
            "confirmation_id": action.id,
            "status": "rejected",
            "tool_name": action.tool_name,
        }

    def to_view(self, action: PendingAction) -> PendingActionView:
        """Convert ORM model to API view."""
        return PendingActionView(
            id=action.id,
            tenant_id=action.tenant_id,
            user_id=action.user_id,
            conversation_id=action.conversation_id,
            action_type=action.action_type,
            tool_name=action.tool_name,
            arguments=action.arguments_json or {},
            status=action.status,
            risk_level=action.risk_level,
            confirmation_prompt=action.confirmation_prompt,
            thread_id=action.langgraph_thread_id,
            confirmation_id=action.id,
            result=action.execution_result_json or {},
            created_at=action.created_at,
            expires_at=action.expire_at,
            expired=self._is_expired(action),
        )

    def _is_expired(self, action: PendingAction) -> bool:
        """判断确认动作是否过期。

        新模型直接使用 expire_at；对旧数据或轻量测试对象，则按 created_at + TTL 计算。
        """
        expire_at = getattr(action, "expire_at", None)
        if expire_at is None:
            created_at = getattr(action, "created_at", None)
            if created_at is None:
                return False
            expire_at = created_at + timedelta(
                seconds=self._settings.pending_action_ttl_seconds
            )
        return expire_at < datetime.now(timezone.utc)

    @staticmethod
    def _build_idempotency_key(
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Build a stable idempotency key for duplicate-click protection."""
        raw = json.dumps(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "tool_name": tool_name,
                "arguments": arguments,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_confirmation_prompt(*, tool: ToolSpec, arguments: dict[str, Any]) -> str:
        """Build user-facing confirmation text for a side-effect action."""
        order_id = str(arguments.get("order_id") or "未提供")
        reason = str(arguments.get("reason") or arguments.get("detail") or "未提供")
        return (
            f"即将执行高风险操作：{tool.description}\n"
            f"- 工具名称：{tool.name}\n"
            f"- 操作类型：{tool.action_type}\n"
            f"- 风险等级：{tool.risk_level}\n"
            f"- 订单号：{order_id}\n"
            f"- 原因：{reason}\n\n"
            "该操作可能创建退款、补偿、换货或转人工工单。请确认是否继续提交。"
        )

    def _ensure_pending_and_not_expired(self, action: PendingAction) -> None:
        """Validate action state before approval or rejection."""
        if action.status != "pending":
            raise AppError(
                "Confirmation is not pending",
                code="confirmation_not_pending",
                status_code=409,
            )

        if self._is_expired(action):
            raise AppError(
                "Confirmation has expired",
                code="confirmation_expired",
                status_code=409,
            )
