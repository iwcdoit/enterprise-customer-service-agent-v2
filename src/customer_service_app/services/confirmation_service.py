from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.schemas import PendingActionView
from customer_service_app.infrastructure.db.models import PendingAction
from customer_service_app.infrastructure.db.repositories import PendingActionRepository


class ConfirmationService:
    """Manage high-risk tool actions that need explicit confirmation."""

    def __init__(self, session: AsyncSession):
        self._repo = PendingActionRepository(session)

    async def create_pending_action(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        tool_name: str,
        arguments: dict,
    ) -> PendingActionView:
        action = await self._repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        return self._to_view(action)

    async def list_pending_actions(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> list[PendingActionView]:
        actions = await self._repo.list_pending(tenant_id=tenant_id, user_id=user_id)
        return [self._to_view(action) for action in actions]

    async def approve(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
        comment: str | None,
    ) -> PendingActionView:
        action = await self._get_pending_action(
            tenant_id=tenant_id,
            user_id=user_id,
            action_id=action_id,
        )
        action = await self._repo.mark_approved(action, comment=comment)
        return self._to_view(action)

    async def reject(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
        comment: str | None,
    ) -> PendingActionView:
        action = await self._get_pending_action(
            tenant_id=tenant_id,
            user_id=user_id,
            action_id=action_id,
        )
        action = await self._repo.mark_rejected(action, comment=comment)
        return self._to_view(action)

    async def _get_pending_action(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action_id: str,
    ) -> PendingAction:
        action = await self._repo.get_owned(
            tenant_id=tenant_id,
            user_id=user_id,
            action_id=action_id,
        )
        if action is None:
            raise AppError("Pending action not found", code="not_found", status_code=404)
        if action.status != "pending":
            raise AppError("Pending action has already been decided", code="action_decided")
        return action

    @staticmethod
    def _to_view(action: PendingAction) -> PendingActionView:
        return PendingActionView(
            id=action.id,
            tenant_id=action.tenant_id,
            user_id=action.user_id,
            conversation_id=action.conversation_id,
            tool_name=action.tool_name,
            arguments=dict(action.arguments_json or {}),
            status=action.status,
            comment=action.comment,
        )
