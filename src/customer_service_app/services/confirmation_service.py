from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.core.config import Settings, get_settings
from customer_service_app.core.exceptions import AppError
from customer_service_app.domain.schemas import PendingActionView
from customer_service_app.infrastructure.db.models import PendingAction
from customer_service_app.infrastructure.db.repositories import PendingActionRepository
from customer_service_app.infrastructure.mcp.approval import build_confirmation_id


class ConfirmationService:
    """Manage high-risk tool actions that need explicit confirmation."""

    def __init__(self, session: AsyncSession, settings: Settings | None = None):
        self._repo = PendingActionRepository(session)
        self._settings = settings or get_settings()

    async def create_pending_action(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        thread_id: str | None = None,
        confirmation_id: str | None = None,
        tool_name: str,
        arguments: dict,
    ) -> PendingActionView:
        resolved_confirmation_id = confirmation_id or build_confirmation_id(
            tool_name=tool_name,
            tenant_id=tenant_id,
            user_id=user_id,
            arguments=arguments,
        )
        action = await self._repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            confirmation_id=resolved_confirmation_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        return self._to_view(action)

    async def list_pending_actions(
        self,
        *,
        tenant_id: str,
        user_id: str,
        include_expired: bool = False,
    ) -> list[PendingActionView]:
        actions = await self._repo.list_pending(tenant_id=tenant_id, user_id=user_id)
        views = [self._to_view(action) for action in actions]
        if include_expired:
            return views
        return [view for view in views if not view.expired]

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
        if self._is_expired(action):
            raise AppError(
                "Pending action has expired",
                code="action_expired",
                status_code=409,
            )
        return action

    def _expires_at(self, action: PendingAction) -> datetime | None:
        """Return the expiration time for displaying pending-action state."""

        created_at = action.created_at
        if created_at is None:
            return None
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at + timedelta(seconds=self._settings.pending_action_ttl_seconds)

    def _is_expired(self, action: PendingAction) -> bool:
        """Return whether a pending action is older than the configured TTL."""

        expires_at = self._expires_at(action)
        if expires_at is None:
            return False
        return datetime.now(timezone.utc) > expires_at

    def _to_view(self, action: PendingAction) -> PendingActionView:
        expires_at = self._expires_at(action)
        return PendingActionView(
            id=action.id,
            tenant_id=action.tenant_id,
            user_id=action.user_id,
            conversation_id=action.conversation_id,
            thread_id=action.thread_id,
            confirmation_id=action.confirmation_id,
            tool_name=action.tool_name,
            arguments=dict(action.arguments_json or {}),
            status=action.status,
            comment=action.comment,
            created_at=action.created_at,
            expires_at=expires_at,
            expired=bool(expires_at and datetime.now(timezone.utc) > expires_at),
        )
