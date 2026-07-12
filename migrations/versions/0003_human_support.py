from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0003_human_support"
down_revision: str | None = "0002_pending_action_confirmation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("service_mode", sa.String(length=32), nullable=False, server_default="bot"),
    )
    op.create_index("ix_conversations_service_mode", "conversations", ["service_mode"])

    op.create_table(
        "human_handoff_sessions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(length=64),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column("support_ticket_id", sa.String(length=64), nullable=True),
        sa.Column("origin_thread_id", sa.String(length=128), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="waiting_assignment",
        ),
        sa.Column("queue_name", sa.String(length=64), nullable=False, server_default="general"),
        sa.Column("priority", sa.String(length=32), nullable=False, server_default="normal"),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("assigned_agent_id", sa.String(length=64), nullable=True),
        sa.Column("resolution_code", sa.String(length=64), nullable=True),
        sa.Column("resolution_summary", sa.Text(), nullable=True),
        sa.Column("resolution_metadata_json", sa.JSON(), nullable=False),
        sa.Column("next_mode", sa.String(length=32), nullable=True),
        sa.Column("idempotency_key", sa.String(length=160), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("requested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("idempotency_key", name="uq_handoff_idempotency_key"),
    )
    op.create_index("ix_handoff_tenant", "human_handoff_sessions", ["tenant_id"])
    op.create_index("ix_handoff_user", "human_handoff_sessions", ["user_id"])
    op.create_index("ix_handoff_conversation_id", "human_handoff_sessions", ["conversation_id"])
    op.create_index("ix_handoff_support_ticket_id", "human_handoff_sessions", ["support_ticket_id"])
    op.create_index("ix_handoff_origin_thread_id", "human_handoff_sessions", ["origin_thread_id"])
    op.create_index("ix_handoff_status", "human_handoff_sessions", ["status"])
    op.create_index("ix_handoff_queue_name", "human_handoff_sessions", ["queue_name"])
    op.create_index("ix_handoff_priority", "human_handoff_sessions", ["priority"])
    op.create_index("ix_handoff_assigned_agent_id", "human_handoff_sessions", ["assigned_agent_id"])
    op.create_index("ix_handoff_idempotency_key", "human_handoff_sessions", ["idempotency_key"])
    op.create_index(
        "idx_handoff_queue",
        "human_handoff_sessions",
        ["tenant_id", "status", "priority", "requested_at"],
    )
    op.create_index(
        "idx_handoff_conversation",
        "human_handoff_sessions",
        ["tenant_id", "conversation_id", "status"],
    )


def downgrade() -> None:
    op.drop_table("human_handoff_sessions")
    op.drop_index("ix_conversations_service_mode", table_name="conversations")
    op.drop_column("conversations", "service_mode")
