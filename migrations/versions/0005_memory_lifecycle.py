"""add long-term memory lifecycle tables

Revision ID: 0005_memory_lifecycle
Revises: 0004_pending_action_result
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0005_memory_lifecycle"
down_revision: str | None = "0004_pending_action_result"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("message_start_id", sa.String(length=64), nullable=True),
        sa.Column("message_end_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversation_summaries_tenant_id", "conversation_summaries", ["tenant_id"])
    op.create_index("ix_conversation_summaries_user_id", "conversation_summaries", ["user_id"])
    op.create_index(
        "ix_conversation_summaries_conversation_id",
        "conversation_summaries",
        ["conversation_id"],
    )

    op.create_table(
        "customer_memories",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("memory_key", sa.String(length=128), nullable=False),
        sa.Column("memory_value_json", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("verification_status", sa.String(length=32), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("sensitivity", sa.String(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "user_id",
            "memory_type",
            "memory_key",
            name="uq_customer_memory_owner_key",
        ),
    )
    op.create_index("ix_customer_memories_tenant_id", "customer_memories", ["tenant_id"])
    op.create_index("ix_customer_memories_user_id", "customer_memories", ["user_id"])
    op.create_index("ix_customer_memories_memory_type", "customer_memories", ["memory_type"])
    op.create_index("ix_customer_memories_memory_key", "customer_memories", ["memory_key"])
    op.create_index(
        "ix_customer_memories_verification_status",
        "customer_memories",
        ["verification_status"],
    )
    op.create_index(
        "idx_customer_memory_lookup",
        "customer_memories",
        ["tenant_id", "user_id", "memory_type", "memory_key"],
    )

    op.create_table(
        "memory_preferences",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column(
            "long_term_memory_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "user_id", name="uq_memory_preference_owner"),
    )
    op.create_index("ix_memory_preferences_tenant_id", "memory_preferences", ["tenant_id"])
    op.create_index("ix_memory_preferences_user_id", "memory_preferences", ["user_id"])


def downgrade() -> None:
    op.drop_table("memory_preferences")
    op.drop_table("customer_memories")
    op.drop_table("conversation_summaries")
