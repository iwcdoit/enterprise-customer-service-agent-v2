from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_tenant_id", "conversations", ["tenant_id"])
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("idx_conversation_tenant_user", "conversations", ["tenant_id", "user_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    op.create_table(
        "orders",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("order_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("logistics_company", sa.String(length=100), nullable=True),
        sa.Column("tracking_number", sa.String(length=100), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_orders_tenant_id", "orders", ["tenant_id"])
    op.create_index("ix_orders_user_id", "orders", ["user_id"])
    op.create_index("ix_orders_order_id", "orders", ["order_id"])
    op.create_index("idx_order_tenant_user_order", "orders", ["tenant_id", "user_id", "order_id"])

    op.create_table(
        "support_tickets",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index("ix_support_tickets_tenant_id", "support_tickets", ["tenant_id"])
    op.create_index("ix_support_tickets_user_id", "support_tickets", ["user_id"])
    op.create_index("ix_support_tickets_conversation_id", "support_tickets", ["conversation_id"])
    op.create_index("ix_support_tickets_idempotency_key", "support_tickets", ["idempotency_key"])

    op.create_table(
        "pending_actions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("arguments_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pending_actions_tenant_id", "pending_actions", ["tenant_id"])
    op.create_index("ix_pending_actions_user_id", "pending_actions", ["user_id"])
    op.create_index("ix_pending_actions_conversation_id", "pending_actions", ["conversation_id"])
    op.create_index("ix_pending_actions_tool_name", "pending_actions", ["tool_name"])
    op.create_index("ix_pending_actions_status", "pending_actions", ["status"])
    op.create_index("idx_pending_action_owner", "pending_actions", ["tenant_id", "user_id", "status"])

    op.create_table(
        "tenant_usage_daily",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("llm_calls", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tenant_usage_daily_tenant_id", "tenant_usage_daily", ["tenant_id"])
    op.create_index("ix_tenant_usage_daily_usage_date", "tenant_usage_daily", ["usage_date"])
    op.create_index(
        "idx_tenant_usage_day",
        "tenant_usage_daily",
        ["tenant_id", "usage_date"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("idx_tenant_usage_day", table_name="tenant_usage_daily")
    op.drop_index("ix_tenant_usage_daily_usage_date", table_name="tenant_usage_daily")
    op.drop_index("ix_tenant_usage_daily_tenant_id", table_name="tenant_usage_daily")
    op.drop_table("tenant_usage_daily")

    op.drop_index("idx_pending_action_owner", table_name="pending_actions")
    op.drop_index("ix_pending_actions_status", table_name="pending_actions")
    op.drop_index("ix_pending_actions_tool_name", table_name="pending_actions")
    op.drop_index("ix_pending_actions_conversation_id", table_name="pending_actions")
    op.drop_index("ix_pending_actions_user_id", table_name="pending_actions")
    op.drop_index("ix_pending_actions_tenant_id", table_name="pending_actions")
    op.drop_table("pending_actions")

    op.drop_index("ix_support_tickets_idempotency_key", table_name="support_tickets")
    op.drop_index("ix_support_tickets_conversation_id", table_name="support_tickets")
    op.drop_index("ix_support_tickets_user_id", table_name="support_tickets")
    op.drop_index("ix_support_tickets_tenant_id", table_name="support_tickets")
    op.drop_table("support_tickets")

    op.drop_index("idx_order_tenant_user_order", table_name="orders")
    op.drop_index("ix_orders_order_id", table_name="orders")
    op.drop_index("ix_orders_user_id", table_name="orders")
    op.drop_index("ix_orders_tenant_id", table_name="orders")
    op.drop_table("orders")

    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_table("messages")

    op.drop_index("idx_conversation_tenant_user", table_name="conversations")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_index("ix_conversations_tenant_id", table_name="conversations")
    op.drop_table("conversations")
