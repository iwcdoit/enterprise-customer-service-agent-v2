from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0002_pending_action_confirmation"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("pending_actions", sa.Column("thread_id", sa.String(length=64), nullable=True))
    op.add_column(
        "pending_actions",
        sa.Column("confirmation_id", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_pending_actions_thread_id", "pending_actions", ["thread_id"])
    op.create_index("ix_pending_actions_confirmation_id", "pending_actions", ["confirmation_id"])


def downgrade() -> None:
    op.drop_index("ix_pending_actions_confirmation_id", table_name="pending_actions")
    op.drop_index("ix_pending_actions_thread_id", table_name="pending_actions")
    op.drop_column("pending_actions", "confirmation_id")
    op.drop_column("pending_actions", "thread_id")
