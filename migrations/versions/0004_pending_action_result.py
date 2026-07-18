"""persist graph confirmation execution results

Revision ID: 0004_pending_action_result
Revises: 0003_human_support
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_pending_action_result"
down_revision = "0003_human_support"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pending_actions",
        sa.Column("result_json", sa.JSON(), nullable=False, server_default=sa.text("('{}')")),
    )
    op.add_column(
        "pending_actions",
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pending_actions", "error_message")
    op.drop_column("pending_actions", "result_json")
