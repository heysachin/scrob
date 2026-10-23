"""add push_on_pull to media_server_connections

Revision ID: pushonpull496
Revises: trktpbk495
Create Date: 2026-10-23

Adds push_on_pull flag to media_server_connections to allow automatically pushing
watched history and playback progress to Nuvio when inbound pull events occur.
"""

from alembic import op
import sqlalchemy as sa


revision = "pushonpull496"
down_revision = "trktpbk495"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "media_server_connections",
        sa.Column("push_on_pull", sa.Boolean(), nullable=False, server_default="true"),
    )


def downgrade() -> None:
    op.drop_column("media_server_connections", "push_on_pull")
