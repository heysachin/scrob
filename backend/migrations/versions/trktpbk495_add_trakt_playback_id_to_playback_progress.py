"""add trakt_playback_id to playback_progress

Revision ID: trktpbk495
Revises: trktpkce494
Create Date: 2026-10-22

Stores the Trakt playback ID on PlaybackProgress to allow correlating and safely
updating or cleaning up Trakt-synced continue-watching entries.
"""

from alembic import op
import sqlalchemy as sa


revision = "trktpbk495"
down_revision = "trktpkce494"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "playback_progress",
        sa.Column("trakt_playback_id", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("playback_progress", "trakt_playback_id")
