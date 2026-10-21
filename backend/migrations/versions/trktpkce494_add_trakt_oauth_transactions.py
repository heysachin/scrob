"""add trakt_oauth_transactions table for PKCE state persistence

Revision ID: trktpkce494
Revises: wesrc477
Create Date: 2026-10-22

Stores short-lived Trakt OAuth 2.0 PKCE transactions (state hash, code_verifier,
user_id, expiration) to support secretless Trakt connections.
"""

from alembic import op
import sqlalchemy as sa


revision = "trktpkce494"
down_revision = "wesrc477"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trakt_oauth_transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("code_verifier", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("redirect_uri", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("state_hash", name="uq_trakt_oauth_transactions_state_hash"),
    )
    op.create_index("ix_trakt_oauth_transactions_state_hash", "trakt_oauth_transactions", ["state_hash"])
    op.create_index("ix_trakt_oauth_transactions_user_id", "trakt_oauth_transactions", ["user_id"])
    op.create_index("ix_trakt_oauth_transactions_expires_at", "trakt_oauth_transactions", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_trakt_oauth_transactions_expires_at", table_name="trakt_oauth_transactions")
    op.drop_index("ix_trakt_oauth_transactions_user_id", table_name="trakt_oauth_transactions")
    op.drop_index("ix_trakt_oauth_transactions_state_hash", table_name="trakt_oauth_transactions")
    op.drop_table("trakt_oauth_transactions")
