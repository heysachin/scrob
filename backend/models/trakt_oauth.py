"""Trakt OAuth 2.0 PKCE transactions table for server-side state persistence."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class TraktOAuthTransaction(Base):
    """Temporary server-side transaction preserving the PKCE code_verifier between
    the authorization initiation and the redirect callback.
    """

    __tablename__ = "trakt_oauth_transactions"

    id            : Mapped[int]                 = mapped_column(Integer, primary_key=True)
    state_hash    : Mapped[str]                 = mapped_column(String(64), unique=True, nullable=False, index=True)
    code_verifier : Mapped[str]                 = mapped_column(String(255), nullable=False)
    user_id       : Mapped[int]                 = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    client_id     : Mapped[str]                 = mapped_column(String(255), nullable=False)
    redirect_uri  : Mapped[str]                 = mapped_column(String(500), nullable=False)
    created_at    : Mapped[datetime]            = mapped_column(DateTime, server_default=func.now(), nullable=False)
    expires_at    : Mapped[datetime]            = mapped_column(DateTime, nullable=False, index=True)
    consumed_at   : Mapped[Optional[datetime]]  = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="trakt_oauth_transactions")
