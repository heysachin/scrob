"""Trakt OAuth 2.0 PKCE (RFC 7636) utilities."""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from core.config import settings as app_settings

logger = logging.getLogger(__name__)

TRAKT_AUTH_URL = "https://auth.trakt.tv/oauth/authorize"
TRAKT_TOKEN_URL = "https://api.trakt.tv/oauth/token"
TIMEOUT = 30.0


def _b64url(data: bytes) -> str:
    """Base64-URL encode bytes without padding."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def generate_code_verifier() -> str:
    """Generate a cryptographically secure random PKCE code verifier.

    RFC 7636 Section 4.1 specifies unreserved characters [A-Z], [a-z], [0-9],
    '-', '.', '_', '~', with length 43-128 characters.
    secrets.token_urlsafe(64) produces 86 URL-safe characters.
    """
    return secrets.token_urlsafe(64)


def generate_code_challenge(verifier: str) -> str:
    """Generate an S256 code challenge from a code verifier.

    code_challenge = BASE64URL-ENCODE(SHA256(ASCII(code_verifier))) without padding.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _b64url(digest)


def generate_oauth_state() -> str:
    """Generate a cryptographically secure random OAuth state."""
    return secrets.token_urlsafe(32)


def hash_oauth_state(state: str) -> str:
    """Compute the SHA-256 hex digest of the OAuth state for safe database storage."""
    return hashlib.sha256(state.encode("ascii")).hexdigest()


def get_trakt_redirect_uri() -> str:
    """Derive the trusted Trakt OAuth PKCE callback URL.

    Uses TRAKT_REDIRECT_URL if explicitly configured; otherwise uses
    SCROB_PUBLIC_URL or SERVER_URL.
    """
    if getattr(app_settings, "trakt_redirect_url", None):
        return app_settings.trakt_redirect_url
    base = getattr(app_settings, "scrob_public_url", None) or app_settings.server_url
    return f"{base.rstrip('/')}/api/proxy/trakt/auth/pkce/callback"


def build_authorization_url(
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
) -> str:
    """Build the Trakt OAuth 2.0 PKCE authorization URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{TRAKT_AUTH_URL}?{urlencode(params)}"


class TraktOAuthError(Exception):
    """Exception raised when Trakt OAuth authorization or token exchange fails."""


async def exchange_code_for_token(
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Exchange a PKCE authorization code for Trakt access and refresh tokens.

    Does not send client_secret.
    """
    payload = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            TRAKT_TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        if not resp.is_success:
            logger.warning(
                "Trakt PKCE token exchange failed with status %d",
                resp.status_code,
            )
            raise TraktOAuthError(f"Trakt token exchange failed (HTTP {resp.status_code})")
        data = resp.json()
        if not isinstance(data, dict) or "access_token" not in data:
            raise TraktOAuthError("Invalid token response from Trakt")
        return data
