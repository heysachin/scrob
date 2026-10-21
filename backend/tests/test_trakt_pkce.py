import hashlib
import hmac
import json
import os
import re
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

from fastapi import FastAPI, HTTPException
import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core import trakt as trakt_client
from core import trakt_pkce
from core.config import settings as app_settings
from core.limiter import limiter
from core.timeutil import utcnow
from db import get_db
from dependencies import get_current_user
from models.trakt_oauth import TraktOAuthTransaction
from models.users import User, UserSettings
from routers import trakt as trakt_router


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


def _sqlite_json_deserializer(value):
    if isinstance(value, str) and value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return json.loads(value)


def _make_engine():
    return create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        json_deserializer=_sqlite_json_deserializer,
    )


class TraktPkceUtilsTests(unittest.TestCase):
    """Test PKCE generation and RFC 7636 conformance."""

    def test_code_verifier_format_and_length(self):
        verifier = trakt_pkce.generate_code_verifier()
        # RFC 7636 §4.1: min 43, max 128 characters
        self.assertGreaterEqual(len(verifier), 43)
        self.assertLessEqual(len(verifier), 128)
        # RFC 7636 §4.1 permitted charset: [A-Za-z0-9-._~]
        self.assertTrue(re.match(r"^[A-Za-z0-9\-._~]+$", verifier))

    def test_code_challenge_rfc7636_test_vector(self):
        # RFC 7636 Appendix B test vector
        test_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        expected_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        challenge = trakt_pkce.generate_code_challenge(test_verifier)
        self.assertEqual(challenge, expected_challenge)
        # Verify no '=' padding
        self.assertFalse(challenge.endswith("="))

    def test_oauth_state_generation(self):
        state1 = trakt_pkce.generate_oauth_state()
        state2 = trakt_pkce.generate_oauth_state()
        self.assertNotEqual(state1, state2)
        self.assertGreaterEqual(len(state1), 32)
        # Hash matches sha256
        state_hash = trakt_pkce.hash_oauth_state(state1)
        expected_hash = hashlib.sha256(state1.encode("ascii")).hexdigest()
        self.assertEqual(state_hash, expected_hash)

    def test_authorization_url_parameters(self):
        url = trakt_pkce.build_authorization_url(
            client_id="my-client-id",
            redirect_uri="https://scrob.sachindev.com/api/proxy/trakt/auth/pkce/callback",
            code_challenge="challenge123",
            state="state123",
        )
        self.assertTrue(url.startswith("https://auth.trakt.tv/oauth/authorize?"))
        self.assertIn("response_type=code", url)
        self.assertIn("client_id=my-client-id", url)
        self.assertIn("code_challenge=challenge123", url)
        self.assertIn("code_challenge_method=S256", url)
        self.assertIn("state=state123", url)
        self.assertIn("redirect_uri=https%3A%2F%2Fscrob.sachindev.com%2Fapi%2Fproxy%2Ftrakt%2Fauth%2Fpkce%2Fcallback", url)

    def test_redirect_uri_derivation_precedence(self):
        original_public = getattr(app_settings, "scrob_public_url", None)
        original_redirect = getattr(app_settings, "trakt_redirect_url", None)
        original_server = app_settings.server_url

        try:
            # 1. Default server_url
            app_settings.scrob_public_url = None
            app_settings.trakt_redirect_url = None
            app_settings.server_url = "http://localhost:7330"
            self.assertEqual(
                trakt_pkce.get_trakt_redirect_uri(),
                "http://localhost:7330/api/proxy/trakt/auth/pkce/callback",
            )

            # 2. SCROB_PUBLIC_URL overrides server_url
            app_settings.scrob_public_url = "https://scrob.sachindev.com"
            self.assertEqual(
                trakt_pkce.get_trakt_redirect_uri(),
                "https://scrob.sachindev.com/api/proxy/trakt/auth/pkce/callback",
            )

            # 3. TRAKT_REDIRECT_URL overrides SCROB_PUBLIC_URL
            app_settings.trakt_redirect_url = "https://custom.domain/trakt/callback"
            self.assertEqual(
                trakt_pkce.get_trakt_redirect_uri(),
                "https://custom.domain/trakt/callback",
            )
        finally:
            app_settings.scrob_public_url = original_public
            app_settings.trakt_redirect_url = original_redirect
            app_settings.server_url = original_server


class TraktPkceFlowTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests for Trakt PKCE endpoints and session persistence."""

    async def asyncSetUp(self):
        limiter.enabled = False
        self.addCleanup(setattr, limiter, "enabled", True)

        self.engine = _make_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(User.__table__.create)
            await conn.run_sync(UserSettings.__table__.create)
            await conn.run_sync(TraktOAuthTransaction.__table__.create)

        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)

        async with self.Session() as s:
            u1 = User(id=1, email="alice@test.local", username="alice", api_key="k1")
            u2 = User(id=2, email="bob@test.local", username="bob", api_key="k2")
            s.add_all([u1, u2])
            s.add(UserSettings(user_id=1, trakt_client_id="alice-client-id"))
            s.add(UserSettings(user_id=2, trakt_client_id="bob-client-id"))
            await s.commit()

        self.app = FastAPI()
        self.app.include_router(trakt_router.router, prefix="/trakt")

        async def override_get_db():
            async with self.Session() as session:
                yield session

        self.app.dependency_overrides[get_db] = override_get_db

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_pkce_start_success(self):
        current_u = User(id=1, email="alice@test.local", username="alice", api_key="k1")
        self.app.dependency_overrides[get_current_user] = lambda: current_u

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.post("/trakt/auth/pkce/start")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("authorization_url", data)
            self.assertEqual(data["expires_in"], 600)
            # Verifier must NEVER be exposed
            self.assertNotIn("verifier", data)
            self.assertNotIn("code_verifier", data)
            self.assertIn("client_id=alice-client-id", data["authorization_url"])

        # Check DB transaction was stored
        async with self.Session() as s:
            txs = (await s.execute(select(TraktOAuthTransaction))).scalars().all()
            self.assertEqual(len(txs), 1)
            self.assertEqual(txs[0].user_id, 1)
            self.assertEqual(txs[0].client_id, "alice-client-id")
            self.assertIsNotNone(txs[0].code_verifier)
            self.assertIsNone(txs[0].consumed_at)

    async def test_pkce_start_with_payload_client_id(self):
        current_u = User(id=1, email="alice@test.local", username="alice", api_key="k1")
        self.app.dependency_overrides[get_current_user] = lambda: current_u

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/trakt/auth/pkce/start",
                json={"client_id": "new-typed-client-id"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("client_id=new-typed-client-id", data["authorization_url"])

        async with self.Session() as s:
            us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            self.assertEqual(us.trakt_client_id, "new-typed-client-id")

    async def test_pkce_start_missing_client_id(self):
        current_u = User(id=1, email="alice@test.local", username="alice", api_key="k1")
        self.app.dependency_overrides[get_current_user] = lambda: current_u

        async with self.Session() as s:
            us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            us.trakt_client_id = None
            await s.commit()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.post("/trakt/auth/pkce/start", json={})
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Trakt Client ID is not configured", resp.json()["detail"])

    async def test_pkce_callback_success(self):
        # Create unconsumed transaction for user 1
        raw_state = "valid-random-state-12345"
        state_hash = trakt_pkce.hash_oauth_state(raw_state)
        verifier = "test-verifier-string-43-chars-long-1234567890"

        async with self.Session() as s:
            tx = TraktOAuthTransaction(
                state_hash=state_hash,
                code_verifier=verifier,
                user_id=1,
                client_id="alice-client-id",
                redirect_uri="https://scrob.sachindev.com/api/proxy/trakt/auth/pkce/callback",
                expires_at=utcnow() + timedelta(minutes=10),
            )
            s.add(tx)
            await s.commit()

        mock_token_data = {
            "access_token": "trakt-access-token-123",
            "refresh_token": "trakt-refresh-token-456",
            "expires_in": 7200,
            "created_at": 1000000,
            "token_type": "Bearer",
            "scope": "public",
        }

        with patch.object(trakt_pkce, "exchange_code_for_token", AsyncMock(return_value=mock_token_data)) as mock_exchange:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/trakt/auth/pkce/callback?code=trakt-code-abc&state={raw_state}",
                    follow_redirects=False,
                )
                self.assertEqual(resp.status_code, 303)
                self.assertEqual(resp.headers["Location"], "/connections?trakt=success")

            mock_exchange.assert_called_once_with(
                client_id="alice-client-id",
                code="trakt-code-abc",
                redirect_uri="https://scrob.sachindev.com/api/proxy/trakt/auth/pkce/callback",
                code_verifier=verifier,
            )

        # Check DB updates
        async with self.Session() as s:
            tx_saved = (await s.execute(select(TraktOAuthTransaction).where(TraktOAuthTransaction.state_hash == state_hash))).scalar_one()
            self.assertIsNotNone(tx_saved.consumed_at)

            user_settings = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            self.assertEqual(user_settings.trakt_access_token, "trakt-access-token-123")
            self.assertEqual(user_settings.trakt_refresh_token, "trakt-refresh-token-456")
            self.assertIsNotNone(user_settings.trakt_token_expires_at)

    async def test_pkce_callback_user_denial(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/trakt/auth/pkce/callback?error=access_denied&error_description=User+denied",
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            self.assertEqual(resp.headers["Location"], "/connections?trakt=denied")

    async def test_pkce_callback_missing_code_or_state(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/trakt/auth/pkce/callback?code=123",
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            self.assertEqual(resp.headers["Location"], "/connections?trakt=error&message=missing_parameters")

    async def test_pkce_callback_incorrect_state(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/trakt/auth/pkce/callback?code=123&state=nonexistent-state",
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            self.assertEqual(resp.headers["Location"], "/connections?trakt=error&message=invalid_or_expired_state")

    async def test_pkce_callback_expired_state(self):
        raw_state = "expired-state-123"
        state_hash = trakt_pkce.hash_oauth_state(raw_state)

        async with self.Session() as s:
            tx = TraktOAuthTransaction(
                state_hash=state_hash,
                code_verifier="test-verifier",
                user_id=1,
                client_id="alice-client-id",
                redirect_uri="https://test/callback",
                expires_at=utcnow() - timedelta(seconds=1),  # already expired
            )
            s.add(tx)
            await s.commit()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.get(
                f"/trakt/auth/pkce/callback?code=123&state={raw_state}",
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            self.assertEqual(resp.headers["Location"], "/connections?trakt=error&message=invalid_or_expired_state")

    async def test_pkce_callback_replayed_state(self):
        raw_state = "replayed-state-123"
        state_hash = trakt_pkce.hash_oauth_state(raw_state)

        async with self.Session() as s:
            tx = TraktOAuthTransaction(
                state_hash=state_hash,
                code_verifier="test-verifier",
                user_id=1,
                client_id="alice-client-id",
                redirect_uri="https://test/callback",
                expires_at=utcnow() + timedelta(minutes=10),
                consumed_at=utcnow(),  # already consumed
            )
            s.add(tx)
            await s.commit()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            resp = await client.get(
                f"/trakt/auth/pkce/callback?code=123&state={raw_state}",
                follow_redirects=False,
            )
            self.assertEqual(resp.status_code, 303)
            self.assertEqual(resp.headers["Location"], "/connections?trakt=error&message=state_already_used")

    async def test_pkce_callback_token_exchange_failure(self):
        raw_state = "exchange-fail-state"
        state_hash = trakt_pkce.hash_oauth_state(raw_state)

        async with self.Session() as s:
            tx = TraktOAuthTransaction(
                state_hash=state_hash,
                code_verifier="test-verifier",
                user_id=1,
                client_id="alice-client-id",
                redirect_uri="https://test/callback",
                expires_at=utcnow() + timedelta(minutes=10),
            )
            s.add(tx)
            await s.commit()

        with patch.object(
            trakt_pkce,
            "exchange_code_for_token",
            AsyncMock(side_effect=trakt_pkce.TraktOAuthError("invalid code")),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/trakt/auth/pkce/callback?code=bad-code&state={raw_state}",
                    follow_redirects=False,
                )
                self.assertEqual(resp.status_code, 303)
                self.assertEqual(resp.headers["Location"], "/connections?trakt=error&message=token_exchange_failed")

    async def test_concurrent_authorization_attempts_isolated(self):
        # Two users initiate authorization concurrently
        s1 = trakt_pkce.generate_oauth_state()
        s2 = trakt_pkce.generate_oauth_state()

        async with self.Session() as s:
            tx1 = TraktOAuthTransaction(
                state_hash=trakt_pkce.hash_oauth_state(s1),
                code_verifier="verifier-alice",
                user_id=1,
                client_id="client-alice",
                redirect_uri="https://test/callback",
                expires_at=utcnow() + timedelta(minutes=10),
            )
            tx2 = TraktOAuthTransaction(
                state_hash=trakt_pkce.hash_oauth_state(s2),
                code_verifier="verifier-bob",
                user_id=2,
                client_id="client-bob",
                redirect_uri="https://test/callback",
                expires_at=utcnow() + timedelta(minutes=10),
            )
            s.add_all([tx1, tx2])
            await s.commit()

        mock_token_data = {
            "access_token": "token-for-bob",
            "refresh_token": "refresh-for-bob",
            "expires_in": 7200,
        }

        # Bob completes his callback
        with patch.object(trakt_pkce, "exchange_code_for_token", AsyncMock(return_value=mock_token_data)):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/trakt/auth/pkce/callback?code=bob-code&state={s2}",
                    follow_redirects=False,
                )
                self.assertEqual(resp.status_code, 303)

        async with self.Session() as s:
            # Bob's transaction is consumed
            tx2_db = (await s.execute(select(TraktOAuthTransaction).where(TraktOAuthTransaction.user_id == 2))).scalar_one()
            self.assertIsNotNone(tx2_db.consumed_at)

            # Bob's user settings have the new token
            bob_settings = (await s.execute(select(UserSettings).where(UserSettings.user_id == 2))).scalar_one()
            self.assertEqual(bob_settings.trakt_access_token, "token-for-bob")

            # Alice's transaction is NOT consumed
            tx1_db = (await s.execute(select(TraktOAuthTransaction).where(TraktOAuthTransaction.user_id == 1))).scalar_one()
            self.assertIsNone(tx1_db.consumed_at)

            # Alice's settings are untouched
            alice_settings = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            self.assertIsNone(alice_settings.trakt_access_token)


class TraktPkceRefreshTests(unittest.IsolatedAsyncioTestCase):
    """Test token refresh and synchronization without client_secret."""

    async def asyncSetUp(self):
        self.engine = _make_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(User.__table__.create)
            await conn.run_sync(UserSettings.__table__.create)

        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_ensure_valid_token_secretless_refresh(self):
        now_ts = int(datetime.now(timezone.utc).timestamp())
        async with self.Session() as s:
            settings = UserSettings(
                user_id=1,
                trakt_client_id="client-id-no-secret",
                trakt_client_secret=None,  # No secret!
                trakt_access_token="old-expired-token",
                trakt_refresh_token="valid-refresh-token",
                trakt_token_expires_at=now_ts - 3600,  # Expired 1 hour ago
            )
            s.add(settings)
            await s.commit()

        # Mock validate_token to return False (token expired)
        # Mock refresh_access_token to return refreshed tokens with rotated refresh_token
        new_token_data = {
            "access_token": "new-refreshed-token",
            "refresh_token": "new-rotated-refresh-token",
            "expires_in": 7200,
        }

        with (
            patch.object(trakt_router.trakt_client, "validate_token", AsyncMock(return_value=False)),
            patch.object(trakt_router.trakt_client, "refresh_access_token", AsyncMock(return_value=new_token_data)) as mock_refresh,
        ):
            async with self.Session() as s:
                us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
                token = await trakt_router.ensure_valid_trakt_token(s, us)
                self.assertEqual(token, "new-refreshed-token")
                self.assertEqual(us.trakt_access_token, "new-refreshed-token")
                self.assertEqual(us.trakt_refresh_token, "new-rotated-refresh-token")

            # Check call args of refresh_access_token
            mock_refresh.assert_called_once_with(
                "client-id-no-secret",
                None,
                "valid-refresh-token",
                redirect_uri=trakt_pkce.get_trakt_redirect_uri(),
            )

    async def test_legacy_refresh_with_secret_preserved(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = httpx.Response(
                200,
                json={"access_token": "fresh", "refresh_token": "rot", "expires_in": 7200},
                request=httpx.Request("POST", "https://api.trakt.tv/oauth/token"),
            )
            mock_post.return_value = mock_resp

            res = await trakt_client.refresh_access_token(
                client_id="cid",
                client_secret="csecret",
                refresh_token="rt",
            )
            self.assertEqual(res["access_token"], "fresh")
            body = mock_post.call_args.kwargs["json"]
            self.assertEqual(body["client_secret"], "csecret")
            self.assertEqual(body["client_id"], "cid")
            self.assertEqual(body["grant_type"], "refresh_token")

    async def test_secretless_refresh_omits_client_secret(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = httpx.Response(
                200,
                json={"access_token": "fresh", "refresh_token": "rot", "expires_in": 7200},
                request=httpx.Request("POST", "https://api.trakt.tv/oauth/token"),
            )
            mock_post.return_value = mock_resp

            res = await trakt_client.refresh_access_token(
                client_id="cid",
                client_secret=None,
                refresh_token="rt",
                redirect_uri="https://scrob.sachindev.com/api/proxy/trakt/auth/pkce/callback",
            )
            self.assertEqual(res["access_token"], "fresh")
            body = mock_post.call_args.kwargs["json"]
            self.assertNotIn("client_secret", body)
            self.assertEqual(body["redirect_uri"], "https://scrob.sachindev.com/api/proxy/trakt/auth/pkce/callback")

    async def test_secretless_disconnect_and_revoke(self):
        current_u = User(id=1, email="alice@test.local", username="alice", api_key="k1")
        async with self.Session() as s:
            s.add(current_u)
            us = UserSettings(
                user_id=1,
                trakt_client_id="cid",
                trakt_client_secret=None,
                trakt_access_token="tok",
                trakt_refresh_token="ref",
            )
            s.add(us)
            await s.commit()

        app = FastAPI()
        app.include_router(trakt_router.router, prefix="/trakt")
        app.dependency_overrides[get_current_user] = lambda: current_u

        async def override_get_db():
            async with self.Session() as s:
                yield s

        app.dependency_overrides[get_db] = override_get_db

        with patch.object(trakt_router.trakt_client, "revoke_token", AsyncMock()) as mock_revoke:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.delete("/trakt/auth/disconnect")
                self.assertEqual(resp.status_code, 200)

            mock_revoke.assert_called_once_with("cid", None, "tok")

        async with self.Session() as s:
            us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            self.assertIsNone(us.trakt_access_token)
            self.assertIsNone(us.trakt_refresh_token)

    def test_require_trakt_config_allows_pkce_without_secret(self):
        settings = UserSettings(user_id=1, trakt_client_id="cid", trakt_client_secret=None)
        # Should not raise
        trakt_router._require_trakt_config(settings)

        # Missing client_id raises 503
        settings_no_id = UserSettings(user_id=1, trakt_client_id=None)
        with self.assertRaises(HTTPException) as cm:
            trakt_router._require_trakt_config(settings_no_id)
        self.assertEqual(cm.exception.status_code, 503)

        # Device config still demands secret
        with self.assertRaises(HTTPException) as cm:
            trakt_router._require_trakt_device_config(settings)
        self.assertEqual(cm.exception.status_code, 503)
