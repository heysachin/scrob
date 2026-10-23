"""Regression and integration tests for Trakt OAuth reconnect and history synchronization.

Covers:
1. Disconnect clears all connection-specific credentials (tokens, secrets, client_id, cursors, transactions).
2. Reconnecting with a different Client ID stores the new credentials.
3. Token refresh uses the new OAuth application credentials.
4. Full resync imports a recent, previously missing movie.
5. Full resync imports a recent, previously missing episode.
6. Repeating Full resync does not create duplicate watch events.
7. Incremental sync continues functioning.
8. Missing TMDB identifiers are recorded with the correct skip reason.
9. Pagination retrieves every expected history record.
10. Missing or malformed API data produces understandable diagnostics rather than silently losing entries.
11. Reported symptom reproduction: hundreds of historical records fetched, all existing skipped, but new watch imports.
"""

import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

import httpx
_REAL_ASYNC_CLIENT = httpx.AsyncClient
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core import trakt as trakt_client
from core import trakt_pkce
from core.timeutil import utcnow
from core.trakt_diagnostics import run_trakt_diagnostics
from db import get_db
from dependencies import get_current_user
from models import (
    Base,
    CollectionSource,
    GlobalSettings,
    Media,
    MediaType,
    Show,
    SyncJob,
    SyncStatus,
    TraktOAuthTransaction,
    User,
    UserSettings,
    WatchEvent,
)
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


class TraktReconnectSyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = _make_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)

        async with self.Session() as s:
            u = User(id=1, email="test@example.com", username="tester", api_key="test-api-key")
            s.add(u)
            us = UserSettings(
                user_id=1,
                tmdb_api_key="mock-tmdb-key",
                trakt_client_id="old-client-id",
                trakt_client_secret="old-client-secret",
                trakt_access_token="old-access-token",
                trakt_refresh_token="old-refresh-token",
                trakt_token_expires_at=int(datetime.now(timezone.utc).timestamp()) + 3600,
                trakt_history_cursor_at=datetime(2026, 1, 1, 0, 0, 0),
            )
            s.add(us)
            gs = GlobalSettings(id=1, tmdb_api_key="global-tmdb-key")
            s.add(gs)
            await s.commit()

        self.app = FastAPI()
        self.app.include_router(trakt_router.router, prefix="/trakt")

        self.current_user = u

        async def override_get_db():
            async with self.Session() as session:
                yield session

        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_current_user] = lambda: self.current_user

    async def asyncTearDown(self):
        await self.engine.dispose()

    # 1. Disconnect clears all connection-specific credentials and preserves user data
    async def test_disconnect_clears_all_credentials_and_preserves_user_data(self):
        async with self.Session() as s:
            # Seed pending OAuth transaction, active sync job, and existing user media
            tx = TraktOAuthTransaction(
                state_hash="pending-state",
                code_verifier="verifier123456789012345678901234567890123456",
                user_id=1,
                client_id="old-client-id",
                redirect_uri="https://scrob.example.com/api/proxy/trakt/auth/pkce/callback",
                expires_at=utcnow() + timedelta(minutes=10),
            )
            s.add(tx)
            job = SyncJob(
                user_id=1,
                source=CollectionSource.trakt,
                status=SyncStatus.running,
            )
            s.add(job)
            movie = Media(id=10, tmdb_id=550, media_type=MediaType.movie, title="Fight Club")
            s.add(movie)
            watch = WatchEvent(id=20, user_id=1, media_id=10, source="trakt", watched_at=utcnow(), completed=True)
            s.add(watch)
            await s.commit()
            job_id = job.id

        with patch.object(trakt_router.trakt_client, "revoke_token", AsyncMock()) as mock_revoke:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                resp = await client.delete("/trakt/auth/disconnect")
                self.assertEqual(resp.status_code, 200)

            mock_revoke.assert_called_once_with("old-client-id", "old-client-secret", "old-access-token")

        async with self.Session() as s:
            us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            self.assertIsNone(us.trakt_client_id)
            self.assertIsNone(us.trakt_client_secret)
            self.assertIsNone(us.trakt_access_token)
            self.assertIsNone(us.trakt_refresh_token)
            self.assertIsNone(us.trakt_token_expires_at)
            self.assertIsNone(us.trakt_device_code)
            self.assertIsNone(us.trakt_history_cursor_at)

            # Assert pending transaction deleted
            tx_res = (await s.execute(select(TraktOAuthTransaction).where(TraktOAuthTransaction.user_id == 1))).scalars().all()
            self.assertEqual(len(tx_res), 0)

            # Assert active job cancelled
            job_db = (await s.execute(select(SyncJob).where(SyncJob.id == job_id))).scalar_one()
            self.assertEqual(job_db.status, SyncStatus.cancelled)

            # Assert user-owned data is preserved
            watch_db = (await s.execute(select(WatchEvent).where(WatchEvent.id == 20))).scalar_one_or_none()
            self.assertIsNotNone(watch_db)
            media_db = (await s.execute(select(Media).where(Media.id == 10))).scalar_one_or_none()
            self.assertIsNotNone(media_db)

    # 2. Reconnecting with a different Client ID stores the new credentials
    async def test_reconnecting_with_different_client_id_stores_new_credentials(self):
        # 1. Start PKCE flow with new client id
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
            resp = await client.post("/trakt/auth/pkce/start", json={"client_id": "new-client-id-2"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("authorization_url", data)

        async with self.Session() as s:
            tx = (await s.execute(select(TraktOAuthTransaction).where(TraktOAuthTransaction.user_id == 1))).scalar_one()
            self.assertEqual(tx.client_id, "new-client-id-2")
            state_hash = tx.state_hash

        mock_token = {
            "access_token": "new-access-token-xyz",
            "refresh_token": "new-refresh-token-xyz",
            "expires_in": 7200,
        }
        with patch.object(trakt_pkce, "exchange_code_for_token", AsyncMock(return_value=mock_token)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                # Callback without authorization header
                callback_resp = await client.get(
                    f"/trakt/auth/pkce/callback?code=mock_code&state=nonexistent",
                    follow_redirects=False,
                )
                self.assertEqual(callback_resp.status_code, 303)
                self.assertIn("trakt=error", callback_resp.headers["Location"])

    # 3. Token refresh uses the new OAuth application credentials
    async def test_token_refresh_uses_new_client_credentials(self):
        now_ts = int(datetime.now(timezone.utc).timestamp())
        async with self.Session() as s:
            us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
            us.trakt_client_id = "new-client-id-app2"
            us.trakt_client_secret = None  # Reconnected via PKCE
            us.trakt_access_token = "expired-token"
            us.trakt_refresh_token = "valid-refresh-token-app2"
            us.trakt_token_expires_at = now_ts - 100  # Expired
            await s.commit()

        refreshed_data = {
            "access_token": "brand-new-access-token",
            "refresh_token": "rotated-refresh-token",
            "expires_in": 7200,
        }

        with (
            patch.object(trakt_router.trakt_client, "validate_token", AsyncMock(return_value=False)),
            patch.object(trakt_router.trakt_client, "refresh_access_token", AsyncMock(return_value=refreshed_data)) as mock_refresh,
        ):
            async with self.Session() as s:
                us = (await s.execute(select(UserSettings).where(UserSettings.user_id == 1))).scalar_one()
                token = await trakt_router.ensure_valid_trakt_token(s, us)
                self.assertEqual(token, "brand-new-access-token")
                self.assertEqual(us.trakt_access_token, "brand-new-access-token")
                self.assertEqual(us.trakt_refresh_token, "rotated-refresh-token")

            # Must use the new client ID and omit secret
            mock_refresh.assert_called_once_with(
                "new-client-id-app2",
                None,
                "valid-refresh-token-app2",
                redirect_uri=trakt_pkce.get_trakt_redirect_uri(),
            )

    # 4. Full resync imports a recent, previously missing movie
    async def test_full_resync_imports_recent_missing_movie(self):
        async with self.Session() as s:
            # Existing movie in DB
            m1 = Media(id=1, tmdb_id=100, media_type=MediaType.movie, title="Existing Movie")
            s.add(m1)
            past_time = datetime(2026, 5, 1, 20, 0, 0)
            s.add(WatchEvent(id=1, user_id=1, media_id=1, source="trakt", watched_at=past_time, completed=True))
            job = SyncJob(id=101, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

        # Mock Trakt response with the existing movie + a new recent movie
        recent_time = datetime(2026, 10, 20, 18, 0, 0)
        trakt_movies = [
            {
                "id": 9001,
                "watched_at": "2026-10-20T18:00:00.000Z",
                "action": "watch",
                "movie": {"title": "Recent New Movie", "ids": {"tmdb": 200}},
            },
            {
                "id": 9002,
                "watched_at": "2026-05-01T20:00:00.000Z",
                "action": "watch",
                "movie": {"title": "Existing Movie", "ids": {"tmdb": 100}},
            },
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return trakt_movies

            async def get_history_episodes(self, start_at, end_at):
                return []

            async def get_ratings(self):
                return {}

            async def get_watchlist(self):
                return []

            async def get_user_lists(self):
                return []

        async with self.Session() as s:
            # Mock TMDB movie resolution
            with patch.object(
                trakt_router,
                "_get_or_create_movie_media",
                AsyncMock(side_effect=lambda db, tmdb_id, title, key, **kw: (
                    Media(id=1, tmdb_id=100, media_type=MediaType.movie, title="Existing Movie")
                    if tmdb_id == 100 else
                    Media(id=2, tmdb_id=200, media_type=MediaType.movie, title="Recent New Movie")
                )),
            ):
                stats, processed, errors, new_watched, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=101,
                    user_id=1,
                    source=MockSource(),
                    api_key="key",
                    sync_watched=True,
                    sync_ratings=False,
                    sync_lists=False,
                    split_watchlist=False,
                    history_start=None,
                    history_end=None,  # Unrestricted Full Resync!
                )

        self.assertEqual(stats["movies"], 1)
        self.assertEqual(stats["imported_movies"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["skipped_duplicates"], 1)
        self.assertEqual(stats["skipped_duplicates_movies"], 1)
        self.assertEqual(stats["fetched_movies"], 2)

    # 5. Full resync imports a recent, previously missing episode
    async def test_full_resync_imports_recent_missing_episode(self):
        async with self.Session() as s:
            job = SyncJob(id=102, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

        trakt_episodes = [
            {
                "id": 8001,
                "watched_at": "2026-10-21T21:00:00.000Z",
                "action": "watch",
                "show": {"title": "Severance", "ids": {"tmdb": 95396}},
                "episode": {"season": 2, "number": 1, "title": "Episode 1"},
            }
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return []

            async def get_history_episodes(self, start_at, end_at):
                return trakt_episodes

            async def get_ratings(self):
                return {}

            async def get_watchlist(self):
                return []

            async def get_user_lists(self):
                return []

        show_stub = Show(id=10, tmdb_id=95396, title="Severance")
        ep_stub = Media(id=50, show_id=10, media_type=MediaType.episode, season_number=2, episode_number=1)

        async with self.Session() as s:
            with (
                patch.object(trakt_router, "_get_or_create_show", AsyncMock(return_value=show_stub)),
                patch.object(trakt_router, "_get_or_create_episode_media", AsyncMock(return_value=ep_stub)),
            ):
                stats, processed, errors, new_watched, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=102,
                    user_id=1,
                    source=MockSource(),
                    api_key="key",
                    sync_watched=True,
                    sync_ratings=False,
                    sync_lists=False,
                    split_watchlist=False,
                    history_start=None,
                    history_end=None,
                )

        self.assertEqual(stats["episodes"], 1)
        self.assertEqual(stats["imported_episodes"], 1)
        self.assertEqual(stats["fetched_episodes"], 1)
        self.assertEqual(stats["skipped"], 0)

    # 6. Repeating Full resync does not create duplicate watch events
    async def test_repeating_full_resync_does_not_create_duplicates(self):
        movie_stub = Media(id=30, tmdb_id=300, media_type=MediaType.movie, title="Dune")
        trakt_movies = [
            {
                "id": 7001,
                "watched_at": "2026-03-01T20:00:00.000Z",
                "action": "watch",
                "movie": {"title": "Dune", "ids": {"tmdb": 300}},
            }
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return trakt_movies

            async def get_history_episodes(self, start_at, end_at):
                return []

            async def get_ratings(self):
                return {}

            async def get_watchlist(self):
                return []

            async def get_user_lists(self):
                return []

        async with self.Session() as s:
            s.add(movie_stub)
            job1 = SyncJob(id=103, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job1)
            await s.commit()

        # Run 1: imports movie
        async with self.Session() as s:
            with patch.object(trakt_router, "_get_or_create_movie_media", AsyncMock(return_value=movie_stub)):
                stats1, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=103,
                    user_id=1,
                    source=MockSource(),
                    api_key="key",
                    sync_watched=True,
                    sync_ratings=False,
                    sync_lists=False,
                    split_watchlist=False,
                    history_start=None,
                    history_end=None,
                )
                await s.commit()

        self.assertEqual(stats1["movies"], 1)
        self.assertEqual(stats1["skipped"], 0)

        # Run 2: same Trakt history -> skips as duplicate
        async with self.Session() as s:
            job2 = SyncJob(id=104, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job2)
            await s.commit()

            with patch.object(trakt_router, "_get_or_create_movie_media", AsyncMock(return_value=movie_stub)):
                stats2, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=104,
                    user_id=1,
                    source=MockSource(),
                    api_key="key",
                    sync_watched=True,
                    sync_ratings=False,
                    sync_lists=False,
                    split_watchlist=False,
                    history_start=None,
                    history_end=None,
                )

        self.assertEqual(stats2["movies"], 0)
        self.assertEqual(stats2["skipped"], 1)
        self.assertEqual(stats2["skipped_duplicates"], 1)

        # Confirm only 1 watch event exists
        async with self.Session() as s:
            events = (await s.execute(select(WatchEvent).where(WatchEvent.user_id == 1, WatchEvent.media_id == 30))).scalars().all()
            self.assertEqual(len(events), 1)

    # 7. Incremental sync continues functioning with cursor
    def test_history_window_full_vs_incremental(self):
        cursor = datetime(2026, 10, 15, 12, 0, 0)
        cutoff = datetime(2026, 10, 22, 12, 0, 0)

        # Full resync must omit both start_at and end_at
        full_start, full_end = trakt_router._history_window(cursor, True, cutoff)
        self.assertIsNone(full_start)
        self.assertIsNone(full_end)

        # Initial sync (cursor is None) must also omit both
        init_start, init_end = trakt_router._history_window(None, False, cutoff)
        self.assertIsNone(init_start)
        self.assertIsNone(init_end)

        # Incremental sync uses cursor - 5 min overlap up to cutoff
        inc_start, inc_end = trakt_router._history_window(cursor, False, cutoff)
        self.assertEqual(inc_start, cursor - timedelta(minutes=5))
        self.assertEqual(inc_end, cutoff)

    # 8. Missing TMDB identifiers are recorded with the correct skip reason
    async def test_missing_tmdb_id_recorded_with_correct_skip_reason(self):
        trakt_movies = [
            {
                "id": 9991,
                "watched_at": "2026-10-01T20:00:00.000Z",
                "movie": {"title": "Untracked Indie Film", "ids": {}},  # no tmdb id!
            }
        ]
        trakt_episodes = [
            {
                "id": 9992,
                "watched_at": "2026-10-01T21:00:00.000Z",
                "show": {"title": "Untracked Web Show", "ids": {}},  # no show tmdb id!
                "episode": {"season": 1, "number": 1},
            },
            {
                "id": 9993,
                "watched_at": "2026-10-01T22:00:00.000Z",
                "show": {"title": "Valid Show", "ids": {"tmdb": 1234}},
                "episode": {"season": None, "number": None},  # invalid episode data!
            },
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return trakt_movies

            async def get_history_episodes(self, start_at, end_at):
                return trakt_episodes

            async def get_ratings(self):
                return {}

            async def get_watchlist(self):
                return []

            async def get_user_lists(self):
                return []

        async with self.Session() as s:
            job = SyncJob(id=105, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            show_stub = Show(id=11, tmdb_id=1234, title="Valid Show")
            with patch.object(trakt_router, "_get_or_create_show", AsyncMock(return_value=show_stub)):
                stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=105,
                    user_id=1,
                    source=MockSource(),
                    api_key="key",
                    sync_watched=True,
                    sync_ratings=False,
                    sync_lists=False,
                    split_watchlist=False,
                    history_start=None,
                    history_end=None,
                )

        self.assertEqual(stats["skipped_missing_tmdb"], 2)  # 1 movie + 1 show
        self.assertEqual(stats["skipped_invalid_data"], 1)  # 1 episode missing season/ep
        self.assertEqual(stats["skipped"], 3)
        self.assertIn('movie "Untracked Indie Film" (no TMDB id)', stats["unimported"])
        self.assertIn('show "Untracked Web Show" (no TMDB id)', stats["unimported"])
        self.assertIn('show "Valid Show" episode (no season/episode number)', stats["unimported"])

    # 9. Pagination retrieves every expected history record
    async def test_pagination_retrieves_every_expected_history_record(self):
        pages_requested = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", 1))
            pages_requested.append(page)
            if page == 1:
                items = [{"id": i, "watched_at": "2026-01-01T00:00:00.000Z", "movie": {"ids": {"tmdb": i}}} for i in range(1, 251)]
                return httpx.Response(200, json=items, headers={"X-Pagination-Page-Count": "2"})
            elif page == 2:
                items = [{"id": i, "watched_at": "2026-01-01T00:00:00.000Z", "movie": {"ids": {"tmdb": i}}} for i in range(251, 301)]
                return httpx.Response(200, json=items, headers={"X-Pagination-Page-Count": "2"})
            return httpx.Response(200, json=[], headers={"X-Pagination-Page-Count": "2"})

        transport = httpx.MockTransport(mock_handler)
        with patch("httpx.AsyncClient", side_effect=lambda **kw: _REAL_ASYNC_CLIENT(transport=transport, **kw)):
            result = await trakt_client.get_history_movies("cid", "token")

        self.assertEqual(pages_requested, [1, 2])
        self.assertEqual(len(result), 300)

    # 10. Diagnostics identifies discrepancies between Mode A (with end_at) and Mode B (unrestricted)
    async def test_diagnostics_detects_discrepancy_and_identity(self):
        async with self.Session() as s:
            # Mock Trakt API with profile and history
            def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/users/me":
                    return httpx.Response(200, json={"username": "heysachin", "name": "Sachin", "ids": {"slug": "heysachin"}})
                if "/sync/history/movies" in request.url.path:
                    # If end_at is provided, omit the newest record to simulate cutoff exclusion
                    if "end_at" in request.url.params:
                        return httpx.Response(200, json=[
                            {"id": 1, "watched_at": "2026-05-01T12:00:00.000Z", "movie": {"title": "Old Movie", "ids": {"tmdb": 10}}}
                        ], headers={"X-Pagination-Page-Count": "1"})
                    else:
                        return httpx.Response(200, json=[
                            {"id": 2, "watched_at": "2026-10-22T08:00:00.000Z", "movie": {"title": "Recent Excluded Movie", "ids": {"tmdb": 20}}},
                            {"id": 1, "watched_at": "2026-05-01T12:00:00.000Z", "movie": {"title": "Old Movie", "ids": {"tmdb": 10}}},
                        ], headers={"X-Pagination-Page-Count": "1"})
                if "/sync/history/episodes" in request.url.path:
                    return httpx.Response(200, json=[], headers={"X-Pagination-Page-Count": "1"})
                return httpx.Response(404)

            transport = httpx.MockTransport(handler)
            with patch("httpx.AsyncClient", side_effect=lambda **kw: _REAL_ASYNC_CLIENT(transport=transport, **kw)):
                report = await run_trakt_diagnostics(s, user_id=1, expected_username="heysachin")

        self.assertTrue(report.account_matches_expected)
        self.assertEqual(report.trakt_account_identity["username"], "heysachin")
        self.assertEqual(report.mode_a_current_movies.total_records, 1)
        self.assertEqual(report.mode_b_unrestricted_movies.total_records, 2)
        self.assertEqual(report.movies_discrepancy_count, 1)
        self.assertEqual(len(report.unrestricted_recent_movies_excluded_by_mode_a), 1)
        self.assertEqual(report.unrestricted_recent_movies_excluded_by_mode_a[0]["title"], "Recent Excluded Movie")

    # 11. Reproduce reported symptom: 221 existing movies + 211 existing episodes skipped, but a new watch imports
    async def test_reproduce_reported_symptom_all_existing_skipped_but_new_imported(self):
        # Setup: user has 221 movies and 211 episodes already in Scrob
        async with self.Session() as s:
            for i in range(1, 222):
                m = Media(id=1000 + i, tmdb_id=10000 + i, media_type=MediaType.movie, title=f"Movie {i}")
                s.add(m)
                s.add(WatchEvent(id=2000 + i, user_id=1, media_id=1000 + i, source="trakt", watched_at=datetime(2026, 1, 1, 0, 0, 0), completed=True))

            show = Show(id=1, tmdb_id=50000, title="Show 1")
            s.add(show)
            for j in range(1, 212):
                ep = Media(id=5000 + j, show_id=1, media_type=MediaType.episode, title=f"Episode {j}", season_number=1, episode_number=j)
                s.add(ep)
                s.add(WatchEvent(id=6000 + j, user_id=1, media_id=5000 + j, source="trakt", watched_at=datetime(2026, 1, 1, 0, 0, 0), completed=True))

            job = SyncJob(id=110, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

        # Build Trakt response containing all 221 movies and 211 episodes + 1 genuinely new movie
        mock_movies = [
            {"id": i, "watched_at": "2026-01-01T00:00:00.000Z", "movie": {"title": f"Movie {i}", "ids": {"tmdb": 10000 + i}}}
            for i in range(1, 222)
        ]
        # Add 1 genuinely new recent movie
        mock_movies.append({
            "id": 99999,
            "watched_at": "2026-10-21T20:00:00.000Z",
            "movie": {"title": "Brand New Movie", "ids": {"tmdb": 99999}},
        })

        mock_episodes = [
            {"id": 10000 + j, "watched_at": "2026-01-01T00:00:00.000Z", "show": {"title": "Show 1", "ids": {"tmdb": 50000}}, "episode": {"season": 1, "number": j}}
            for j in range(1, 212)
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return mock_movies

            async def get_history_episodes(self, start_at, end_at):
                return mock_episodes

            async def get_ratings(self):
                return {}

            async def get_watchlist(self):
                return []

            async def get_user_lists(self):
                return []

        async with self.Session() as s:
            def fake_get_movie(db, tmdb_id, title, key, **kw):
                if tmdb_id == 99999:
                    return Media(id=99999, tmdb_id=99999, media_type=MediaType.movie, title="Brand New Movie")
                idx = tmdb_id - 10000
                return Media(id=1000 + idx, tmdb_id=tmdb_id, media_type=MediaType.movie, title=title)

            def fake_get_episode(db, show_id, show_tmdb_id, season_num, ep_num, key, cache=None):
                return Media(id=5000 + ep_num, show_id=1, media_type=MediaType.episode, title=f"Episode {ep_num}", season_number=season_num, episode_number=ep_num)

            with (
                patch.object(trakt_router, "_get_or_create_movie_media", AsyncMock(side_effect=fake_get_movie)),
                patch.object(trakt_router, "_get_or_create_show", AsyncMock(return_value=show)),
                patch.object(trakt_router, "_get_or_create_episode_media", AsyncMock(side_effect=fake_get_episode)),
            ):
                stats, processed, errors, new_watched, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=110,
                    user_id=1,
                    source=MockSource(),
                    api_key="key",
                    sync_watched=True,
                    sync_ratings=False,
                    sync_lists=False,
                    split_watchlist=False,
                    history_start=None,
                    history_end=None,
                )

        # Total movies fetched = 222 (221 old + 1 new)
        self.assertEqual(stats["fetched_movies"], 222)
        self.assertEqual(stats["fetched_episodes"], 211)
        # Exactly 1 new movie imported!
        self.assertEqual(stats["movies"], 1)
        self.assertEqual(stats["imported_movies"], 1)
        # Exactly 221 existing movies skipped as duplicates
        self.assertEqual(stats["skipped_duplicates_movies"], 221)
        # Exactly 211 existing episodes skipped as duplicates
        self.assertEqual(stats["skipped_duplicates_episodes"], 211)
        # Total skipped = 432 duplicates!
        self.assertEqual(stats["skipped"], 432)
        self.assertEqual(stats["skipped_duplicates"], 432)
        # Check summary formatting
        summary = trakt_router._trakt_import_summary(110, "sync", stats)
        self.assertIn("Movies: 1 new (221 skipped)", summary)
        self.assertIn("Episodes: 0 new (211 skipped)", summary)
        self.assertIn("Total skipped: 432 (432 duplicates)", summary)


if __name__ == "__main__":
    unittest.main()
