"""Automated tests for Trakt Continue Watching (Playback Progress) synchronization.

Covers:
1. Movie playback imported to PlaybackProgress (progress %, seconds, paused_at, trakt_playback_id).
2. Episode playback imported to PlaybackProgress.
3. Unfinished playback is never stored as a completed WatchEvent.
4. Repeated sync is idempotent (updates progress, does not create duplicate rows).
5. Conflict handling: older Trakt playback does NOT overwrite newer local progress (e.g. from Plex).
6. Conflict handling: newer Trakt playback overwrites older local progress.
7. Stale playback skipped if movie/episode was already completed at or after paused_at.
8. Completed watch event imported during sync cleans up corresponding PlaybackProgress.
9. Removed Trakt playback entry (present previously, now gone from Trakt) is cleaned up safely,
   while non-Trakt local progress is preserved.
10. Missing TMDB identifiers and malformed metadata handled safely without losing entries or crashing.
11. Empty playback API response handled cleanly.
12. Trakt core API methods call /sync/playback/movies and /sync/playback/episodes.
"""

import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

import httpx
_REAL_ASYNC_CLIENT = httpx.AsyncClient
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core import trakt as trakt_client
from core.timeutil import utcnow
from models import (
    Base,
    CollectionSource,
    GlobalSettings,
    Media,
    MediaType,
    PlaybackProgress,
    Show,
    SyncJob,
    SyncStatus,
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


class BaseMockSource:
    async def get_history_movies(self, start_at=None, end_at=None): return []
    async def get_history_episodes(self, start_at=None, end_at=None): return []
    async def get_playback_movies(self): return []
    async def get_playback_episodes(self): return []
    async def get_ratings(self): return {}
    async def get_watchlist(self): return []
    async def get_user_lists(self): return []


class TraktPlaybackSyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            json_deserializer=_sqlite_json_deserializer,
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self.Session = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

        async with self.Session() as s:
            user = User(id=1, username="testuser", email="test@example.com", api_key="test-api-key")
            s.add(user)
            settings = UserSettings(
                user_id=1,
                tmdb_api_key="mock-tmdb-key",
                trakt_client_id="test_client_id",
                trakt_client_secret="test_client_secret",
                trakt_access_token="test_access_token",
                trakt_refresh_token="test_refresh_token",
                trakt_sync_watched=True,
            )
            s.add(settings)
            gs = GlobalSettings(id=1, tmdb_api_key="global-tmdb-key")
            s.add(gs)
            await s.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    # 1. Movie playback imported to PlaybackProgress with correct progress %, runtime seconds, paused_at, and trakt_playback_id
    async def test_movie_playback_imported_to_playback_progress(self):
        movie_media = Media(
            id=101,
            tmdb_id=550,
            media_type=MediaType.movie,
            title="Fight Club",
            runtime=139,  # 139 mins = 8340s
        )

        trakt_playback_movies = [
            {
                "id": 987654,
                "progress": 45.0,  # 45%
                "paused_at": "2026-10-22T08:30:00.000Z",
                "type": "movie",
                "movie": {
                    "title": "Fight Club",
                    "year": 1999,
                    "ids": {"tmdb": 550, "trakt": 123},
                },
            }
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return []
            async def get_history_episodes(self, start_at, end_at):
                return []
            async def get_playback_movies(self):
                return trakt_playback_movies
            async def get_playback_episodes(self):
                return []
            async def get_ratings(self):
                return {}
            async def get_watchlist(self):
                return []
            async def get_user_lists(self):
                return []

        async with self.Session() as s:
            s.add(movie_media)
            job = SyncJob(id=201, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=201,
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

        self.assertEqual(stats["playback_movies"], 1)
        self.assertEqual(stats["fetched_playback_movies"], 1)

        async with self.Session() as s:
            # Verify PlaybackProgress created
            pp = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 101))).scalar_one()
            self.assertAlmostEqual(pp.progress_percent, 0.45, places=3)
            # 45% of 139 minutes (8340 seconds) = 3753 seconds
            self.assertEqual(pp.progress_seconds, 3753)
            self.assertEqual(pp.trakt_playback_id, 987654)
            self.assertEqual(pp.updated_at, datetime(2026, 10, 22, 8, 30, 0))

            # Verify NO WatchEvent created (must not store unfinished playback as completed watch event)
            events = (await s.execute(select(WatchEvent).where(WatchEvent.user_id == 1, WatchEvent.media_id == 101))).scalars().all()
            self.assertEqual(len(events), 0)

    # 2. Episode playback imported to PlaybackProgress with show and episode mapping
    async def test_episode_playback_imported_to_playback_progress(self):
        show = Show(id=10, tmdb_id=1399, title="Game of Thrones")
        ep_media = Media(
            id=202,
            show_id=10,
            media_type=MediaType.episode,
            title="Winter Is Coming",
            season_number=1,
            episode_number=1,
            runtime=62,  # 62 mins = 3720s
        )

        trakt_playback_episodes = [
            {
                "id": 987655,
                "progress": 50.0,
                "paused_at": "2026-10-22T09:00:00.000Z",
                "type": "episode",
                "show": {"title": "Game of Thrones", "ids": {"tmdb": 1399}},
                "episode": {"season": 1, "number": 1, "title": "Winter Is Coming", "ids": {"tmdb": 63056}},
            }
        ]

        class MockSource:
            async def get_history_movies(self, start_at, end_at):
                return []
            async def get_history_episodes(self, start_at, end_at):
                return []
            async def get_playback_movies(self):
                return []
            async def get_playback_episodes(self):
                return trakt_playback_episodes
            async def get_ratings(self):
                return {}
            async def get_watchlist(self):
                return []
            async def get_user_lists(self):
                return []

        async with self.Session() as s:
            s.add(show)
            s.add(ep_media)
            job = SyncJob(id=202, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            with (
                patch.object(trakt_router, "_get_or_create_show", autospec=True, return_value=show),
                patch.object(trakt_router, "_get_or_create_episode_media", autospec=True, return_value=ep_media),
            ):
                stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s,
                    job_id=202,
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

        self.assertEqual(stats["playback_episodes"], 1)

        async with self.Session() as s:
            pp = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 202))).scalar_one()
            self.assertAlmostEqual(pp.progress_percent, 0.50, places=3)
            # 50% of 62 mins (3720s) = 1860s
            self.assertEqual(pp.progress_seconds, 1860)
            self.assertEqual(pp.trakt_playback_id, 987655)
            self.assertEqual(pp.updated_at, datetime(2026, 10, 22, 9, 0, 0))

    # 3. Repeated sync is idempotent (updates existing row, never creates duplicates)
    async def test_idempotent_playback_sync_no_duplicates(self):
        movie_media = Media(id=103, tmdb_id=600, media_type=MediaType.movie, title="Movie 3", runtime=100)

        def make_source(progress_pct):
            class Source(BaseMockSource):
                async def get_playback_movies(self):
                    return [{
                        "id": 999001,
                        "progress": progress_pct,
                        "paused_at": "2026-10-22T10:00:00.000Z",
                        "type": "movie",
                        "movie": {"title": "Movie 3", "ids": {"tmdb": 600}},
                    }]
            return Source()

        async with self.Session() as s:
            s.add(movie_media)
            job1 = SyncJob(id=203, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job1)
            await s.commit()

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                await trakt_router._apply_trakt_import(
                    s, job_id=203, user_id=1, source=make_source(30.0), api_key="k",
                    sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                    history_start=None, history_end=None,
                )

        # Run again with progress updated to 55%
        async with self.Session() as s:
            job2 = SyncJob(id=204, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job2)
            await s.commit()

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                await trakt_router._apply_trakt_import(
                    s, job_id=204, user_id=1, source=make_source(55.0), api_key="k",
                    sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                    history_start=None, history_end=None,
                )

        async with self.Session() as s:
            rows = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 103))).scalars().all()
            self.assertEqual(len(rows), 1, "Must update existing row, not create duplicates")
            self.assertAlmostEqual(rows[0].progress_percent, 0.55, places=3)

    # 4. Conflict handling: older Trakt playback does NOT overwrite newer local progress (e.g. from Plex)
    async def test_conflict_handling_preserves_newer_local_progress(self):
        movie_media = Media(id=104, tmdb_id=700, media_type=MediaType.movie, title="Movie 4", runtime=120)

        async with self.Session() as s:
            s.add(movie_media)
            # Local progress updated at 11:00 UTC (e.g. from Plex webhook) with 75%
            local_pp = PlaybackProgress(
                user_id=1,
                media_id=104,
                progress_percent=0.75,
                progress_seconds=5400,
                updated_at=datetime(2026, 10, 22, 11, 0, 0),
                trakt_playback_id=None,
            )
            s.add(local_pp)
            job = SyncJob(id=205, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            # Trakt has older progress from 09:00 UTC with 35%
            class OlderTraktSource(BaseMockSource):
                async def get_playback_movies(self):
                    return [{
                        "id": 999002,
                        "progress": 35.0,
                        "paused_at": "2026-10-22T09:00:00.000Z",
                        "type": "movie",
                        "movie": {"title": "Movie 4", "ids": {"tmdb": 700}},
                    }]

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s, job_id=205, user_id=1, source=OlderTraktSource(), api_key="k",
                    sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                    history_start=None, history_end=None,
                )

        # Verified that older Trakt playback was skipped as stale
        self.assertEqual(stats["playback_movies"], 0)
        self.assertEqual(stats["skipped_playback_stale"], 1)

        async with self.Session() as s:
            pp = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 104))).scalar_one()
            self.assertAlmostEqual(pp.progress_percent, 0.75, places=3, msg="Local progress must remain intact")
            self.assertEqual(pp.updated_at, datetime(2026, 10, 22, 11, 0, 0))

    # 5. Conflict handling: newer Trakt playback updates older local progress
    async def test_conflict_handling_overwrites_older_local_progress(self):
        movie_media = Media(id=105, tmdb_id=800, media_type=MediaType.movie, title="Movie 5", runtime=120)

        async with self.Session() as s:
            s.add(movie_media)
            # Local progress older: 08:00 UTC at 20%
            local_pp = PlaybackProgress(
                user_id=1,
                media_id=105,
                progress_percent=0.20,
                progress_seconds=1440,
                updated_at=datetime(2026, 10, 22, 8, 0, 0),
            )
            s.add(local_pp)
            job = SyncJob(id=206, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            # Trakt newer: 10:30 UTC at 65%
            class NewerTraktSource(BaseMockSource):
                async def get_playback_movies(self):
                    return [{
                        "id": 999003,
                        "progress": 65.0,
                        "paused_at": "2026-10-22T10:30:00.000Z",
                        "type": "movie",
                        "movie": {"title": "Movie 5", "ids": {"tmdb": 800}},
                    }]

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s, job_id=206, user_id=1, source=NewerTraktSource(), api_key="k",
                    sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                    history_start=None, history_end=None,
                )

        self.assertEqual(stats["playback_movies"], 1)

        async with self.Session() as s:
            pp = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 105))).scalar_one()
            self.assertAlmostEqual(pp.progress_percent, 0.65, places=3)
            self.assertEqual(pp.updated_at, datetime(2026, 10, 22, 10, 30, 0))
            self.assertEqual(pp.trakt_playback_id, 999003)

    # 6. Stale playback is skipped if media has already been completed at or after paused_at
    async def test_stale_playback_skipped_if_already_completed(self):
        movie_media = Media(id=106, tmdb_id=900, media_type=MediaType.movie, title="Completed Movie")

        async with self.Session() as s:
            s.add(movie_media)
            # Completed watch event at 12:00 UTC
            s.add(WatchEvent(
                id=301,
                user_id=1,
                media_id=106,
                source="trakt",
                watched_at=datetime(2026, 10, 22, 12, 0, 0),
                completed=True,
            ))
            job = SyncJob(id=207, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            # Trakt returns old in-progress playback from 08:00 UTC (before completion)
            class StaleTraktSource(BaseMockSource):
                async def get_playback_movies(self):
                    return [{
                        "id": 999004,
                        "progress": 50.0,
                        "paused_at": "2026-10-22T08:00:00.000Z",
                        "type": "movie",
                        "movie": {"title": "Completed Movie", "ids": {"tmdb": 900}},
                    }]

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                    s, job_id=207, user_id=1, source=StaleTraktSource(), api_key="k",
                    sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                    history_start=None, history_end=None,
                )

        self.assertEqual(stats["skipped_playback_stale"], 1)

        async with self.Session() as s:
            pps = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 106))).scalars().all()
            self.assertEqual(len(pps), 0, "Must not resurrect PlaybackProgress for completed item")

    # 7. Completed watch event imported during sync cleans up corresponding PlaybackProgress
    async def test_completed_watch_in_sync_clears_playback_progress(self):
        movie_media = Media(id=107, tmdb_id=1001, media_type=MediaType.movie, title="Newly Finished Movie")

        async with self.Session() as s:
            s.add(movie_media)
            # Initially user was in progress
            s.add(PlaybackProgress(
                user_id=1,
                media_id=107,
                progress_percent=0.60,
                progress_seconds=3600,
                updated_at=datetime(2026, 10, 22, 9, 0, 0),
                trakt_playback_id=55555,
            ))
            job = SyncJob(id=208, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            # Now sync brings in completed watch from Trakt history
            class FinishedSource(BaseMockSource):
                async def get_history_movies(self, start_at=None, end_at=None):
                    return [{
                        "id": 111111,
                        "watched_at": "2026-10-22T11:00:00.000Z",
                        "movie": {"title": "Newly Finished Movie", "ids": {"tmdb": 1001}},
                    }]

            with patch.object(trakt_router, "_get_or_create_movie_media", autospec=True, return_value=movie_media):
                stats, _, _, new_watched, _ = await trakt_router._apply_trakt_import(
                    s, job_id=208, user_id=1, source=FinishedSource(), api_key="k",
                    sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                    history_start=None, history_end=None,
                )

        self.assertEqual(stats["movies"], 1)
        self.assertIn(107, new_watched)

        async with self.Session() as s:
            # PlaybackProgress must have been cleared
            pps = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 107))).scalars().all()
            self.assertEqual(len(pps), 0, "PlaybackProgress must be cleared when item is watched to completion")

    # 8. Removed Trakt playback entry is cleaned up safely, while non-Trakt progress is preserved
    async def test_removed_trakt_playback_entry_cleaned_up(self):
        m_trakt = Media(id=108, tmdb_id=1002, media_type=MediaType.movie, title="Trakt Paused Movie")
        m_plex = Media(id=109, tmdb_id=1003, media_type=MediaType.movie, title="Plex Paused Movie")

        async with self.Session() as s:
            s.add(m_trakt)
            s.add(m_plex)
            # Item 1 was synced from Trakt (has trakt_playback_id)
            s.add(PlaybackProgress(
                user_id=1,
                media_id=108,
                progress_percent=0.40,
                progress_seconds=2400,
                trakt_playback_id=88888,
            ))
            # Item 2 was created by Plex webhook (trakt_playback_id is None)
            s.add(PlaybackProgress(
                user_id=1,
                media_id=109,
                progress_percent=0.70,
                progress_seconds=4200,
                trakt_playback_id=None,
            ))
            job = SyncJob(id=209, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            # Next Trakt sync returns empty playback (user dismissed it or finished it on Trakt)
            class EmptyPlaybackSource(BaseMockSource):
                pass

            stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                s, job_id=209, user_id=1, source=EmptyPlaybackSource(), api_key="k",
                sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                history_start=None, history_end=None,
            )

        self.assertEqual(stats["removed_playback"], 1)

        async with self.Session() as s:
            # Trakt playback removed
            pp_trakt = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 108))).scalar_one_or_none()
            self.assertIsNone(pp_trakt)
            # Plex playback strictly preserved
            pp_plex = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1, PlaybackProgress.media_id == 109))).scalar_one_or_none()
            self.assertIsNotNone(pp_plex)
            self.assertAlmostEqual(pp_plex.progress_percent, 0.70)

    # 9. Missing TMDB IDs in playback are handled safely without crashing
    async def test_missing_tmdb_id_in_playback_handled_safely(self):
        class MissingTmdbSource(BaseMockSource):
            async def get_playback_movies(self):
                return [{
                    "id": 1234,
                    "progress": 50.0,
                    "paused_at": "2026-10-22T08:00:00.000Z",
                    "type": "movie",
                    "movie": {"title": "No TMDB Movie", "ids": {}},  # No TMDB ID!
                }]
            async def get_playback_episodes(self):
                return [{
                    "id": 1235,
                    "progress": 50.0,
                    "paused_at": "2026-10-22T08:00:00.000Z",
                    "type": "episode",
                    "show": {"title": "No TMDB Show", "ids": {}},  # No TMDB ID!
                    "episode": {"season": 1, "number": 1},
                }]

        async with self.Session() as s:
            job = SyncJob(id=210, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                s, job_id=210, user_id=1, source=MissingTmdbSource(), api_key="k",
                sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                history_start=None, history_end=None,
            )

        self.assertEqual(stats["skipped_missing_tmdb"], 2)
        self.assertEqual(stats["playback_movies"], 0)
        self.assertEqual(stats["playback_episodes"], 0)

    # 10. Empty playback API response handled cleanly
    async def test_empty_playback_api_responses_handled_cleanly(self):
        class EmptySource(BaseMockSource):
            pass

        async with self.Session() as s:
            job = SyncJob(id=211, user_id=1, source=CollectionSource.trakt, status=SyncStatus.pending)
            s.add(job)
            await s.commit()

            stats, _, _, _, _ = await trakt_router._apply_trakt_import(
                s, job_id=211, user_id=1, source=EmptySource(), api_key="k",
                sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                history_start=None, history_end=None,
            )

        self.assertEqual(stats["playback_movies"], 0)
        self.assertEqual(stats["playback_episodes"], 0)
        self.assertEqual(stats["fetched_playback_movies"], 0)
        self.assertEqual(stats["fetched_playback_episodes"], 0)

    # 11. Trakt core API methods call /sync/playback/movies and /sync/playback/episodes
    async def test_trakt_core_get_playback_movies_and_episodes(self):
        requested_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            if "/sync/playback/movies" in request.url.path:
                return httpx.Response(200, json=[{"id": 1, "progress": 40.0, "type": "movie"}], headers={"X-Pagination-Page-Count": "1"})
            if "/sync/playback/episodes" in request.url.path:
                return httpx.Response(200, json=[{"id": 2, "progress": 60.0, "type": "episode"}], headers={"X-Pagination-Page-Count": "1"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        with patch("httpx.AsyncClient", side_effect=lambda **kw: _REAL_ASYNC_CLIENT(transport=transport, **kw)):
            movies = await trakt_client.get_playback_movies("client_id", "access_token")
            episodes = await trakt_client.get_playback_episodes("client_id", "access_token")

        self.assertEqual(len(movies), 1)
        self.assertEqual(movies[0]["id"], 1)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["id"], 2)
        self.assertTrue(any("/sync/playback/movies" in url for url in requested_urls))
        self.assertTrue(any("/sync/playback/episodes" in url for url in requested_urls))


if __name__ == "__main__":
    unittest.main()
