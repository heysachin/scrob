import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from models import (
    Base,
    Media,
    MediaType,
    PlaybackProgress,
    Show,
    User,
    UserSettings,
    WatchEvent,
)
from models.connections import MediaServerConnection
from routers import history as history_router
from routers import sync as sync_router
from routers import trakt as trakt_router


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


def _sqlite_json_deserializer(value):
    if isinstance(value, str) and value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return json.loads(value)


class NuvioAutoPushAndTimingTests(unittest.IsolatedAsyncioTestCase):
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
            user = User(id=1, username="testuser", email="test@example.com", api_key="test-key")
            s.add(user)
            settings = UserSettings(user_id=1, tmdb_api_key="mock-key")
            s.add(settings)
            await s.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    # 1. Test _nuvio_progress_item handles progress_seconds <= 0 with progress_percent > 0
    def test_nuvio_progress_item_handles_zero_seconds_fallback(self):
        movie = Media(id=1, media_type=MediaType.movie, tmdb_data={"external_ids": {"imdb_id": "tt1234567"}}, runtime=0)
        progress = PlaybackProgress(
            user_id=1,
            media_id=1,
            progress_percent=0.50,
            progress_seconds=0,
            updated_at=datetime(2026, 10, 23, 12, 0, 0),
        )
        item = sync_router._nuvio_progress_item(progress, movie)
        self.assertIsNotNone(item)
        self.assertEqual(item["content_id"], "tt1234567")
        self.assertEqual(item["content_type"], "movie")
        # 50% of fallback 120 mins (7,200,000 ms) = 3,600,000 ms
        self.assertEqual(item["duration"], 7_200_000)
        self.assertEqual(item["position"], 3_600_000)

    # 2. Test Continue Watching purges completed items and deduplicates shows by true last watched
    async def test_continue_watching_purges_completed_and_deduplicates_by_show(self):
        async with self.Session() as s:
            show = Show(id=10, tmdb_id=100, title="Show 1")
            s.add(show)
            ep1 = Media(id=101, show_id=10, media_type=MediaType.episode, title="Ep 1", season_number=1, episode_number=1)
            ep2 = Media(id=102, show_id=10, media_type=MediaType.episode, title="Ep 2", season_number=1, episode_number=2)
            ep3 = Media(id=103, show_id=10, media_type=MediaType.episode, title="Ep 3", season_number=1, episode_number=3)
            movie1 = Media(id=201, media_type=MediaType.movie, title="Finished Movie")
            movie2 = Media(id=202, media_type=MediaType.movie, title="Active Movie")
            s.add_all([ep1, ep2, ep3, movie1, movie2])

            # Movie 1 was paused on Monday, but completed on Tuesday
            s.add(PlaybackProgress(user_id=1, media_id=201, progress_percent=0.4, progress_seconds=2400, updated_at=datetime(2026, 10, 20, 10, 0, 0)))
            s.add(WatchEvent(user_id=1, media_id=201, completed=True, watched_at=datetime(2026, 10, 21, 10, 0, 0)))

            # Movie 2 was paused on Wednesday (active)
            s.add(PlaybackProgress(user_id=1, media_id=202, progress_percent=0.7, progress_seconds=4200, updated_at=datetime(2026, 10, 22, 10, 0, 0)))

            # Show 1 Ep 1 was paused on Monday, but completed on Tuesday
            s.add(PlaybackProgress(user_id=1, media_id=101, progress_percent=0.3, progress_seconds=1000, updated_at=datetime(2026, 10, 20, 10, 0, 0)))
            s.add(WatchEvent(user_id=1, media_id=101, completed=True, watched_at=datetime(2026, 10, 21, 12, 0, 0)))

            # Show 1 Ep 2 was completed on Wednesday
            s.add(WatchEvent(user_id=1, media_id=102, completed=True, watched_at=datetime(2026, 10, 22, 12, 0, 0)))

            # Show 1 Ep 3 was paused on Thursday (active)
            s.add(PlaybackProgress(user_id=1, media_id=103, progress_percent=0.5, progress_seconds=1800, updated_at=datetime(2026, 10, 23, 10, 0, 0)))

            await s.commit()

            # Now call get_continue_watching
            user = (await s.execute(select(User).where(User.id == 1))).scalar_one()
            result = await history_router.get_continue_watching(db=s, current_user=user)
            cw_items = result["continue_watching"]

            # Expected:
            # - Movie 1 (completed) is purged!
            # - Show 1 Ep 1 (superseded) is purged!
            # - Exactly 2 items remain: Show 1 Ep 3 and Movie 2
            # - Show 1 Ep 3 (Thursday 10:00) is before Movie 2 (Wednesday 10:00)
            self.assertEqual(len(cw_items), 2)
            self.assertEqual(cw_items[0]["media"]["id"], 103)  # Show 1 Ep 3
            self.assertEqual(cw_items[1]["media"]["id"], 202)  # Movie 2

            # Also verify purged from PlaybackProgress table
            all_pps = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.user_id == 1))).scalars().all()
            pp_media_ids = {pp.media_id for pp in all_pps}
            self.assertEqual(pp_media_ids, {103, 202})

    # 3. Test _fan_out_changes_to_other_connections pushes playback to Nuvio
    async def test_fan_out_pushes_playback_to_nuvio_connection(self):
        async with self.Session() as s:
            conn = MediaServerConnection(
                id=1,
                user_id=1,
                type="nuvio",
                name="Nuvio",
                url="https://api.nuvio.tv",
                token="test-token",
                server_user_id="1",
                push_playback=True,
            )
            s.add(conn)
            movie = Media(id=301, media_type=MediaType.movie, title="Test Movie", tmdb_data={"external_ids": {"imdb_id": "tt9999999"}})
            s.add(movie)
            s.add(PlaybackProgress(user_id=1, media_id=301, progress_percent=0.45, progress_seconds=2700, updated_at=datetime(2026, 10, 23, 10, 0, 0)))
            await s.commit()

            with patch("core.nuvio.push_watch_progress", AsyncMock()) as mock_push:
                await sync_router._fan_out_changes_to_other_connections(
                    s,
                    user_id=1,
                    exclude_connection_id=None,
                    new_watched_ids=set(),
                    new_ratings={},
                    new_playback_media_ids={301},
                )
                self.assertEqual(mock_push.call_count, 1)
                args, kwargs = mock_push.call_args
                items = args[3]
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["content_id"], "tt9999999")
                self.assertEqual(items[0]["content_type"], "movie")

    # 4. Test _nuvio_imdb_id resolves directly from Media.imdb_id even without tmdb_data
    def test_nuvio_imdb_id_resolves_from_direct_column(self):
        movie_with_direct_imdb = Media(id=401, media_type=MediaType.movie, title="Direct IMDb Movie", imdb_id="tt1122334", tmdb_data=None)
        self.assertEqual(sync_router._nuvio_imdb_id(movie_with_direct_imdb), "tt1122334")

        watched_item = sync_router._nuvio_watched_item(movie_with_direct_imdb, datetime(2026, 10, 23, 10, 0, 0))
        self.assertIsNotNone(watched_item)
        self.assertEqual(watched_item["content_id"], "tt1122334")
        self.assertEqual(watched_item["content_type"], "movie")

        progress = PlaybackProgress(user_id=1, media_id=401, progress_percent=0.5, progress_seconds=3600, updated_at=datetime(2026, 10, 23, 10, 0, 0))
        progress_item = sync_router._nuvio_progress_item(progress, movie_with_direct_imdb)
        self.assertIsNotNone(progress_item)
        self.assertEqual(progress_item["content_id"], "tt1122334")
        self.assertEqual(progress_item["content_type"], "movie")

    # 5. Test Trakt playback movies are NOT skipped by past completed watch events (only strictly newer watches)
    async def test_trakt_playback_movie_not_skipped_by_past_watch(self):
        async with self.Session() as s:
            movie = Media(id=501, tmdb_id=5001, media_type=MediaType.movie, title="Rewatched Movie", imdb_id="tt5001")
            s.add(movie)
            # Past completed watch from Monday
            s.add(WatchEvent(user_id=1, media_id=501, completed=True, watched_at=datetime(2026, 10, 20, 10, 0, 0)))
            job = trakt_router.SyncJob(id=501, user_id=1, source="trakt", status="pending")
            s.add(job)
            await s.commit()

            class TraktPlaybackSource:
                async def get_history_movies(self, start_at=None, end_at=None): return []
                async def get_history_episodes(self, start_at=None, end_at=None): return []
                async def get_playback_episodes(self): return []
                async def get_ratings(self): return {}
                async def get_watchlist(self): return []
                async def get_user_lists(self): return []
                async def get_playback_movies(self):
                    return [{
                        "id": 9991,
                        "progress": 40.0,
                        "paused_at": "2026-10-23T15:00:00.000Z",
                        "type": "movie",
                        "movie": {
                            "title": "Rewatched Movie",
                            "ids": {"tmdb": 5001, "imdb": "tt5001"},
                        },
                    }]

            stats, _, _, new_watched, _ = await trakt_router._apply_trakt_import(
                s, job_id=501, user_id=1, source=TraktPlaybackSource(), api_key=None,
                sync_watched=True, sync_ratings=False, sync_lists=False, split_watchlist=False,
                history_start=None, history_end=None,
            )

            self.assertEqual(stats["playback_movies"], 1)
            self.assertEqual(stats.get("skipped_playback_stale", 0), 0)
            pps = (await s.execute(select(PlaybackProgress).where(PlaybackProgress.media_id == 501))).scalars().all()
            self.assertEqual(len(pps), 1)
            self.assertEqual(pps[0].progress_percent, 0.40)

    # 6. Test push_on_pull=True automatically fans out on pull events
    async def test_pull_event_fans_out_to_nuvio_when_push_on_pull_true(self):
        async with self.Session() as s:
            conn = MediaServerConnection(
                id=601,
                user_id=1,
                type="nuvio",
                name="Nuvio Cloud",
                url="https://api.nuvio.tv",
                token="test-token",
                server_user_id="1",
                push_watched=False,
                push_playback=False,
                push_on_pull=True,
            )
            s.add(conn)
            movie = Media(id=601, media_type=MediaType.movie, title="Pull Movie", imdb_id="tt6000001")
            s.add(movie)
            s.add(WatchEvent(user_id=1, media_id=601, completed=True, watched_at=datetime(2026, 10, 23, 10, 0, 0)))
            s.add(PlaybackProgress(user_id=1, media_id=601, progress_percent=0.6, progress_seconds=3600, updated_at=datetime(2026, 10, 23, 12, 0, 0)))
            await s.commit()

            with patch("core.nuvio.push_watched_items", AsyncMock()) as mock_push_watched, \
                 patch("core.nuvio.push_watch_progress", AsyncMock()) as mock_push_progress:
                await sync_router._fan_out_changes_to_other_connections(
                    s,
                    user_id=1,
                    exclude_connection_id=None,
                    new_watched_ids={601},
                    new_ratings={},
                    new_playback_media_ids={601},
                    is_pull=True,
                )
                self.assertEqual(mock_push_watched.call_count, 1)
                self.assertEqual(mock_push_progress.call_count, 1)
                # Verify items sent to Nuvio
                w_items = mock_push_watched.call_args[0][3]
                p_items = mock_push_progress.call_args[0][3]
                self.assertEqual(w_items[0]["content_id"], "tt6000001")
                self.assertEqual(p_items[0]["content_id"], "tt6000001")

    # 7. Test push_on_pull=False blocks fan-out on pull events
    async def test_pull_event_does_not_fan_out_when_push_on_pull_false(self):
        async with self.Session() as s:
            conn = MediaServerConnection(
                id=701,
                user_id=1,
                type="nuvio",
                name="Nuvio Cloud",
                url="https://api.nuvio.tv",
                token="test-token",
                server_user_id="1",
                push_watched=False,
                push_playback=False,
                push_on_pull=False,
            )
            s.add(conn)
            movie = Media(id=701, media_type=MediaType.movie, title="Pull Movie 2", imdb_id="tt7000002")
            s.add(movie)
            await s.commit()

            with patch("core.nuvio.push_watched_items", AsyncMock()) as mock_push_watched:
                await sync_router._fan_out_changes_to_other_connections(
                    s,
                    user_id=1,
                    exclude_connection_id=None,
                    new_watched_ids={701},
                    new_ratings={},
                    is_pull=True,
                )
                self.assertEqual(mock_push_watched.call_count, 0)

    # 8. Test non-Nuvio servers (e.g. Plex) are not pushed during pull events (is_pull=True)
    async def test_pull_event_does_not_push_to_other_media_servers(self):
        async with self.Session() as s:
            plex_conn = MediaServerConnection(
                id=801,
                user_id=1,
                type="plex",
                name="Home Plex",
                url="http://192.168.1.100:32400",
                token="plex-token",
                push_watched=True,
            )
            s.add(plex_conn)
            movie = Media(id=801, media_type=MediaType.movie, title="Plex Movie", tmdb_id=8001)
            s.add(movie)
            await s.commit()

            with patch("core.plex.mark_watched", AsyncMock()) as mock_plex_watched:
                await sync_router._fan_out_changes_to_other_connections(
                    s,
                    user_id=1,
                    exclude_connection_id=None,
                    new_watched_ids={801},
                    new_ratings={},
                    is_pull=True,
                )
                self.assertEqual(mock_plex_watched.call_count, 0)

    # 9. Test _push_nuvio_playback_progress respects push_on_pull=True
    async def test_push_nuvio_playback_progress_respects_push_on_pull(self):
        async with self.Session() as s:
            conn = MediaServerConnection(
                id=901,
                user_id=1,
                type="nuvio",
                name="Nuvio Cloud",
                url="https://api.nuvio.tv",
                token="test-token",
                server_user_id="1",
                push_playback=False,
                push_on_pull=True,
            )
            s.add(conn)
            movie = Media(id=901, media_type=MediaType.movie, title="Progress Movie", imdb_id="tt9000001")
            s.add(movie)
            s.add(PlaybackProgress(user_id=1, media_id=901, progress_percent=0.75, progress_seconds=4500, updated_at=datetime(2026, 10, 23, 14, 0, 0)))
            await s.commit()

            with patch("core.nuvio.push_watch_progress", AsyncMock()) as mock_push_progress:
                await sync_router._push_nuvio_playback_progress(s, user_id=1, media_ids=[901])
                self.assertEqual(mock_push_progress.call_count, 1)
