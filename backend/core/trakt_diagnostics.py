"""Trakt Synchronization Diagnostics Utility.

Read-only diagnostic tool to inspect Trakt watch history retrieval and account identity.
Compares:
  A. Current Full resync behavior (arguments Scrob sends: start_at=None, end_at=cutoff)
  B. Unrestricted history (unbounded: start_at=None, end_at=None)

Does not modify the database or any synchronization state.
"""

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core import trakt as trakt_client
from core.timeutil import utcnow
from core.watch_dedup import load_existing_watch_times
from models.media import Media, MediaType
from models.show import Show
from models.users import User, UserSettings

logger = logging.getLogger(__name__)


@dataclass
class HistoryRunDiagnostics:
    media_type: str  # "movies" or "episodes"
    mode: str        # "current_full_resync" or "unrestricted"
    start_at_arg: Optional[str]
    end_at_arg: Optional[str]
    total_records: int = 0
    pages_fetched: int = 0
    records_per_page: list[int] = field(default_factory=list)
    most_recent_watched_at: Optional[str] = None
    oldest_watched_at: Optional[str] = None
    target_match_found: bool = False
    target_match_details: Optional[dict] = None
    sample_recent_ids: list[int] = field(default_factory=list)


@dataclass
class TraktDiagnosticReport:
    timestamp_utc: str
    user_id: int
    trakt_account_identity: Optional[dict] = None
    account_matches_expected: Optional[bool] = None
    mode_a_current_movies: Optional[HistoryRunDiagnostics] = None
    mode_b_unrestricted_movies: Optional[HistoryRunDiagnostics] = None
    mode_a_current_episodes: Optional[HistoryRunDiagnostics] = None
    mode_b_unrestricted_episodes: Optional[HistoryRunDiagnostics] = None
    movies_discrepancy_count: int = 0
    episodes_discrepancy_count: int = 0
    unrestricted_recent_movies_excluded_by_mode_a: list[dict] = field(default_factory=list)
    unrestricted_recent_episodes_excluded_by_mode_a: list[dict] = field(default_factory=list)
    local_db_status: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


async def _fetch_history_with_diagnostics(
    client: httpx.AsyncClient,
    path: str,
    media_type: str,
    mode: str,
    client_id: str,
    access_token: str,
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    target_tmdb_id: Optional[int] = None,
    target_title: Optional[str] = None,
    target_watched_at: Optional[datetime] = None,
) -> tuple[HistoryRunDiagnostics, list[dict]]:
    """Fetch history from Trakt recording pagination details without modifying state."""
    headers = trakt_client._headers(client_id, access_token)
    extra_params = {
        key: trakt_client._iso_utc(value)
        for key, value in (("start_at", start_at), ("end_at", end_at))
        if value is not None
    }

    diag = HistoryRunDiagnostics(
        media_type=media_type,
        mode=mode,
        start_at_arg=extra_params.get("start_at"),
        end_at_arg=extra_params.get("end_at"),
    )

    all_items: list[dict] = []
    page = 1
    while True:
        params = dict(extra_params)
        params.update({"page": page, "limit": trakt_client.PAGE_SIZE})

        response = await client.get(
            f"{trakt_client.TRAKT_BASE}{path}",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        page_items = response.json()
        if not isinstance(page_items, list):
            break

        count = len(page_items)
        diag.pages_fetched += 1
        diag.records_per_page.append(count)
        all_items.extend(page_items)

        if not page_items:
            break

        try:
            page_count = int(response.headers.get("X-Pagination-Page-Count", page))
        except (TypeError, ValueError):
            page_count = page

        if page >= page_count:
            break
        page += 1

    diag.total_records = len(all_items)
    if all_items:
        # Trakt history is reverse-chronological by default
        diag.most_recent_watched_at = all_items[0].get("watched_at")
        diag.oldest_watched_at = all_items[-1].get("watched_at")
        diag.sample_recent_ids = [item.get("id") for item in all_items[:10] if item.get("id")]

    # Check for target watch if supplied
    for item in all_items:
        item_id = item.get("id")
        item_watched_at = item.get("watched_at")
        movie_or_show = item.get("movie") or item.get("show") or {}
        ids = movie_or_show.get("ids", {})
        title = movie_or_show.get("title", "")
        tmdb_id = ids.get("tmdb")

        matched = False
        if target_tmdb_id and tmdb_id == target_tmdb_id:
            matched = True
        elif target_title and target_title.lower() in title.lower():
            matched = True
        elif target_watched_at and item_watched_at:
            parsed = trakt_client._parse_trakt_datetime(item_watched_at)
            if parsed and abs((parsed - target_watched_at).total_seconds()) <= 300:
                matched = True

        if matched:
            diag.target_match_found = True
            diag.target_match_details = {
                "trakt_history_id": item_id,
                "title": title,
                "tmdb_id": tmdb_id,
                "watched_at": item_watched_at,
            }
            break

    return diag, all_items


async def run_trakt_diagnostics(
    db: AsyncSession,
    user_id: int,
    expected_username: Optional[str] = None,
    target_tmdb_id: Optional[int] = None,
    target_title: Optional[str] = None,
    target_watched_at: Optional[datetime] = None,
) -> TraktDiagnosticReport:
    """Run full safe, read-only diagnostic comparison for a user's Trakt integration."""
    now_utc = datetime.now(timezone.utc)
    report = TraktDiagnosticReport(
        timestamp_utc=now_utc.isoformat(),
        user_id=user_id,
    )

    # 1. Inspect user credentials (read-only)
    settings_res = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    settings = settings_res.scalar_one_or_none()

    if not settings or not settings.trakt_access_token or not settings.trakt_client_id:
        report.summary = "Trakt is not connected or credentials missing."
        return report

    client_id = settings.trakt_client_id
    access_token = settings.trakt_access_token

    # 2. Check authenticated identity via /users/me
    profile = await trakt_client.get_user_profile(client_id, access_token)
    report.trakt_account_identity = profile
    if profile and expected_username:
        actual_user = (profile.get("username") or "").lower()
        report.account_matches_expected = (actual_user == expected_username.lower())

    # 3. Mode A: Current full resync behavior (start_at=None, end_at=cutoff)
    history_cutoff = now_utc.replace(tzinfo=None)
    # Replicate current Scrob _history_window(cursor=None, full_resync=True, cutoff=history_cutoff)
    mode_a_start = None
    mode_a_end = history_cutoff

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        # Movies: Mode A
        diag_a_movies, items_a_movies = await _fetch_history_with_diagnostics(
            http_client,
            "/sync/history/movies",
            "movies",
            "current_full_resync",
            client_id,
            access_token,
            start_at=mode_a_start,
            end_at=mode_a_end,
            target_tmdb_id=target_tmdb_id,
            target_title=target_title,
            target_watched_at=target_watched_at,
        )
        report.mode_a_current_movies = diag_a_movies

        # Movies: Mode B (Unrestricted)
        diag_b_movies, items_b_movies = await _fetch_history_with_diagnostics(
            http_client,
            "/sync/history/movies",
            "movies",
            "unrestricted",
            client_id,
            access_token,
            start_at=None,
            end_at=None,
            target_tmdb_id=target_tmdb_id,
            target_title=target_title,
            target_watched_at=target_watched_at,
        )
        report.mode_b_unrestricted_movies = diag_b_movies

        # Episodes: Mode A
        diag_a_episodes, items_a_episodes = await _fetch_history_with_diagnostics(
            http_client,
            "/sync/history/episodes",
            "episodes",
            "current_full_resync",
            client_id,
            access_token,
            start_at=mode_a_start,
            end_at=mode_a_end,
            target_tmdb_id=target_tmdb_id,
            target_title=target_title,
            target_watched_at=target_watched_at,
        )
        report.mode_a_current_episodes = diag_a_episodes

        # Episodes: Mode B (Unrestricted)
        diag_b_episodes, items_b_episodes = await _fetch_history_with_diagnostics(
            http_client,
            "/sync/history/episodes",
            "episodes",
            "unrestricted",
            client_id,
            access_token,
            start_at=None,
            end_at=None,
            target_tmdb_id=target_tmdb_id,
            target_title=target_title,
            target_watched_at=target_watched_at,
        )
        report.mode_b_unrestricted_episodes = diag_b_episodes

    # 4. Compare discrepancies
    ids_a_movies = {item["id"] for item in items_a_movies if "id" in item}
    ids_b_movies = {item["id"] for item in items_b_movies if "id" in item}
    excluded_movie_ids = ids_b_movies - ids_a_movies
    report.movies_discrepancy_count = len(excluded_movie_ids)

    for item in items_b_movies:
        if item.get("id") in excluded_movie_ids:
            report.unrestricted_recent_movies_excluded_by_mode_a.append({
                "trakt_id": item.get("id"),
                "title": item.get("movie", {}).get("title"),
                "tmdb_id": item.get("movie", {}).get("ids", {}).get("tmdb"),
                "watched_at": item.get("watched_at"),
            })

    ids_a_episodes = {item["id"] for item in items_a_episodes if "id" in item}
    ids_b_episodes = {item["id"] for item in items_b_episodes if "id" in item}
    excluded_episode_ids = ids_b_episodes - ids_a_episodes
    report.episodes_discrepancy_count = len(excluded_episode_ids)

    for item in items_b_episodes:
        if item.get("id") in excluded_episode_ids:
            report.unrestricted_recent_episodes_excluded_by_mode_a.append({
                "trakt_id": item.get("id"),
                "show_title": item.get("show", {}).get("title"),
                "episode": f"S{item.get('episode', {}).get('season')}E{item.get('episode', {}).get('number')}",
                "watched_at": item.get("watched_at"),
            })

    # 5. Local database duplicate check (read-only)
    existing_times = await load_existing_watch_times(db, user_id)
    report.local_db_status = {
        "distinct_media_with_watches": len(existing_times),
        "total_local_watch_events": sum(len(times) for times in existing_times.values()),
        "stored_trakt_cursor": settings.trakt_history_cursor_at.isoformat() if settings.trakt_history_cursor_at else None,
    }

    # Summary analysis
    summary_lines = []
    if profile:
        summary_lines.append(f"Authenticated as Trakt user: '{profile.get('username')}' (slug: {profile.get('slug')}).")
    summary_lines.append(
        f"Movies: Mode A (current full resync) fetched {diag_a_movies.total_records} plays ({diag_a_movies.pages_fetched} pages); "
        f"Mode B (unrestricted) fetched {diag_b_movies.total_records} plays ({diag_b_movies.pages_fetched} pages). "
        f"Difference: {report.movies_discrepancy_count} excluded by end_at parameter."
    )
    summary_lines.append(
        f"Episodes: Mode A fetched {diag_a_episodes.total_records} plays ({diag_a_episodes.pages_fetched} pages); "
        f"Mode B fetched {diag_b_episodes.total_records} plays ({diag_b_episodes.pages_fetched} pages). "
        f"Difference: {report.episodes_discrepancy_count} excluded by end_at parameter."
    )
    report.summary = "\n".join(summary_lines)
    return report


async def _main():
    parser = argparse.ArgumentParser(description="Trakt Synchronization Diagnostics")
    parser.add_argument("--user-id", type=int, default=1, help="Scrob User ID (default: 1)")
    parser.add_argument("--expected-user", type=str, default=None, help="Expected Trakt username")
    parser.add_argument("--target-title", type=str, default=None, help="Specific title to search for in history")
    parser.add_argument("--target-tmdb", type=int, default=None, help="Specific TMDB ID to search for")
    args = parser.parse_args()

    from db import engine
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        report = await run_trakt_diagnostics(
            session,
            user_id=args.user_id,
            expected_username=args.expected_user,
            target_title=args.target_title,
            target_tmdb_id=args.target_tmdb,
        )

    import json
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
