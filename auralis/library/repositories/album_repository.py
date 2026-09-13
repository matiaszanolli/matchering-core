"""
Album Repository
~~~~~~~~~~~~~~~

Data access layer for album operations

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

from pathlib import Path
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload, selectinload, with_expression

from ..artwork import create_artwork_extractor
from ..models import Album, Artist, Track
from .base import BaseRepository


def _track_count_subquery() -> Any:
    """Correlated COUNT of an album's tracks, evaluated per row by the DB
    engine — no JOIN, no row multiplication. Albums with no tracks yield 0
    (not NULL), so no COALESCE is needed. Mirrors the pattern already used
    in ArtistRepository.get_all() / PlaylistRepository.get_all() (#4777)."""
    return (
        select(func.count(Track.id))
        .where(Track.album_id == Album.id)
        .correlate(Album)
        .scalar_subquery()
    )


def _total_duration_subquery() -> Any:
    """Correlated SUM of an album's track durations. COALESCE because SUM
    over zero rows is NULL, unlike COUNT (#4777)."""
    return (
        select(func.coalesce(func.sum(Track.duration), 0.0))
        .where(Track.album_id == Album.id)
        .correlate(Album)
        .scalar_subquery()
    )


# Named eager-load constants (#5028), mirroring track_repository.py's
# _track_eager_options() / genre_repository.py's _GENRE_LOAD_OPTIONS
# convention. #4236 fixed a DetachedInstanceError caused by exactly this
# duplication (one method's inline tuple fixed, its sibling's copy left
# stale) and its own proposed fix recommended this extraction — done here so
# a future read path can't silently omit it again.

# Single-album detail reads: get_by_id/get_by_title hand the album straight
# to to_dict(), which reads both album.artist and album.tracks.
#
# Track relationships used by Track.to_dict() are eager-loaded too
# (#5170/#5260). Everything here is expunged before it is returned, so a lazy
# relationship access downstream otherwise degrades to an empty wire value.
_ALBUM_DETAIL_OPTIONS = (
    joinedload(Album.artist),
    selectinload(Album.tracks).selectinload(Track.genres),
    selectinload(Album.tracks).selectinload(Track.artists),
    selectinload(Album.tracks).joinedload(Track.album),
)

# Paginated/listing reads: track_count_expr/total_duration_expr cover what
# the list serializer needs — no eager Album.tracks (fixes #4777's N+1).
_ALBUM_LIST_OPTIONS = (
    joinedload(Album.artist),
    with_expression(Album.track_count_expr, _track_count_subquery()),
    with_expression(Album.total_duration_expr, _total_duration_subquery()),
)


class AlbumRepository(BaseRepository):
    """Repository for album database operations"""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        super().__init__(session_factory)

        # Initialize artwork extractor
        artwork_dir = Path.home() / ".auralis" / "artwork"
        self.artwork_extractor = create_artwork_extractor(str(artwork_dir))

    def get_by_id(self, album_id: int) -> Album | None:
        """Get album by ID with relationships loaded"""
        with self._session_scope() as session:
            album = session.execute(
                select(Album)
                .options(*_ALBUM_DETAIL_OPTIONS)
                .where(Album.id == album_id)
            ).scalars().unique().first()
            if album:
                # Expunge the album AND all eagerly-loaded related objects so
                # callers can access album.tracks after the session closes
                # without hitting DetachedInstanceError (fixes #2406).
                session.expunge_all()
            return album

    def get_by_title(self, title: str) -> Album | None:
        """Get album by title with relationships loaded"""
        with self._session_scope() as session:
            album = session.execute(
                select(Album)
                .options(*_ALBUM_DETAIL_OPTIONS)
                .where(Album.title == title)
            ).scalars().unique().first()
            if album:
                # Expunge the album AND all eagerly-loaded related objects so
                # callers can access album.tracks after the session closes
                # without hitting DetachedInstanceError (fixes #2406 / #4236).
                session.expunge_all()
            return album

    def get_all(self, limit: int = 50, offset: int = 0, order_by: str = 'title') -> tuple[list[Album], int]:
        """
        Get all albums with pagination and total count

        Args:
            limit: Maximum number of albums to return
            offset: Number of albums to skip
            order_by: Column to order by ('title', 'year', 'created_at')

        Returns:
            Tuple of (albums list, total count)
        """
        with self._session_scope() as session:
            # Get total count
            total = session.execute(
                select(func.count()).select_from(Album)
            ).scalar_one()

            # Get albums for current page (whitelist to prevent arbitrary attribute access)
            VALID_ORDER_COLUMNS = {'title', 'year', 'created_at'}
            if order_by not in VALID_ORDER_COLUMNS:
                order_by = 'title'
            order_column = getattr(Album, order_by, Album.title)
            albums = session.execute(
                select(Album)
                .options(*_ALBUM_LIST_OPTIONS)
                .order_by(order_column.asc())
                .limit(limit)
                .offset(offset)
            ).scalars().unique().all()

            # Expunge every album AND its eagerly-loaded artist (fixes #2406 /
            # #4236 — a per-item expunge(album) does not cascade to related
            # objects, leaving them DetachedInstanceError-prone). tracks is no
            # longer eager-loaded here (#4777) — track_count_expr/
            # total_duration_expr cover what the serializer actually needs.
            if albums:
                session.expunge_all()

            return albums, total

    def get_recent(self, limit: int = 50, offset: int = 0) -> list[Album]:
        """Get recently added albums with pagination"""
        with self._session_scope() as session:
            albums = session.execute(
                select(Album)
                .options(*_ALBUM_LIST_OPTIONS)
                .order_by(Album.created_at.desc())
                .limit(limit)
                .offset(offset)
            ).scalars().unique().all()
            if albums:
                session.expunge_all()
            return albums

    def search(self, query: str, limit: int = 50, offset: int = 0, order_by: str = 'title') -> tuple[list[Album], int]:
        """
        Search albums by title or artist name

        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Column to order by ('title', 'year', 'created_at')

        Returns:
            Tuple of (matching albums, total match count) — total enables correct
            pagination (fixes #2482: estimated total was always wrong).
        """
        with self._session_scope() as session:
            # Escape LIKE metacharacters so a query containing '%' or '_' does
            # not accidentally match all rows (fixes #2405).
            escaped = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            search_term = f"%{escaped}%"
            search_filter = or_(
                Album.title.ilike(search_term, escape='\\'),
                Artist.name.ilike(search_term, escape='\\')
            )

            total = session.execute(
                select(func.count(Album.id))
                .join(Album.artist, isouter=True)
                .where(search_filter)
            ).scalar_one_or_none() or 0

            # Whitelist order_by to prevent arbitrary attribute access
            VALID_ORDER_COLUMNS = {'title', 'year', 'created_at'}
            if order_by not in VALID_ORDER_COLUMNS:
                order_by = 'title'
            order_column = getattr(Album, order_by, Album.title)

            results = session.execute(
                select(Album)
                .join(Album.artist, isouter=True)
                .where(search_filter)
                # Deliberately not _ALBUM_LIST_OPTIONS (#5028 CONSISTENCY check):
                # this query already JOINs Album.artist explicitly for the WHERE
                # filter above, so joinedload(Album.artist) on the same
                # relationship would attach a second, conflicting join.
                # selectinload sidesteps that with its own separate query — the
                # other two expressions are identical to _ALBUM_LIST_OPTIONS.
                .options(
                    selectinload(Album.artist),
                    with_expression(Album.track_count_expr, _track_count_subquery()),
                    with_expression(Album.total_duration_expr, _total_duration_subquery()),
                )
                .order_by(order_column.asc())
                .limit(limit)
                .offset(offset)
            ).scalars().unique().all()

            if results:
                session.expunge_all()
            return results, total

    def extract_and_save_artwork(self, album_id: int) -> str | None:
        """
        Extract artwork from album's tracks and save it

        Args:
            album_id: Album ID to extract artwork for

        Returns:
            Path to saved artwork, or None if extraction failed
        """
        with self._session_scope() as session:
            album = session.execute(
                select(Album)
                # Deliberately narrower than _ALBUM_DETAIL_OPTIONS (#5028
                # CONSISTENCY check): only album.tracks is read below, never
                # album.artist, so there's no reason to pay for the join.
                .options(selectinload(Album.tracks))
                .where(Album.id == album_id)
            ).scalars().first()

            if not album or not album.tracks:
                return None

            # Try to extract artwork from the first track
            for track in album.tracks:
                if track.filepath and Path(track.filepath).exists():
                    artwork_path = self.artwork_extractor.extract_artwork(
                        track.filepath, album_id
                    )

                    if artwork_path:
                        # Update album with artwork path
                        previous_path = album.artwork_path
                        album.artwork_path = artwork_path
                        session.commit()
                        session.refresh(album)
                        # Remove the superseded file now that the new path is
                        # committed (mirrors delete_artwork below; #4850).
                        # Every write generates a unique
                        # album_{id}_{content_hash}.ext filename, so leaving
                        # this unlinked orphaned one more file on every
                        # re-extract with no sweeper anywhere to reclaim it.
                        if previous_path and previous_path != artwork_path:
                            self.artwork_extractor.delete_artwork(previous_path)
                        return artwork_path

            return None

    def update_artwork(self, album_id: int, artwork_path: str) -> bool:
        """
        Update album artwork path

        Args:
            album_id: Album ID
            artwork_path: Path to artwork file

        Returns:
            True if updated successfully
        """
        with self._session_scope() as session:
            try:
                album = session.execute(select(Album).where(Album.id == album_id)).scalars().first()
                if album:
                    album.artwork_path = artwork_path
                    session.commit()
                    return True
                return False
            except Exception:
                session.rollback()  # fixes #2238: prevent dirty session in pool
                raise

    def delete_artwork(self, album_id: int) -> bool:
        """
        Delete album artwork

        Args:
            album_id: Album ID

        Returns:
            True if deleted successfully
        """
        with self._session_scope() as session:
            try:
                album = session.execute(select(Album).where(Album.id == album_id)).scalars().first()
                if album and album.artwork_path:
                    # Delete file
                    self.artwork_extractor.delete_artwork(album.artwork_path)
                    # Clear database reference
                    album.artwork_path = None
                    session.commit()
                    return True
                return False
            except Exception:
                session.rollback()  # fixes #2238: prevent dirty session in pool
                raise

    def update_artwork_path(self, album_id: int, artwork_path: str) -> Album | None:
        """
        Update album artwork path.

        Args:
            album_id: Album ID
            artwork_path: Path to artwork file

        Returns:
            Updated album or None if not found

        Raises:
            Exception: If update fails
        """
        with self._session_scope() as session:
            try:
                album = session.execute(select(Album).where(Album.id == album_id)).scalars().first()
                if not album:
                    return None

                previous_path = album.artwork_path
                album.artwork_path = artwork_path
                session.commit()
                session.refresh(album)
                # Every other read path here eager-loads artist + tracks because
                # Album.to_dict() reads both; refresh() does not re-apply query
                # options, so force them in before detaching (#4641). The artwork
                # router hands this album straight to to_dict().
                _ = album.artist, album.tracks
                session.expunge(album)
                # Remove the superseded file now that the new path is
                # committed (mirrors delete_artwork below; #4850) -- the
                # download flow through here left the previous file on disk
                # forever on every re-download that produced different bytes.
                if previous_path and previous_path != artwork_path:
                    self.artwork_extractor.delete_artwork(previous_path)
                return album
            except Exception:
                session.rollback()
                raise
