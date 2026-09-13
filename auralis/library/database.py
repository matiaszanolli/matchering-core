"""
Auralis Library Database
~~~~~~~~~~~~~~~~~~~~~~~~

Owns the SQLite database itself: migration, engine, pragmas, session factory,
file-permission hardening, scan-slot accounting and orderly shutdown. Hands out
repositories through a :class:`RepositoryFactory`.

This is the composition root. It replaced ``LibraryManager`` on the production
startup path in #4619, and #4915 deleted that class outright. Historical note,
since the name still turns up in older comments: ``LibraryManager`` had emitted
a ``DeprecationWarning`` since v1.1.0 yet was still constructed on every backend
boot, because the bootstrap responsibilities below lived on it and
``RepositoryFactory`` (which only takes a ready-made session factory) could not
stand in for it. Splitting them apart is what finally made the removal possible.

#5162 then renamed the backend globals key to ``library_database``; it had gone
on naming the deleted class for a month. The old spelling is deliberately not
repeated here — the point of that rename was that grepping the old name should
return nothing.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import atexit
import os
import threading
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ..utils.logging import error, info, warning
from .constants import DEFAULT_DB_PATH
from .migration_manager import check_and_migrate_database
from .models import Base
from .repositories import (
    AlbumRepository,
    ArtistRepository,
    FingerprintRepository,
    FingerprintStatsRepository,
    GenreRepository,
    PlaylistRepository,
    QueueRepository,
    RepositoryFactory,
    StatsRepository,
    TrackRepository,
)
from .repositories.settings_repository import SettingsRepository

# #4523: the same-process thread lock that used to live here (#4232) moved into
# `migration_manager.migration_lock`, which now acquires thread *and* process
# exclusion together. Holding a second lock here would deadlock against it, and
# keeping it here only ever covered one of the two migration entry points —
# `migrations.normalize_existing_artists` takes `migration_lock` directly and
# never passed through this constructor.


class LibraryDatabase:
    """
    The library's SQLite database and everything needed to talk to it.

    Responsibilities:
    - Run pending schema migrations exactly once, under a process-wide lock
    - Create the engine with the WAL/pragma configuration the fingerprint
      writers depend on
    - Restrict the DB file and its WAL sidecars to owner-only (#2577, #4347)
    - Expose repositories via :attr:`repositories` (a ``RepositoryFactory``)
    - Account for concurrent scan slots (#2438)
    - Checkpoint the WAL and dispose the pool on shutdown (#2066)

    Example:
        ```python
        db = LibraryDatabase()
        tracks, total = db.repositories.tracks.get_all(limit=50)
        ```

    The repository accessors (:attr:`tracks`, :attr:`albums`, …) are
    conveniences that forward to :attr:`repositories`; they exist so components
    that only need one repository can take this object and reach it directly.
    """

    def __init__(self, database_path: str | None = None) -> None:
        """
        Open (migrating if needed) the library database.

        Args:
            database_path: Path to the SQLite database file. Defaults to
                ``~/.auralis/library.db``.

        Raises:
            Exception: If the schema migration fails — the caller must not
                proceed against a half-migrated database.
        """
        # #4824: harden ~/.auralis whenever the database lives at the default
        # location, not only when the caller omitted the path. fetch_artwork.py
        # passes DEFAULT_DB_PATH explicitly, so it skipped this block and the
        # directory was created later by migration_lock's plain mkdir at the
        # process umask (typically 0o755). A caller-chosen location elsewhere
        # is deliberately left alone — that directory is not ours to chmod.
        if database_path is None or os.path.realpath(database_path) == os.path.realpath(DEFAULT_DB_PATH):
            DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # mkdir(mode=) is ignored when the dir already exists, so re-restrict
            # it to owner-only even for a pre-existing ~/.auralis (#4347).
            try:
                os.chmod(DEFAULT_DB_PATH.parent, 0o700)
            except OSError:
                pass
        if database_path is None:
            database_path = str(DEFAULT_DB_PATH)

        self.database_path = database_path

        # Check and migrate database before initializing engine. `migration_lock`
        # (in migration_manager) now serializes both same-process threads and
        # other processes (#4232, #4523), so no additional lock is needed here.
        info("Checking database version...")
        if not check_and_migrate_database(database_path, auto_backup=True):
            error("Database migration failed!")
            raise Exception("Failed to migrate database to current version")

        # Configure SQLite for safe, frequent fingerprint writes
        # WAL mode + synchronous=NORMAL enables fast writes with durability guarantees
        connect_args = {
            'timeout': 15,  # 15s busy timeout (WAL makes contention rare)
            'check_same_thread': False,
        }

        # SQLite allows only one writer at a time. With WAL mode, readers don't
        # block writers (and vice versa), so a small pool is sufficient even under
        # heavy concurrent load. Sessions are short-lived (session-per-method),
        # so connections are returned quickly. (#2086)
        # `Engine | None` because shutdown() clears it (#3769) — __del__ gates
        # its own shutdown attempt on the None.
        self.engine: Engine | None = create_engine(
            f"sqlite:///{database_path}",
            echo=False,
            connect_args=connect_args,
            pool_pre_ping=True,  # Verify connections before use
            pool_size=5,  # Sufficient for SQLite with WAL concurrent readers
            max_overflow=5,  # Up to 10 total connections
        )

        # Restrict database file permissions to owner-only (#2577)
        db_path = Path(database_path)
        if db_path.exists():
            try:
                os.chmod(db_path, 0o600)
            except OSError:
                pass

        # Configure SQLite pragmas for reliable fingerprinting persistence
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            """Set SQLite pragmas for safe, fast writes"""
            cursor = dbapi_connection.cursor()
            # Enable Write-Ahead Logging for better concurrent write performance
            cursor.execute("PRAGMA journal_mode=WAL")
            # Set synchronous to NORMAL (safer than OFF, faster than FULL)
            # Ensures fingerprints are durably written to WAL before returning
            cursor.execute("PRAGMA synchronous=NORMAL")
            # SQLite applies cache_size PER CONNECTION, not as an aggregate. With
            # pool_size=5 + max_overflow=5 (up to 10 connections), 8MB/connection
            # caps the worst-case aggregate at 80MB instead of 640MB. (#4779)
            cursor.execute("PRAGMA cache_size=-8192")  # 8MB cache per connection
            # Optimize for frequent writes
            cursor.execute("PRAGMA temp_store=MEMORY")
            # Foreign key enforcement for data integrity
            cursor.execute("PRAGMA foreign_keys=ON")
            # Increase max page count for larger working set
            cursor.execute("PRAGMA max_page_count=1073741823")  # ~4GB max database
            # Explicit busy timeout at PRAGMA level for consistent behavior (#2091)
            cursor.execute("PRAGMA busy_timeout=60000")  # 60s
            cursor.close()

        self.SessionLocal = sessionmaker(self.engine)

        # Create tables if they don't exist (for fresh databases). This opens a
        # connection, which fires the connect event above and switches the DB to
        # WAL mode — creating the -wal/-shm sidecars.
        Base.metadata.create_all(self.engine)

        # WAL sidecars are created lazily by SQLite with the process umask (often
        # world/group-readable) and never chmod'd, unlike the main DB above.
        # Restrict them to owner-only so recently-played/library data isn't
        # readable by other local accounts on a custom path or a pre-existing
        # loose ~/.auralis (#4347).
        for _suffix in ("-wal", "-shm"):
            _sidecar = Path(database_path + _suffix)
            if _sidecar.exists():
                try:
                    os.chmod(_sidecar, 0o600)
                except OSError:
                    pass

        # Single source of repositories. Everything that needs the DB goes
        # through here — no component builds its own factory off SessionLocal.
        self.repositories = RepositoryFactory(self.SessionLocal)

        # #4842: the v017->v018 migration adds tracks.filepath_key but cannot
        # fill it — the key is case-folded only on case-insensitive platforms,
        # and SQLite's ASCII-only lower() disagrees with str.casefold() on the
        # non-ASCII paths a music library is full of. Fill it here, where
        # make_filepath_key() is available and remains the single authority.
        # Idempotent: matches nothing on every start after the first.
        try:
            self.repositories.tracks.backfill_filepath_keys()
        except Exception as e:  # noqa: BLE001 - never block startup on a backfill
            warning(f"filepath_key backfill skipped: {e}")

        # Thread-safe locking for delete operations (prevents race conditions)
        self._delete_lock = threading.RLock()

        # Scan concurrency tracking (#2438)
        self._scan_slots_lock = threading.Lock()
        self._active_scans: int = 0

        # Per-directory scan dedup (#3455 / #4509): must live here, not on
        # LibraryScanner, because every caller (LibraryAutoScanner._do_scan,
        # LibraryManager.scan_directories, before #4915) constructed a FRESH LibraryScanner
        # per invocation — an instance attribute would never be shared
        # between two overlapping scans, so the guard would only ever see
        # its own empty set. This object is the one thing already shared
        # across those call sites (it's what the #2438 scan-slot counter
        # above uses for the same reason).
        self._active_scan_paths_lock = threading.Lock()
        self._active_scan_paths: set[str] = set()

        # Clean up incomplete fingerprints from interrupted sessions (crash recovery)
        self._cleanup_incomplete_fingerprints()

        # Register cleanup handler for graceful shutdown (#2066)
        atexit.register(self.shutdown)

        info(f"Auralis library database opened: {database_path}")

    # ------------------------------------------------------------------
    # Repository access
    # ------------------------------------------------------------------

    @property
    def session_factory(self) -> Any:
        """The SQLAlchemy sessionmaker, under the RepositoryFactory's name."""
        return self.SessionLocal

    @property
    def tracks(self) -> TrackRepository:
        return self.repositories.tracks

    @property
    def albums(self) -> AlbumRepository:
        return self.repositories.albums

    @property
    def artists(self) -> ArtistRepository:
        return self.repositories.artists

    @property
    def genres(self) -> GenreRepository:
        return self.repositories.genres

    @property
    def playlists(self) -> PlaylistRepository:
        return self.repositories.playlists

    @property
    def stats(self) -> StatsRepository:
        return self.repositories.stats

    @property
    def fingerprints(self) -> FingerprintRepository:
        return self.repositories.fingerprints

    @property
    def fingerprint_stats(self) -> FingerprintStatsRepository:
        return self.repositories.fingerprint_stats

    @property
    def queue(self) -> QueueRepository:
        return self.repositories.queue

    @property
    def settings(self) -> SettingsRepository:
        return self.repositories.settings

    def get_session(self) -> Any:
        """Get a new database session."""
        return self.SessionLocal()

    # ------------------------------------------------------------------
    # Scan concurrency (#2438)
    # ------------------------------------------------------------------

    def try_acquire_scan_slot(self) -> tuple[bool, int]:
        """
        Atomically reserve a concurrent scan slot (#2438).

        Reads max_concurrent_scans from settings on every call so that user
        changes take effect for the next scan without a restart.

        Returns:
            (acquired, max_allowed)
            acquired=False means the caller must not start a scan.
        """
        try:
            settings = self.settings.get_settings()
            max_scans: int = max(1, (settings.max_concurrent_scans or 1) if settings else 1)
        except Exception:
            max_scans = 1  # conservative fallback if settings are unavailable

        with self._scan_slots_lock:
            if self._active_scans >= max_scans:
                return False, max_scans
            self._active_scans += 1
            return True, max_scans

    def release_scan_slot(self) -> None:
        """Release a slot previously acquired by try_acquire_scan_slot()."""
        with self._scan_slots_lock:
            self._active_scans = max(0, self._active_scans - 1)

    def try_reserve_scan_paths(self, paths: list[str]) -> list[str]:
        """
        Atomically reserve directory paths for an exclusive scan (#3455 / #4509).

        Args:
            paths: Normalized (os.path.normpath(os.path.abspath(...))) directory
                paths the caller is about to scan.

        Returns:
            The subset of `paths` already reserved by another in-flight scan.
            Empty means none clashed and all of `paths` are now reserved for
            the caller — who must call `release_scan_paths(paths)` when done,
            success or failure alike.
        """
        with self._active_scan_paths_lock:
            clashing = [p for p in paths if p in self._active_scan_paths]
            if clashing:
                return clashing
            self._active_scan_paths.update(paths)
            return []

    def release_scan_paths(self, paths: list[str]) -> None:
        """Release paths previously reserved by try_reserve_scan_paths()."""
        with self._active_scan_paths_lock:
            self._active_scan_paths.difference_update(paths)

    def is_scanning(self) -> bool:
        """True if at least one scan currently holds a slot.

        Live read of the same counter try_acquire_scan_slot()/
        release_scan_slot() maintain — used by GET /api/library/scan/status
        so a client that reconnects mid-scan (or after one finished while it
        was offline) can resync `isScanning` instead of relying solely on a
        WebSocket broadcast it may have missed (#4821).
        """
        with self._scan_slots_lock:
            return self._active_scans > 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Release all database resources.

        Performs critical cleanup operations:
        1. WAL checkpoint (TRUNCATE) - flush WAL to main database file
        2. PRAGMA optimize - save query planner statistics
        3. Connection pool disposal - close all open connections

        This prevents WAL file corruption, leftover lockfiles, and lost
        uncommitted transactions.

        Automatically called via atexit on normal shutdown; safe to call
        manually and safe to call twice.

        Fixes #2066 - No resource cleanup on shutdown
        """
        if not hasattr(self, 'engine') or self.engine is None:
            return  # Already shut down or never initialized

        try:
            info("🛑 Shutting down library database...")

            # 1. Checkpoint WAL to ensure all data is in main database file
            with self.engine.connect() as conn:
                try:
                    # TRUNCATE mode: checkpoint and delete WAL/SHM files
                    result = conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                    checkpoint_result = result.fetchone()
                    if checkpoint_result:
                        info(f"WAL checkpoint completed: {checkpoint_result}")
                except Exception as e:
                    error(f"WAL checkpoint failed: {e}")

                # 2. Optimize query planner statistics
                try:
                    conn.execute(text("PRAGMA optimize"))
                    info("Query planner optimized")
                except Exception as e:
                    error(f"PRAGMA optimize failed: {e}")

            # 3. Dispose connection pool (closes all connections)
            self.engine.dispose()
            info("Connection pool disposed")

            info("✅ Library database shutdown complete")

        except Exception as e:
            error(f"Error during library database shutdown: {e}")
        finally:
            # #3769: clear the engine reference in `finally` so a raise
            # from `engine.dispose()` (or the PRAGMA path above it)
            # doesn't leave a stale handle around. `__del__` checks
            # `self.engine is None` to gate its shutdown attempt, so a
            # stale handle meant `__del__` re-invoked shutdown from GC
            # — where loggers may already be torn down.
            self.engine = None

    def __del__(self) -> None:
        """
        Destructor - cleanup when object is garbage collected.

        Fallback cleanup in case the atexit handler doesn't run or the
        object is destroyed before application exit.
        """
        try:
            self.shutdown()
        except Exception:
            pass  # Suppress errors during garbage collection

    def _cleanup_incomplete_fingerprints(self) -> None:
        """
        Clean up incomplete fingerprints from interrupted fingerprinting sessions.

        When workers claim tracks, they create placeholder fingerprints (with LUFS=-100)
        to prevent race conditions. If the system crashes before processing completes,
        these placeholders remain in the database and must be cleaned up on restart
        so those tracks can be re-processed.

        This is CRITICAL for resumable large-scale fingerprinting jobs.
        """
        try:
            count = self.fingerprint_stats.cleanup_incomplete_fingerprints()

            if count > 0:
                warning(
                    f"Cleaned up {count} incomplete fingerprints from "
                    f"interrupted session. These tracks will be re-processed."
                )
            else:
                info("No incomplete fingerprints found - resuming clean session")

        except Exception as e:
            error(f"Error cleaning up incomplete fingerprints: {e}")
