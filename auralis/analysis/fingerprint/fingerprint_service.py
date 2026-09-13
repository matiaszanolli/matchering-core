"""
Unified Fingerprinting Service

Consolidates all fingerprinting logic (cache lookup, computation, storage) into
a single interface used by both playback and batch processing.

Replaces:
  - FingerprintGenerator (backend fingerprint_generator.py)
  - MasteringTargetService (backend mastering_target_service.py)
  - Embedded logic in AudioFingerprintAnalyzer usage

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from auralis.analysis.fingerprint.audio_fingerprint_analyzer import (
    AudioFingerprintAnalyzer,
)
from auralis.analysis.fingerprint.fingerprint_storage import FingerprintStorage
from auralis.analysis.fingerprint.windowed_compute import compute_windowed_fingerprint
from auralis.library.repositories.fingerprint_repository import FingerprintRepository
from auralis.library.repositories.track_repository import TrackRepository

logger = logging.getLogger(__name__)

# All 25 fingerprint dimension keys — must match TrackFingerprint column names.
_FP_KEYS: tuple[str, ...] = (
    'sub_bass_pct', 'bass_pct', 'low_mid_pct', 'mid_pct', 'upper_mid_pct',
    'presence_pct', 'air_pct', 'lufs', 'crest_db', 'bass_mid_ratio',
    'tempo_bpm', 'rhythm_stability', 'transient_density', 'silence_ratio',
    'spectral_centroid', 'spectral_rolloff', 'spectral_flatness',
    'harmonic_ratio', 'pitch_stability', 'chroma_energy',
    'dynamic_range_variation', 'loudness_variation_std', 'peak_consistency',
    'stereo_width', 'phase_correlation',
)


def _make_engine(db_path: Path):
    """Create a minimal SQLAlchemy engine matching LibraryDatabase's configuration."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={'timeout': 15, 'check_same_thread': False},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        # #4510: foreign-key enforcement is per-connection in SQLite, and the
        # other two engines on this same library.db (database.py,
        # migration_engine.py) both enable it. Without it, fingerprint writes
        # through this self-created engine would accept a stale track_id the
        # main engine rejects.
        cursor.execute("PRAGMA foreign_keys=ON")

    return engine


class FingerprintService:
    """
    Unified fingerprinting service with 3-tier caching:
    1. Database (SQLite) - fastest, persistent
    2. .25d file cache - fast, portable
    3. On-demand computation - slower but ensures fresh data

    Single interface for all fingerprint operations.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        session_factory: Callable[[], Session] | None = None,
    ):
        """
        Initialize fingerprinting service.

        Args:
            db_path: Path to SQLite database (default: ~/.auralis/library.db)
            session_factory: Optional SQLAlchemy session factory. When provided
                             the service uses the caller's connection pool.
                             When omitted a minimal engine is created from db_path.
        """
        if db_path is None:
            from auralis.library.constants import DEFAULT_DB_PATH
            db_path = DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.analyzer = AudioFingerprintAnalyzer()

        if session_factory is None:
            self._engine = _make_engine(self.db_path)
            session_factory = sessionmaker(self._engine)
        else:
            self._engine = None

        self._session_factory = session_factory
        self._fingerprint_repo = FingerprintRepository(session_factory)
        self._track_repo = TrackRepository(session_factory)

    def close(self) -> None:
        """Dispose the connection pool of the engine this service created.

        Only an engine the service built itself (no ``session_factory`` was
        injected) is disposed — when a caller passes its own factory, that caller
        owns the engine and must dispose it. Without this the private engine's
        pool leaked until GC finalizers ran (#4501, same class as #2395/#3746).
        Idempotent: safe to call more than once.
        """
        engine = self._engine
        if engine is not None:
            self._engine = None
            engine.dispose()

    def __del__(self) -> None:
        # GC safety net for owners that forget to call close(); the explicit
        # teardown wiring in the owners is the real fix (#4501).
        try:
            self.close()
        except Exception:
            pass

    def get_or_compute(self, audio_path: Path, audio: np.ndarray | None = None, sr: int | None = None) -> dict | None:
        """
        Get fingerprint using 3-tier cache strategy, or compute new one.

        Args:
            audio_path: Path to audio file
            audio: Optional pre-loaded audio (if None, will load from file)
            sr: Optional sample rate (required if audio provided)

        Returns:
            25D fingerprint dictionary or None on failure

        Priority:
            1. Database cache (SQLite, fastest)
            2. .25d file cache (FingerprintStorage)
            3. On-demand computation (AudioFingerprintAnalyzer)
        """
        try:
            # Tier 1: Check database cache
            fingerprint = self._load_from_database(str(audio_path))
            if fingerprint:
                logger.debug(f"Fingerprint cache hit (database): {audio_path.name}")
                return fingerprint

            # Tier 2: Check .25d file cache
            fingerprint = self._load_from_file_cache(audio_path)
            if fingerprint:
                logger.debug(f"Fingerprint cache hit (.25d file): {audio_path.name}")
                # Save to database for future faster access
                self._save_to_database(str(audio_path), fingerprint)
                return fingerprint

            # Tier 3: Compute new fingerprint
            logger.info(f"Computing fingerprint: {audio_path.name}")
            fingerprint = self._compute_fingerprint(audio_path, audio, sr)

            if fingerprint:
                # Cache to both database and .25d file
                self._save_to_database(str(audio_path), fingerprint)
                FingerprintStorage.save(audio_path, fingerprint, {})
                logger.debug(f"Fingerprint cached for: {audio_path.name}")
                return fingerprint

            return None

        except Exception as e:
            logger.error(f"Fingerprint retrieval failed: {e}")
            return None

    def _load_from_database(self, filepath: str) -> dict | None:
        """Load fingerprint from database via FingerprintRepository."""
        try:
            if not self.db_path.exists():
                return None

            track_id = self._track_repo.get_id_by_filepath(filepath)
            if track_id is None:
                return None

            # get_by_track_id() already excludes the claim placeholder (lufs
            # sentinel) and stale-algorithm-version rows at the query level
            # (#4822) — this used to be re-checked here too, duplicating the
            # same two conditions FingerprintRepository now enforces for
            # every caller, not just this one. Returning None (rather than a
            # stale/incomplete row) makes the on-demand path self-healing: it
            # recomputes and rewrites (#4595 — background queue re-
            # fingerprints outdated rows separately in its Phase 2 pass, but
            # until it reaches this track the cached row must not be served).
            fp = self._fingerprint_repo.get_by_track_id(track_id)
            if fp is None:
                return None

            result = {key: getattr(fp, key) for key in _FP_KEYS}
            if not self._band_pct_valid(result):
                # Filename only, matching this file's other INFO logs (#4929) —
                # the full path leaks OS-username/install-layout info (#4351/#4366).
                logger.info(f"Discarding stale DB fingerprint (band-pct sum != 1): {Path(filepath).name}")
                return None
            return result

        except Exception as e:
            logger.debug(f"Database fingerprint lookup failed: {e}")
            return None

    # Frequency band keys that must sum to ~1.0 in a valid fingerprint.
    _BAND_PCT_KEYS: tuple[str, ...] = (
        'sub_bass_pct', 'bass_pct', 'low_mid_pct', 'mid_pct',
        'upper_mid_pct', 'presence_pct', 'air_pct',
    )

    @staticmethod
    def _band_pct_valid(fp: dict) -> bool:
        """Return True if the seven frequency-band fractions sum to 1 ± 0.05."""
        total = sum(fp.get(k, 0.0) for k in FingerprintService._BAND_PCT_KEYS)
        return 0.95 <= total <= 1.05

    def _load_from_file_cache(self, audio_path: Path) -> dict | None:
        """Load fingerprint from .25d file cache, discarding stale entries."""
        try:
            cached_data = FingerprintStorage.load(audio_path)
            if cached_data:
                fingerprint, _ = cached_data
                if not self._band_pct_valid(fingerprint):
                    logger.info(
                        f"Discarding stale .25d cache (band-pct sum != 1): {audio_path.name}"
                    )
                    return None
                return fingerprint
            return None
        except Exception as e:
            logger.debug(f"File cache lookup failed: {e}")
            return None

    def _compute_fingerprint(
        self,
        audio_path: Path,
        audio: np.ndarray | None = None,
        sr: int | None = None
    ) -> dict | None:
        """Compute fingerprint using the shared windowed implementation.

        #4595: the body+probe windowing that used to live here is now the single
        implementation in `windowed_compute`, shared with the batch library-scan
        path (`services/fingerprint_extractor.py`). Both previously computed
        fingerprints with materially different sampling strategies, and whichever
        wrote the DB row first won permanently.
        """
        return compute_windowed_fingerprint(self.analyzer, audio_path, audio, sr)

    def _save_to_database(self, filepath: str, fingerprint: dict) -> bool:
        """Save fingerprint to database via FingerprintRepository."""
        try:
            if not self.db_path.exists():
                return False

            track_id = self._track_repo.get_id_by_filepath(filepath)
            if track_id is None:
                # Track not in library yet; nothing to associate the fingerprint with.
                return False

            fp_data = {key: fingerprint[key] for key in _FP_KEYS if key in fingerprint}
            return self._fingerprint_repo.upsert(track_id, fp_data) is not None

        except Exception as e:
            logger.debug(f"Database save failed: {e}")
            return False

    @staticmethod
    def _numpy_to_python(obj):
        """Recursively convert NumPy types to native Python types.

        #3765: previously `else: return obj` swallowed unhandled NumPy
        types (e.g. `np.complex128`, `np.datetime64`) — they'd be
        passed through verbatim and then silently break JSON
        serialisation downstream of any future analyzer that emitted
        them. Native Python primitives pass through; NumPy types
        outside the handled set fail loud.
        """
        if isinstance(obj, dict):
            return {k: FingerprintService._numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(FingerprintService._numpy_to_python(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            # Any other NumPy scalar (np.complex128, np.datetime64,
            # np.str_, ...) — raise loud rather than pass through.
            raise TypeError(
                f"Cannot serialise NumPy type {type(obj).__name__}; "
                f"add an explicit branch to _numpy_to_python()."
            )
        else:
            return obj
