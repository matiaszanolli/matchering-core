"""FingerprintService's self-created engine enforces foreign keys (#4510).

SQLite's `foreign_keys` pragma is per-connection. `LibraryDatabase` and
`migration_engine` enable it on their connections to library.db, but
`FingerprintService._make_engine` — used whenever no session factory is
injected — set WAL/synchronous/busy_timeout and not this one, so writes through
it silently accepted a track_id the main engine would reject.
"""

from sqlalchemy import text

from auralis.analysis.fingerprint.fingerprint_service import _make_engine


def test_connections_have_foreign_keys_on(tmp_path):
    engine = _make_engine(tmp_path / "library.db")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_a_fk_violating_insert_is_rejected(tmp_path):
    engine = _make_engine(tmp_path / "library.db")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE tracks (id INTEGER PRIMARY KEY)"))
            conn.execute(text(
                "CREATE TABLE fingerprints (id INTEGER PRIMARY KEY, "
                "track_id INTEGER NOT NULL REFERENCES tracks(id))"
            ))
        import sqlalchemy.exc
        import pytest

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            with engine.begin() as conn:
                conn.execute(text("INSERT INTO fingerprints (track_id) VALUES (999)"))
    finally:
        engine.dispose()


def test_the_existing_pragmas_are_still_applied(tmp_path):
    engine = _make_engine(tmp_path / "library.db")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"
            assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 60000
    finally:
        engine.dispose()
