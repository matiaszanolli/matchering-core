"""~/.auralis is owner-only however the default database path is reached (#4824).

The #4347 hardening ran only for `LibraryDatabase(database_path=None)`.
`fetch_artwork.py` passes DEFAULT_DB_PATH explicitly, so it skipped the block
and `migration_lock.py` later created the directory with a plain
`mkdir(parents=True, exist_ok=True)` at the process umask (typically 0o755).
"""

import os
import stat
import sys

import pytest

import auralis.library.database as dbmod

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def fake_default(tmp_path, monkeypatch):
    path = tmp_path / "home" / ".auralis" / "library.db"
    monkeypatch.setattr(dbmod, "DEFAULT_DB_PATH", path)
    return path


def test_explicit_default_path_hardens_a_new_directory(fake_default):
    db = dbmod.LibraryDatabase(database_path=str(fake_default))
    try:
        assert _mode(fake_default.parent) == 0o700
    finally:
        db.shutdown()


def test_explicit_default_path_restricts_an_existing_loose_directory(fake_default):
    fake_default.parent.mkdir(parents=True)
    os.chmod(fake_default.parent, 0o755)

    db = dbmod.LibraryDatabase(database_path=str(fake_default))
    try:
        assert _mode(fake_default.parent) == 0o700
    finally:
        db.shutdown()


def test_omitted_path_still_hardens(fake_default):
    db = dbmod.LibraryDatabase()
    try:
        assert db.database_path == str(fake_default)
        assert _mode(fake_default.parent) == 0o700
    finally:
        db.shutdown()


def test_a_caller_chosen_directory_elsewhere_is_left_alone(fake_default, tmp_path):
    custom_dir = tmp_path / "shared-libraries"
    custom_dir.mkdir()
    os.chmod(custom_dir, 0o755)

    db = dbmod.LibraryDatabase(database_path=str(custom_dir / "library.db"))
    try:
        assert _mode(custom_dir) == 0o755
    finally:
        db.shutdown()
