"""
Regression tests: path leak via result_data (#3848), metadata 404 detail
(#3849), and PathValidationError's own text reflected into a 400 (#4807).

All three findings belong to the same disclosure class as #3322 (server
filesystem path in API responses). These tests assert the sanitised form is
in place.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))

from core.processing_engine import ProcessingJob, ProcessingStatus


# ---------------------------------------------------------------------------
# #3848: ProcessingJob.to_dict() — result_data must not expose output_path
# ---------------------------------------------------------------------------

class TestResultDataPathSanitisation:
    """result_data stored by process_job() must not contain the absolute output_path."""

    def _make_completed_job(self, output_path: str) -> ProcessingJob:
        job = ProcessingJob.__new__(ProcessingJob)
        job.job_id = "test-job"
        job.input_path = "/home/user/music/input.flac"
        job.output_path = output_path
        job.mode = "adaptive"
        job.settings = {}
        job.status = ProcessingStatus.COMPLETED
        job.progress = 100.0
        job.error_message = None
        job.created_at = datetime.now()
        job.started_at = datetime.now()
        job.completed_at = datetime.now()
        # Simulate what process_job() now stores (filename-only key)
        job.result_data = {
            "output_file": Path(output_path).name,
            "sample_rate": 44100,
            "duration": 180.0,
        }
        return job

    def test_result_data_has_no_output_path_key(self):
        """to_dict() result_data must not contain the absolute-path key 'output_path'."""
        job = self._make_completed_job("/tmp/auralis_processing/uuid_processed.wav")
        d = job.to_dict()
        result = d.get("result_data") or {}
        assert "output_path" not in result, (
            "result_data must not expose the absolute output_path (#3848)"
        )

    def test_result_data_contains_filename_only(self):
        """to_dict() result_data must expose only the filename via 'output_file'."""
        abs_path = "/tmp/auralis_processing/abc-def_processed.wav"
        job = self._make_completed_job(abs_path)
        d = job.to_dict()
        result = d.get("result_data") or {}
        assert result.get("output_file") == "abc-def_processed.wav"

    def test_to_dict_absolute_path_absent_from_all_fields(self):
        """The full /tmp path must not appear anywhere in the to_dict() output."""
        abs_path = "/tmp/auralis_processing/secret_processed.wav"
        job = self._make_completed_job(abs_path)
        d = job.to_dict()
        full_repr = str(d)
        assert "/tmp/auralis_processing" not in full_repr, (
            "Absolute output path leaked into to_dict() output (#3848)"
        )


# ---------------------------------------------------------------------------
# #3849: metadata router 404 detail must not embed str(FileNotFoundError)
# ---------------------------------------------------------------------------

def _make_metadata_router_and_app():
    """Build a minimal FastAPI app with just the metadata router.

    Module-level (not a TestMetadata404DetailSanitisation-only helper) so
    TestScanFoldersAndMetadataPathValidationSanitisation below can share it
    rather than duplicating it for a sibling finding in the same disclosure
    class (#4807).
    """
    from fastapi import FastAPI
    from routers.metadata import create_metadata_router

    app = FastAPI()
    router = create_metadata_router(
        get_repository_factory=lambda: None,
        broadcast_manager=MagicMock(),
    )
    app.include_router(router)
    return app


def _make_metadata_repos(filepath: str) -> MagicMock:
    repos = MagicMock()
    track = MagicMock()
    track.id = 1
    track.filepath = filepath
    track.format = "flac"
    repos.tracks.get_by_id = MagicMock(return_value=track)
    return repos


class TestMetadata404DetailSanitisation:
    """FileNotFoundError caught in metadata endpoints must not leak the filepath."""

    _make_router_and_app = staticmethod(_make_metadata_router_and_app)
    _make_repos = staticmethod(_make_metadata_repos)

    def test_get_editable_fields_404_has_no_filepath(self):
        """GET /fields 404 detail must not embed the absolute filepath."""
        from fastapi.testclient import TestClient

        app = self._make_router_and_app()
        abs_path = "/home/alice/private/music/secret.flac"
        repos = self._make_repos(abs_path)

        with (
            patch("routers.metadata.require_repository_factory", return_value=repos),
            patch("routers.metadata.validate_file_path", side_effect=FileNotFoundError(
                f"[Errno 2] No such file or directory: '{abs_path}'"
            )),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/api/metadata/tracks/1/fields")

        assert response.status_code == 404
        detail = response.json().get("detail", "")
        assert abs_path not in detail, (
            f"Absolute path leaked in 404 detail: {detail!r} (#3849)"
        )
        assert "1" in detail  # track_id should be present

    def test_get_track_metadata_404_has_no_filepath(self):
        """GET /metadata 404 detail must not embed the absolute filepath."""
        from fastapi.testclient import TestClient

        app = self._make_router_and_app()
        abs_path = "/home/alice/private/music/secret.flac"
        repos = self._make_repos(abs_path)

        with (
            patch("routers.metadata.require_repository_factory", return_value=repos),
            patch("routers.metadata.validate_file_path", side_effect=FileNotFoundError(
                f"[Errno 2] No such file or directory: '{abs_path}'"
            )),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/api/metadata/tracks/1")

        assert response.status_code == 404
        detail = response.json().get("detail", "")
        assert abs_path not in detail, (
            f"Absolute path leaked in 404 detail: {detail!r} (#3849)"
        )

    def test_update_track_metadata_404_has_no_filepath(self):
        """PUT /metadata 404 detail must not embed the absolute filepath."""
        from fastapi.testclient import TestClient

        app = self._make_router_and_app()
        abs_path = "/home/alice/private/music/secret.flac"
        repos = self._make_repos(abs_path)

        with (
            patch("routers.metadata.require_repository_factory", return_value=repos),
            patch("routers.metadata.validate_file_path", side_effect=FileNotFoundError(
                f"[Errno 2] No such file or directory: '{abs_path}'"
            )),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.put(
                    "/api/metadata/tracks/1",
                    json={"title": "Test"},
                )

        assert response.status_code == 404
        detail = response.json().get("detail", "")
        assert abs_path not in detail, (
            f"Absolute path leaked in 404 detail: {detail!r} (#3849)"
        )


# ---------------------------------------------------------------------------
# #4807: PathValidationError's own text names the resolved path and every
# allowed directory. Four routes reflected str(e) verbatim into a 400 body --
# a single bad request could enumerate the user's entire configured library
# layout. Same disclosure class as #3849 above, at the PathValidationError/400
# call sites rather than the FileNotFoundError/404 ones.
# ---------------------------------------------------------------------------

_BAIT_MESSAGE = (
    "Path '/etc/shadow' is outside allowed directories. "
    "Allowed directories: /home/alice/Music, /home/alice/Documents, /home/alice/Podcasts"
)


class TestPathValidationErrorRouteSanitisation:
    """The four HTTP call sites must not reflect str(PathValidationError) at
    all, regardless of what the exception's own message says -- proven here
    with a deliberately maximal "bait" message (a resolved path plus a full
    allowed-directories enumeration) standing in for whatever
    validate_file_path/validate_user_chosen_directory actually raise today or
    in the future."""

    def test_metadata_get_fields_400_has_no_validation_detail(self):
        from fastapi.testclient import TestClient
        from security.path_security import PathValidationError

        app = _make_metadata_router_and_app()
        repos = _make_metadata_repos("/home/alice/private/music/secret.flac")

        with (
            patch("routers.metadata.require_repository_factory", return_value=repos),
            patch("routers.metadata.validate_file_path", side_effect=PathValidationError(_BAIT_MESSAGE)),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/api/metadata/tracks/1/fields")

        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Allowed directories" not in detail
        assert "/home/alice" not in detail
        assert "/etc/shadow" not in detail

    def test_metadata_get_track_400_has_no_validation_detail(self):
        from fastapi.testclient import TestClient
        from security.path_security import PathValidationError

        app = _make_metadata_router_and_app()
        repos = _make_metadata_repos("/home/alice/private/music/secret.flac")

        with (
            patch("routers.metadata.require_repository_factory", return_value=repos),
            patch("routers.metadata.validate_file_path", side_effect=PathValidationError(_BAIT_MESSAGE)),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/api/metadata/tracks/1")

        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Allowed directories" not in detail
        assert "/home/alice" not in detail

    def test_metadata_update_track_400_has_no_validation_detail(self):
        from fastapi.testclient import TestClient
        from security.path_security import PathValidationError

        app = _make_metadata_router_and_app()
        repos = _make_metadata_repos("/home/alice/private/music/secret.flac")

        with (
            patch("routers.metadata.require_repository_factory", return_value=repos),
            patch("routers.metadata.validate_file_path", side_effect=PathValidationError(_BAIT_MESSAGE)),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.put("/api/metadata/tracks/1", json={"title": "Test"})

        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Allowed directories" not in detail
        assert "/home/alice" not in detail

    def test_settings_put_scan_folders_400_has_no_validation_detail(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.settings import create_settings_router
        from security.path_security import PathValidationError

        app = FastAPI()
        app.include_router(create_settings_router(lambda: MagicMock()))

        with patch(
            "routers.settings.validate_directory_list",
            side_effect=PathValidationError(_BAIT_MESSAGE),
        ):
            with TestClient(app) as client:
                response = client.put("/api/settings", json={"scan_folders": ["/etc"]})

        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Allowed directories" not in detail
        assert "/home/alice" not in detail

    def test_settings_post_scan_folder_400_has_no_validation_detail(self):
        """POST /api/settings/scan-folders {"folder": "/etc"} -- the issue's
        own repro (#4807)."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.settings import create_settings_router
        from security.path_security import PathValidationError

        app = FastAPI()
        app.include_router(create_settings_router(lambda: MagicMock()))

        with patch(
            "routers.settings.validate_user_chosen_directory",
            side_effect=PathValidationError(_BAIT_MESSAGE),
        ):
            with TestClient(app) as client:
                response = client.post("/api/settings/scan-folders", json={"folder": "/etc"})

        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "Allowed directories" not in detail
        assert "/home/alice" not in detail


class TestValidateFilePathOwnMessage:
    """Unit-level: validate_file_path's "outside allowed directories" branch
    no longer enumerates every allowed directory in the exception it raises,
    with the full detail (resolved path + every allowed dir) still available
    server-side at DEBUG -- matching config/startup.py's existing
    DEBUG-for-sensitive-paths convention (#3844/#4376) rather than losing the
    detail outright."""

    def test_raised_message_has_no_directory_enumeration(self, tmp_path, caplog):
        from security.path_security import PathValidationError, validate_file_path

        outside = tmp_path / "outside" / "secret.flac"
        outside.parent.mkdir()
        outside.write_text("x")
        allowed = [tmp_path / "music"]
        (tmp_path / "music").mkdir()

        with pytest.raises(PathValidationError) as exc_info:
            validate_file_path(str(outside), allowed_base_dirs=allowed)

        msg = str(exc_info.value)
        assert "Allowed directories" not in msg
        assert str(allowed[0]) not in msg
        # The existing shape sibling tests already pin (outside allowed
        # directories) must survive.
        assert "outside allowed directories" in msg.lower()

    def test_full_detail_still_reaches_debug(self, tmp_path, caplog):
        from security.path_security import PathValidationError, validate_file_path

        outside = tmp_path / "outside" / "secret.flac"
        outside.parent.mkdir()
        outside.write_text("x")
        allowed = [tmp_path / "music"]
        (tmp_path / "music").mkdir()

        with caplog.at_level("DEBUG", logger="security.path_security"):
            with pytest.raises(PathValidationError):
                validate_file_path(str(outside), allowed_base_dirs=allowed)

        full_detail = " ".join(r.message for r in caplog.records)
        assert "Allowed directories" in full_detail
        assert str(allowed[0]) in full_detail
