"""
Tests for Enhancement Router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the enhancement settings API endpoints.

Coverage:
- POST /api/player/enhancement/toggle - Enable/disable enhancement
- POST /api/player/enhancement/preset - Change preset
- POST /api/player/enhancement/intensity - Adjust intensity
- GET /api/player/enhancement/status - Get current settings
- GET /api/player/mastering/recommendation/{track_id} - Get mastering recommendation
- GET /api/processing/parameters - Get processing parameters

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))


class TestToggleEnhancement:
    """Test POST /api/player/enhancement/toggle"""

    def test_toggle_enhancement_enable(self, client):
        """Test enabling enhancement"""
        # `enabled` is a JSON body field (ToggleEnhancementRequest), not a query
        # param: #2485 moved every enhancement mutation endpoint off query
        # strings onto Pydantic bodies. Sent as `?enabled=true` the required
        # body is simply absent, so the request 422s on "body: Field required"
        # and the handler never runs (#5089 -- previously masked by the
        # Origin-check 403, which rejected this POST before validation).
        response = client.post("/api/player/enhancement/toggle", json={"enabled": True})

        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
        assert data["settings"]["enabled"] is True
        assert "message" in data

    def test_toggle_enhancement_disable(self, client):
        """Test disabling enhancement"""
        # JSON body, not a query param -- see test_toggle_enhancement_enable.
        response = client.post("/api/player/enhancement/toggle", json={"enabled": False})

        assert response.status_code == 200
        data = response.json()
        assert data["settings"]["enabled"] is False

    def test_toggle_enhancement_missing_parameter(self, client):
        """Test toggle without enabled parameter"""
        response = client.post("/api/player/enhancement/toggle")

        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_toggle_enhancement_invalid_type(self, client):
        """Test toggle with invalid boolean value"""
        # Must go in the body to actually exercise boolean coercion: as
        # `?enabled=invalid` this 422'd on the missing body instead, making the
        # test indistinguishable from test_toggle_enhancement_missing_parameter
        # above and blind to the field's type constraint (#5089).
        response = client.post("/api/player/enhancement/toggle", json={"enabled": "invalid"})

        assert response.status_code == 422
        assert response.json()["errors"][0]["field"] == "body.enabled"


class TestSetEnhancementPreset:
    """Test POST /api/player/enhancement/preset"""

    def test_set_preset_valid(self, client):
        """Test setting valid preset"""
        # JSON body (SetPresetRequest), not a query param -- see
        # test_toggle_enhancement_enable re: #2485. /api/player/enhancement/*
        # carries no RateLimitMiddleware rule, so the five iterations need no
        # reset_rate_limits (only /api/files/upload, /api/processing,
        # /api/library/scan and /api/similarity are limited).
        for preset in ["adaptive", "gentle", "warm", "bright", "punchy"]:
            response = client.post("/api/player/enhancement/preset", json={"preset": preset})

            assert response.status_code == 200
            data = response.json()
            assert data["settings"]["preset"] == preset

    def test_set_preset_case_insensitive(self, client):
        """Test that preset names are case-insensitive"""
        # SetPresetRequest.lowercase_preset is a mode="before" validator, so the
        # canonical Literal still enforces the closed set after lowering.
        response = client.post("/api/player/enhancement/preset", json={"preset": "WARM"})

        assert response.status_code == 200
        data = response.json()
        assert data["settings"]["preset"] == "warm"

    def test_set_preset_invalid(self, client):
        """Test setting invalid preset"""
        response = client.post(
            "/api/player/enhancement/preset", json={"preset": "invalid_preset"}
        )

        # 422, not 400: the preset constraint is `EnhancementPresetLiteral`
        # (#4424 made it the single source of truth in schemas.py), so an
        # unknown value is rejected by Pydantic during request validation and
        # surfaces through config/app.py's RequestValidationError handler as
        # {"detail": "Validation error", "errors": [...]}. The handler never
        # runs, so there is no hand-rolled 400 "Invalid preset" to assert on --
        # that phrasing predates the Literal (#5089).
        assert response.status_code == 422
        error = response.json()["errors"][0]
        assert error["field"] == "body.preset"
        # The Literal's own message enumerates the valid presets.
        assert "'adaptive'" in error["message"] and "'punchy'" in error["message"]

    def test_set_preset_missing_parameter(self, client):
        """Test preset change without preset parameter"""
        response = client.post("/api/player/enhancement/preset")

        assert response.status_code == 422


class TestSetEnhancementIntensity:
    """Test POST /api/player/enhancement/intensity"""

    def test_set_intensity_valid(self, client):
        """Test setting valid intensity values"""
        for intensity in [0.0, 0.5, 1.0]:
            response = client.post("/api/player/enhancement/intensity", json={"intensity": intensity})

            assert response.status_code == 200
            data = response.json()
            assert data["settings"]["intensity"] == intensity

    def test_set_intensity_boundary_values(self, client):
        """Test intensity at exact boundaries"""
        # Test minimum
        response = client.post("/api/player/enhancement/intensity", json={"intensity": 0.0})
        assert response.status_code == 200

        # Test maximum
        response = client.post("/api/player/enhancement/intensity", json={"intensity": 1.0})
        assert response.status_code == 200

    def test_set_intensity_below_minimum(self, client):
        """Intensity below 0.0 is rejected, not silently clamped (#4600).

        This endpoint used to clamp and return 200 carrying a value the caller
        never sent, while PUT /api/settings 422'd the identical input. Both now
        share EnhancementIntensity.
        """
        response = client.post("/api/player/enhancement/intensity", json={"intensity": -0.1})

        assert response.status_code == 422

    def test_set_intensity_above_maximum(self, client):
        """Intensity above 1.0 is rejected, not silently clamped (#4600)."""
        response = client.post("/api/player/enhancement/intensity", json={"intensity": 1.1})

        assert response.status_code == 422

    def test_set_intensity_rejects_non_finite(self, client):
        """NaN was the sharp case: the old clamp turned it into 1.0 (#4600)."""
        for body in ('{"intensity": NaN}', '{"intensity": Infinity}'):
            response = client.post(
                "/api/player/enhancement/intensity",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422, body

    def test_set_intensity_missing_parameter(self, client):
        """Test intensity change without body"""
        response = client.post("/api/player/enhancement/intensity")

        assert response.status_code == 422

    def test_set_intensity_invalid_type(self, client):
        """Test intensity with non-numeric value"""
        response = client.post("/api/player/enhancement/intensity", json={"intensity": "invalid"})

        assert response.status_code == 422


class TestGetEnhancementStatus:
    """Test GET /api/player/enhancement/status"""

    def test_get_enhancement_status(self, client):
        """Test getting current enhancement status"""
        response = client.get("/api/player/enhancement/status")

        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "preset" in data
        assert "intensity" in data
        assert isinstance(data["enabled"], bool)
        assert isinstance(data["preset"], str)
        assert isinstance(data["intensity"], (int, float))

    def test_get_enhancement_status_accepts_get_only(self, client):
        """Test that status endpoint only accepts GET"""
        response = client.post("/api/player/enhancement/status")
        assert response.status_code == 405  # Method Not Allowed


class TestGetMasteringRecommendation:
    """Test GET /api/player/mastering/recommendation/{track_id} (fixes #2731)

    The endpoint resolves filepath from the DB by track_id — no filepath
    query parameter is accepted.
    """

    @staticmethod
    def _mock_recommendation(**overrides):
        """A recommendation mock shaped like MasteringRecommendation.to_response().

        The route serializes with ``rec.to_response(track_id)``, not
        ``rec.to_dict()`` (#3840: to_dict() omits track_id/is_hybrid and leaks
        the internal-only created/alternative_profiles keys). A mock that only
        stubs ``to_dict`` therefore returns a bare Mock from ``to_response``,
        which the route's ``isinstance(result, dict)`` guard collapses to ``{}``
        -- and ``response_model=MasteringRecommendationResponse`` then raises
        ResponseValidationError naming all 8 required fields (#5089).

        The payload carries every required field of that response model so these
        tests assert the routing behaviour they are named for rather than
        tripping over serialization.
        """
        from unittest.mock import Mock

        payload = {
            "track_id": 1,
            "primary_profile_id": "pop",
            "primary_profile_name": "Pop",
            "confidence_score": 0.9,
            "predicted_loudness_change": 1.5,
            "predicted_crest_change": -0.5,
            "predicted_centroid_change": 120.0,
            "weighted_profiles": [],
            "reasoning": "",
            "is_hybrid": False,
        }
        payload.update(overrides)

        rec = Mock()
        rec.to_response.return_value = payload
        return rec

    @pytest.fixture
    def repos(self, client):
        """Swap the real RepositoryFactory for a mock on the real app.

        This class used to build its own FastAPI app around
        create_enhancement_router(...) with mock closures. That can never work
        in a session that also touches main.app: routers/enhancement.py binds
        its handlers to a MODULE-LEVEL ``router = APIRouter(...)`` singleton, so
        every create_enhancement_router() call APPENDS another copy of each
        route to that one object. main.app's startup registers the first copy
        (bound to the real, globals-backed closures), and the standalone app
        then included *both* copies -- Starlette matches the earlier one, so the
        mock repos were ignored and the endpoint queried the developer's real
        library.db, returning a real uploads/*.wav filepath. Running the class
        by itself hid this, because main.app was never built (#5089).

        Injecting through main.globals_dict instead exercises the route that is
        actually registered, and mirrors how the router is really wired:
        config/routes.py passes ``get_repository_factory=get_component(
        'repository_factory')``, a lambda that reads globals_dict at request
        time -- the same lazy-resolution pattern test_files_api.py uses.
        """
        from unittest.mock import Mock

        import main
        from cache import streamlined_cache_manager

        mock_repos = Mock()
        mock_repos.tracks.get_by_id = Mock(return_value=None)

        original = main.globals_dict.get('repository_factory')
        main.globals_dict['repository_factory'] = mock_repos

        # REST and playback share the singleton recommendation cache (#5280).
        # Clear it so repeated track ids in this class remain isolated.
        streamlined_cache_manager.clear_mastering_recommendations()
        try:
            yield mock_repos
        finally:
            main.globals_dict['repository_factory'] = original
            streamlined_cache_manager.clear_mastering_recommendations()

    def test_returns_404_for_nonexistent_track(self, client, repos):
        """Call with nonexistent track_id → 404"""
        response = client.get("/api/player/mastering/recommendation/999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_rejects_a_stored_path_outside_allowed_directories(self, client, repos, monkeypatch):
        """#4817: the DB filepath is re-validated before any file I/O, and the
        400 does not echo the path or the allowed-directory list (#4807)."""
        from unittest.mock import Mock, patch
        from security.path_security import validate_file_path as real_validate

        monkeypatch.setattr("routers.enhancement.validate_file_path", real_validate)
        track = Mock()
        track.filepath = "/definitely/not/an/allowed/dir/song.flac"
        repos.tracks.get_by_id.return_value = track

        with patch("core.chunked_processor.ChunkedAudioProcessor") as MockProc:
            response = client.get("/api/player/mastering/recommendation/1")

        assert response.status_code == 400
        assert "/definitely" not in response.json()["detail"]
        MockProc.assert_not_called()

    def test_resolves_filepath_from_db(self, client, repos):
        """Call with valid track_id → filepath resolved from DB, not query param"""
        from unittest.mock import Mock, patch

        track = Mock()
        track.filepath = "/music/song.flac"
        repos.tracks.get_by_id.return_value = track

        mock_rec = self._mock_recommendation()

        with patch("core.chunked_processor.ChunkedAudioProcessor") as MockProc:
            MockProc.return_value.get_mastering_recommendation.return_value = mock_rec
            response = client.get("/api/player/mastering/recommendation/1")

        assert response.status_code == 200
        MockProc.assert_called_once()
        call_kwargs = MockProc.call_args[1]
        assert call_kwargs["filepath"] == "/music/song.flac"
        assert call_kwargs["track_id"] == 1

    def test_no_filepath_query_param_accepted(self, client, repos):
        """The filepath query parameter must not influence the endpoint"""
        from unittest.mock import Mock, patch

        track = Mock()
        track.filepath = "/music/real.flac"
        repos.tracks.get_by_id.return_value = track

        mock_rec = self._mock_recommendation()

        with patch("core.chunked_processor.ChunkedAudioProcessor") as MockProc:
            MockProc.return_value.get_mastering_recommendation.return_value = mock_rec
            response = client.get(
                "/api/player/mastering/recommendation/1?filepath=/evil/path.wav"
            )

        assert response.status_code == 200
        call_kwargs = MockProc.call_args[1]
        assert call_kwargs["filepath"] == "/music/real.flac"

    def test_custom_confidence_threshold(self, client, repos):
        """Custom confidence_threshold parameter is passed through"""
        from unittest.mock import Mock, patch

        track = Mock()
        track.filepath = "/music/song.flac"
        repos.tracks.get_by_id.return_value = track

        mock_rec = self._mock_recommendation()

        with patch("core.chunked_processor.ChunkedAudioProcessor") as MockProc:
            MockProc.return_value.get_mastering_recommendation.return_value = mock_rec
            response = client.get(
                "/api/player/mastering/recommendation/1?confidence_threshold=0.8"
            )

        assert response.status_code == 200
        MockProc.return_value.get_mastering_recommendation.assert_called_once_with(
            confidence_threshold=0.8
        )

    def test_second_request_uses_shared_cache(self, client, repos):
        """The REST path must not construct a second processor on a cache hit."""
        from unittest.mock import Mock, patch

        track = Mock()
        track.filepath = "/music/song.flac"
        repos.tracks.get_by_id.return_value = track
        mock_rec = self._mock_recommendation()

        with patch("core.chunked_processor.ChunkedAudioProcessor") as MockProc:
            MockProc.return_value.get_mastering_recommendation.return_value = mock_rec
            first = client.get("/api/player/mastering/recommendation/1")
            second = client.get("/api/player/mastering/recommendation/1")

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json() == first.json()
        MockProc.assert_called_once()

    def test_returns_500_when_analysis_times_out(self, client, repos):
        """#5248: a hung ChunkedAudioProcessor construction/analysis (e.g. a
        corrupt-header file — sf.info() has no timeout of its own for
        natively-decodable formats) must surface as a clean 500 naming the
        timeout, not hang the request indefinitely.

        Patches asyncio.wait_for (not asyncio.to_thread) so the earlier,
        unrelated `await asyncio.to_thread(repos.tracks.get_by_id, ...)`
        lookup is unaffected — wait_for is only used at the one new call
        site this fix added. The side_effect closes the real
        `asyncio.to_thread(_run_recommendation)` coroutine it intercepts
        before raising, so nothing is left dangling for the GC to warn
        about.
        """
        from unittest.mock import Mock, patch

        track = Mock()
        track.filepath = "/music/song.flac"
        repos.tracks.get_by_id.return_value = track

        async def _timeout_side_effect(coro, timeout=None):
            coro.close()
            raise TimeoutError

        with patch("routers.enhancement.asyncio.wait_for", side_effect=_timeout_side_effect):
            response = client.get("/api/player/mastering/recommendation/1")

        assert response.status_code == 500
        assert "timed out" in response.json()["detail"].lower()


class TestGetProcessingParameters:
    """Test GET /api/processing/parameters"""

    def test_get_processing_parameters_returns_defaults(self, client):
        """Test that endpoint returns default values when no track processed"""
        response = client.get("/api/processing/parameters")

        assert response.status_code == 200
        data = response.json()

        # Check all required fields present
        assert "spectral_balance" in data
        assert "dynamic_range" in data
        assert "energy_level" in data
        assert "target_lufs" in data
        assert "peak_target_db" in data
        assert "bass_boost" in data
        assert "air_boost" in data
        assert "compression_amount" in data
        assert "expansion_amount" in data
        assert "stereo_width" in data

    def test_processing_parameters_field_types(self, client):
        """Test that all fields have correct types"""
        response = client.get("/api/processing/parameters")
        data = response.json()

        # All values should be numeric
        for key, value in data.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"

    def test_processing_parameters_value_ranges(self, client):
        """Test that coordinate values are in valid range"""
        response = client.get("/api/processing/parameters")
        data = response.json()

        # Coordinates should be 0-1
        assert 0.0 <= data["spectral_balance"] <= 1.0
        assert 0.0 <= data["dynamic_range"] <= 1.0
        assert 0.0 <= data["energy_level"] <= 1.0

    def test_processing_parameters_accepts_get_only(self, client):
        """Test that endpoint only accepts GET requests"""
        response = client.post("/api/processing/parameters")
        assert response.status_code == 405  # Method Not Allowed


class TestClearProcessingCacheRemoved:
    """POST /api/player/enhancement/cache/clear was removed (fixes #3835 /
    BE-PE-2). It operated on `processing_cache`, a dict nothing in the
    codebase ever wrote to — the endpoint always reported "0 items removed"
    regardless of real cache state, an actively misleading API surface. The
    real invalidation path (multi-tier buffer manager) already runs from
    set_enhancement_intensity/set_enhancement_preset; see #2504.
    """

    def test_clear_cache_endpoint_no_longer_exists(self, client):
        """The dead endpoint must not resolve to any API route. 404 in dev
        mode; 405 when main.py's StaticFiles catch-all mount is active
        (production-like test config) and "handles" the unmatched path as a
        potential static GET, rejecting the POST method — either way, no
        enhancement-router handler runs."""
        response = client.post("/api/player/enhancement/cache/clear")
        assert response.status_code in (404, 405)


class TestEnhancementIntegration:
    """Integration tests for enhancement endpoints"""

    def test_workflow_enable_change_preset_adjust_intensity(self, client):
        """Test complete workflow: enable → change preset → adjust intensity"""
        # All three take JSON bodies, not query params (#2485 / #5089) -- see
        # test_toggle_enhancement_enable.
        # 1. Enable enhancement
        response = client.post("/api/player/enhancement/toggle", json={"enabled": True})
        assert response.status_code == 200

        # 2. Change preset
        response = client.post("/api/player/enhancement/preset", json={"preset": "warm"})
        assert response.status_code == 200

        # 3. Adjust intensity
        response = client.post("/api/player/enhancement/intensity", json={"intensity": 0.7})
        assert response.status_code == 200

        # 4. Verify final state
        response = client.get("/api/player/enhancement/status")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["preset"] == "warm"
        assert data["intensity"] == 0.7

    def test_multiple_preset_changes(self, client):
        """Test changing presets multiple times"""
        presets = ["gentle", "warm", "bright", "punchy", "adaptive"]

        for preset in presets:
            # JSON body, not a query param (#2485 / #5089).
            response = client.post("/api/player/enhancement/preset", json={"preset": preset})
            assert response.status_code == 200
            assert response.json()["settings"]["preset"] == preset

# The recommendation/pre-warm paths re-check a DB filepath with
# validate_file_path before any file I/O (#4817/#4818). These tests exercise
# what happens *after* that check with fabricated paths like /music/x.flac, so
# stand the check in with a pass-through; the guard itself is covered by the
# tests below that restore the real validator.
@pytest.fixture(autouse=True)
def _accept_fabricated_track_paths(monkeypatch):
    from pathlib import Path as _Path
    monkeypatch.setattr("routers.enhancement.validate_file_path", lambda filepath, *a, **k: _Path(filepath))

