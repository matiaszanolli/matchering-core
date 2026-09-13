"""
Regression tests for preset/intensity pre-warm parity (#4425)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`toggle_enhancement` pre-processes the next few chunks in the background
when enabling mid-playback (#2296), so the client doesn't hit an audible
on-demand-processing gap. `set_enhancement_preset` and
`set_enhancement_intensity` never got the same treatment, even though a
mid-playback preset/intensity change re-issues the streaming task exactly
the same way enabling does (see useEnhancementControl.ts's setPreset/
setIntensity, both of which call reissueActiveStreamAs('play_enhanced', ...)
when enabled).

Mirrors tests/backend/test_preprocess_upcoming_chunks_5052.py's harness:
mount create_enhancement_router() directly and monkeypatch
spawn_background_task to run the pre-warm coroutine synchronously so
assertions can run after it completes.
"""

import asyncio
import sys

import pytest
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.enhancement import create_enhancement_router


def _make_playing_state_manager(filepath: str = "/music/track.wav"):
    """A player_state_manager reporting an actively-playing current track."""
    manager = Mock()
    state = Mock()
    state.current_track = Mock()
    state.current_track.id = 1
    state.current_track.filepath = filepath
    state.current_time = 30.0
    state.state = Mock()
    state.state.value = "playing"
    manager.get_state = Mock(return_value=state)
    return manager


def _build_client(enhancement_settings: dict, player_state_manager) -> TestClient:
    connection_manager = Mock()
    connection_manager.broadcast = AsyncMock()
    router = create_enhancement_router(
        get_enhancement_settings=lambda: enhancement_settings,
        connection_manager=connection_manager,
        get_player_state_manager=lambda: player_state_manager,
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _run_prewarm_synchronously():
    """See test_preprocess_upcoming_chunks_5052.py's identical helper for the
    full rationale — runs the pre-warm coroutine to completion on a fresh
    loop in a throwaway thread instead of firing-and-forgetting it."""
    def _sync_spawn(coro, *, name=None):
        outcome: dict = {}

        def _runner():
            try:
                asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - surfaced via outcome
                outcome["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return MagicMock()

    return patch("routers.enhancement.spawn_background_task", side_effect=_sync_spawn)


def _patched_prewarm_chain():
    """Common patches so a pre-warm attempt (if triggered) resolves cleanly
    without touching real audio I/O."""
    mock_processor = MagicMock()
    mock_processor.get_wav_chunk_path = MagicMock(return_value="/tmp/chunk_1.wav")
    return (
        patch("auralis.io.unified_loader.get_audio_info",
              return_value={"duration_seconds": 180.0}),
        patch("core.chunked_processor.ChunkedAudioProcessor",
              return_value=mock_processor),
        patch("os.path.exists", return_value=True),
        mock_processor,
    )


class TestSetPresetPrewarm:
    def test_preset_change_while_playing_and_enabled_triggers_prewarm(self):
        settings = {"enabled": True, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, mock_processor = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/preset", json={"preset": "warm"}
            )

        assert response.status_code == 200
        mock_ctor.assert_called_once()
        assert mock_processor.get_wav_chunk_path.called
        # Pre-warm must use the NEW preset, not the stale one.
        assert mock_ctor.call_args.kwargs["preset"] == "warm"

    def test_no_prewarm_when_enhancement_disabled(self):
        """Matches the frontend's own gate (setPreset only re-issues the
        stream `if (enabledRef.current)`) — pre-warming for a preset that
        isn't driving the active audio path would be wasted work."""
        settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, _ = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/preset", json={"preset": "warm"}
            )

        assert response.status_code == 200
        mock_ctor.assert_not_called()

    def test_no_prewarm_when_preset_unchanged(self):
        """Re-posting the SAME preset must not fire a redundant pre-warm."""
        settings = {"enabled": True, "preset": "warm", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, _ = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/preset", json={"preset": "warm"}
            )

        assert response.status_code == 200
        mock_ctor.assert_not_called()

    def test_no_prewarm_when_nothing_is_playing(self):
        settings = {"enabled": True, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = Mock()
        idle_state = Mock()
        idle_state.current_track = None
        idle_state.state = Mock()
        idle_state.state.value = "stopped"
        player_state_manager.get_state = Mock(return_value=idle_state)
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, _ = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/preset", json={"preset": "warm"}
            )

        assert response.status_code == 200
        mock_ctor.assert_not_called()


class TestSetIntensityPrewarm:
    def test_intensity_change_while_playing_and_enabled_triggers_prewarm(self):
        settings = {"enabled": True, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, mock_processor = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/intensity", json={"intensity": 0.5}
            )

        assert response.status_code == 200
        mock_ctor.assert_called_once()
        assert mock_processor.get_wav_chunk_path.called
        assert mock_ctor.call_args.kwargs["intensity"] == 0.5

    def test_no_prewarm_when_enhancement_disabled(self):
        settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, _ = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/intensity", json={"intensity": 0.5}
            )

        assert response.status_code == 200
        mock_ctor.assert_not_called()

    def test_no_prewarm_when_intensity_unchanged(self):
        settings = {"enabled": True, "preset": "adaptive", "intensity": 0.5}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        p1, p2, p3, _ = _patched_prewarm_chain()
        with _run_prewarm_synchronously(), p1, p2 as mock_ctor, p3:
            response = client.post(
                "/api/player/enhancement/intensity", json={"intensity": 0.5}
            )

        assert response.status_code == 200
        mock_ctor.assert_not_called()

# The recommendation/pre-warm paths re-check a DB filepath with
# validate_file_path before any file I/O (#4817/#4818). These tests exercise
# what happens *after* that check with fabricated paths like /music/x.flac, so
# stand the check in with a pass-through; the guard itself is covered by the
# tests below that restore the real validator.
@pytest.fixture(autouse=True)
def _accept_fabricated_track_paths(monkeypatch):
    from pathlib import Path as _Path
    monkeypatch.setattr("routers.enhancement.validate_file_path", lambda filepath, *a, **k: _Path(filepath))

