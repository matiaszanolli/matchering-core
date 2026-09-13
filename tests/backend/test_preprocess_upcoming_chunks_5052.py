"""
Regression tests for chunk pre-warm probe/leak fixes (#5052)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`_preprocess_upcoming_chunks()` (routers/enhancement.py, triggered by
POST /api/player/enhancement/toggle when enabling mid-playback) used bare
soundfile.info() to probe track duration. libsndfile cannot open
.m4a/.aac/.wma, so the probe raised, the function's catch-all swallowed it,
and pre-warm silently no-op'd for those formats (CP-2) — exactly the stall
this pre-fetch exists to prevent.

Secondary defect (CP-3): the throwaway ChunkedAudioProcessor this function
builds was never closed. For any format needing SeekableSource.resolve()'s
temp-dir conversion — the same set CP-2 fails the probe for — that leaks a
full-track temp WAV until process restart. CP-2's fix alone would unmask
CP-3's leak, so both fixes ship together.

These tests mount create_enhancement_router() directly (not the full app),
so there's no OriginCheckMiddleware to work around, and monkeypatch
spawn_background_task to await the pre-warm coroutine synchronously instead
of firing-and-forgetting it, so assertions can run after it completes.
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


def _make_playing_state_manager(filepath: str = "/music/track.m4a"):
    """A player_state_manager reporting an actively-playing current track —
    the precondition toggle_enhancement checks before firing pre-warm. Real
    (not Mock) id/filepath/current_time, since toggle_enhancement's own
    post-launch log line formats current_time with :.1f."""
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
    """spawn_background_task normally asyncio.create_task()s the coroutine
    and returns immediately. Patch it to run the coroutine to completion
    before the request returns, so the test can assert on its effects
    deterministically. TestClient's request handling already runs inside a
    live event loop on this thread, so the replacement can't just
    new_event_loop().run_until_complete() here (Python refuses to run a
    second loop on a thread that already has one running) — instead it runs
    the coroutine to completion on a fresh loop in a throwaway thread and
    blocks (via join) until that finishes, which is safe since it isn't the
    loop actually driving this coroutine's own execution.
    """
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


class TestPrewarmProbesFfmpegOnlyFormats:
    def test_prewarm_processes_chunks_for_m4a_track(self):
        """#5052 (CP-2): an .m4a track's duration must be determined via
        get_audio_info() (ffprobe-backed), not bare sf.info() (which raises
        for this format) — pre-warm must actually run, not silently no-op."""
        settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        mock_processor = MagicMock()
        mock_processor.get_wav_chunk_path = MagicMock(return_value="/tmp/chunk_1.wav")

        with _run_prewarm_synchronously(), \
             patch("auralis.io.unified_loader.get_audio_info",
                   return_value={"duration_seconds": 180.0}) as mock_get_info, \
             patch("core.chunked_processor.ChunkedAudioProcessor",
                   return_value=mock_processor) as mock_ctor, \
             patch("os.path.exists", return_value=True):
            response = client.post(
                "/api/player/enhancement/toggle", json={"enabled": True}
            )

        assert response.status_code == 200
        mock_get_info.assert_called_once()
        # The probe succeeded, so the processor must actually have been
        # constructed and asked to build chunks — the pre-#5052 bug never
        # reached this point for an .m4a-shaped duration probe.
        mock_ctor.assert_called_once()
        assert mock_processor.get_wav_chunk_path.called

    def test_prewarm_logs_and_exits_cleanly_when_probe_genuinely_fails(self):
        """When get_audio_info() itself can't determine a duration (e.g. a
        corrupt file), pre-warm must still fail closed without constructing
        a processor — the graceful-failure path must survive #5052's fix."""
        settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        with _run_prewarm_synchronously(), \
             patch("auralis.io.unified_loader.get_audio_info",
                   return_value={"error": "ffprobe failed"}), \
             patch("core.chunked_processor.ChunkedAudioProcessor") as mock_ctor:
            response = client.post(
                "/api/player/enhancement/toggle", json={"enabled": True}
            )

        assert response.status_code == 200
        mock_ctor.assert_not_called()


class TestPrewarmProcessorIsAlwaysClosed:
    def test_processor_closed_after_successful_prewarm(self):
        """#5052 (CP-3): the throwaway processor must be released once
        pre-warm finishes successfully."""
        settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        mock_processor = MagicMock()
        mock_processor.get_wav_chunk_path = MagicMock(return_value="/tmp/chunk_1.wav")

        with _run_prewarm_synchronously(), \
             patch("auralis.io.unified_loader.get_audio_info",
                   return_value={"duration_seconds": 180.0}), \
             patch("core.chunked_processor.ChunkedAudioProcessor",
                   return_value=mock_processor), \
             patch("os.path.exists", return_value=True):
            client.post("/api/player/enhancement/toggle", json={"enabled": True})

        mock_processor.close.assert_called_once()

    def test_processor_closed_even_when_a_chunk_raises_mid_loop(self):
        """The close() call must survive an exception from any single
        chunk's processing (try/finally), not just the happy path."""
        settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
        player_state_manager = _make_playing_state_manager()
        client = _build_client(settings, player_state_manager)

        mock_processor = MagicMock()
        mock_processor.get_wav_chunk_path = MagicMock(
            side_effect=RuntimeError("simulated chunk processing failure")
        )

        with _run_prewarm_synchronously(), \
             patch("auralis.io.unified_loader.get_audio_info",
                   return_value={"duration_seconds": 180.0}), \
             patch("core.chunked_processor.ChunkedAudioProcessor",
                   return_value=mock_processor):
            client.post("/api/player/enhancement/toggle", json={"enabled": True})

        mock_processor.close.assert_called_once()

# The recommendation/pre-warm paths re-check a DB filepath with
# validate_file_path before any file I/O (#4817/#4818). These tests exercise
# what happens *after* that check with fabricated paths like /music/x.flac, so
# stand the check in with a pass-through; the guard itself is covered by the
# tests below that restore the real validator.
@pytest.fixture(autouse=True)
def _accept_fabricated_track_paths(monkeypatch):
    from pathlib import Path as _Path
    monkeypatch.setattr("routers.enhancement.validate_file_path", lambda filepath, *a, **k: _Path(filepath))



def test_prewarm_skips_a_stored_path_outside_allowed_directories(monkeypatch):
    """#4818: the background task re-validates the player-state filepath and
    short-circuits before probing or constructing a processor (WIRING check)."""
    from security.path_security import validate_file_path as real_validate

    monkeypatch.setattr("routers.enhancement.validate_file_path", real_validate)
    settings = {"enabled": False, "preset": "adaptive", "intensity": 1.0}
    client = _build_client(
        settings, _make_playing_state_manager("/definitely/not/allowed/track.m4a")
    )

    with _run_prewarm_synchronously(), \
         patch("auralis.io.unified_loader.get_audio_info") as probe, \
         patch("core.chunked_processor.ChunkedAudioProcessor") as ctor:
        response = client.post("/api/player/enhancement/toggle", json={"enabled": True})

    assert response.status_code == 200
    probe.assert_not_called()
    ctor.assert_not_called()

