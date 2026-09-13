"""
Player Streaming Mixin
~~~~~~~~~~~~~~~~~~~~~~

Real-time audio chunk output and auto-advance triggering for AudioPlayer,
extracted from enhanced_audio_player.py (#4249). This is the hot-path
callback invoked by the audio output thread — a distinct concern from
user-initiated transport controls (play/pause/stop/seek) and queue
navigation (next/previous/add), even though it ultimately calls
``self.next_track()`` (provided by ``PlayerQueueNavigationMixin``) when a
track ends.

Mixed into AudioPlayer (rather than composed as a separate object) so the
existing lock discipline — every method below acquires
``self.file_manager._audio_lock`` directly on the player instance, exactly
as before the extraction — continues to work unchanged. Moving code here
does NOT change lock acquisition order or scope relative to the fixed state
of #4141 / #3735 / #5105.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import threading
from collections.abc import Callable

import numpy as np

from .audio_file_manager import AudioFileManager
from .config import PlayerConfig
from .playback_controller import PlaybackController
from .queue_controller import QueueController
from .realtime import RealtimeProcessor


class PlayerStreamingMixin:
    """Audio chunk generation + auto-advance, delegating to component objects.

    Instance state below is initialized by AudioPlayer.__init__ or provided
    by sibling mixins, not here — declared here only so type checkers know
    this mixin depends on it.
    """

    config: PlayerConfig
    file_manager: AudioFileManager
    playback: PlaybackController
    queue: QueueController
    processor: RealtimeProcessor
    auto_advance: bool
    _auto_advancing: threading.Event
    _advance_generation: int
    _advance_thread: threading.Thread | None
    _stop_requested: threading.Event
    _cleanup_in_progress: threading.Event
    # Provided by PlayerQueueNavigationMixin. Bare annotation only (no
    # assignment) — see PlayerQueueNavigationMixin for why this must not
    # become a real attribute in this class's __dict__.
    next_track: Callable[[], bool]

    def get_audio_chunk(self, chunk_size: int | None = None) -> np.ndarray:
        """
        Get a chunk of processed audio for playback.

        Args:
            chunk_size: Size of audio chunk to return

        Returns:
            Processed audio chunk (stereo, float32)
        """
        if chunk_size is None:
            chunk_size = self.config.buffer_size

        # Hold _audio_lock across the is_loaded check, position read, chunk
        # fetch, end-of-track test, AND the auto-advance test-and-spawn so a
        # concurrent stop()/load_file() cannot unload audio between
        # is_loaded() and the slice (#3295), AND so two concurrent callers
        # cannot both pass the `not _auto_advancing.is_set()` check and spawn
        # duplicate advance threads (#3434 — Event.is_set + .set is TOCTOU).
        # Spawning a thread doesn't block, so we never deadlock; the advance
        # thread acquires the lock independently when it runs.
        with self.file_manager._audio_lock:
            # Return silence if not playing
            if not self.file_manager.is_loaded() or not self.playback.is_playing():
                return np.zeros((chunk_size, 2), dtype=np.float32)

            # Atomically read position and advance to prevent seek race (#2153)
            pos = self.playback.read_and_advance_position(chunk_size)

            # Get raw audio chunk using the captured position
            chunk = self.file_manager.get_audio_chunk(pos, chunk_size)

            # Check for end of track — use atomic flag to prevent concurrent auto-advance
            end_of_track = pos + chunk_size >= self.file_manager.get_total_samples()

            if end_of_track:
                # #3692: gate on has_next_track() instead of
                # is_queue_empty(). is_queue_empty() returns False for
                # "1 track in queue already playing" — we'd spawn
                # auto-advance threads at ~21 Hz against a phantom next
                # track. has_next_track() returns True only when
                # peek_next() would return non-None.
                if self.auto_advance and self.queue.has_next_track():
                    if not self._auto_advancing.is_set():
                        self._auto_advancing.set()
                        self._advance_generation += 1
                        gen = self._advance_generation
                        advance_thread = threading.Thread(
                            target=self._auto_advance_next,
                            args=(gen,),
                            daemon=True
                        )
                        self._advance_thread = advance_thread  # joined in cleanup() (#3694)
                        advance_thread.start()

        # Apply advanced real-time processing (outside the lock — pure CPU
        # work on the captured chunk, no shared state touched).
        processed_chunk = self.processor.process_chunk(chunk)

        return processed_chunk

    def _auto_advance_next(self, generation: int) -> None:
        """Auto-advance to next track (background thread, runs at most once)"""
        try:
            if self._stop_requested.is_set():
                return  # User called stop() — don't start next track (#3296)
            # #3727: also bail if cleanup() is in progress. A 2 s join
            # timeout in cleanup (slow disk I/O during a load_file
            # already in flight) could allow this thread to continue
            # past the _stop_requested check and then call next_track()
            # — which would swap audio_data into a freshly-cleared
            # player. This guard combined with the post-cleanup join
            # makes the test invariant "audio_data is None after
            # cleanup()" hold deterministically.
            if self._cleanup_in_progress.is_set():
                return
            if self.playback.is_playing():
                self.next_track()
        except Exception:
            # next_track() failed (e.g. queue race before lock fix, file error).
            # Stop playback so the audio callback's early-return guard fires and
            # prevents this method from being re-triggered on every subsequent
            # chunk — which would happen because _auto_advancing is cleared in
            # the finally block regardless of outcome (fixes #2441).
            self.playback.stop()
        finally:
            # #3718: hold `_audio_lock` for the compare-and-clear so it is
            # atomic w.r.t. the spawn site in get_audio_chunk() (which
            # also holds _audio_lock when incrementing _advance_generation
            # and spawning the next thread). Previously the unlocked
            # compare allowed: thread A reads gen==5 → True, then
            # get_audio_chunk increments to 6 and spawns thread B, then
            # A clears _auto_advancing while B is still running — the
            # next get_audio_chunk sees the flag clear and spawns thread
            # C, leaving B and C concurrent. The lock guarantees the
            # generation read and the clear happen as one critical
            # section relative to the spawner. #3350 introduced the
            # compare; this fix closes the residual race window.
            with self.file_manager._audio_lock:
                if self._advance_generation == generation:
                    self._auto_advancing.clear()
