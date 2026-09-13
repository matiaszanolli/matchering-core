"""
Auralis Audio Player
~~~~~~~~~~~~~~~~~~~~

Real-time audio player with advanced DSP processing and library integration

Refactored from monolithic design into 5 focused components:
- PlaybackController: State machine (PLAYING, PAUSED, STOPPED, etc.)
- AudioFileManager: File I/O and audio data access
- QueueController: Queue and playlist management
- GaplessPlaybackEngine: Prebuffering and seamless transitions
- IntegrationManager: Library, callbacks, statistics

Uses Facade pattern to maintain backward-compatible API.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import threading
from collections.abc import Callable
from typing import Any

from ..analysis.fingerprint.fingerprint_service import FingerprintService
from ..utils.logging import info, warning
from .audio_file_manager import AudioFileManager
from .config import PlayerConfig
from .fingerprint_loader_mixin import PlayerFingerprintLoaderMixin
from .gapless_playback_engine import GaplessPlaybackEngine
from .integration_manager import IntegrationManager
from .playback_controller import PlaybackController, PlaybackState
from .player_callbacks_mixin import PlayerCallbacksMixin
from .player_file_loading_mixin import PlayerFileLoadingMixin
from .player_properties_mixin import PlayerPropertiesMixin
from .player_queue_navigation_mixin import PlayerQueueNavigationMixin
from .player_streaming_mixin import PlayerStreamingMixin
from .queue_controller import QueueController
from .realtime import RealtimeProcessor


class AudioPlayer(
    PlayerFileLoadingMixin,
    PlayerQueueNavigationMixin,
    PlayerStreamingMixin,
    PlayerCallbacksMixin,
    PlayerFingerprintLoaderMixin,
    PlayerPropertiesMixin,
):
    """
    Real-time audio player with advanced DSP and library integration.

    Facade that coordinates 5 specialized components for clean separation of concerns:
    - PlaybackController: State machine
    - AudioFileManager: File I/O
    - QueueController: Queue/playlist
    - GaplessPlaybackEngine: Prebuffering
    - IntegrationManager: Library/callbacks

    File loading, queue navigation, streaming/auto-advance,
    callbacks/introspection, fingerprint-scheduling, and the
    backward-compatible property surface are mixed in (see the imports
    above) rather than composed as separate objects, so existing code/tests
    that read or patch attributes like ``player._track_generation`` or
    ``player.position`` directly continue to work unchanged (#4249). This
    class itself keeps only ``__init__``, transport (play/pause/stop/seek),
    and cleanup — every method here or in a mixin touches ``_audio_lock``
    with the exact same acquisition order/scope as before the split.

    Features:
    - Advanced real-time DSP processing
    - Automatic mastering with multiple profiles
    - Queue management and playlist support
    - Gapless playback with prebuffering
    - Library integration and auto-reference selection
    - Performance monitoring and statistics

    API compatible with AudioPlayer.

    Phase 6C: Fully migrated to RepositoryFactory pattern (no LibraryManager fallback)
    """

    def __init__(
        self,
        config: PlayerConfig | None = None,
        get_repository_factory: Callable[[], Any] | None = None,
    ) -> None:
        """
        Initialize the enhanced audio player with components.

        Args:
            config: Player configuration (PlayerConfig)
            get_repository_factory: Callable that returns RepositoryFactory instance (REQUIRED)
        """
        if config is None:
            config = PlayerConfig()

        # Validate required parameter
        if get_repository_factory is None:
            raise ValueError("get_repository_factory is required")

        self.config = config
        self.get_repository_factory = get_repository_factory

        # Initialize components
        self.playback = PlaybackController()
        self.file_manager = AudioFileManager(config.sample_rate)
        self.queue = QueueController(get_repository_factory)
        self.processor = RealtimeProcessor(config)
        self.gapless = GaplessPlaybackEngine(self.file_manager, self.queue)
        self.integration = IntegrationManager(
            self.playback,
            self.file_manager,
            self.queue,
            self.processor,
            get_repository_factory,
        )

        # Fingerprinting service for adaptive mastering
        self.fingerprint_service = FingerprintService()
        self._current_fingerprint: dict | None = None
        # Protects _current_fingerprint against a background loader writing
        # concurrently with the playback thread reading it for adaptive DSP
        # parameters (fixes #2491).
        self._fingerprint_lock = threading.Lock()
        # Monotonic counter incremented on each track load; fingerprint
        # callbacks check their generation matches before applying (#3445).
        self._track_generation: int = 0

        # Control flags
        self.auto_advance = True
        self._auto_advancing = threading.Event()
        self._advance_generation = 0  # Monotonic counter for compare-and-clear (#3350)
        self._stop_requested = threading.Event()  # Prevents auto-advance after stop() (#3296)
        # #5240: mirrors `_stop_requested` for the equivalent pause-race window.
        # next_track()/previous_track() capture `was_playing` before a blocking
        # track transition, then unconditionally resume via `playback.play()`
        # afterward. `_stop_requested` lets that resume be overridden by a
        # concurrent stop() — but pause() had no equivalent signal, so a user
        # pausing mid-transition was silently overridden back to PLAYING. Set
        # in pause(), cleared in play(), checked (like _stop_requested) both
        # before attempting the resume and again immediately after.
        self._pause_requested = threading.Event()
        # #3694: hold a reference to the most recently spawned auto-advance
        # thread so cleanup() can join it. Without this, an in-flight advance
        # thread that has already passed its _stop_requested check can call
        # load_file() *after* cleanup() returns, leaving audio_data non-None
        # post-teardown (test flakiness; benign in Electron production where
        # the process exits immediately after cleanup).
        self._advance_thread: threading.Thread | None = None
        # #3727: covers the timeout path of #3694. cleanup() sets this
        # before the join; _auto_advance_next() checks it after the
        # is_playing() guard so a thread that gets past the
        # _stop_requested check (timeout window in cleanup) still aborts
        # before invoking next_track() and the subsequent audio_data
        # swap. Without this, a 2 s join timeout on slow I/O could
        # leave audio_data non-None after cleanup() returns.
        self._cleanup_in_progress = threading.Event()

        info("Enhanced AudioPlayer initialized (refactored architecture, RepositoryFactory support enabled, fingerprinting enabled)")

    # ========== Playback Control (delegates to PlaybackController) ==========

    def play(self) -> bool:
        """Start playback"""
        if not self.file_manager.is_loaded():
            warning("No audio file loaded")
            return False
        # #3669: clear `_stop_requested` AFTER `playback.play()` succeeds.
        # Previous order (clear → play) allowed a concurrent stop() to set
        # the flag between the clear and the state transition, leaving
        # state=PLAYING with _stop_requested=SET. Auto-advance is then
        # permanently suppressed for the rest of the session
        # (PlayerStreamingMixin._auto_advance_next checks
        # `_stop_requested.is_set()` and bails).
        started = self.playback.play()
        if started:
            self._stop_requested.clear()
            self._pause_requested.clear()
        return started

    def pause(self) -> bool:
        """Pause playback"""
        # #5240: mirrors `_stop_requested.set()` in stop() — records that the
        # user explicitly requested a pause, so a resume already in flight
        # (next_track()/previous_track()) can detect and honor it instead of
        # silently overriding it back to PLAYING.
        paused = self.playback.pause()
        if paused:
            self._pause_requested.set()
        return paused

    def stop(self) -> bool:
        """Stop playback"""
        self._stop_requested.set()
        self._auto_advancing.clear()
        return self.playback.stop()

    def seek(self, position_seconds: float) -> bool:
        """
        Seek to a position in seconds.

        Args:
            position_seconds: Target position in seconds

        Returns:
            bool: True if successful
        """
        # #3713: hold `_audio_lock` across the ENTIRE seek, including the
        # `playback.seek()` call. The original #3357 fix only snapshotted
        # `max_samples` and `sample_rate` atomically — but a gapless
        # `advance_with_prebuffer` could still swap to a shorter track
        # between the snapshot and the seek, leaving the clamp using the
        # old (larger) length. The new track would then receive a
        # position > its own length → empty slice → silence + immediate
        # auto-advance. PlaybackController.seek() takes only its own
        # `_lock`; the canonical nesting `_audio_lock → PlaybackController._lock`
        # is already used by `get_audio_chunk` so no inversion risk.
        #
        # #3781: `defer_notifications()` is the OUTER context manager (exits
        # last) so `playback.seek()`'s notify — which re-enters `_audio_lock`
        # via `IntegrationManager._on_playback_state_change` — fires only
        # after `_audio_lock` is released below, closing the deadlock against
        # `get_playback_info()`'s opposite `_position_lock` -> `_audio_lock`
        # order.
        with self.playback.defer_notifications(), self.file_manager._audio_lock:
            if not self.file_manager.is_loaded():
                warning("No audio file loaded")
                return False
            max_samples = self.file_manager.get_total_samples()
            sample_rate = self.file_manager.sample_rate
            position_samples = int(position_seconds * sample_rate)
            return self.playback.seek(position_samples, max_samples)

    @property
    def state(self) -> PlaybackState:
        """Get current playback state"""
        return self.playback.state

    def is_playing(self) -> bool:
        """Check if currently playing"""
        return self.playback.is_playing()

    # ========== Position (kept on AudioPlayer, not PlayerPropertiesMixin) ==========
    # test_no_direct_attribute_bypass inspects `type(player).__dict__["position"]`
    # directly (no MRO walk), so this property must live on this class itself.

    @property
    def position(self) -> int:
        """Get current position in samples"""
        return self.playback.position

    @position.setter
    def position(self, value: int) -> None:
        """Set position in samples — thread-safe via PlaybackController.seek()"""
        # Hold _audio_lock across the read-then-seek (#4141), like
        # AudioPlayer.seek(): a gapless transition between get_total_samples()
        # and playback.seek() could otherwise swap in a shorter track and leave
        # max_samples stale, seeking past the new track's end. RLock so the
        # nested get_total_samples() acquisition is safe.
        # #3781: defer_notifications() outer so seek()'s notify fires after
        # _audio_lock releases (see AudioPlayer.seek() for full rationale).
        with self.playback.defer_notifications(), self.file_manager._audio_lock:
            max_samples = self.file_manager.get_total_samples()
            self.playback.seek(value, max_samples)

    # ========== Cleanup ==========

    def cleanup(self) -> None:
        """Clean up resources"""
        # #3727: signal in-flight advance threads BEFORE stop() so the
        # _auto_advance_next early-bail (after the _stop_requested
        # check) sees the cleanup signal. Combined with the join below
        # this closes the timeout-path window left by #3694 alone.
        self._cleanup_in_progress.set()
        # #3438: bump the fingerprint generation counter so any in-flight
        # `_load_fingerprint_for_file` thread (spawned by the most recent
        # track load, still running because they're daemon threads) will
        # fail its `self._track_generation != generation` staleness check
        # under `_fingerprint_lock` and discard rather than write into the
        # freshly-cleaned processor. The existing #3719 lock discipline
        # handles the write-side; the cleanup-time bump just guarantees
        # there's always a newer generation to compare against post-stop.
        with self._fingerprint_lock:
            self._track_generation += 1
        self.stop()
        # #3694: wait for any in-flight auto-advance thread to finish before
        # clearing audio_data. _stop_requested was set by stop() above, but
        # an advance thread already past that gate can still call load_file()
        # and re-populate audio_data after clear_all(). Daemon=True saves us
        # at process exit, but tests that reuse the player (or assert on
        # post-cleanup state) need a deterministic barrier.
        # #4227: read _advance_thread under _audio_lock (it is written under the
        # same lock in PlayerStreamingMixin.get_audio_chunk()). A raw read can
        # tear under free-threaded 3.14+, skipping the join so an in-flight
        # advance thread survives cleanup. Join outside the lock so a slow
        # thread can't deadlock against a writer holding _audio_lock.
        with self.file_manager._audio_lock:
            advance_thread = self._advance_thread
        if advance_thread is not None and advance_thread.is_alive():
            advance_thread.join(timeout=2.0)
            if advance_thread.is_alive():
                warning("Auto-advance thread did not exit within cleanup timeout")
        self.file_manager.clear_all()
        self.gapless.cleanup()
        self.integration.cleanup()
        # Dispose the fingerprint service's self-created engine pool (#4501).
        try:
            self.fingerprint_service.close()
        except Exception:
            pass
        info("AudioPlayer cleanup completed")
