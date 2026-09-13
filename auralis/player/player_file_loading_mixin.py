"""
Player File Loading Mixin
~~~~~~~~~~~~~~~~~~~~~~~~~

Track/reference file loading for AudioPlayer, extracted from
enhanced_audio_player.py (#4249). Split out from queue navigation
(``player_queue_navigation_mixin.py``) purely to keep both modules under
the project's 300-line convention — the two remain a tightly coupled pair
(every navigation entry point loads a file as part of advancing) and are
always mixed into ``AudioPlayer`` together.

Mixed into AudioPlayer (rather than composed as a separate object) so the
existing lock discipline — every method below acquires
``self.file_manager._audio_lock`` / ``self.playback`` locks directly on the
player instance, exactly as before the extraction — continues to work
unchanged. Moving code here does NOT change lock acquisition order or scope
relative to the fixed state of #4141 / #3735 / #5105.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

from collections.abc import Callable

from .audio_file_manager import AudioFileManager
from .gapless_playback_engine import GaplessPlaybackEngine
from .integration_manager import IntegrationManager
from .playback_controller import PlaybackController
from .realtime import RealtimeProcessor


class PlayerFileLoadingMixin:
    """Track/reference file loading, delegating to component objects.

    Instance state below is initialized by AudioPlayer.__init__ or provided
    by sibling mixins, not here — declared here only so type checkers know
    this mixin depends on it.
    """

    file_manager: AudioFileManager
    playback: PlaybackController
    gapless: GaplessPlaybackEngine
    integration: IntegrationManager
    processor: RealtimeProcessor
    # Provided by PlayerFingerprintLoaderMixin. Bare annotation only (no
    # assignment) — this must NOT become a real attribute in this class's
    # __dict__, or it would shadow PlayerFingerprintLoaderMixin's actual
    # implementation via MRO regardless of base-class order.
    _schedule_fingerprint_load: Callable[[str], None]

    def load_file(self, file_path: str) -> bool:
        """
        Load an audio file for playback with automatic fingerprinting for adaptive mastering.

        Args:
            file_path: Path to the audio file

        Returns:
            bool: True if successful
        """
        if self.file_manager.load_file(file_path):
            # #3667: hold _audio_lock while the playback controller resets
            # position so no concurrent get_audio_chunk() can observe the
            # new audio_data at the old position. file_manager.load_file
            # has already swapped audio_data inside _audio_lock — re-taking
            # the RLock here (it's reentrant) keeps swap+reset visible to
            # readers as a single critical section.
            # #3781: defer_notifications() outer so load_and_stop()'s notify
            # fires after _audio_lock releases (see seek() for full rationale).
            with self.playback.defer_notifications(), self.file_manager._audio_lock:
                self.playback.load_and_stop()

            self._schedule_fingerprint_load(file_path)

            # Start prebuffering next track
            self.gapless.start_prebuffering()

            self.integration._notify_callbacks({
                'action': 'file_loaded',
                'file': file_path
            })
            return True
        else:
            self.playback.set_error()
            return False

    def load_reference(self, file_path: str) -> bool:
        """
        Load a reference file for real-time mastering.

        Args:
            file_path: Path to the reference audio file

        Returns:
            bool: True if successful
        """
        ref_data = self.file_manager.load_reference(file_path)
        if ref_data is not None:
            self.processor.set_reference_audio(ref_data)
            self.integration._notify_callbacks({
                'action': 'reference_loaded',
                'file': file_path
            })
        return ref_data is not None

    def load_track_from_library(self, track_id: int) -> bool:
        """
        Load a track from the library by ID.

        Args:
            track_id: Database ID of the track

        Returns:
            bool: True if successful
        """
        if self.integration.load_track_from_library(track_id):
            # Mirror load_file (#3667): hold _audio_lock so the position
            # reset is atomic with the audio swap already done by
            # integration.load_track_from_library -> file_manager.load_file.
            # The DB session is closed before we reach here, so no
            # lock-ordering issue.
            # #3781: defer_notifications() outer so load_and_stop()'s notify
            # fires after _audio_lock releases (see seek() for full rationale).
            # #3785: capture current_file inside the same _audio_lock section
            # rather than as a separate raw read after it releases — closes
            # the window where a concurrent load/advance could swap
            # current_file between load_and_stop() and the read.
            with self.playback.defer_notifications(), self.file_manager._audio_lock:
                self.playback.load_and_stop()
                current_file = self.file_manager.current_file
            # IntegrationManager.load_track_from_library() calls
            # file_manager.load_file() internally, which sets current_file.
            # Schedule the fingerprint loader here (the player wrapper
            # bypasses AudioPlayer.load_file()) so adaptive mastering picks
            # up the new track instead of keeping the previous one (#3463).
            if current_file:
                self._schedule_fingerprint_load(current_file)
            self.gapless.start_prebuffering()
            return True
        else:
            self.playback.set_error()
            return False
