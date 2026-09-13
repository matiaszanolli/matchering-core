"""
Player Fingerprint Loader Mixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Background-thread 25D fingerprint scheduling for adaptive mastering,
extracted from AudioPlayer (#4249).

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import threading
from pathlib import Path

from ..analysis.fingerprint.fingerprint_service import FingerprintService
from ..utils.logging import debug, info, warning
from .realtime import RealtimeProcessor


class PlayerFingerprintLoaderMixin:
    """Schedules and applies 25D fingerprints in the background.

    Mixed into AudioPlayer (rather than composed as a separate object) so
    that ``_track_generation``, ``_fingerprint_lock``, ``_current_fingerprint``,
    and the two methods below stay directly on the player instance. Existing
    tests patch/read these via ``player._schedule_fingerprint_load`` and
    ``player._track_generation`` directly; a composed helper object would
    require changing every call site instead of just moving code.

    Instance state below is initialized by AudioPlayer.__init__, not here —
    declared here only so type checkers know this mixin depends on it.
    """

    fingerprint_service: FingerprintService
    processor: RealtimeProcessor
    _current_fingerprint: dict | None
    _fingerprint_lock: threading.Lock
    _track_generation: int

    def _schedule_fingerprint_load(self, file_path: str) -> None:
        """
        Bump the generation counter and spawn a background fingerprint loader.

        Called by every track-load entry point (``load_file``,
        ``load_track_from_library``, gapless ``next_track``) so that stale
        in-flight fingerprint threads are invalidated and the new track gets
        its adaptive-mastering fingerprint applied (#3445, #3463).

        Increment + publish runs under ``_fingerprint_lock`` so two
        concurrent loads don't both observe the same value and pass the
        same generation to their loaders (#3473). Plain ``self._x += 1`` is
        LOAD_ATTR/STORE_ATTR at bytecode level — not atomic.

        Args:
            file_path: Path of the now-current audio file.
        """
        with self._fingerprint_lock:
            self._track_generation += 1
            generation = self._track_generation

        threading.Thread(
            target=self._load_fingerprint_for_file,
            args=(file_path, generation),
            daemon=True,
            name="fingerprint-loader",
        ).start()

    def _load_fingerprint_for_file(self, file_path: str, generation: int) -> None:
        """
        Load 25D fingerprint for file and apply to processor for adaptive mastering.

        Uses 3-tier caching: database → .25d file → on-demand computation.
        Non-blocking operation (uses cache hits when available).

        Args:
            file_path: Path to audio file
            generation: Track generation counter — result is discarded if the
                track has changed since this thread was started (#3445).
        """
        try:
            audio_path = Path(file_path)
            fingerprint = self.fingerprint_service.get_or_compute(audio_path)

            # #3719: hold `_fingerprint_lock` across the staleness check
            # AND the processor write. Previously the check was an
            # unlocked atomic int read (#3445) and the
            # `processor.set_fingerprint()` write was OUTSIDE the lock
            # entirely. Race: thread T_A (fp for track 5) passes the
            # check; user skips to track 6 → T_B spawned; T_B hits cache,
            # passes check, calls set_fingerprint(fp6); T_A (still in
            # flight) then calls set_fingerprint(fp5) — overwriting fp6.
            # Holding the lock across check+act serialises the two threads
            # so the newer generation always wins.
            if fingerprint:
                with self._fingerprint_lock:
                    if self._track_generation != generation:
                        debug(f"Discarding stale fingerprint for {audio_path.name} (generation {generation} != {self._track_generation})")
                        return
                    self._current_fingerprint = fingerprint
                    self.processor.set_fingerprint(fingerprint)
                info(f"Fingerprint loaded for adaptive mastering: "
                     f"LUFS {fingerprint.get('lufs', 0):.1f} dB, "
                     f"crest {fingerprint.get('crest_db', 0):.1f} dB")
            else:
                debug(f"Failed to load fingerprint for {audio_path.name}, using profile-based mastering")
                with self._fingerprint_lock:
                    if self._track_generation != generation:
                        return
                    self._current_fingerprint = None
                    self.processor.set_fingerprint(None)

        except Exception as e:
            warning(f"Error loading fingerprint: {e}")
            # Same lock discipline on the error path — check generation
            # and write under the same lock so a newer load doesn't get
            # its fingerprint reset by this stale error handler.
            with self._fingerprint_lock:
                if self._track_generation != generation:
                    return
                self._current_fingerprint = None
                self.processor.set_fingerprint(None)
