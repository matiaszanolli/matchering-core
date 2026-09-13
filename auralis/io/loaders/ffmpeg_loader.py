"""
FFmpeg Loader
~~~~~~~~~~~~~

Audio loading using FFmpeg for MP3/M4A/AAC/OGG/WMA

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import functools
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from ...utils.logging import Code, ModuleError, debug, warning
from .soundfile_loader import load_with_soundfile

# FFmpeg decodes untrusted media as an ambient, unversioned system binary
# (#4344) — no minimum-version enforcement previously existed beyond a bare
# presence probe. 4.0 (2018) predates it by enough years that we warn rather
# than refuse: a hard refusal on an old-but-present binary would silently
# break playback for users whose distro ships an older-but-still-patched
# build, and this heuristic can't distinguish "old" from "old but backported
# security fixes". Bundling a pinned FFmpeg build was considered (tracked
# alongside SEC-M2) but deferred: per-platform binary packaging + licensing
# review is out of scope here, so a version floor + warning is the interim
# mitigation until (or instead of) bundling.
MINIMUM_FFMPEG_VERSION = (4, 0, 0)

# Rejects any file_path FFmpeg/libavformat would parse as a protocol
# specifier rather than a plain filesystem path. A bare `"://" in path`
# substring check (the guard this replaced) misses `pipe:0`, `concat:a|b`,
# `data:...` and every other colon-only protocol, which never contain "://"
# (#4834). Requires at least two characters before the colon so a
# single-letter Windows drive path ("C:\\Users\\...") is never misclassified
# as a protocol — every real FFmpeg protocol name is 2+ characters.
_PROTOCOL_PREFIX_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]+:')


def reject_protocol_path(file_path_str: str) -> None:
    """Raise ``ModuleError`` if *file_path_str* looks like an FFmpeg protocol
    specifier (``concat:``, ``pipe:``, ``http://``, ...) instead of a plain
    filesystem path. Shared by both ffprobe/ffmpeg call sites (#4834)."""
    if _PROTOCOL_PREFIX_RE.match(file_path_str):
        raise ModuleError(
            f"{Code.ERROR_UNSUPPORTED_FORMAT}: URL/protocol inputs are not allowed ({file_path_str})"
        )


def _parse_ffmpeg_version(version_output: str) -> tuple[int, ...] | None:
    """Extract the (major, minor, ...) version tuple from `ffmpeg -version` output.

    First line looks like: "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright ..."
    or "ffmpeg version n6.0 Copyright ...". Returns None if unparseable (e.g.
    an unusual distro-patched version string) rather than guessing.
    """
    match = re.search(r"ffmpeg version n?(\d+(?:\.\d+)+)", version_output)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def _terminate_process(proc: "subprocess.Popen[str]") -> None:
    """Terminate an FFmpeg child promptly: SIGTERM, escalate to SIGKILL.

    Best-effort — the child is going away regardless, so any error while
    signalling it is swallowed.
    """
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


def _run_ffmpeg_cancellable(
    cmd: list[str],
    timeout: float,
    cancel_event: "threading.Event | None",
) -> "subprocess.CompletedProcess[str]":
    """Run an FFmpeg command, honouring a cooperative cancellation token.

    With no ``cancel_event`` this is a plain blocking ``subprocess.run`` —
    byte-for-byte the previous behaviour, so library-scan and other non-job
    callers are unaffected.

    With a ``cancel_event``, FFmpeg runs under ``Popen`` and is polled so that a
    ``cancel_event.set()`` from another thread (``ProcessingEngine.cancel_job``)
    terminates the child within ~100 ms instead of parking a worker thread and a
    CPU core for up to ``timeout`` seconds (#4496). On cancel the child is
    terminated and ``asyncio.CancelledError`` is raised so the abort surfaces as
    a clean cancellation rather than a spurious decode failure.
    """
    if cancel_event is None:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if cancel_event.is_set():
        raise asyncio.CancelledError()

    # FFmpeg writes the decoded WAV to a file (not stdout), so stdout stays
    # empty and stderr carries only modest progress text — the pipe buffers
    # never fill, so repeated poll-timeout communicate() calls cannot deadlock.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = time.monotonic()
    poll_interval = 0.1
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=poll_interval)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            if cancel_event.is_set():
                _terminate_process(proc)
                raise asyncio.CancelledError()
            if time.monotonic() - start > timeout:
                _terminate_process(proc)
                raise  # -> ModuleError(ERROR_FFMPEG_TIMEOUT) in the caller

# Upper bound on plausible audio bitrate, used only when ffprobe reports no
# duration (e.g. true-VBR MP3 without a Xing/VBRI header, #4128). Estimating
# duration as file_bits / this value yields a *lower bound* on the real
# duration, so the pre-decode guard rejects only genuinely oversized files
# (>~1.8 GB at MAX_DURATION_SECONDS=7200) and never a normal one. The
# post-decode duration check remains the backstop for everything in between.
_MAX_PLAUSIBLE_BITRATE_BPS = 2_000_000


@functools.lru_cache(maxsize=1)
def check_ffmpeg() -> bool:
    """Check if FFmpeg is available, warning if its version is below the floor.

    Memoized for the process lifetime (#4117): FFmpeg availability does not
    change within a run, so probing once avoids forking an `ffmpeg -version`
    subprocess on every FFmpeg-routed file load (one redundant probe per file
    during bulk scans). Call ``check_ffmpeg.cache_clear()`` to force a re-probe
    (e.g. in tests that toggle availability).

    A version below MINIMUM_FFMPEG_VERSION only warns (#4344) — presence is
    still all that's required to proceed, since a hard refusal on a heuristic
    version parse would risk breaking playback on a legitimately-patched but
    unusually-versioned system build.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False

        version = _parse_ffmpeg_version(result.stdout)
        if version is None:
            warning("Could not determine FFmpeg version from `ffmpeg -version` output")
        elif version < MINIMUM_FFMPEG_VERSION:
            min_str = ".".join(str(p) for p in MINIMUM_FFMPEG_VERSION)
            got_str = ".".join(str(p) for p in version)
            warning(
                f"FFmpeg {got_str} is older than the recommended minimum {min_str} — "
                "older builds carry known demuxer/decoder CVEs; consider upgrading."
            )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=1)
def check_ffprobe() -> bool:
    """Check if the ffprobe binary is available.

    ffprobe is a separate binary from ffmpeg (#4119): an environment may have
    ffmpeg but not ffprobe. Memoized for the process lifetime like
    ``check_ffmpeg`` (#4117); call ``check_ffprobe.cache_clear()`` to re-probe.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-version'],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _probe_audio(file_path: Path) -> dict:
    """
    Probe audio file with ffprobe.

    Returns a dict with keys:
        duration    float | None  – total duration in seconds
        sample_rate int  | None  – native sample rate (Hz)
        channels    int  | None  – number of channels
    """
    result_dict: dict = {'duration': None, 'sample_rate': None, 'channels': None}
    try:
        import json

        ffprobe_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            '--',
            str(file_path)
        ]

        result = subprocess.run(
            ffprobe_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            warning(f"ffprobe failed: {result.stderr}")
            return result_dict

        probe_data = json.loads(result.stdout)

        duration = probe_data.get('format', {}).get('duration')
        if duration:
            result_dict['duration'] = float(duration)

        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                sr = stream.get('sample_rate')
                ch = stream.get('channels')
                if sr:
                    result_dict['sample_rate'] = int(sr)
                if ch:
                    result_dict['channels'] = int(ch)
                break

    # #4119: catch FileNotFoundError (ffprobe binary absent) so it does not
    # escape and get mislabeled ERROR_FFMPEG_CONVERSION by the caller; degrade
    # to the empty result_dict (load_with_ffmpeg also guards via check_ffprobe).
    except FileNotFoundError:
        warning("ffprobe binary not found; skipping probe (install ffprobe for accurate metadata)")
    # Keep this tuple narrow — do NOT add a trailing `Exception` (#3697).
    # A catch-all here swallows programming errors and mislabels every load as
    # `Code.ERROR_CORRUPTED`.
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as e:
        warning(f"Could not probe audio with ffprobe: {e}")

    return result_dict


def load_with_ffmpeg(
    file_path: Path,
    temp_folder: str | None = None,
    cancel_event: "threading.Event | None" = None,
    offset: float | None = None,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio using FFmpeg conversion to WAV.

    If ``cancel_event`` is provided and set from another thread mid-decode, the
    FFmpeg child is terminated promptly and ``asyncio.CancelledError`` is raised
    (#4496). When it is ``None`` (the default for every non-job caller) the
    decode runs exactly as before.

    ``offset``/``duration`` (seconds) bound the decode to a window, mapping to
    ffmpeg's ``-ss``/``-t`` (#5110). Both default to ``None``, which decodes the
    whole file exactly as before — every existing call site is unaffected.

    ``-ss`` is placed *before* ``-i`` for input seeking, which skips to the
    window rather than decoding and discarding everything ahead of it. That is
    the whole point: fingerprint windowing needs ~150 s out of a file that may
    be hours long, and decoding all of it was up to ~50x the necessary CPU with
    a peak footprint in the GB range.
    """

    # Check if FFmpeg is available
    if not check_ffmpeg():
        raise ModuleError(f"{Code.ERROR_FFMPEG_NOT_FOUND}: FFmpeg required for {file_path.suffix}")

    # ffprobe is a separate binary used by _probe_audio below; guard it here so
    # its absence surfaces as ERROR_FFMPEG_NOT_FOUND rather than being
    # mislabeled ERROR_FFMPEG_CONVERSION further down (#4119).
    if not check_ffprobe():
        raise ModuleError(f"{Code.ERROR_FFMPEG_NOT_FOUND}: ffprobe required for {file_path.suffix}")

    # Ensure the input path is a regular file and not a URL/protocol
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise ModuleError(f"{Code.ERROR_FILE_NOT_FOUND}: {file_path}")
    # Guard against ffmpeg protocol specifiers (http://, pipe:, concat:, data:, etc.)
    file_path_str = str(file_path)
    reject_protocol_path(file_path_str)

    # Probe source format: duration, sample rate, and channel count
    probe = _probe_audio(file_path)
    expected_duration = probe['duration']
    # Fail fast when FFprobe cannot determine sample rate or channel count.
    # Silently assuming 44100 Hz / 2 ch caused 48 kHz and other files to be
    # permanently resampled to the wrong rate (fixes #2495).
    if probe['sample_rate'] is None or probe['channels'] is None:
        raise ModuleError(
            f"{Code.ERROR_CORRUPTED}: Could not probe sample rate / channel count for "
            f"'{file_path}'. FFprobe output may be malformed or the container unsupported."
        )
    source_sample_rate = probe['sample_rate']
    source_channels = probe['channels']

    # #3671: bail early if the source duration exceeds MAX_DURATION_SECONDS.
    # Without this an N-hour podcast / DJ mix wrote an N×900 MB temp WAV to
    # /tmp (RAM-backed on Linux) before any check ran, peaking RSS at ~2.7 GB
    # for a 90-minute MP3. Importing here avoids a circular import with
    # auralis.io.loader.
    from auralis.io.loader import MAX_DURATION_SECONDS, oversize_decode_detail
    if expected_duration is not None:
        if expected_duration > MAX_DURATION_SECONDS:
            raise ModuleError(
                f"{Code.ERROR_FFMPEG_CONVERSION}: Audio file exceeds maximum duration "
                f"({expected_duration:.0f}s > {MAX_DURATION_SECONDS}s): {file_path}"
            )
        # ffprobe already gave us sample_rate and channels above, so bound the
        # actual decoded size rather than trusting duration as a proxy (#4875).
        detail = oversize_decode_detail(
            expected_duration, source_sample_rate, source_channels
        )
        if detail:
            raise ModuleError(
                f"{Code.ERROR_FFMPEG_CONVERSION}: {detail}: {file_path}"
            )
    else:
        # #4128: ffprobe returned no duration (true-VBR MP3 without Xing/VBRI).
        # Fall back to a file-size-based lower-bound estimate so an oversized
        # file is rejected before FFmpeg decodes it to a multi-hundred-MB temp WAV.
        min_duration = (file_path.stat().st_size * 8) / _MAX_PLAUSIBLE_BITRATE_BPS
        if min_duration > MAX_DURATION_SECONDS:
            raise ModuleError(
                f"{Code.ERROR_FFMPEG_CONVERSION}: Audio file exceeds maximum duration "
                f"(size implies >= {min_duration:.0f}s at "
                f"{_MAX_PLAUSIBLE_BITRATE_BPS // 1000} kbps > {MAX_DURATION_SECONDS}s): {file_path}"
            )

    # Create temporary WAV file
    if temp_folder:
        temp_dir = Path(temp_folder)
        temp_dir.mkdir(exist_ok=True)
    else:
        temp_dir = Path(tempfile.gettempdir())

    # Use mkstemp for unique temp filenames — prevents collision when two
    # threads concurrently load files with the same stem (#2908).
    fd, temp_wav_str = tempfile.mkstemp(suffix='.wav', dir=str(temp_dir), prefix='auralis_')
    os.close(fd)  # Close fd so FFmpeg can write to the path
    temp_wav = Path(temp_wav_str)

    try:
        # Convert to WAV using FFmpeg
        debug(f"Converting {file_path} to WAV using FFmpeg")

        # #3672: `-ac 2` lets FFmpeg apply its proper surround downmix matrix
        # (center → L+R at -3 dB, surround channels distributed). Previously
        # we passed `-ac {source_channels}` and then `soundfile_loader`
        # truncated to `[:, :2]` — which silently dropped the center channel
        # (vocals/dialogue) for 5.1/7.1 content.
        #
        # #4611/#4597: that downmix is now gated on `source_channels > 2`.
        # Applying it unconditionally also *up*-mixed genuine mono into fake
        # stereo, which #3672 never intended: `get_audio_info()` reported the
        # true channel count from ffprobe while `load_audio()` handed back a
        # 2-D array, so the two disagreed about the same file and the returned
        # dimensionality depended on file extension rather than audio content.
        # Gating here matches `soundfile_loader`, which downmixes only when
        # `shape[1] > 2`.
        needs_downmix = source_channels > 2
        ffmpeg_cmd = ['ffmpeg']
        # -ss BEFORE -i is input seeking: ffmpeg jumps to the offset instead of
        # decoding and discarding everything ahead of it (#5110).
        if offset is not None and offset > 0:
            ffmpeg_cmd += ['-ss', f'{offset:.6f}']
        ffmpeg_cmd += [
            '-i', file_path_str,
            '-acodec', 'pcm_s16le',            # 16-bit PCM
            '-ar', str(source_sample_rate),    # Preserve native sample rate
        ]
        if duration is not None and duration > 0:
            ffmpeg_cmd += ['-t', f'{duration:.6f}']
        if needs_downmix:
            ffmpeg_cmd += ['-ac', '2']         # Surround → stereo (#3672)
        ffmpeg_cmd += [
            '-y',                              # Overwrite output
            str(temp_wav)
        ]
        if offset is not None or duration is not None:
            debug(
                f"FFmpeg: bounded decode offset={offset}s duration={duration}s "
                f"(instead of the full file)"
            )
        if needs_downmix:
            debug(f"FFmpeg: converting at {source_sample_rate} Hz, "
                  f"downmixing {source_channels} → 2 ch")
        else:
            debug(f"FFmpeg: converting at {source_sample_rate} Hz, "
                  f"preserving {source_channels} ch")

        result = _run_ffmpeg_cancellable(
            ffmpeg_cmd,
            timeout=300,  # 5 minute timeout
            cancel_event=cancel_event,
        )

        if result.returncode != 0:
            raise ModuleError(f"{Code.ERROR_FFMPEG_CONVERSION}: {result.stderr}")

        # Load the converted WAV file
        audio_data, sample_rate = load_with_soundfile(temp_wav)

        # Validate duration against original file metadata.
        #
        # #5110: with a bounded decode the output is deliberately shorter than
        # the source, so comparing against the *file's* duration would flag
        # every windowed read as severely truncated. Compare against the span
        # that was actually requested instead — the check still catches a decode
        # that returned far less than asked for, which is what it is for.
        effective_expected = expected_duration
        if effective_expected is not None and (offset is not None or duration is not None):
            remaining = max(0.0, effective_expected - (offset or 0.0))
            effective_expected = min(duration, remaining) if duration else remaining

        if effective_expected is not None and effective_expected > 0:
            actual_duration = len(audio_data) / sample_rate
            expected_duration = effective_expected
            duration_percentage = (actual_duration / expected_duration) * 100

            if duration_percentage < 10:
                # Severely truncated - raise error
                raise ModuleError(
                    f"{Code.ERROR_TRUNCATED_FILE}: File is severely truncated "
                    f"({duration_percentage:.1f}% complete, expected {expected_duration:.2f}s, got {actual_duration:.2f}s)"
                )
            elif duration_percentage < 90:
                # Moderately truncated - log warning
                warning(
                    f"{Code.WARNING_TRUNCATED_FILE}: File appears truncated "
                    f"({duration_percentage:.1f}% complete, expected {expected_duration:.2f}s, got {actual_duration:.2f}s)"
                )

        return audio_data, sample_rate

    except subprocess.TimeoutExpired:
        raise ModuleError(f"{Code.ERROR_FFMPEG_TIMEOUT}: Conversion timed out")
    except ModuleError:
        # #3695: don't re-wrap already-specific ModuleError raises (e.g.
        # ERROR_TRUNCATED_FILE from soundfile_loader). Matches the pattern
        # in soundfile_loader.py:80-82. Without this, every internal
        # diagnostic code is overwritten by ERROR_FFMPEG_CONVERSION.
        raise
    except Exception as e:
        raise ModuleError(f"{Code.ERROR_FFMPEG_CONVERSION}: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_wav.exists():
            try:
                temp_wav.unlink()
                debug(f"Cleaned up temporary file: {temp_wav}")
            except Exception:
                warning(f"Failed to clean up temporary file: {temp_wav}")
