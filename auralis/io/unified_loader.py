"""
Unified Audio Loader
~~~~~~~~~~~~~~~~~~~~

Enhanced audio file loading supporting multiple formats and processing modes

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.

Unified audio loading system combining Matchering and Auralis capabilities
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..utils.logging import Code, ModuleError, debug, info, warning
from .formats import FFMPEG_FORMATS, SUPPORTED_FORMATS
from .loaders import check_ffprobe, load_with_ffmpeg, load_with_soundfile, reject_protocol_path
# check_ffmpeg is unused within this module's own logic (post-#4119, the
# ffprobe guard uses check_ffprobe exclusively — see #4540) but is re-exported
# so test_ffprobe_error_masking_4540.py can monkeypatch unified_loader.check_ffmpeg
# to isolate that regression guard from real ffmpeg-binary detection in CI (#4888).
from .loaders import check_ffmpeg  # noqa: F401
from .processing import resample_audio, validate_audio

# SUPPORTED_FORMATS / FFMPEG_FORMATS live in auralis.io.formats (the single
# source of truth) and are re-exported here for backward compatibility with
# existing importers (#4109).


def load_audio(
    file_path: str | Path,
    file_type: str = "audio",
    temp_folder: str | None = None,
    target_sample_rate: int | None = None,
    force_stereo: bool = False,
    normalize_on_load: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file with format detection and conversion

    Args:
        file_path: Path to the audio file
        file_type: Type of file being loaded ("target", "reference", or "audio")
        temp_folder: Temporary folder for FFmpeg conversions
        target_sample_rate: Resample to this sample rate (optional)
        force_stereo: Convert mono to stereo if True
        normalize_on_load: Normalize audio on loading
        cancel_event: Optional cooperative cancellation token. When set from
            another thread during an FFmpeg decode, the FFmpeg child is
            terminated promptly and ``asyncio.CancelledError`` is raised (#4496).
            Ignored for natively-decodable formats (no subprocess to cancel).

    Returns:
        tuple: (audio_data, sample_rate)

    Raises:
        ModuleError: If file cannot be loaded or is invalid
    """
    file_path = Path(file_path)

    debug(f"Loading {file_type} audio file: {file_path}")

    # Validate file exists
    if not file_path.exists():
        raise ModuleError(f"{Code.ERROR_FILE_NOT_FOUND}: {file_path}")

    # Check file size
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ModuleError(f"{Code.ERROR_EMPTY_FILE}: {file_path}")

    debug(f"File size: {file_size / (1024*1024):.2f} MB")

    # Get file extension
    file_ext = file_path.suffix.lower()

    if file_ext not in SUPPORTED_FORMATS:
        raise ModuleError(f"{Code.ERROR_UNSUPPORTED_FORMAT}: {file_ext}")

    info(f"Detected format: {SUPPORTED_FORMATS[file_ext]}")

    # Load audio based on format
    if file_ext in FFMPEG_FORMATS:
        audio_data, sample_rate = load_with_ffmpeg(
            file_path, temp_folder, cancel_event=cancel_event
        )
    else:
        audio_data, sample_rate = load_with_soundfile(file_path)

    # #3671: enforce the same duration ceiling that loader.py applies for
    # FFmpeg-routed formats. load_with_ffmpeg now also enforces it pre-decode
    # (#3671 again, in ffmpeg_loader.py), but this post-decode guard also
    # covers the soundfile path (long WAV/FLAC) and any future caller of
    # load_audio() that bypasses the FFmpeg pipeline.
    from auralis.io.loader import MAX_DURATION_SECONDS, oversize_decode_detail
    duration = len(audio_data) / max(sample_rate, 1)
    if duration > MAX_DURATION_SECONDS:
        raise ModuleError(
            f"{Code.ERROR_CORRUPTED}: Audio file exceeds maximum duration "
            f"({duration:.0f}s > {MAX_DURATION_SECONDS}s): {file_path}"
        )
    # Backstop for the same blind spot (#4875). The buffer is already resident
    # here, so this cannot prevent that allocation — but it still stops the
    # downstream validate/sanitize/resample copies, each of which multiplies
    # an already-oversized buffer, and it covers any caller reaching
    # load_audio() without passing one of the pre-decode guards.
    channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
    detail = oversize_decode_detail(duration, sample_rate, channels)
    if detail:
        raise ModuleError(f"{Code.ERROR_CORRUPTED}: {detail}: {file_path}")

    # Validate audio data
    audio_data, sample_rate = validate_audio(audio_data, sample_rate, file_type)

    # Apply post-processing options
    if target_sample_rate and target_sample_rate != sample_rate:
        # Only downsample, never upsample (resampling reduces quality)
        if target_sample_rate < sample_rate:
            original_sr = sample_rate
            audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
            debug(f"Downsampled from {original_sr} Hz to {target_sample_rate} Hz")
        else:
            debug(f"Skipping upsample: target {target_sample_rate} Hz >= current {sample_rate} Hz (would degrade quality)")

    if force_stereo and audio_data.ndim == 1:
        audio_data = np.column_stack([audio_data, audio_data])
        debug("Converted mono to stereo")

    if normalize_on_load:
        # #3749: dtype-preserving normalize. `np.max(np.abs(audio_data))`
        # returns a numpy scalar (typically float64 even on float32
        # input), so `audio_data / peak * 0.98` can silently promote
        # under older NumPy promotion rules. Cast `peak` to the input
        # dtype defensively — same drift class as #3658 / #3659 /
        # #3744 / #3752.
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            peak_typed = audio_data.dtype.type(peak)
            audio_data = audio_data / peak_typed * audio_data.dtype.type(0.98)
            debug("Normalized audio on load")

    info(f"Successfully loaded {file_type}: {audio_data.shape[0]} samples, "
         f"{sample_rate} Hz, {audio_data.ndim} channels")

    return audio_data, sample_rate


def get_audio_info(file_path: str | Path) -> dict[str, Any]:
    """
    Get information about an audio file without fully loading it

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary containing audio file information
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ModuleError(f"{Code.ERROR_FILE_NOT_FOUND}: {file_path}")

    file_ext = file_path.suffix.lower()
    file_size = file_path.stat().st_size

    info_dict = {
        'file_path': str(file_path),
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'format': SUPPORTED_FORMATS.get(file_ext, 'Unknown'),
        'extension': file_ext
    }

    try:
        if file_ext in FFMPEG_FORMATS:
            # Use FFprobe for non-native formats
            audio_info = _get_info_with_ffprobe(file_path)
        else:
            # Use soundfile for native formats
            audio_info = _get_info_with_soundfile(file_path)

        info_dict.update(audio_info)

    except Exception as e:
        info_dict['error'] = str(e)
        warning(f"Could not get audio info for {file_path}: {e}")

    return info_dict


def _get_info_with_soundfile(file_path: Path) -> dict[str, Any]:
    """Get audio info using soundfile"""
    info = sf.info(str(file_path))

    return {
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'frames': info.frames,
        'duration_seconds': info.duration,
        'format': info.format,
        'subtype': info.subtype
    }


def _get_info_with_ffprobe(file_path: Path) -> dict[str, Any]:
    """Get audio info using FFprobe"""
    # #4540: guard on ffprobe, not ffmpeg. They are separate binaries and an
    # environment can have one without the other — the exact gap #4119 closed
    # in ffmpeg_loader._probe_audio, which never reached this second copy.
    if not check_ffprobe():
        raise ModuleError(f"{Code.ERROR_FFMPEG_NOT_FOUND}: FFprobe required")

    # This second ffprobe call site had no protocol guard at all (#4834) —
    # ffmpeg_loader._probe_audio gained one in #4119, but it never reached here.
    reject_protocol_path(str(file_path))

    try:
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
            raise ModuleError(f"FFprobe failed: {result.stderr}")

        probe_data = json.loads(result.stdout)

        # Find audio stream
        audio_stream = None
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break

        if not audio_stream:
            raise ModuleError("No audio stream found")

        duration = float(probe_data.get('format', {}).get('duration', 0))

        def safe_int(value: Any, default: int = 0) -> int:
            """Convert to int, returning default for non-numeric values like 'N/A'."""
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        return {
            'sample_rate': safe_int(audio_stream.get('sample_rate', 0)),
            'channels': safe_int(audio_stream.get('channels', 0)),
            'duration_seconds': duration,
            'codec': audio_stream.get('codec_name', 'Unknown'),
            'bit_rate': safe_int(audio_stream.get('bit_rate', 0))
        }

    except subprocess.TimeoutExpired:
        raise ModuleError("FFprobe timed out")
    except FileNotFoundError:
        # #4540: check_ffprobe() above is memoized, so the binary can still
        # vanish between the check and the call. Without this the error escaped
        # as a bare FileNotFoundError.
        raise ModuleError(f"{Code.ERROR_FFMPEG_NOT_FOUND}: FFprobe not found")
    except json.JSONDecodeError:
        raise ModuleError("Invalid FFprobe output")


def batch_load_info(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """
    Get information for multiple audio files

    Args:
        file_paths: List of file paths

    Returns:
        List of audio info dictionaries
    """
    info_list = []

    for file_path in file_paths:
        try:
            info_dict = get_audio_info(file_path)
            info_list.append(info_dict)
        except Exception as e:
            info_list.append({
                'file_path': str(file_path),
                'error': str(e)
            })

    return info_list


# Convenience functions
def load_target(file_path: str | Path, **kwargs: Any) -> tuple[np.ndarray, int]:
    """Load target audio file"""
    return load_audio(file_path, file_type="target", **kwargs)


def load_reference(file_path: str | Path, **kwargs: Any) -> tuple[np.ndarray, int]:
    """Load reference audio file"""
    return load_audio(file_path, file_type="reference", **kwargs)


def is_audio_file(file_path: str | Path) -> bool:
    """Check if file is a supported audio format"""
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS


def get_supported_formats() -> list[str]:
    """Get list of supported audio formats"""
    return list(SUPPORTED_FORMATS.keys())