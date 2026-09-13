"""
Files Router
~~~~~~~~~~~~

Handles file operations: file uploads and supported format queries.

Endpoints:
- POST /api/files/upload - Upload audio files
- GET /api/audio/formats - Get supported audio formats

Note: POST /api/library/scan is handled exclusively by routers/library.py
which supports multiple directories, recursive scanning, and progress callbacks
via WebSocket (fixes #2123).

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import logging
import shutil
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from auralis.utils.logging import sanitize_log_value

from .dependencies import require_repository_factory
from core.processing_engine import _safe_error_message


# Upload limits — single source of truth in config.limits (#4033). MAX_UPLOAD_FILES
# bounds how many files one multipart request may carry so a single request can't
# monopolise the backend / bloat the DB (#4349); MAX_UPLOAD_BYTES is the per-file
# byte cap enforced at the read site below.
from config.limits import MAX_UPLOAD_BYTES as _MAX_UPLOAD_BYTES
from config.limits import MAX_UPLOAD_FILES as _MAX_UPLOAD_FILES

# Magic byte signatures for supported audio formats (issue #2415).
# NOTE: every extension in formats.SUPPORTED_FORMATS must be recognised here or
# well-formed uploads are silently rejected (#4498). The prefix list covers the
# formats with a fixed leading signature; AIFF (FORM…AIFF/AIFC) and MP4/M4A/AAC
# (ftyp at offset 4) need the two-part checks in _has_valid_audio_magic().
# tests/backend/test_upload_magic_bytes.py::test_every_supported_format_is_recognised
# guards this against future drift.
_AUDIO_MAGIC: tuple[bytes, ...] = (
    b'\xff\xfb', b'\xff\xf3', b'\xff\xf2',  # MP3 (sync word variants)
    b'\xff\xf1', b'\xff\xf9',                # AAC (ADTS, MPEG-4 / MPEG-2)
    b'ID3',                                  # MP3 with ID3 tag
    b'RIFF',                                 # WAV
    b'fLaC',                                 # FLAC
    b'OggS',                                 # OGG/Vorbis/Opus
    b'.snd',                                 # AU (Sun/NeXT)
    # WMA / ASF — 16-byte ASF header-object GUID (#4498)
    b'\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c',
)


def _has_valid_audio_magic(data: bytes) -> bool:
    """Return True if the first bytes match a known audio container."""
    if len(data) < 8:
        return False
    # MP4/M4A/AAC: 4-byte box size then ASCII 'ftyp'
    if data[4:8] == b'ftyp':
        return True
    # AIFF/AIFC: 'FORM' then the form-type 'AIFF'/'AIFC' at offset 8 (#4498)
    if data[0:4] == b'FORM' and data[8:12] in (b'AIFF', b'AIFC'):
        return True
    return any(data.startswith(m) for m in _AUDIO_MAGIC)

# Import only if available
try:
    from auralis.io.unified_loader import load_audio, SUPPORTED_FORMATS
    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False

logger = logging.getLogger(__name__)


class UploadResultItem(BaseModel):
    """Per-file outcome of an upload batch.

    The success and failure branches emit different key sets — only
    `filename`, `status` and `message` are common — so the track fields are
    optional rather than required.
    """
    filename: str = Field(description="Client-supplied file name")
    status: Literal["success", "error"] = Field(description="Per-file outcome")
    message: str = Field(description="Human-readable detail")
    track_id: int | None = Field(default=None, description="Created track ID, on success")
    title: str | None = Field(default=None, description="Derived track title, on success")
    duration: float | None = Field(default=None, description="Duration in seconds, on success")
    sample_rate: int | None = Field(default=None, description="Sample rate in Hz, on success")


class UploadResultResponse(BaseModel):
    """Result of an audio file upload batch.

    Partial success is normal: the route returns 200 with a mix of
    `status: "success"` and `status: "error"` entries rather than failing the
    whole batch.
    """
    results: list[UploadResultItem] = Field(
        default_factory=list,
        description="One entry per uploaded file, in request order",
    )


class SupportedFormatsResponse(BaseModel):
    """Audio formats, sample rates, and bit depths the backend accepts."""
    input_formats: list[str] = Field(description="Accepted input file extensions")
    output_formats: list[str] = Field(description="Available output file extensions")
    sample_rates: list[int] = Field(description="Supported sample rates in Hz")
    bit_depths: list[int] = Field(description="Supported bit depths")


# ============================================================================
# DEPENDENCY WIRING (#4670)
#
# create_files_router() used to be a closure: both handlers below were nested
# inside it purely to reach get_repository_factory via closure capture, which
# made a handler impossible to import or call without first building the whole
# router. Handlers are now module level; they reach the same callables through
# FastAPI Depends() instead.
#
# Factory calls can coexist in one process (the live app plus throwaway test
# apps), so each router publishes its own deps through a request ContextVar.
# The module-level holder is only the fallback for direct handler calls where
# no router-level dependency ran (#5262).
#
# A handler's Depends() default is only consulted when FastAPI itself invokes
# it for a real request; a direct unit-test call passes the dependency
# explicitly as a keyword argument and never touches _FilesDeps at all --
# that's the seam #4670 asked for.
# ============================================================================

class _FilesDeps:
    def __init__(
        self,
        get_repository_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.get_repository_factory = get_repository_factory


_deps = _FilesDeps()
_active_deps: ContextVar[_FilesDeps | None] = ContextVar(
    "_files_active_deps", default=None
)


def _current_deps() -> _FilesDeps:
    return _active_deps.get() or _deps


def _make_deps_binder(deps: _FilesDeps) -> Callable[[], Any]:
    """Publish one router instance's dependencies for its request."""

    async def _bind_deps() -> None:
        _active_deps.set(deps)

    return _bind_deps


def _get_repos() -> Any:
    """Get repository factory for accessing repositories."""
    get_repository_factory = _current_deps().get_repository_factory
    if get_repository_factory:
        return require_repository_factory(get_repository_factory)
    raise HTTPException(status_code=503, detail="Repository factory not available")


def _get_repos_provider() -> Callable[[], Any]:
    """Hand the handler the repository *provider*, not the repositories.

    The original nested handler called get_repos() partway through its body —
    after the empty-batch (400) and file-count (413) checks — so a request
    rejected by either of those never touched the repository factory.
    Resolving the factory through Depends() directly would hoist its 503 ahead
    of both, changing which status an over-large batch gets on a backend with
    no repositories, so the handler keeps calling it itself at the original
    point.
    """
    return _get_repos


async def upload_files(
    files: list[UploadFile] = File(...),
    get_repos: Callable[[], Any] = Depends(_get_repos_provider),
) -> dict[str, Any]:
    """
    Upload audio files for processing.

    Saves uploaded files to library and extracts metadata.

    Args:
        files: List of uploaded files

    Returns:
        dict: Upload results for each file with metadata

    Raises:
        HTTPException: If no files provided or library unavailable
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Cap the file count before decoding anything (#4349): reject an
    # over-large batch with 413 rather than looping over every element.
    if len(files) > _MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (max {_MAX_UPLOAD_FILES})",
        )

    repos = get_repos()

    if not HAS_LIBRARY:
        raise HTTPException(status_code=503, detail="Audio processing library not available")

    results: list[dict[str, Any]] = []
    supported_extensions = tuple(SUPPORTED_FORMATS.keys())

    for file in files:
        try:
            # Validate file type
            if not file.filename or not file.filename.lower().endswith(supported_extensions):
                # Log rejected uploads for security audit trail (fixes #2421).
                logger.warning(
                    f"Rejected upload of unsupported file type: {file.filename!r}"
                )
                results.append({
                    "filename": file.filename or "",
                    "status": "error",
                    "message": "Unsupported file type"
                })
                continue

            # Save uploaded file to temporary location.
            # Cap the read at the source instead of buffering the whole
            # body first (fixes #3494 / BE-NEW-36 — prior code did
            # `await file.read()` with no size limit and only checked the
            # cap AFTER reading, so a 2 GB POST OOM'd the backend before
            # rejection). +1 lets us detect overflow via length compare.
            suffix = Path(file.filename).suffix if file.filename else ""
            content = await file.read(_MAX_UPLOAD_BYTES + 1)
            if len(content) > _MAX_UPLOAD_BYTES:
                logger.warning(f"Rejected oversized upload: {file.filename!r} ({len(content)} bytes)")
                results.append({
                    "filename": file.filename or "",
                    "status": "error",
                    "message": "File exceeds maximum upload size of 500 MB"
                })
                continue

            # Validate magic bytes before invoking audio parser (issue #2415)
            if not _has_valid_audio_magic(content):
                logger.warning(
                    f"Rejected upload with invalid audio content: {file.filename!r}"
                )
                results.append({
                    "filename": file.filename or "",
                    "status": "error",
                    "message": "File content does not match any known audio format"
                })
                continue

            # Sync ops below — temp write, audio decode, file move, DB
            # insert — are wrapped in asyncio.to_thread so the event
            # loop isn't blocked for the (often multi-hundred-ms)
            # duration of an upload (fixes #3494 / BE-NEW-36).
            def _write_temp(content_bytes: bytes, suffix_str: str) -> str:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_str) as tmp:
                    tmp.write(content_bytes)
                    return tmp.name

            temp_path = await asyncio.to_thread(_write_temp, content, suffix)

            try:
                # Load audio to get metadata (sample rate, duration)
                audio_data, sample_rate = await asyncio.to_thread(load_audio, temp_path)
                duration = len(audio_data) / sample_rate

                # Move to permanent library storage before committing the DB record
                # (issue #2392: storing temp_path then deleting it made every upload
                # unplayable immediately after the finally block ran).
                upload_dir = Path.home() / ".auralis" / "uploads"
                await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)
                permanent_path = upload_dir / f"{uuid4().hex}{suffix}"
                await asyncio.to_thread(shutil.move, str(temp_path), str(permanent_path))

                # Extract file name without extension
                file_stem = Path(file.filename).stem if file.filename else "track"

                # Create track info dictionary for library
                track_info = {
                    "filepath": str(permanent_path),  # permanent, not temp (issue #2392)
                    "filename": file.filename,
                    "title": file_stem,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": 1 if audio_data.ndim == 1 else audio_data.shape[1]
                }

                # Add track to library via repository
                track = await asyncio.to_thread(repos.tracks.add, track_info)

                if track:
                    results.append({
                        "filename": file.filename or "",
                        "status": "success",
                        "message": "File uploaded and added to library",
                        "track_id": track.id,
                        "title": track.title,
                        "duration": float(track.duration),
                        "sample_rate": track.sample_rate
                    })
                    logger.info(f"Successfully uploaded and processed: {sanitize_log_value(file.filename)}")
                else:
                    results.append({
                        "filename": file.filename or "",
                        "status": "error",
                        "message": "Failed to add track to library"
                    })

            except Exception as e:
                logger.error(f"Audio processing error for {sanitize_log_value(file.filename)}: {e}", exc_info=True)
                results.append({
                    "filename": file.filename or "",
                    "status": "error",
                    "message": f"Failed to process audio: {_safe_error_message(e)}"
                })

            finally:
                # shutil.move already removed the temp file on success; unlink is a
                # no-op (missing_ok) in that case. On failure the temp file still
                # exists and must be cleaned up here.
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"Failed to clean temp file: {e}")

        except Exception as e:
            logger.error(f"Upload error for {sanitize_log_value(file.filename)}: {e}", exc_info=True)
            results.append({
                "filename": file.filename or "",
                "status": "error",
                "message": _safe_error_message(e)
            })

    return {"results": results}


async def get_supported_formats() -> dict[str, Any]:
    """
    Get supported audio formats.

    Returns information about supported input/output formats,
    sample rates, and bit depths.

    Returns:
        dict: Supported formats, sample rates, and bit depths
    """
    # #4768: these used to be hand-typed literals that had drifted — input
    # omitted .aiff/.aif/.au/.wma/.opus, all of which the upload validator in
    # this same file accepts. Derive each list from what actually enforces it.
    from typing import get_args

    from auralis.io.formats import SUPPORTED_FORMATS

    # What ProcessingSettings will accept for a processing job. Not every
    # (format, bit_depth) pair is valid — its model validator rejects the ones
    # libsndfile cannot write — so these are the options, not a matrix.
    #
    # routes.py treats the processing router as optional and tolerates it
    # failing to import. When it cannot, no processing job can be submitted,
    # so the truthful answer is "no output formats", not a crash here.
    try:
        from routers.processing_api import ProcessingSettings
    except Exception:
        output_formats: list[str] = []
        bit_depths: list[int] = []
    else:
        fields = ProcessingSettings.model_fields
        output_formats = sorted(f".{fmt}" for fmt in get_args(fields["output_format"].annotation))
        bit_depths = sorted(get_args(fields["bit_depth"].annotation))

    return {
        # The decoder's single source of truth (#4109), same set upload uses.
        "input_formats": sorted(SUPPORTED_FORMATS),
        "output_formats": output_formats,
        "sample_rates": [44100, 48000, 88200, 96000, 192000],
        "bit_depths": bit_depths,
    }


def create_files_router(
    get_repository_factory: Callable[[], Any] | None = None
) -> APIRouter:
    """
    Factory function to create files router with dependencies.

    Args:
        get_repository_factory: Callable that returns RepositoryFactory instance

    Returns:
        APIRouter: Configured router instance

    Note:
        Directory scanning is handled by routers/library.py (fixes #2123).
    """
    global _deps

    deps = _FilesDeps(get_repository_factory)
    _deps = deps

    router = APIRouter(
        tags=["files"],
        dependencies=[Depends(_make_deps_binder(deps))],
    )

    router.add_api_route("/api/files/upload", upload_files, methods=["POST"], response_model=UploadResultResponse)
    router.add_api_route("/api/audio/formats", get_supported_formats, methods=["GET"], response_model=SupportedFormatsResponse)

    return router
