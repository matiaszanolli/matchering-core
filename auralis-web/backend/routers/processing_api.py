#!/usr/bin/env python3

"""
Processing API Routes
~~~~~~~~~~~~~~~~~~~~~

FastAPI routes for audio processing functionality.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from core.processing_engine import ProcessingEngine, ProcessingJob, ProcessingStatus
from pydantic import BaseModel, Field, ValidationError, model_validator
from security.path_security import PathValidationError, validate_file_path

from .dependencies import with_error_handling
from .errors import NotFoundError

logger = logging.getLogger(__name__)

# Upload security constants (#2560). Single source of truth in config.limits (#4033).
from config.limits import MAX_UPLOAD_BYTES as _MAX_UPLOAD_BYTES
from config.limits import UPLOAD_TEMP_DIRNAME
# Derived from the single source of truth (auralis.io.formats) so the upload
# allowlist tracks exactly what the loader can decode (#4109).
from auralis.io.formats import AUDIO_EXTENSIONS as _ALLOWED_AUDIO_EXTENSIONS
# The mastering engine's real preset definitions — the presets endpoint
# projects these rather than restating them (#5220).
from auralis.core.config.preset_profiles import PresetProfile, create_preset_profiles


def _write_upload(temp_dir: Path, input_path: Path, content: bytes) -> None:
    """Create the upload dir and write *content* with exclusive create.

    Runs in a worker thread via asyncio.to_thread (#4653). "xb" makes the open
    fail with FileExistsError instead of following a symlink planted at the
    path between name selection and open (#2170).
    """
    temp_dir.mkdir(exist_ok=True)
    with open(input_path, "xb") as f:
        f.write(content)


def _is_valid_audio_magic(data: bytes) -> bool:
    """Return True if data starts with a known audio format magic signature."""
    if len(data) < 8:
        return False
    if data[:4] == b"RIFF":                          # WAV
        return True
    if data[:4] == b"fLaC":                          # FLAC
        return True
    if data[:4] == b"OggS":                          # OGG/Opus
        return True
    if data[:3] == b"ID3":                           # MP3 with ID3v2 tag
        return True
    if data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"):  # MP3 sync word
        return True
    if data[4:8] == b"ftyp":                         # M4A/MP4 (MPEG-4 container)
        return True
    if data[:4] in (b"FORM", b"AIFF"):               # AIFF
        return True
    return False

# Pydantic models for request/response
class ProcessingSettings(BaseModel):
    """Processing settings from UI"""
    # Literal, not a bare str (#4735): _build_config dispatches on this with an
    # if/elif chain that has no else, so an unrecognised value silently skipped
    # set_processing_mode() entirely and ran adaptive under whatever mode the
    # config defaulted to. Now a 422 at the route boundary.
    mode: Literal["adaptive", "reference", "hybrid"] = "adaptive"
    # Literal, not a bare str (#4746): output_format is interpolated straight
    # into the output filename extension, and bit_depth is mapped through a
    # subtype table (core/processing_engine.py) with a silent PCM_16
    # fallback for anything unrecognised — a bad value used to reach
    # libsndfile and fail deep inside the save step as a generic job
    # failure, not a 422 at submit time.
    output_format: Literal["wav", "flac", "mp3"] = "wav"
    bit_depth: Literal[16, 24, 32] = 16
    sample_rate: int | None = None  # None = keep original

    # EQ settings
    eq: dict[str, Any] | None = None

    # Dynamics settings
    dynamics: dict[str, Any] | None = None

    # Level matching settings
    level_matching: dict[str, Any] | None = None

    # Genre override
    genre_override: str | None = None

    @model_validator(mode="after")
    def _validate_format_bit_depth_combo(self) -> "ProcessingSettings":
        """Reject (output_format, bit_depth) pairs libsndfile cannot
        actually write, rather than letting them fail deep inside the save
        step (core/processing_engine.py's subtype_map) as a generic,
        misleading job failure (#4746).

        Verified against the installed libsndfile (soundfile.available_subtypes):
        - WAV:  PCM_16 / PCM_24 / PCM_32 all valid.
        - FLAC: PCM_16 / PCM_24 valid; FLAC has no 32-bit PCM subtype.
        - MP3:  only MPEG_LAYER_I/II/III subtypes exist — none of the PCM_*
          subtypes bit_depth maps to are valid for MP3, so no (mp3, bit_depth)
          combination can currently succeed via this pipeline.
        """
        if self.output_format == "mp3":
            raise ValueError(
                "output_format='mp3' is not currently supported — MP3 is a "
                "lossy format with no PCM bit depth, and the save pipeline "
                "only writes PCM subtypes. Use 'wav' or 'flac'."
            )
        if self.output_format == "flac" and self.bit_depth == 32:
            raise ValueError(
                "output_format='flac' does not support bit_depth=32 "
                "(FLAC's maximum is 24-bit PCM). Use bit_depth=16 or 24, "
                "or output_format='wav' for 32-bit."
            )
        return self


class ProcessRequest(BaseModel):
    """Request to process audio"""
    input_path: str
    settings: ProcessingSettings
    reference_path: str | None = None


class ProcessResponse(BaseModel):
    """Response after submitting processing job"""
    job_id: str
    # ProcessingStatus, not bare str (#3896): the enum is the authority on the
    # value set and, being a str Enum, serialises to the same JSON while making
    # OpenAPI publish the five valid values instead of an opaque "string".
    status: ProcessingStatus
    message: str


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: ProcessingStatus
    progress: float
    error_message: str | None = None
    result_data: dict[str, Any] | None = None


class CancelJobResponse(BaseModel):
    """Response after cancelling a job"""
    message: str
    job_id: str


class JobListResponse(BaseModel):
    """Response listing processing jobs"""
    jobs: list[JobStatusResponse]
    total: int


class QueueStatusResponse(BaseModel):
    """Current processing queue status"""
    queued: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total: int = 0

    model_config = {"extra": "allow"}


class PresetsResponse(BaseModel):
    """Available processing presets"""
    presets: dict[str, Any]


def _job_status_response(job: ProcessingJob) -> JobStatusResponse:
    """Serialize one job identically for the detail and list endpoints."""
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        result_data=job.result_data,
    )


class ProcessingParametersResponse(BaseModel):
    """Live auto-mastering parameters from the continuous-space system.

    `is_default` distinguishes "no measurement yet" from real measurements
    that happen to land on the default values (#3779) — both branches return
    200 with the same field set.
    """
    is_default: bool = Field(description="True when these are placeholders, not measurements")
    spectral_balance: float = Field(description="Spectral-balance coordinate (0–1)")
    dynamic_range: float = Field(description="Dynamic-range coordinate (0–1)")
    energy_level: float = Field(description="Energy-level coordinate (0–1)")
    target_lufs: float = Field(description="Target integrated loudness (LUFS)")
    peak_target_db: float = Field(description="Target true peak (dBFS)")
    bass_boost: float = Field(description="Low-shelf gain (dB)")
    air_boost: float = Field(description="High-shelf gain (dB)")
    compression_amount: float = Field(description="Compression amount (0–1)")
    expansion_amount: float = Field(description="Expansion amount (0–1)")
    stereo_width: float = Field(description="Target stereo width")


class CleanupResponse(BaseModel):
    """Response after cleaning up old jobs"""
    message: str
    removed: int


# ============================================================================
# DEPENDENCY WIRING (#4670)
#
# create_processing_router() used to be a 570-line closure: every handler
# below was nested inside it purely to reach get_processing_engine /
# get_enhancement_settings through closure capture, which made a handler
# impossible to import or call without first building the whole router.
# Handlers are now module level; they reach the same callables through
# FastAPI Depends() instead.
#
# _ProcessingDeps holds the raw callables the factory receives. `_deps` is the
# module-level holder, populated by create_processing_router() itself -- in
# production that happens exactly once per process (config/routes.py calls the
# factory a single time at startup).
#
# Unlike the player router (#4670's first slice), this factory *is* called
# more than once per process: tests/backend/test_processing_api.py builds a
# fresh router per test around a mock engine while main.app's own router --
# built at import time with the real engine getter *and* the enhancement
# settings getter -- is still live in the same process. A single module-level
# holder would be last-writer-wins across those routers, so a later test
# driving main.app (e.g. tests/backend/test_processing_parameters.py) would
# silently get the mock engine and a None enhancement-settings getter.
# Each factory call therefore also builds its own _ProcessingDeps, published
# for the duration of a request by a router-level dependency (_bind_deps)
# into the _current_deps ContextVar: router-level dependencies are solved
# before the handler's own Depends(), and a ContextVar set inside an async
# dependency stays visible for the rest of that request's task, so every
# provider below resolves against the deps of the router that matched.
# `_deps` stays as the fallback for anything resolved outside a request.
#
# A handler's Depends() default is only consulted when FastAPI itself invokes
# it for a real request; a direct unit-test call passes the dependency
# explicitly as a keyword argument and never touches _ProcessingDeps or the
# ContextVar at all -- that is the seam #4670 asked for.
# ============================================================================

class _ProcessingDeps:
    """Raw dependencies one create_processing_router() call was handed."""

    def __init__(
        self,
        get_processing_engine: Callable[[], ProcessingEngine | None] | None = None,
        get_enhancement_settings: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.get_processing_engine: Callable[[], ProcessingEngine | None] = (
            get_processing_engine if get_processing_engine is not None else (lambda: None)
        )
        self.get_enhancement_settings = get_enhancement_settings


_deps = _ProcessingDeps()

_current_deps: ContextVar[_ProcessingDeps] = ContextVar(
    "auralis_processing_deps", default=_deps
)


def _make_deps_binder(deps: _ProcessingDeps) -> Callable[[], Any]:
    """Build the router-level dependency that publishes `deps` per request.

    One binder per create_processing_router() call; see the DEPENDENCY WIRING
    note above for why the deps are not read straight off a single global.
    """
    async def _bind_deps() -> None:
        _current_deps.set(deps)

    return _bind_deps


def _get_processing_engine() -> ProcessingEngine | None:
    """Live ProcessingEngine, or None when it has not been initialised."""
    return _current_deps.get().get_processing_engine()


def _get_enhancement_settings() -> Callable[[], dict[str, Any]] | None:
    """The enhancement-settings *getter*, or None when the factory omitted it.

    Returns the callable rather than the settings dict so GET /parameters can
    still tell "not wired up" (503) from "wired up and empty" (#5073), exactly
    as it did when it read the closure variable directly.
    """
    return _current_deps.get().get_enhancement_settings


@with_error_handling("submit processing job")
async def process_audio(
    request: ProcessRequest,
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> ProcessResponse:
    """
    Submit an audio file for processing.
    Returns a job ID that can be used to track progress.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    try:
        # Validate input path against allowed directories (#2559)
        try:
            validated_input = validate_file_path(
                request.input_path, context="input_path"
            )
        except PathValidationError:
            # validate_file_path logs it once with the context above (#4925).
            raise HTTPException(status_code=400, detail="Invalid or inaccessible input path")

        # A "reference" job with no reference is not a meaningful adaptive
        # fallback — the caller asked for their reference to be matched.
        # Fail fast and say so, rather than silently returning
        # adaptive-mastered audio labelled as a reference job (#4735).
        if request.settings.mode == "reference" and not request.reference_path:
            raise HTTPException(
                status_code=422,
                detail="mode='reference' requires a reference_path",
            )

        validated_reference: Path | None = None
        if request.reference_path:
            try:
                validated_reference = validate_file_path(
                    request.reference_path, context="reference_path"
                )
            except PathValidationError:
                # validate_file_path logs it once with the context above (#4925).
                raise HTTPException(status_code=400, detail="Invalid or inaccessible reference path")

        # Create processing job
        job = await engine.create_job(
            input_path=str(validated_input),
            settings=request.settings.model_dump(),
            mode=request.settings.mode,
            reference_path=str(validated_reference) if validated_reference else None
        )

        # Submit to queue
        try:
            job_id = await engine.submit_job(job)
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=503,
                detail="Processing queue is full, please try again later",
            )

        # Debug, not info (#3844): input_path is an absolute media-library path.
        logger.debug(f"Processing job {job_id} submitted for {request.input_path}")

        return ProcessResponse(
            job_id=job_id,
            status=ProcessingStatus.QUEUED,
            message="Processing job submitted successfully"
        )

    # @with_error_handling maps anything else — notably OperationalError
    # -> retryable 503 (#4605). This arm is kept only so the try: stays
    # valid without re-indenting the whole body.
    except HTTPException:
        raise


@with_error_handling("upload and process")
async def upload_and_process(
    file: UploadFile = File(...),
    settings: str = Form(...),  # JSON string of ProcessingSettings
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> ProcessResponse:
    """
    Upload an audio file and immediately submit for processing.
    Combines file upload and processing submission in one request.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    try:
        # Parse settings from JSON string. Malformed client input (bad
        # JSON, or a shape ProcessingSettings rejects) is a 400, not the
        # generic 500 both used to fall through to via the bare
        # `except Exception` below — that loose behavior was masked by
        # a test assertion permitting 500 as an acceptable outcome
        # (#4788) until it was tightened and caught this.
        import json
        try:
            settings_dict = json.loads(settings)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {e}")
        try:
            processing_settings = ProcessingSettings(**settings_dict)
        except (TypeError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid processing settings: {e}")

        # Save uploaded file to temp location
        temp_dir = Path(tempfile.gettempdir()) / UPLOAD_TEMP_DIRNAME

        # Enforce size limit before reading the whole body (#2560)
        content = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {_MAX_UPLOAD_BYTES // 1024 // 1024} MB)"
            )

        # Reject files whose magic bytes don't match a known audio format (#2560)
        if not _is_valid_audio_magic(content):
            raise HTTPException(
                status_code=415,
                detail="Unsupported or invalid audio file format"
            )

        # Use a UUID filename to prevent client-controlled path injection (#2560).
        # Open with "xb" (exclusive create) to prevent TOCTOU: if another process
        # created a symlink at this path between path selection and open, the
        # kernel raises FileExistsError rather than following the symlink (fixes #2170).
        original_ext = Path(file.filename or "").suffix.lower()
        if original_ext not in _ALLOWED_AUDIO_EXTENSIONS:
            original_ext = ".bin"
        input_path = temp_dir / f"{uuid.uuid4()}{original_ext}"
        # #4653: the mkdir and the up-to-500 MB write used to run directly on the
        # event loop, stalling WebSocket audio delivery and every other request
        # for the length of the write. Same shape as files.py's _write_temp
        # (#3494). The "xb" exclusive-create mode is kept — it is the #2170
        # anti-TOCTOU guard, not incidental.
        await asyncio.to_thread(_write_upload, temp_dir, input_path, content)

        # Debug, not info (#3844): avoid logging absolute filesystem paths.
        logger.debug(f"Uploaded file saved to {input_path}")

        # Create and submit job — clean up temp file on failure (#3223)
        try:
            job = await engine.create_job(
                input_path=str(input_path),
                settings=processing_settings.model_dump(),
                mode=processing_settings.mode
            )

            try:
                job_id = await engine.submit_job(job)
            except asyncio.QueueFull:
                raise HTTPException(
                    status_code=503,
                    detail="Processing queue is full, please try again later",
                )

            return ProcessResponse(
                job_id=job_id,
                status=ProcessingStatus.QUEUED,
                message=f"File {file.filename} uploaded and queued for processing"
            )
        except Exception:
            # Clean up orphaned temp file on any failure after write
            input_path.unlink(missing_ok=True)
            raise

    # See the note on process_audio above (#4605).
    except HTTPException:
        raise


@with_error_handling("get job status")
async def get_job_status(
    job_id: str,
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> JobStatusResponse:
    """Get the status of a processing job"""
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    job = await engine.get_job(job_id)
    if not job:
        raise NotFoundError("Job")

    return _job_status_response(job)


@with_error_handling("download job result")
async def download_result(
    job_id: str,
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> FileResponse:
    """
    Download the processed audio file.
    Only available when job status is 'completed'.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    job = await engine.get_job(job_id)
    if not job:
        raise NotFoundError("Job")

    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status.value})")

    # A job that reached COMPLETED without output_path set is an
    # engine-layer bug, not a client error — but Path(None) raises an
    # unhandled TypeError rather than a typed response (#4736).
    if not job.output_path:
        logger.error(f"Job {job_id} is COMPLETED but has no output_path set")
        raise HTTPException(status_code=500, detail="Job completed but produced no output file")

    output_path = Path(job.output_path).resolve()

    # Validate output path is within the expected temp directory (#2561)
    allowed_output_base = Path(tempfile.gettempdir()).resolve()
    try:
        output_path.relative_to(allowed_output_base)
    except ValueError:
        logger.error(f"Job {job_id} output path outside expected directory: {output_path}")
        raise HTTPException(status_code=500, detail="Output path configuration error")

    if not output_path.exists():
        raise NotFoundError("Output file")

    # Determine media type based on file extension
    media_types = {
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
    }
    media_type = media_types.get(output_path.suffix, "application/octet-stream")

    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=f"auralis_processed{output_path.suffix}"
    )


@with_error_handling("cancel job")
async def cancel_job(
    job_id: str,
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> dict[str, Any]:
    """Cancel a queued or processing job"""
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    success = await engine.cancel_job(job_id)
    if not success:
        # Check if job exists to provide correct error
        job = await engine.get_job(job_id)
        if not job:
            raise NotFoundError("Job")
        raise HTTPException(status_code=400, detail="Job cannot be cancelled (already completed)")

    return {"message": "Job cancelled successfully", "job_id": job_id}


@with_error_handling("list jobs")
async def list_jobs(
    status: ProcessingStatus | None = None,
    limit: int = Query(50, ge=1, le=1000),
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> JobListResponse:
    """List all processing jobs, optionally filtered by status.

    `status` is the enum rather than `str` + a hand-rolled check (#3896):
    FastAPI now rejects an unknown value at the boundary with 422, matching
    how `limit` already behaves, and OpenAPI documents the valid values.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    jobs = engine.get_all_jobs()

    # Filter by status if provided. FastAPI has already coerced and
    # validated the value, so no manual membership check is needed.
    if status:
        jobs = [j for j in jobs if j.status == status]

    total = len(jobs)
    limited_jobs = jobs[:limit]

    return JobListResponse(
        jobs=[_job_status_response(job) for job in limited_jobs],
        total=total,
    )


@with_error_handling("get queue status")
async def get_queue_status(
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> dict[str, Any]:
    """Get current processing queue status"""
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    return engine.get_queue_status()


def _preset_to_payload(profile: PresetProfile) -> dict[str, Any]:
    """Project a `PresetProfile` onto the public presets payload.

    Units are the profile's own: EQ gains and compressor/limiter thresholds in
    dB, attack/release in ms, ratio as N:1, blends and intensities in 0.0-1.0.
    """
    return {
        "name": profile.name,
        "description": profile.description,
        # Every profile is applied through the adaptive mastering path;
        # PresetProfile has no mode of its own.
        "mode": "adaptive",
        "settings": {
            "eq": {
                "enabled": profile.eq_blend > 0.0,
                "blend": profile.eq_blend,
                "low": profile.low_shelf_gain,
                "low_mid": profile.low_mid_gain,
                "mid": profile.mid_gain,
                "high_mid": profile.high_mid_gain,
                "high": profile.high_shelf_gain,
            },
            "dynamics": {
                "enabled": profile.dynamics_blend > 0.0,
                "blend": profile.dynamics_blend,
                "compressor": {
                    "threshold": profile.compression_threshold,
                    "ratio": profile.compression_ratio,
                    "attack": profile.compression_attack,
                    "release": profile.compression_release,
                },
                "limiter": {
                    "threshold": profile.limiter_threshold,
                    "release": profile.limiter_release,
                },
            },
            "level_matching": {
                "enabled": True,
                "target_lufs": profile.target_lufs,
                "peak_target_db": profile.peak_target_db,
            },
        },
    }


async def get_processing_presets() -> dict[str, Any]:
    """Get available processing presets.

    #5220: this used to return a hand-typed dict of 5 presets whose EQ and
    compressor numbers were invented — unitless integers that matched nothing
    the mastering engine applies — and which silently omitted the engine's
    6th preset, "live". The catalog is now projected from
    `create_preset_profiles()`, the same source `HybridProcessor` masters
    with, so the two cannot drift apart again.

    Note the split with #4861: that issue tracks "live" being unreachable
    through `schemas.VALID_PRESETS` / `EnhancementPresetLiteral` and the
    frontend's `ENHANCEMENT_PRESETS`, which this endpoint does not feed and
    this change does not touch.
    """
    return {
        "presets": {
            name: _preset_to_payload(profile)
            for name, profile in create_preset_profiles().items()
        }
    }


async def get_processing_parameters(
    get_enhancement_settings: Callable[[], dict[str, Any]] | None = Depends(
        _get_enhancement_settings
    ),
) -> dict[str, Any]:
    """
    Get current processing parameters from the continuous space system.
    This shows what the auto-mastering engine is doing in real-time.

    Reads from the global content profile cache populated by ChunkedAudioProcessor
    during streaming playback.

    Moved here from routers/enhancement.py (#5073) so GET /api/processing/parameters
    is gated behind HAS_PROCESSING like its 8 siblings under this router, instead
    of being unconditionally registered while the rest of the /api/processing
    namespace can be absent in a degraded build.

    Returns:
        dict: Processing parameters including coordinates, targets, and adjustments
    """
    if get_enhancement_settings is None:
        raise HTTPException(status_code=503, detail="Enhancement settings not available")

    try:
        from core.chunked_processor import get_last_content_profile

        # Get current preset
        preset = get_enhancement_settings().get("preset", "adaptive")

        # Try to get profile from ChunkedAudioProcessor global cache
        profile = get_last_content_profile(preset)

        if profile is None:
            # No processing data yet - return default values.
            # #3779: include `is_default: True` so clients can
            # distinguish "no data yet" from "real measurements
            # that coincidentally landed at the default values".
            logger.debug(f"No processing profile found for preset '{preset}' - returning defaults")
            return {
                "is_default": True,
                "spectral_balance": 0.5,
                "dynamic_range": 0.5,
                "energy_level": 0.5,
                "target_lufs": -14.0,
                "peak_target_db": -1.0,
                "bass_boost": 0.0,
                "air_boost": 0.0,
                "compression_amount": 0.0,
                "expansion_amount": 0.0,
                "stereo_width": 0.75
            }

        # Extract coordinates (ProcessingCoordinates dataclass or dict)
        coords = profile.get('coordinates')
        params = profile.get('parameters')

        if coords is None or params is None:
            # Legacy mode or no continuous space data.
            # #3779: same is_default marker as the no-profile branch.
            logger.debug(f"Profile for preset '{preset}' missing coordinates or parameters")
            return {
                "is_default": True,
                "spectral_balance": 0.5,
                "dynamic_range": 0.5,
                "energy_level": 0.5,
                "target_lufs": -14.0,
                "peak_target_db": -1.0,
                "bass_boost": 0.0,
                "air_boost": 0.0,
                "compression_amount": 0.0,
                "expansion_amount": 0.0,
                "stereo_width": 0.75
            }

        # Extract values (handle both dataclass and dict formats)
        def get_attr(obj: Any, attr: str, default: Any = 0.0) -> Any:
            """Get attribute from dataclass or dict"""
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        # Convert ProcessingCoordinates and ProcessingParameters to dict
        # #3779: `is_default: False` confirms these are measured values
        # from a live ChunkedAudioProcessor profile, not the defaults.
        result = {
            "is_default": False,
            "spectral_balance": get_attr(coords, 'spectral_balance', 0.5),
            "dynamic_range": get_attr(coords, 'dynamic_range', 0.5),
            "energy_level": get_attr(coords, 'energy_level', 0.5),
            "target_lufs": get_attr(params, 'target_lufs', -14.0),
            "peak_target_db": get_attr(params, 'peak_target_db', -1.0),
            "bass_boost": get_attr(params, 'eq_curve', {}).get('low_shelf_gain', 0.0),
            "air_boost": get_attr(params, 'eq_curve', {}).get('high_shelf_gain', 0.0),
            "compression_amount": get_attr(params, 'compression_params', {}).get('amount', 0.0),
            "expansion_amount": get_attr(params, 'expansion_params', {}).get('amount', 0.0),
            "stereo_width": get_attr(params, 'stereo_width_target', 0.75)
        }

        logger.debug(f"📊 Returning processing parameters for preset '{preset}': {result}")
        return result

    except HTTPException:
        raise
    except Exception:
        # Don't silently mask the failure as 'all-systems-nominal' —
        # surface it as 500 so operators see a real error count
        # increment and clients can decide whether to retry (#3562 /
        # BE-NEW-104). The legitimate empty-profile case still falls
        # through the normal 200-OK path above.
        logger.exception("Failed to get processing parameters")
        raise HTTPException(
            status_code=500,
            detail="Failed to get processing parameters",
        )


async def cleanup_old_jobs(
    max_age_hours: float = Query(24, gt=0),
    engine: ProcessingEngine | None = Depends(_get_processing_engine),
) -> dict[str, Any]:
    """Clean up completed jobs older than specified hours"""
    if not engine:
        raise HTTPException(status_code=503, detail="Processing engine not available")

    removed_count = await engine.cleanup_old_jobs(max_age_hours)

    return {
        "message": f"Cleaned up jobs older than {max_age_hours} hours",
        "removed": removed_count
    }


def create_processing_router(
    get_processing_engine: Callable[[], ProcessingEngine | None],
    get_enhancement_settings: Callable[[], dict[str, Any]] | None = None,
) -> APIRouter:
    """
    Factory that assembles the processing router from the module-level
    handlers above, wiring the engine getter in through Depends() rather than
    a closure (#4670). It keeps the factory shape every other router uses,
    which is what replaced the module-level mutable + ``set_processing_engine``
    setter this router started out with (fixes #3862 / BE-RH-11).

    Args:
        get_processing_engine: Callable returning the live ``ProcessingEngine``
            instance, or ``None`` if not yet initialised.
        get_enhancement_settings: Callable returning the live enhancement
            settings dict, used only by GET /parameters to resolve the active
            preset. Optional so existing callers/tests that don't touch that
            route need no changes; that route 503s if omitted (#5073).

    Returns:
        Configured ``APIRouter`` for ``/api/processing``.
    """
    deps = _ProcessingDeps(get_processing_engine, get_enhancement_settings)
    # Keep the module-level holder pointing at the most recent call, so
    # anything resolving outside a request still sees a wired-up getter.
    _deps.get_processing_engine = deps.get_processing_engine
    _deps.get_enhancement_settings = deps.get_enhancement_settings

    router = APIRouter(
        prefix="/api/processing",
        tags=["audio-processing"],
        dependencies=[Depends(_make_deps_binder(deps))],
    )

    router.add_api_route("/process", process_audio, methods=["POST"], response_model=ProcessResponse)
    router.add_api_route("/upload-and-process", upload_and_process, methods=["POST"], response_model=ProcessResponse)
    router.add_api_route("/job/{job_id}", get_job_status, methods=["GET"], response_model=JobStatusResponse)
    router.add_api_route("/job/{job_id}/download", download_result, methods=["GET"])
    router.add_api_route("/job/{job_id}/cancel", cancel_job, methods=["POST"], response_model=CancelJobResponse)
    router.add_api_route("/jobs", list_jobs, methods=["GET"], response_model=JobListResponse)
    router.add_api_route("/queue/status", get_queue_status, methods=["GET"], response_model=QueueStatusResponse)
    router.add_api_route("/presets", get_processing_presets, methods=["GET"], response_model=PresetsResponse)
    router.add_api_route("/parameters", get_processing_parameters, methods=["GET"], response_model=ProcessingParametersResponse)
    router.add_api_route("/jobs/cleanup", cleanup_old_jobs, methods=["DELETE"], response_model=CleanupResponse)

    return router
