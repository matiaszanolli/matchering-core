"""
Settings Router
~~~~~~~~~~~~~~~

REST API for user settings management.

Endpoints:
- GET  /api/settings                    — get current settings
- PUT  /api/settings                    — update settings
- POST /api/settings/scan-folders       — add a scan folder
- POST /api/settings/scan-folders/delete — remove a scan folder
- POST /api/settings/reset              — reset to defaults

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from helpers import seed_enhancement_settings
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator
from schemas import (  # shared preset enum (#4424) / intensity constraint (#4600)
    VALID_PRESETS,
    EnhancementIntensity,
    EnhancementPresetLiteral,
)
from security.path_security import (
    PathValidationError,
    clear_extra_allowed_directories,
    register_allowed_directory,
    unregister_allowed_directory,
    validate_directory_list,
    validate_user_chosen_directory,
)
from websocket.outbound_messages import broadcast_typed

from .dependencies import with_error_handling
from .errors import NotFoundError

logger = logging.getLogger(__name__)

# Fields that also live in the runtime `enhancement_settings` dict (#4409),
# under different names (see seed_enhancement_settings). A PUT that touches
# any of these must re-seed the live dict and broadcast — otherwise a
# Settings-dialog save only takes effect at the next backend restart (#4587).
_ENHANCEMENT_SETTINGS_FIELDS = frozenset({
    'default_preset', 'auto_enhance', 'enhancement_intensity',
})


class _ScanFolderRequest(BaseModel):
    folder: str


class SettingsUpdateRequest(BaseModel):
    """Typed, validated body for ``PUT /api/settings`` (#3837 / BE-SCH-2).

    Every field is optional so the endpoint remains a partial update. ``extra='forbid'``
    turns a misspelled field name into a 422 instead of a silent no-op (the
    SettingsRepository whitelist would otherwise drop unknown keys without complaint).
    Field set mirrors ``SettingsRepository.update_settings`` whitelist plus the
    separately-handled ``scan_folders`` (JSON) and ``file_types`` (CSV) columns.
    """

    model_config = ConfigDict(extra="forbid")

    # Library
    scan_folders: list[str] | None = None
    file_types: list[str] | None = None
    auto_scan: bool | None = None
    scan_interval: int | None = Field(default=None, ge=0)
    # Playback
    crossfade_enabled: bool | None = None
    crossfade_duration: float | None = Field(default=None, ge=0)
    gapless_enabled: bool | None = None
    replay_gain_enabled: bool | None = None
    # 0.0-1.0, NOT the 0-100 scale used by SetVolumeRequest and PlayerState
    # (#4711). Two different scales share the field name across this API; the
    # persisted UserSettings.volume column is normalised (default 0.8), so the
    # scale is stated here rather than renamed — a rename would need a column
    # migration and a stored 0.8 misread as 0-100 is near-silent.
    volume: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Default playback volume, normalised 0.0-1.0. Note this is "
                    "NOT the 0-100 scale used by POST /api/player/volume.",
    )
    # Audio output
    output_device: str | None = None
    bit_depth: int | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    # Interface
    theme: str | None = None
    language: str | None = None
    show_visualizations: bool | None = None
    mini_player_on_close: bool | None = None
    # Enhancement
    # Constrained to the shared preset enum so an invalid value 422s at the
    # boundary instead of silently falling back to adaptive downstream (#4424).
    default_preset: EnhancementPresetLiteral | None = None
    auto_enhance: bool | None = None
    # Shared constraint (#4600) — same definition the enhancement router uses,
    # so the two cannot drift back apart.
    enhancement_intensity: EnhancementIntensity | None = None
    # Advanced
    cache_size: int | None = Field(default=None, ge=0)
    max_concurrent_scans: int | None = Field(default=None, ge=1)
    enable_analytics: bool | None = None
    debug_mode: bool | None = None


class SettingsResponse(BaseModel):
    """Response shape for settings endpoints (mirrors ``UserSettings.to_dict()``).

    ``extra='allow'`` keeps forward-compatibility with the dict the repository emits
    (e.g. ``id`` / ``created_at`` / ``updated_at``) without having to enumerate them.
    """

    model_config = ConfigDict(extra="allow")

    scan_folders: list[str] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)
    auto_scan: bool | None = None
    scan_interval: int | None = None
    crossfade_enabled: bool | None = None
    crossfade_duration: float | None = None
    gapless_enabled: bool | None = None
    replay_gain_enabled: bool | None = None
    # Same normalised 0.0-1.0 scale as SettingsUpdateRequest.volume, and
    # bounded identically (#4711) — it was unconstrained `float | None`, so
    # the scale was enforced on the way in but not on the way out.
    volume: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Default playback volume, normalised 0.0-1.0. Note this is "
                    "NOT the 0-100 scale used by POST /api/player/volume.",
    )
    output_device: str | None = None
    bit_depth: int | None = None
    sample_rate: int | None = None
    theme: str | None = None
    language: str | None = None
    show_visualizations: bool | None = None
    mini_player_on_close: bool | None = None
    # #4647: these mirror SettingsUpdateRequest's constraints, so OpenAPI
    # advertises the same closed preset enum and 0.0-1.0 intensity on the way
    # out as on the way in. The validators below keep that from turning a
    # legacy or hand-edited row into a response-validation 500 on
    # GET /api/settings: an off-list value degrades to null with a warning —
    # the same tolerance helpers.seed_enhancement_settings applies to the
    # stored preset (#4710).
    default_preset: EnhancementPresetLiteral | None = None
    auto_enhance: bool | None = None
    enhancement_intensity: EnhancementIntensity | None = None
    cache_size: int | None = None
    max_concurrent_scans: int | None = None
    enable_analytics: bool | None = None
    debug_mode: bool | None = None

    @field_validator("default_preset", mode="before")
    @classmethod
    def _degrade_unknown_preset(cls, value: object) -> object:
        if value is None or value in VALID_PRESETS:
            return value
        logger.warning(
            f"Stored default_preset {value!r} is not one of {VALID_PRESETS}; "
            "reporting it as null"
        )
        return None

    @field_validator("enhancement_intensity", mode="before")
    @classmethod
    def _degrade_invalid_intensity(cls, value: object) -> object:
        if value is None:
            return None
        try:
            as_float = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            as_float = math.nan
        if isinstance(value, bool) or not math.isfinite(as_float) or not 0.0 <= as_float <= 1.0:
            logger.warning(
                f"Stored enhancement_intensity {value!r} is outside 0.0-1.0; "
                "reporting it as null"
            )
            return None
        return as_float


class SettingsUpdateResponse(BaseModel):
    """Envelope returned by mutating settings endpoints."""

    model_config = ConfigDict(extra="forbid")

    message: str
    settings: SettingsResponse


def create_settings_router(
    get_settings_repo: Callable[[], Any],
    get_auto_scanner: Callable[[], Any] | None = None,
    get_enhancement_settings: Callable[[], dict[str, Any]] | None = None,
    connection_manager: Any = None,
) -> APIRouter:
    """
    Factory function to create the settings router.

    Args:
        get_settings_repo: Callable returning a SettingsRepository instance
        get_auto_scanner: Optional callable returning a LibraryAutoScanner (for
                          reload_config() calls after settings changes)
        get_enhancement_settings: Optional callable returning the shared,
                          shared-by-reference runtime `enhancement_settings`
                          dict (auralis-web/backend/main.py). When given, a
                          PUT/reset that touches an enhancement-related field
                          re-seeds this dict and broadcasts the change so it
                          takes effect on the live session immediately,
                          instead of only at the next backend restart (#4587)
                          — the write-back half of the sync #4409 started.
        connection_manager: Optional WebSocket connection manager used to
                          broadcast `enhancement_settings_changed` after a
                          re-seed. No-op (re-seed only, no broadcast) if not
                          given.

    Returns:
        APIRouter: Configured router instance
    """
    router = APIRouter(tags=["settings"])

    def _repo() -> Any:
        repo = get_settings_repo()
        if not repo:
            raise HTTPException(status_code=503, detail="Settings not available")
        return repo

    async def _notify_scanner() -> None:
        """Tell the auto-scanner to reload its config immediately."""
        if get_auto_scanner is None:
            return
        scanner = get_auto_scanner()
        if scanner and hasattr(scanner, 'reload_config'):
            try:
                await scanner.reload_config()
            except Exception as exc:
                logger.warning(f"Failed to notify auto-scanner of settings change: {exc}")

    async def _sync_enhancement_settings(settings: Any) -> None:
        """Re-seed the live runtime dict from just-persisted settings and
        broadcast (#4587) — the DB-to-runtime half of keeping the Settings
        dialog and the Enhancement Panel in sync. Mirrors
        config/startup.py's one-time seed at process start, using the exact
        same `seed_enhancement_settings` helper so the two can never drift
        onto different mapping/validation logic.

        `enhancement_settings` is mutated in place (seed_enhancement_settings's
        own contract) — every router holding a reference to it (system,
        enhancement) observes the new values without re-fetching anything.
        """
        if get_enhancement_settings is None:
            return
        live_settings = get_enhancement_settings()
        seed_enhancement_settings(live_settings, settings)
        if connection_manager is None:
            return
        try:
            await broadcast_typed(
                connection_manager,
                "enhancement_settings_changed",
                {
                    "enabled": live_settings["enabled"],
                    "preset": live_settings["preset"],
                    "intensity": live_settings["intensity"],
                },
            )
        except Exception as exc:
            logger.warning(f"Failed to broadcast enhancement_settings_changed: {exc}")

    @router.get("/api/settings", response_model=SettingsResponse)
    @with_error_handling("get settings")
    async def get_settings() -> dict[str, Any]:
        """Get current user settings."""
        settings = await asyncio.to_thread(_repo().get_settings)
        if not settings:
            raise NotFoundError("Settings")
        return settings.to_dict()

    @router.put("/api/settings", response_model=SettingsUpdateResponse)
    @with_error_handling("update settings")
    async def update_settings(updates: SettingsUpdateRequest) -> dict[str, Any]:
        """
        Update user settings.

        Accepts a partial, typed set of settings fields. Unknown field names are
        rejected with HTTP 422 (``extra='forbid'``) instead of silently no-op'ing
        through the SettingsRepository whitelist (#3837 / BE-SCH-2). Only fields the
        client actually sent are applied (``exclude_unset``), preserving partial-update
        semantics.

        ``scan_folders`` is the same column POST /api/settings/scan-folders writes,
        so it gets the identical path validation and allowlist-registration
        treatment here (#4765) — otherwise a folder added via this generic route
        skips both, and files under it 400 as "invalid filepath" until restart.
        """
        payload = updates.model_dump(exclude_unset=True)

        if 'scan_folders' in payload and payload['scan_folders'] is not None:
            try:
                payload['scan_folders'] = validate_directory_list(payload['scan_folders'])
            except PathValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Snapshot the previous list (if scan_folders is being written) so the
        # allowlist can be diffed against it after the write — additions get
        # registered, removals get unregistered, regardless of which route
        # made the change (fixes the #3842 "no inverse" gap for this route too).
        previous_folders: set[str] = set()
        if 'scan_folders' in payload:
            previous_settings = await asyncio.to_thread(_repo().get_settings)
            if previous_settings and previous_settings.scan_folders:
                raw = previous_settings.scan_folders
                previous_folders = set(json.loads(raw) if isinstance(raw, str) else raw)

        settings = await asyncio.to_thread(_repo().update_settings, payload)

        if 'scan_folders' in payload:
            new_folders = set(payload['scan_folders'] or [])
            for added in new_folders - previous_folders:
                register_allowed_directory(Path(added))
            for removed in previous_folders - new_folders:
                unregister_allowed_directory(Path(removed))

        await _notify_scanner()

        # #4587: only re-seed/broadcast when this save actually touched an
        # enhancement-related field — an unrelated save (e.g. theme) should
        # not fire a WS message enhancement consumers have no reason to see.
        if _ENHANCEMENT_SETTINGS_FIELDS & payload.keys():
            await _sync_enhancement_settings(settings)

        return {"message": "Settings updated", "settings": settings.to_dict()}

    @router.post("/api/settings/scan-folders", response_model=SettingsUpdateResponse)
    @with_error_handling("add scan folder")
    async def add_scan_folder(body: _ScanFolderRequest) -> dict[str, Any]:
        """Add a folder to the list of scanned directories."""
        if not body.folder or not body.folder.strip():
            raise HTTPException(status_code=400, detail="folder must be a non-empty path")
        try:
            validated = validate_user_chosen_directory(body.folder.strip())
        except PathValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        settings = await asyncio.to_thread(_repo().add_scan_folder, str(validated))
        # Register so validate_file_path accepts files under this folder
        register_allowed_directory(validated)
        await _notify_scanner()
        return {"message": f"Scan folder added: {body.folder}", "settings": settings.to_dict()}

    @router.post("/api/settings/scan-folders/delete", response_model=SettingsUpdateResponse)
    @with_error_handling("remove scan folder")
    async def remove_scan_folder(body: _ScanFolderRequest) -> dict[str, Any]:
        """Remove a folder from the list of scanned directories."""
        settings = await asyncio.to_thread(_repo().remove_scan_folder, body.folder)
        # Undo the registration made in add_scan_folder so validate_file_path()
        # stops trusting this folder for the rest of the session (fixes #3842),
        # instead of only reverting on the next backend restart.
        unregister_allowed_directory(Path(body.folder))
        await _notify_scanner()
        return {"message": f"Scan folder removed: {body.folder}", "settings": settings.to_dict()}

    @router.post("/api/settings/reset", response_model=SettingsUpdateResponse)
    @with_error_handling("reset settings")
    async def reset_settings() -> dict[str, Any]:
        """Reset all settings to their default values."""
        settings = await asyncio.to_thread(_repo().reset_to_defaults)
        # reset_to_defaults wipes the configured scan_folders list, so no
        # previously-registered extra directory should remain trusted (#3842).
        clear_extra_allowed_directories()
        await _notify_scanner()
        # #4587: reset_to_defaults() always rewrites every column (unlike the
        # partial-update PUT above), so the enhancement fields are always
        # touched here — re-seed/broadcast unconditionally rather than
        # checking payload keys, since there is no payload to check.
        await _sync_enhancement_settings(settings)
        return {"message": "Settings reset to defaults", "settings": settings.to_dict()}

    return router
