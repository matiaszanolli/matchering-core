"""
Metadata Router
~~~~~~~~~~~~~~~

Handles track metadata editing operations.

Endpoints:
- GET /api/metadata/tracks/{track_id}/fields - Get editable fields for a track
- GET /api/metadata/tracks/{track_id} - Get current metadata for a track
- PUT /api/metadata/tracks/{track_id} - Update metadata for a track
- POST /api/metadata/batch - Batch update metadata for multiple tracks

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from schemas import TrackId
from security.path_security import PathValidationError, validate_file_path
from websocket.outbound_messages import broadcast_typed

from auralis.library.metadata_editor import MetadataEditor, MetadataUpdate

from .dependencies import require_repository_factory, with_error_handling
from .errors import NotFoundError

logger = logging.getLogger(__name__)
# Note: router is created inside create_metadata_router() for better testability


class MetadataUpdateRequest(BaseModel):
    """Request model for metadata updates"""
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    albumartist: str | None = None
    year: int | None = None
    genre: str | None = None
    track: int | None = Field(None, alias="track_number")
    disc: int | None = Field(None, alias="disc_number")
    comment: str | None = None
    bpm: int | None = None
    composer: str | None = None
    publisher: str | None = None
    lyrics: str | None = None
    copyright: str | None = None

    model_config = {"extra": "forbid", "validate_by_name": True, "validate_by_alias": True}


class BatchMetadataUpdateRequest(BaseModel):
    """Request model for batch metadata updates

    ``metadata`` is typed as :class:`MetadataUpdateRequest` rather than a
    free-form ``dict[str, Any]`` (#4555).  The loose dict flowed unfiltered into
    a ``setattr`` loop over the Track ORM object, so any column name that
    happened to appear in the payload — ``id``, ``filepath``, ``album_id``,
    ``play_count``, ``favorite`` — was written and committed without error.
    Reusing the single-track model (which declares an explicit tag-field list
    and ``extra="forbid"``) makes the two routes accept exactly the same shape
    and rejects unknown keys with a 422.
    """
    track_id: TrackId = Field(..., description="Track ID")
    metadata: MetadataUpdateRequest = Field(..., description="Metadata fields to update")


#: Upper bound on one batch. The route writes tags to every file in the list
#: inside a single request, so an unbounded payload is an unbounded amount of
#: synchronous disk I/O (#4681).
MAX_BATCH_UPDATES = 1000


class BatchMetadataRequest(BaseModel):
    """Request model for batch update endpoint"""
    # Upper bound only. The empty case is NOT a 422: the route answers it with
    # an explicit 400 "No updates provided", pinned by
    # test_metadata_batch_atomicity.py::test_empty_updates_returns_400 and
    # test_metadata_api.py::test_batch_update_empty_list. A min_length here
    # would preempt that with a 422 and change a contract this issue never
    # asked to change.
    updates: list[BatchMetadataUpdateRequest] = Field(
        ...,
        max_length=MAX_BATCH_UPDATES,
        description="List of updates",
    )


class EditableFieldsResponse(BaseModel):
    """Editable tag fields for a track, plus the file's current tag values."""
    track_id: int = Field(description="Track database ID")
    format: str | None = Field(default=None, description="Container/codec format")
    editable_fields: list[str] = Field(
        default_factory=list,
        description="Tag names writable for this format (per TAG_MAPPINGS)",
    )
    current_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Tags currently present in the file; keys vary by format",
    )


class TrackMetadataResponse(BaseModel):
    """Current on-disk tag values for a track."""
    track_id: int = Field(description="Track database ID")
    format: str | None = Field(default=None, description="Container/codec format")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Tags read from the file; keys vary by format",
    )


class MetadataUpdateResponse(BaseModel):
    """Result of writing tags to one track's file.

    `updated_fields` lists what was written to the FILE (tag names). Fields
    with no Track DB column — artist, album, albumartist, genre, bpm,
    composer, publisher, copyright — are intentionally file-only (#4731).
    """
    track_id: int = Field(description="Track database ID")
    success: bool = Field(description="True when the file write succeeded")
    updated_fields: list[str] = Field(
        default_factory=list,
        description="Tag names written to the file",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Tags re-read from the file after the write",
    )


class BatchMetadataResultItem(BaseModel):
    """Per-track outcome within a batch metadata update."""
    track_id: int = Field(description="Track database ID")
    success: bool = Field(description="True when this track's write succeeded")
    updates: dict[str, Any] | None = Field(
        default=None,
        description="Tags written for this track (successful entries only)",
    )
    error: str | None = Field(default=None, description="Failure reason, when unsuccessful")
    rolled_back: bool | None = Field(
        default=None,
        description="True on entries reverted because a sibling write failed",
    )


class BatchMetadataResponse(BaseModel):
    """Aggregate result of a batch metadata update."""
    success: bool = Field(description="True when no track failed")
    total: int = Field(description="Number of updates attempted")
    successful: int = Field(description="Number that succeeded")
    failed: int = Field(description="Number that failed")
    results: list[BatchMetadataResultItem] = Field(
        default_factory=list,
        description="Per-track outcomes",
    )
    rolled_back: bool = Field(description="True when the whole batch was reverted")


# MetadataUpdateRequest fields whose Pydantic name is a mutagen tag name, not
# the Track DB column name (#4731). ``track``/``disc`` have an explicit
# ``alias=`` already; ``comment`` has none, since it doubles as the file-tag
# name in write_metadata's tag dict and only diverges for the DB column.
_METADATA_FIELD_TO_COLUMN: dict[str, str] = {
    'track': 'track_number',
    'disc': 'disc_number',
    'comment': 'comments',
}


def _tag_dict_to_db_columns(tag_updates: dict[str, Any]) -> dict[str, Any]:
    """Translate a mutagen-tag-keyed update dict to Track DB column names.

    ``write_metadata`` needs the tag-named keys (``track``, ``disc``,
    ``comment``) verbatim; the repository's writable-columns allowlist is
    keyed on DB column names (``track_number``, ``disc_number``,
    ``comments``), so passing the tag-named dict straight through silently
    drops those three fields (#4731). Fields with no DB column (``artist``,
    ``album``, ``genre``, ...) pass through unchanged here and are dropped
    downstream by the repository's own allowlist — they're file-only tags,
    not a translation gap.
    """
    return {_METADATA_FIELD_TO_COLUMN.get(k, k): v for k, v in tag_updates.items()}


# ============================================================================
# DEPENDENCY WIRING (#4670)
#
# create_metadata_router() used to be a ~310-line closure: every handler
# below was nested inside it purely to reach get_repository_factory /
# broadcast_manager / metadata_editor via closure capture, which made a
# handler impossible to import or call without first building the whole
# router. Handlers are now module level; they reach the same callables
# through FastAPI Depends() instead.
#
# Factory calls can coexist in one process (the live app plus throwaway test
# apps), so each router publishes its own deps through a request ContextVar.
# The module-level holder is only the fallback for direct handler calls where
# no router-level dependency ran (#5262).
#
# A handler's Depends() default is only consulted when FastAPI itself
# invokes it for a real request; a direct unit-test call passes the
# dependency explicitly as a keyword argument and never touches
# _MetadataDeps at all -- that's the seam #4670 asked for.
#
# Note the handlers still call require_repository_factory(...) themselves
# rather than receiving an already-resolved RepositoryFactory: the batch
# route deliberately answers an empty `updates` list with a 400 BEFORE it
# ever touches the factory, and resolving the factory in a Depends() would
# turn that into a 503 whenever the factory is unavailable. Injecting the
# getter keeps the original ordering verbatim.
# ============================================================================

class _MetadataDeps:
    def __init__(
        self,
        get_repository_factory: Callable[[], Any] = lambda: None,
        broadcast_manager: Any = None,
        metadata_editor: MetadataEditor | None = None,
    ) -> None:
        self.get_repository_factory = get_repository_factory
        self.broadcast_manager = broadcast_manager
        self.metadata_editor = metadata_editor


_deps = _MetadataDeps()
_active_deps: ContextVar[_MetadataDeps | None] = ContextVar(
    "_metadata_active_deps", default=None
)


def _current_deps() -> _MetadataDeps:
    return _active_deps.get() or _deps


def _make_deps_binder(deps: _MetadataDeps) -> Callable[[], Any]:
    """Publish one router instance's dependencies for its request."""

    async def _bind_deps() -> None:
        _active_deps.set(deps)

    return _bind_deps


def _get_repository_factory() -> Callable[[], Any]:
    return _current_deps().get_repository_factory


def _get_broadcast_manager() -> Any:
    return _current_deps().broadcast_manager


def _get_metadata_editor() -> MetadataEditor:
    metadata_editor = _current_deps().metadata_editor
    if metadata_editor is None:
        raise HTTPException(status_code=503, detail="Metadata editor not available")
    return metadata_editor


@with_error_handling("get editable fields")
async def get_editable_fields(
    track_id: Annotated[int, Path(ge=1)],
    get_repository_factory: Callable[[], Any] = Depends(_get_repository_factory),
    metadata_editor: MetadataEditor = Depends(_get_metadata_editor),
) -> dict[str, Any]:
    """
    Get list of editable metadata fields for a track.

    Args:
        track_id: Track ID

    Returns:
        dict: List of editable fields and their current values

    Raises:
        HTTPException: If track not found or file doesn't exist
    """
    try:
        repos = require_repository_factory(get_repository_factory)
        # Get track from database
        track = await asyncio.to_thread(repos.tracks.get_by_id, track_id)

        if not track:
            raise NotFoundError("Track")

        # Validate DB-retrieved filepath before any file I/O (fixes #2302)
        try:
            filepath_str = str(validate_file_path(str(track.filepath)))
        except PathValidationError:
            # PathValidationError's own text names the resolved path and
            # every allowed directory (#4807) -- validate_file_path already
            # logs it once via _logs_rejections; the client gets a fixed,
            # generic detail regardless of which check failed.
            raise HTTPException(status_code=400, detail="Invalid track filepath")
        editable_fields = await asyncio.to_thread(metadata_editor.get_editable_fields, filepath_str)

        # Get current metadata (file I/O — run in thread)
        current_metadata = await asyncio.to_thread(metadata_editor.read_metadata, filepath_str)

        return {
            "track_id": track_id,
            "format": track.format,
            "editable_fields": editable_fields,
            "current_metadata": current_metadata
        }

    except HTTPException:
        raise  # Re-raise HTTPException as-is (don't wrap in 500)
    except FileNotFoundError:
        raise NotFoundError("Audio file", detail=f"Audio file not found for track {track_id}")


@with_error_handling("get track metadata")
async def get_track_metadata(
    track_id: Annotated[int, Path(ge=1)],
    get_repository_factory: Callable[[], Any] = Depends(_get_repository_factory),
    metadata_editor: MetadataEditor = Depends(_get_metadata_editor),
) -> dict[str, Any]:
    """
    Get current metadata for a track.

    Args:
        track_id: Track ID

    Returns:
        dict: Track metadata

    Raises:
        HTTPException: If track not found or file doesn't exist
    """
    try:
        repos = require_repository_factory(get_repository_factory)
        # Get track from database
        track = await asyncio.to_thread(repos.tracks.get_by_id, track_id)

        if not track:
            raise NotFoundError("Track")

        # Validate DB-retrieved filepath before file I/O (fixes #2302)
        try:
            filepath_validated = str(validate_file_path(str(track.filepath)))
        except PathValidationError:
            # PathValidationError's own text names the resolved path and
            # every allowed directory (#4807) -- validate_file_path already
            # logs it once via _logs_rejections; the client gets a fixed,
            # generic detail regardless of which check failed.
            raise HTTPException(status_code=400, detail="Invalid track filepath")

        # Read metadata from file (offloaded to thread to avoid event-loop block, fixes #2317)
        metadata = await asyncio.to_thread(metadata_editor.read_metadata, filepath_validated)

        return {
            "track_id": track_id,
            "format": track.format,
            "metadata": metadata
        }

    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except FileNotFoundError:
        raise NotFoundError("Audio file", detail=f"Audio file not found for track {track_id}")


@with_error_handling("update track metadata")
async def update_track_metadata(
    track_id: Annotated[int, Path(ge=1)],
    request: MetadataUpdateRequest,
    get_repository_factory: Callable[[], Any] = Depends(_get_repository_factory),
    metadata_editor: MetadataEditor = Depends(_get_metadata_editor),
    broadcast_manager: Any = Depends(_get_broadcast_manager),
) -> dict[str, Any]:
    """
    Update metadata for a track.

    Args:
        track_id: Track ID
        request: Metadata fields to update

    Returns:
        dict: Updated track metadata

    Raises:
        HTTPException: If track not found, file doesn't exist, or update fails
    """
    try:
        repos = require_repository_factory(get_repository_factory)
        # Get track from database
        track = await asyncio.to_thread(repos.tracks.get_by_id, track_id)

        if not track:
            raise NotFoundError("Track")

        # Convert request to dict and filter out None values
        metadata_updates = {
            k: v for k, v in request.model_dump().items()
            if v is not None
        }

        if not metadata_updates:
            raise HTTPException(status_code=400, detail="No metadata fields provided")

        # Validate DB-retrieved filepath before any file I/O (fixes #2302)
        try:
            filepath_validated = str(validate_file_path(str(track.filepath)))
        except PathValidationError:
            # PathValidationError's own text names the resolved path and
            # every allowed directory (#4807) -- validate_file_path already
            # logs it once via _logs_rejections; the client gets a fixed,
            # generic detail regardless of which check failed.
            raise HTTPException(status_code=400, detail="Invalid track filepath")

        # Write metadata to file (backup always enforced server-side, fixes #2407).
        # Offloaded to thread to avoid blocking the event loop (fixes #2317).
        success = await asyncio.to_thread(
            metadata_editor.write_metadata,
            filepath_validated,
            metadata_updates,
            True  # backup=True
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to write metadata to file")

        # Update database record using repository. metadata_updates is
        # tag-keyed (matches write_metadata above); translate to DB
        # column names first or track/disc/comment silently vanish (#4731).
        db_metadata_updates = _tag_dict_to_db_columns(metadata_updates)
        updated_track = await asyncio.to_thread(
            lambda: repos.tracks.update_metadata(track_id, **db_metadata_updates)
        )
        if not updated_track:
            raise NotFoundError("Track", detail="Track not found for update")

        # Use the updated track for subsequent operations
        track = updated_track

        # Broadcast metadata updated event. updated_fields lists what was
        # written to the FILE (tag names); fields with no Track DB
        # column (artist, album, albumartist, genre, bpm, composer,
        # publisher, copyright) are intentionally file-only and don't
        # appear in db_metadata_updates above (#4731).
        if broadcast_manager:
            await broadcast_typed(
                broadcast_manager,
                "metadata_updated",
                {
                    "track_id": track_id,
                    "updated_fields": list(metadata_updates.keys()),
                },
            )

        # Read updated metadata (offloaded to thread, fixes #2317)
        # track was refreshed from DB above — re-validate before read (fixes #2302)
        validated_path_for_read = str(validate_file_path(str(track.filepath)))
        updated_metadata = await asyncio.to_thread(metadata_editor.read_metadata, validated_path_for_read)

        logger.info(f"Updated metadata for track {track_id}: {list(metadata_updates.keys())}")

        return {
            "track_id": track_id,
            "success": True,
            "updated_fields": list(metadata_updates.keys()),
            "metadata": updated_metadata
        }

    except HTTPException:
        raise
    except FileNotFoundError:
        raise NotFoundError("Audio file", detail=f"Audio file not found for track {track_id}")
    except ValueError as e:
        # Invalid metadata error
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")


@with_error_handling("batch update metadata")
async def batch_update_metadata(
    request: BatchMetadataRequest,
    get_repository_factory: Callable[[], Any] = Depends(_get_repository_factory),
    metadata_editor: MetadataEditor = Depends(_get_metadata_editor),
    broadcast_manager: Any = Depends(_get_broadcast_manager),
) -> dict[str, Any]:
    """
    Batch update metadata for multiple tracks.

    Args:
        request: List of metadata updates

    Returns:
        dict: Batch update results (success/failure per track)

    Raises:
        HTTPException: If library manager/factory not available or validation fails
    """
    if not request.updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    repos = require_repository_factory(get_repository_factory)

    # Fetch all requested tracks in one WHERE-IN query (fixes #3857 N+1).
    all_track_ids = [u.track_id for u in request.updates]
    track_map: dict[int, Any] = await asyncio.to_thread(
        repos.tracks.get_by_ids, all_track_ids
    )

    # Prepare batch updates — validate filepath for each found track.
    batch_updates = []
    for update_req in request.updates:
        track = track_map.get(update_req.track_id)
        if not track:
            logger.warning(f"Track {update_req.track_id} not found, skipping")
            continue

        # Drop unset fields, mirroring the single-track route (#4555).
        # model_dump() emits field names (track/disc), not the
        # track_number/disc_number aliases, so both routes hand the
        # metadata editor an identically-shaped tag dict.
        metadata_fields = {
            k: v for k, v in update_req.metadata.model_dump().items()
            if v is not None
        }
        if not metadata_fields:
            logger.warning(
                f"No metadata fields for track {update_req.track_id}, skipping"
            )
            continue

        # Validate DB-retrieved filepath before file I/O (fixes #2302)
        try:
            validated_filepath = str(validate_file_path(
                str(track.filepath),
                context=f"track {update_req.track_id}, batch update — skipping",
            ))
        except PathValidationError:
            # validate_file_path logs it once with the context above (#4925).
            continue

        batch_updates.append(MetadataUpdate(
            track_id=update_req.track_id,
            filepath=validated_filepath,
            updates=metadata_fields,
            backup=True  # always enforced server-side (fixes #2407)
        ))

    # Execute batch update (atomic when backup=True).
    # Offloaded to thread to avoid blocking the event loop (fixes #2317).
    results = await asyncio.to_thread(metadata_editor.batch_update, batch_updates)
    rolled_back: bool = results.get('rolled_back', False)

    # Update database for all successful results in one transaction
    # (fixes #3857 N+1 on the update side).
    successful_track_ids = []

    if not rolled_back:
        db_updates: list[tuple[int, dict[str, Any]]] = []
        for result in results.get('results', []):
            if result.get('success'):
                track_id = result['track_id']
                # updates is tag-keyed (echoed from the MetadataUpdate we
                # built above) — translate before the DB write, mirroring
                # the single-track route's fix (#4731).
                updates = _tag_dict_to_db_columns(result.get('updates', {}))
                if updates:
                    db_updates.append((track_id, updates))

        if db_updates:
            successful_track_ids = await asyncio.to_thread(
                repos.tracks.update_metadata_batch, db_updates
            )

    # Broadcast batch update event
    if broadcast_manager and successful_track_ids:
        await broadcast_typed(
            broadcast_manager,
            "metadata_batch_updated",
            {
                "track_ids": successful_track_ids,
                "count": len(successful_track_ids),
            },
        )

    logger.info(
        f"Batch metadata update: {results['successful']}/{results['total']} successful"
        + (", rolled back" if rolled_back else "")
    )

    return {
        "success": results['failed'] == 0,
        "total": results['total'],
        "successful": results['successful'],
        "failed": results['failed'],
        "results": results['results'],
        "rolled_back": rolled_back,
    }


def create_metadata_router(
    get_repository_factory: Callable[[], Any],
    broadcast_manager: Any,
    metadata_editor: MetadataEditor | None = None
) -> APIRouter:
    """
    Factory function to create metadata router with dependencies.

    Args:
        get_repository_factory: Callable that returns RepositoryFactory instance
        broadcast_manager: WebSocket broadcast manager
        metadata_editor: Optional MetadataEditor instance (for testing)

    Returns:
        APIRouter: Configured router instance

    Note:
        Phase 6B: Fully migrated to RepositoryFactory pattern (no LibraryManager fallback).
    """
    # Initialize metadata editor (shared instance) or use provided one
    if metadata_editor is None:
        metadata_editor = MetadataEditor()

    global _deps

    deps = _MetadataDeps(get_repository_factory, broadcast_manager, metadata_editor)
    _deps = deps

    # Create a fresh router instance (important for testing - avoids route pollution)
    router = APIRouter(
        tags=["metadata"],
        dependencies=[Depends(_make_deps_binder(deps))],
    )

    router.add_api_route("/api/metadata/tracks/{track_id}/fields", get_editable_fields, methods=["GET"], response_model=EditableFieldsResponse)
    router.add_api_route("/api/metadata/tracks/{track_id}", get_track_metadata, methods=["GET"], response_model=TrackMetadataResponse)
    router.add_api_route("/api/metadata/tracks/{track_id}", update_track_metadata, methods=["PUT"], response_model=MetadataUpdateResponse)
    router.add_api_route("/api/metadata/batch", batch_update_metadata, methods=["POST"], response_model=BatchMetadataResponse)

    return router
