"""
Artwork Router
~~~~~~~~~~~~~~

Handles album artwork operations: retrieval, extraction, and deletion.

Endpoints:
- GET /api/albums/{album_id}/artwork - Get album artwork file
- POST /api/albums/{album_id}/artwork/extract - Extract artwork from tracks
- DELETE /api/albums/{album_id}/artwork - Delete album artwork

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import logging
import mimetypes
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal

from core.thumbnail_cache import (
    THUMB_TMP_PREFIX,
    artwork_cache_dirs,
    prune_thumbnail_cache,
    purge_thumbnails,
    reap_orphan_temp_files,
    thumb_path_hash,
)
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from websocket.outbound_messages import ArtworkUpdatedPayload, broadcast_typed

from .dependencies import require_repository_factory, with_error_handling
from .errors import NotFoundError

logger = logging.getLogger(__name__)


# NOTE: GET /api/albums/{album_id}/artwork has no response_model on purpose —
# it returns raw image bytes (FileResponse/Response), not JSON. Same for the
# other binary routes in this codebase; only the JSON routes below are typed.


class ArtworkExtractResponse(BaseModel):
    """Result of extracting embedded artwork from an album's tracks."""
    message: str = Field(description="Human-readable confirmation")
    artwork_url: str = Field(description="Artwork API URL, never a filesystem path (#2508)")
    album_id: int = Field(description="Album that was updated")


class ArtworkDeleteResponse(BaseModel):
    """Result of deleting album artwork.

    Idempotent per RFC 7231 §4.3.5: a repeat call after a successful delete
    still returns 200 (#3563).
    """
    message: str = Field(description="Human-readable confirmation")
    album_id: int = Field(description="Album that was updated")


class ArtworkDownloadResponse(BaseModel):
    """Result of downloading album artwork from an online source."""
    message: str = Field(description="Human-readable confirmation")
    artwork_url: str = Field(description="Artwork API URL, never a filesystem path")
    album_id: int = Field(description="Album that was updated")
    artist: str = Field(description="Artist name used for the lookup")
    album: str | None = Field(default=None, description="Album name used for the lookup")


# Downscaled-thumbnail support (#4447). Requested sizes snap UP to one of these
# buckets so the on-disk cache holds at most len(_THUMB_BUCKETS) variants per
# source image instead of an unbounded set of arbitrary sizes.
_THUMB_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024)


def _bucket_size(size: int) -> int:
    """Snap a requested max-dimension up to the nearest cache bucket."""
    for bucket in _THUMB_BUCKETS:
        if size <= bucket:
            return bucket
    return _THUMB_BUCKETS[-1]


# Prefix for in-progress thumbnail renders (#4527). Owned by core.thumbnail_cache
# (which reaps orphans by it, #4532) and re-exported here for the existing
# callers and tests that read it from this module.
_THUMB_TMP_PREFIX = THUMB_TMP_PREFIX

# Per-cache-key render locks (#4527). Two requests for the same album at the
# same bucket used to render concurrently; now the second waits and takes the
# first's result, which also caps the peak memory of a grid scroll. Keyed on the
# full cache key, NOT the album id — a per-album lock would serialise unrelated
# buckets against each other.
_THUMB_LOCKS: dict[str, threading.Lock] = {}
_THUMB_WAITERS: dict[str, int] = {}
_THUMB_GUARD = threading.Lock()


@contextmanager
def _thumb_render_lock(key: str) -> Iterator[None]:
    """Hold the render lock for one cache key, and retire it when idle.

    The waiter count is what makes retiring safe: dropping a lock another
    thread is still blocked on would let a third thread create a second lock
    for the same key and render concurrently again. Without retiring, the dict
    would instead grow one entry per (album, bucket, artwork generation) for the
    life of the process.
    """
    with _THUMB_GUARD:
        lock = _THUMB_LOCKS.setdefault(key, threading.Lock())
        _THUMB_WAITERS[key] = _THUMB_WAITERS.get(key, 0) + 1
    try:
        with lock:
            yield
    finally:
        with _THUMB_GUARD:
            remaining = _THUMB_WAITERS[key] - 1
            if remaining:
                _THUMB_WAITERS[key] = remaining
            else:
                del _THUMB_WAITERS[key]
                _THUMB_LOCKS.pop(key, None)


def _render_thumbnail(
    src: Path, dst: Path, bucket: int, pil_fmt: str, ext: str, thumb_dir: Path
) -> None:
    """Render ``src`` into ``dst`` via a temp file unique to this writer (#4527).

    The previous temp name was derived only from the cache key
    (``dst.suffix + ".tmp"``), so N threads rendering the same album at the same
    bucket interleaved their bytes into one file and each then promoted whatever
    it happened to contain. ``Path.replace()`` is atomic within a filesystem, so
    a per-writer temp is all that is needed for correctness — the lock above is
    an efficiency measure, not the fix.

    Raises on failure, having removed its own temp file; the caller converts
    that to ``None``.
    """
    # Imported here rather than at module scope to preserve the original lazy
    # import: this router is constructed at startup and PIL is not cheap.
    from PIL import Image

    fd, tmp_name = tempfile.mkstemp(
        dir=thumb_dir, prefix=_THUMB_TMP_PREFIX, suffix=ext
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with Image.open(src) as image:
                # thumbnail() preserves aspect ratio and only ever downsizes, so
                # a small source is served as-is rather than upscaled.
                image.thumbnail((bucket, bucket))
                # Rebound to a new name: convert() returns a plain Image while
                # `image` is typed as the ImageFile the context manager yielded,
                # so reassigning it is an incompatible-assignment for mypy.
                out_image = (
                    image.convert("RGB")
                    if pil_fmt == "JPEG" and image.mode not in ("RGB", "L")
                    else image
                )
                out_image.save(handle, format=pil_fmt)
        tmp.replace(dst)
    except BaseException:
        # Leave no orphan behind — a generation-based cache purge keys on the
        # source path hash and would never match a stray temp file.
        tmp.unlink(missing_ok=True)
        raise


def _artwork_dirs() -> tuple[Path, Path]:
    """``(artwork_dir, thumb_dir)`` — one definition for every caller.

    The render path and the purge path MUST agree on where the cache lives, so
    both read it from here rather than each rebuilding the path.
    """
    return artwork_cache_dirs()


def _purge_album_thumbnails(*sources: Path | str | None) -> int:
    """Purge cached thumbnails derived from ``sources``, and reap stale temps.

    Blocking (filesystem) — call via ``asyncio.to_thread``.

    Resolves each source before hashing, because that is what the render path
    hashes: ``get_album_artwork`` passes
    ``Path(album.artwork_path).resolve(strict=False)`` into the cache. Hashing
    the unresolved ``album.artwork_path`` instead would produce a different
    prefix and silently purge nothing whenever the stored path is relative or
    crosses a symlink (#4532).
    """
    _artwork_dir, thumb_dir = _artwork_dirs()

    resolved: list[Path] = []
    for source in sources:
        if not source:
            continue
        try:
            resolved.append(Path(source).resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            logger.warning("Could not resolve artwork path for purge: %s", source)

    return purge_thumbnails(thumb_dir, *resolved) + reap_orphan_temp_files(thumb_dir)


def _thumb_target(media_type: str) -> tuple[str, str, str]:
    """Map a source media type to (PIL format, file extension, response type).

    JPEG stays JPEG; WEBP stays WEBP; everything else (PNG/GIF/unknown) is
    rendered as PNG so transparency is preserved.
    """
    if media_type == "image/jpeg":
        return "JPEG", ".jpg", "image/jpeg"
    if media_type == "image/webp":
        return "WEBP", ".webp", "image/webp"
    return "PNG", ".png", "image/png"


def _get_or_create_thumbnail(
    src: Path, requested_size: int, media_type: str, thumb_dir: Path
) -> tuple[Path, str] | None:
    """Return (thumbnail_path, media_type) for a downscaled copy of ``src``.

    Blocking (PIL + disk IO) — call via ``asyncio.to_thread``. The cache key
    includes the source path hash, bucketed size, and source mtime/size so an
    artwork edit produces a new file and stale thumbnails are never served.
    Returns ``None`` on any failure so the caller can fall back to the original.
    """
    try:
        bucket = _bucket_size(requested_size)
        pil_fmt, ext, resp_type = _thumb_target(media_type)

        stat = src.stat()
        # thumb_path_hash is shared with the purge in core.thumbnail_cache, which
        # globs on this exact prefix — computing it here independently would let
        # the two drift and silently strand every entry (#4532).
        path_hash = thumb_path_hash(src)
        key = f"{path_hash}_{bucket}_{stat.st_mtime_ns:x}_{stat.st_size:x}{ext}"
        dst = thumb_dir / key

        if not dst.exists():
            thumb_dir.mkdir(parents=True, exist_ok=True)
            with _thumb_render_lock(key):
                # Re-check under the lock: whoever held it before us may have
                # rendered this exact key already.
                if not dst.exists():
                    # Sweep dead writers' temp files before adding our own, so
                    # orphans cannot accumulate indefinitely on an install that
                    # never deletes artwork (#4532). Cheap: one glob over a
                    # directory whose size is bounded by live thumbnails, and
                    # only on an actual cache miss.
                    reap_orphan_temp_files(thumb_dir)
                    _render_thumbnail(src, dst, bucket, pil_fmt, ext, thumb_dir)
                    # Never evict the file this request is about to serve. If
                    # one thumbnail somehow exceeds the whole cap, keep that
                    # response valid and reclaim it on a later miss instead.
                    prune_thumbnail_cache(thumb_dir, keep=dst)

        return dst, resp_type
    except Exception:
        logger.exception("Thumbnail generation failed for %s", src)
        return None


# ============================================================================
# DEPENDENCY WIRING (#4670)
#
# create_artwork_router() used to be a 330-line closure: every handler below
# was nested inside it purely to reach connection_manager/get_repository_factory
# via closure capture, which made a handler impossible to import or call
# without first building the whole router. Handlers are now module level; they
# reach the same callables through FastAPI Depends() instead.
#
# Factory calls can coexist in one process (the live app plus throwaway test
# apps), so each router publishes its own deps through a request ContextVar.
# The module-level holder is only the fallback for direct handler calls where
# no router-level dependency ran (#5262).
#
# A handler's Depends() default is only consulted when FastAPI itself
# invokes it for a real request; a direct unit-test call passes the
# service/dependency explicitly as a keyword argument and never touches
# _ArtworkDeps at all -- that's the seam #4670 asked for.
# ============================================================================

class _ArtworkDeps:
    def __init__(
        self,
        connection_manager: Any = None,
        get_repository_factory: Callable[[], Any] = lambda: None,
    ) -> None:
        self.connection_manager = connection_manager
        self.get_repository_factory = get_repository_factory


_deps = _ArtworkDeps()
_active_deps: ContextVar[_ArtworkDeps | None] = ContextVar(
    "_artwork_active_deps", default=None
)


def _current_deps() -> _ArtworkDeps:
    return _active_deps.get() or _deps


def _make_deps_binder(deps: _ArtworkDeps) -> Callable[[], Any]:
    """Publish one router instance's dependencies for its request."""

    async def _bind_deps() -> None:
        _active_deps.set(deps)

    return _bind_deps


def _get_connection_manager() -> Any:
    return _current_deps().connection_manager


async def _broadcast_artwork_updated(
    connection_manager: Any,
    action: Literal["extracted", "downloaded", "deleted"],
    album_id: int,
    artwork_url: str | None = None,
) -> None:
    """Broadcast `artwork_updated` from the one place all three emit sites
    share, so the payload cannot drift out of sync with the frontend's
    ``ArtworkUpdatedMessage`` contract (types/ws/enhancement.ts:71-78, #4676).
    ``artwork_url`` is omitted for 'deleted', matching that contract's
    `artwork_url?: string; // absent for 'deleted'`.
    """
    data: ArtworkUpdatedPayload = {"action": action, "album_id": album_id}
    if artwork_url is not None:
        data["artwork_url"] = artwork_url
    await broadcast_typed(connection_manager, "artwork_updated", data)


def _get_repos() -> Any:
    """Get repository factory for accessing repositories."""
    return require_repository_factory(_current_deps().get_repository_factory)


@with_error_handling("get artwork")
async def get_album_artwork(
    album_id: Annotated[int, PathParam(ge=1)],
    request: Request,
    size: int | None = Query(
        None,
        ge=16,
        le=2048,
        description=(
            "Optional max dimension (px) for a downscaled thumbnail. Snaps "
            "up to a cache bucket; omit for the full-resolution image (#4447)."
        ),
    ),
    repos: Any = Depends(_get_repos),
) -> Response:
    """
    Get album artwork file (with path traversal protection).

    Args:
        album_id: Album ID
        size: Optional max dimension for a downscaled thumbnail variant.

    Returns:
        FileResponse: Artwork image file (or a size-appropriate thumbnail).

    Raises:
        HTTPException: If library manager/factory not available, album/artwork not found,
                     or path validation fails
    """
    # Get album to find artwork path
    album = await asyncio.to_thread(repos.albums.get_by_id, album_id)

    if not album:
        raise NotFoundError("Album")

    if not album.artwork_path:
        raise NotFoundError("Artwork")

    # Security: Validate artwork path is within allowed directory
    # Define allowed artwork directory (shared with the purge path, #4532)
    artwork_dir, thumb_dir = _artwork_dirs()
    # Off the event loop (#4653's CONSISTENCY sweep): this is inside async def.
    await asyncio.to_thread(artwork_dir.mkdir, parents=True, exist_ok=True)

    # Resolve allowed directory (handles symlinks in base path)
    allowed_dir = artwork_dir.resolve()

    # Resolve artwork path (handles symlinks and relative paths)
    # Use strict=False to resolve path even if file doesn't exist (for security validation)
    try:
        requested_path = Path(album.artwork_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Invalid artwork path for album {album_id}: {album.artwork_path} - {e}")
        raise HTTPException(status_code=403, detail="Access denied: invalid path")

    # Security: Check that resolved path is within allowed directory
    # This MUST happen before existence check to prevent path traversal
    # Use is_relative_to() for safe path comparison (prevents traversal attacks)
    if not requested_path.is_relative_to(allowed_dir):
        logger.warning(
            f"Path traversal attempt blocked for album {album_id}: "
            f"requested={requested_path}, allowed_dir={allowed_dir}"
        )
        raise HTTPException(status_code=403, detail="Access denied: path outside artwork directory")

    # Additional check: file must exist (after security validation)
    if not requested_path.exists():
        raise NotFoundError("Artwork")

    # Detect MIME type from file extension first, then fall back to magic bytes
    # so that PNG files with unrecognized/missing extensions are not served
    # as image/jpeg (fixes #2510).
    media_type, _ = mimetypes.guess_type(str(requested_path))
    if not media_type or not media_type.startswith("image/"):
        # Read the first 12 bytes to identify the format via magic bytes
        def _read_header() -> bytes:
            try:
                with open(requested_path, "rb") as _f:
                    return _f.read(12)
            except OSError:
                return b""
        header = await asyncio.to_thread(_read_header)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            media_type = "image/png"
        elif header[:3] == b"\xff\xd8\xff":
            media_type = "image/jpeg"
        elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            media_type = "image/webp"
        elif header[:4] in (b"GIF8", b"GIF9"):
            media_type = "image/gif"
        else:
            media_type = "image/jpeg"  # safest fallback for browsers

    # If a thumbnail size was requested, serve a downscaled variant instead
    # of the full-resolution bitmap (#4447). On any failure we fall back to
    # the original, so a broken/unsupported image never 500s the request.
    serve_path = requested_path
    serve_media_type = media_type
    if size is not None:
        thumbnail = await asyncio.to_thread(
            _get_or_create_thumbnail, requested_path, size, media_type, thumb_dir
        )
        if thumbnail is not None:
            serve_path, serve_media_type = thumbnail

    # Build ETag from the SERVED file's stat for conditional caching (#2864).
    stat = serve_path.stat()
    etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'

    # If client already has this version, return 304 (no body).
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(
            status_code=304,
            headers={
                "ETag": etag,
                "Cache-Control": "public, no-cache",
            },
        )

    # Return artwork file with ETag for conditional caching.
    # no-cache = always revalidate, but 304 avoids re-download
    # when content hasn't changed.
    return FileResponse(
        str(serve_path),
        media_type=serve_media_type,
        headers={
            "ETag": etag,
            "Cache-Control": "public, no-cache",
        },
    )


@with_error_handling("extract artwork")
async def extract_album_artwork(
    album_id: Annotated[int, PathParam(ge=1)],
    repos: Any = Depends(_get_repos),
    connection_manager: Any = Depends(_get_connection_manager),
) -> dict[str, Any]:
    """
    Extract artwork from album tracks.

    Extracts embedded artwork from the album's audio files and saves it.

    Args:
        album_id: Album ID

    Returns:
        dict: Success message and artwork URL

    Raises:
        HTTPException: If library manager/factory not available or extraction fails
    """
    # The superseded generation's thumbnails are keyed on the OLD source
    # path, which the extract below overwrites, so read it first (#4532).
    previous = await asyncio.to_thread(repos.albums.get_by_id, album_id)
    previous_path = previous.artwork_path if previous else None

    artwork_path = await asyncio.to_thread(repos.albums.extract_and_save_artwork, album_id)

    if not artwork_path:
        raise HTTPException(
            status_code=404,
            detail="No artwork found in album tracks"
        )

    # Purge both paths: the old one covers a move to a new file, the new one
    # covers an in-place overwrite (same path, fresh mtime — the old key
    # still sits in the cache). Thumbnails render lazily, so nothing for the
    # new generation exists yet and this cannot delete a live entry.
    await asyncio.to_thread(_purge_album_thumbnails, previous_path, artwork_path)

    # Convert filesystem path to API URL
    artwork_url = f"/api/albums/{album_id}/artwork"

    # Broadcast artwork updated event
    await _broadcast_artwork_updated(connection_manager, "extracted", album_id, artwork_url)

    return {
        "message": "Artwork extracted successfully",
        "artwork_url": artwork_url,  # API URL — consistent with artist serializer (fixes #2508)
        "album_id": album_id
    }


@with_error_handling("delete artwork")
async def delete_album_artwork(
    album_id: Annotated[int, PathParam(ge=1)],
    repos: Any = Depends(_get_repos),
    connection_manager: Any = Depends(_get_connection_manager),
) -> dict[str, Any]:
    """
    Delete album artwork.

    Args:
        album_id: Album ID

    Returns:
        dict: Success message

    Raises:
        HTTPException: If library manager/factory not available or artwork not found
    """
    # Idempotent DELETE per RFC 7231 §4.3.5 — a repeat call after a
    # successful delete should NOT 404 (#3563 / BE-NEW-105). Only
    # 404 when the album itself doesn't exist; if artwork is
    # already gone, return success.
    album = await asyncio.to_thread(repos.albums.get_by_id, album_id)
    if album is None:
        raise NotFoundError("Album", album_id)
    # Read the source path BEFORE the row is cleared — it is the only way
    # back to the derived thumbnails, and delete_artwork discards it (#4532).
    source_path = album.artwork_path
    success = await asyncio.to_thread(repos.albums.delete_artwork, album_id)
    # If repo returns False the artwork was already absent — also
    # success from the client's idempotency perspective.

    # Drop the derived thumbnails. Runs regardless of `success` so a
    # half-finished earlier delete (row gone, cache left) still gets cleaned
    # up, and after the DB write so a purge failure cannot fail the request —
    # purge_thumbnails swallows its own OSErrors for that reason.
    await asyncio.to_thread(_purge_album_thumbnails, source_path)

    # Broadcast artwork updated event (only when something actually changed)
    if success:
        await _broadcast_artwork_updated(connection_manager, "deleted", album_id)

    return {"message": "Artwork deleted successfully", "album_id": album_id}


@with_error_handling("download artwork")
async def download_album_artwork(
    album_id: Annotated[int, PathParam(ge=1)],
    repos: Any = Depends(_get_repos),
    connection_manager: Any = Depends(_get_connection_manager),
) -> dict[str, Any]:
    """
    Download album artwork from online sources.

    Automatically searches and downloads artwork from MusicBrainz and iTunes.

    Args:
        album_id: Album ID

    Returns:
        dict: Success message and artwork path

    Raises:
        HTTPException: If library manager/factory not available or download fails
    """
    # Get album using repository (includes eager loading of artist)
    album = await asyncio.to_thread(repos.albums.get_by_id, album_id)

    if not album:
        raise NotFoundError("Album")

    # Get artist name (from first track if available)
    artist_name = album.artist.name if album.artist else "Unknown Artist"
    album_name = album.title

    # Download artwork using the artwork downloader service
    from services.artwork_downloader import get_artwork_downloader
    downloader = get_artwork_downloader()

    artwork_path = await downloader.download_artwork(
        artist=artist_name,
        album=album_name,
        album_id=album_id
    )

    if not artwork_path:
        raise HTTPException(
            status_code=404,
            detail=f"No artwork found online for '{album_name}' by '{artist_name}'"
        )

    # Captured before update_artwork_path replaces it, so the superseded
    # generation's thumbnails can still be located (#4532).
    previous_path = album.artwork_path

    # Save artwork path to database
    updated_album = await asyncio.to_thread(repos.albums.update_artwork_path, album_id, artwork_path)
    if not updated_album:
        raise NotFoundError("Album")

    # Same both-paths purge as the extract route above.
    await asyncio.to_thread(_purge_album_thumbnails, previous_path, artwork_path)

    # Convert filesystem path to API URL
    artwork_url = f"/api/albums/{album_id}/artwork"

    # Broadcast artwork updated event
    await _broadcast_artwork_updated(connection_manager, "downloaded", album_id, artwork_url)

    return {
        "message": "Artwork downloaded successfully",
        "artwork_url": artwork_url,  # API URL, not filesystem path
        "album_id": album_id,
        "artist": artist_name,
        "album": album_name
    }


def create_artwork_router(
    connection_manager: Any,
    get_repository_factory: Callable[[], Any]
) -> APIRouter:
    """
    Factory function to create artwork router with dependencies.

    Args:
        connection_manager: WebSocket connection manager for broadcasts
        get_repository_factory: Callable that returns RepositoryFactory instance

    Returns:
        APIRouter: Configured router instance

    Note:
        Phase 6B: Fully migrated to RepositoryFactory pattern (no LibraryManager fallback)
    """
    global _deps

    deps = _ArtworkDeps(connection_manager, get_repository_factory)
    _deps = deps

    router = APIRouter(
        tags=["artwork"],
        dependencies=[Depends(_make_deps_binder(deps))],
    )

    # GET stays first: it is the only route on the bare `/artwork` path, and the
    # `/artwork/extract` + `/artwork/download` literals below must keep their
    # original relative order for path matching to be unchanged.
    router.add_api_route("/api/albums/{album_id}/artwork", get_album_artwork, methods=["GET"])
    router.add_api_route(
        "/api/albums/{album_id}/artwork/extract", extract_album_artwork,
        methods=["POST"], response_model=ArtworkExtractResponse,
    )
    router.add_api_route(
        "/api/albums/{album_id}/artwork", delete_album_artwork,
        methods=["DELETE"], response_model=ArtworkDeleteResponse,
    )
    router.add_api_route(
        "/api/albums/{album_id}/artwork/download", download_album_artwork,
        methods=["POST"], response_model=ArtworkDownloadResponse,
    )

    return router
