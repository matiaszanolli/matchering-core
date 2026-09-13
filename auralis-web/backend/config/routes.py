"""
Router Configuration and Registration

Imports all router factories and registers them with the FastAPI application.
Handles dependency injection for each router via lambdas.

:copyright: (C) 2024 Auralis Team
:license: GPLv3
"""

import logging
from typing import Any
from collections.abc import Callable

from fastapi import APIRouter, FastAPI
from routers.albums import create_albums_router
from routers.artists import create_artists_router
from routers.artwork import create_artwork_router
from routers.enhancement import create_enhancement_router
from routers.files import create_files_router
from routers.library import create_library_router
from routers.tracks import create_tracks_router
from routers.library_scan import create_library_scan_router
from routers.fingerprint_status import create_fingerprint_status_router
from routers.metadata import create_metadata_router
from routers.player import create_player_router
from routers.playlists import create_playlists_router
from routers.settings import create_settings_router

# Import router factories
from routers.system import create_system_router
from routers.health import create_health_router

# NOTE: cache_streamlined and similarity router factories are imported locally
# inside their own try/except blocks below (matching processing_api), not at
# module level — their `include_router()` calls are
# already guarded so a broken transitive dependency degrades gracefully
# (fixes #2324), but a module-level import here would raise before that
# protection is ever reached, hard-crashing startup anyway (#3907).

logger = logging.getLogger(__name__)


def setup_routers(app: FastAPI, deps: dict[str, Any]) -> None:
    """
    Register all routers with FastAPI application.

    Args:
        app: FastAPI application instance
        deps: Dictionary of dependencies:
            - HAS_AURALIS: bool
            - HAS_PROCESSING: bool
            - HAS_STREAMLINED_CACHE: bool
            - HAS_SIMILARITY: bool
            - manager: ConnectionManager
            - enhancement_settings: dict
            - chunked_audio_processor_class: class
            - create_track_info_fn: callable
            - globals: Dict with component instances
    """

    HAS_PROCESSING: bool = deps.get('HAS_PROCESSING', False)
    HAS_STREAMLINED_CACHE: bool = deps.get('HAS_STREAMLINED_CACHE', False)
    HAS_SIMILARITY: bool = deps.get('HAS_SIMILARITY', False)
    manager: Any = deps.get('manager')
    enhancement_settings: dict[str, Any] = deps.get('enhancement_settings', {})
    chunked_audio_processor_class: Any = deps.get('chunked_audio_processor_class')
    create_track_info_fn: Any = deps.get('create_track_info_fn')
    globals_dict: dict[str, Any] = deps.get('globals', {})

    # Helper to safely get global components
    def get_component(key: str) -> Callable[[], Any]:
        return lambda: globals_dict.get(key)

    # Include processing API routes (if available)
    if HAS_PROCESSING:
        try:
            from routers.processing_api import create_processing_router
            processing_router = create_processing_router(
                get_component('processing_engine'),
                get_enhancement_settings=lambda: enhancement_settings,
            )
            app.include_router(processing_router)
            logger.debug("✅ Processing API router included")
        except Exception as e:
            # Catch all exceptions (not just ImportError) so syntax errors or
            # missing transitive deps degrade gracefully rather than crashing
            # startup (fixes #2324).
            logger.warning(f"⚠️  Processing API router not available: {e}", exc_info=True)

    # Health and version routes (extracted from system router in #4074)
    health_router: APIRouter = create_health_router(
        HAS_AURALIS=deps.get('HAS_AURALIS', False),
        get_library_database=get_component('library_database'),
    )
    app.include_router(health_router)
    logger.debug("✅ Health router registered")

    # WebSocket system router
    # Issue #2740: Pass get_state_manager so reconnecting WebSocket clients
    # receive a full player state snapshot immediately on connect.
    system_router: APIRouter = create_system_router(
        manager=manager,
        get_processing_engine=get_component('processing_engine'),
        HAS_AURALIS=deps.get('HAS_AURALIS', False),
        get_repository_factory=get_component('repository_factory'),
        get_enhancement_settings=lambda: enhancement_settings,
        get_state_manager=get_component('player_state_manager'),
        # Pass the process-wide StreamlinedCacheManager so AudioStreamController
        # reuses the shared chunk cache across requests (fixes #3855).
        get_cache_manager=get_component('streamlined_cache'),
    )
    app.include_router(system_router)
    logger.debug("✅ System router registered")

    # Create and include settings router (GET/PUT /api/settings, scan-folders management)
    # #4587: get_enhancement_settings/connection_manager let a settings save
    # re-seed the live enhancement_settings dict and broadcast the change,
    # instead of only taking effect at the next backend restart.
    settings_router: APIRouter = create_settings_router(
        get_settings_repo=get_component('settings_repository'),
        get_auto_scanner=get_component('auto_scanner'),
        get_enhancement_settings=lambda: enhancement_settings,
        connection_manager=manager,
    )
    app.include_router(settings_router)
    logger.debug("✅ Settings router registered")

    # Create and include files router (scan, upload, formats)
    files_router: APIRouter = create_files_router(
        get_repository_factory=get_component('repository_factory')
    )
    app.include_router(files_router)
    logger.debug("✅ Files router registered (Phase 2 RepositoryFactory enabled)")

    # Create and include enhancement router
    enhancement_router: APIRouter = create_enhancement_router(
        get_enhancement_settings=lambda: enhancement_settings,
        connection_manager=manager,
        get_multi_tier_buffer=lambda: globals_dict.get('streamlined_cache') if HAS_STREAMLINED_CACHE else None,
        get_player_state_manager=get_component('player_state_manager'),
        get_processing_engine=lambda: globals_dict.get('processing_engine') if HAS_PROCESSING else None,
        get_repository_factory=get_component('repository_factory'),
    )
    app.include_router(enhancement_router)
    logger.debug("✅ Enhancement router registered")

    # Create and include artwork router (with Phase 6B RepositoryFactory refactoring)
    artwork_router: APIRouter = create_artwork_router(
        connection_manager=manager,
        get_repository_factory=get_component('repository_factory')
    )
    app.include_router(artwork_router)
    logger.debug("✅ Artwork router registered (Phase 2 RepositoryFactory enabled)")

    # Create and include playlists router (with Phase 2 RepositoryFactory support)
    playlists_router: APIRouter = create_playlists_router(
        get_repository_factory=get_component('repository_factory'),
        connection_manager=manager
    )
    app.include_router(playlists_router)
    logger.debug("✅ Playlists router registered (Phase 2 RepositoryFactory enabled)")

    # Create and include library router (stats, browse, reset — Phase 6B)
    library_router: APIRouter = create_library_router(
        get_repository_factory=get_component('repository_factory'),
        get_cache_manager=get_component('streamlined_cache'),
        # Reset pauses/restarts all background workers (#4111). resolve_worker
        # looks workers up by the shared BACKGROUND_WORKER_KEYS in the
        # component registry.
        resolve_worker=lambda key: globals_dict.get(key),
    )
    app.include_router(library_router)
    logger.debug("✅ Library router registered (stats/browse/reset)")

    # Track-domain routes (listing, favorites, lyrics)
    tracks_router: APIRouter = create_tracks_router(
        get_repository_factory=get_component('repository_factory'),
    )
    app.include_router(tracks_router)
    logger.debug("✅ Tracks router registered")

    # Scan route with async progress broadcast
    library_scan_router: APIRouter = create_library_scan_router(
        get_library_database=get_component('library_database'),
        connection_manager=manager,
    )
    app.include_router(library_scan_router)
    logger.debug("✅ Library scan router registered")

    # Fingerprint status routes
    fingerprint_status_router: APIRouter = create_fingerprint_status_router(
        get_repository_factory=get_component('repository_factory'),
    )
    app.include_router(fingerprint_status_router)
    logger.debug("✅ Fingerprint status router registered")

    # Create and include metadata router (with Phase 6B RepositoryFactory refactoring)
    metadata_router: APIRouter = create_metadata_router(
        get_repository_factory=get_component('repository_factory'),
        broadcast_manager=manager
    )
    app.include_router(metadata_router)
    logger.debug("✅ Metadata router registered (Phase 2 RepositoryFactory enabled)")

    # Create and include albums router (with Phase 6B RepositoryFactory refactoring)
    albums_router: APIRouter = create_albums_router(
        get_repository_factory=get_component('repository_factory')
    )
    app.include_router(albums_router)
    logger.debug("✅ Albums router registered (Phase 2 RepositoryFactory enabled)")

    # Create and include artists router (with Phase 6B RepositoryFactory refactoring)
    artists_router: APIRouter = create_artists_router(
        get_repository_factory=get_component('repository_factory')
    )
    app.include_router(artists_router)
    logger.debug("✅ Artists router registered (Phase 2 RepositoryFactory enabled)")

    # Create and include player router
    player_router: APIRouter = create_player_router(
        get_library_database=get_component('library_database'),
        get_audio_player=get_component('audio_player'),
        get_player_state_manager=get_component('player_state_manager'),
        connection_manager=manager,
        chunked_audio_processor_class=chunked_audio_processor_class,
        create_track_info_fn=create_track_info_fn,
        get_multi_tier_buffer=lambda: globals_dict.get('streamlined_cache') if HAS_STREAMLINED_CACHE else None,
        get_enhancement_settings=lambda: enhancement_settings
    )
    app.include_router(player_router)
    logger.debug("✅ Player router registered")

    # Include cache management router (if available).
    # Register unconditionally when the module is importable; handlers
    # return 503 until the cache manager is initialised during lifespan
    # (fixes #2756 — router was never registered because globals_dict
    # was still empty at setup_routers() time).
    if HAS_STREAMLINED_CACHE:
        try:
            from routers.cache_streamlined import create_streamlined_cache_router
            cache_router: APIRouter = create_streamlined_cache_router(
                get_cache_manager=lambda: globals_dict.get('streamlined_cache'),
                broadcast_manager=manager
            )
            app.include_router(cache_router)
            logger.info("✅ Streamlined cache router registered")
        except Exception as e:
            logger.warning(f"⚠️  Failed to register streamlined cache router: {e}", exc_info=True)

    # Create and include the similarity router family (if available). Split
    # into three domain routers (#4270): similarity search, graph management,
    # and fingerprint-queue admin. All three keep the /api/similarity prefix so
    # existing route paths are unchanged.
    if HAS_SIMILARITY:
        try:
            from routers.similarity import create_similarity_router
            from routers.similarity_graph import create_similarity_graph_router
            from routers.fingerprint_queue import create_fingerprint_queue_router

            app.include_router(create_similarity_router(
                get_similarity_system=get_component('similarity_system'),
                get_graph_builder=get_component('graph_builder'),
                get_repository_factory=get_component('repository_factory')
            ))
            app.include_router(create_similarity_graph_router(
                get_graph_builder=get_component('graph_builder')
            ))
            app.include_router(create_fingerprint_queue_router(
                get_repository_factory=get_component('repository_factory')
            ))
            logger.info("✅ Similarity router family registered (search, graph, fingerprint-queue)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to register similarity router family: {e}", exc_info=True)

    # NOTE: the WAV streaming / MSE REST router (routers/wav_streaming.py,
    # /api/stream/*) was retired (#4435). Live playback is entirely
    # WebSocket-based (play_normal / play_enhanced); the REST/MSE chunk surface
    # had no production frontend caller — only MSW test mocks — consistent with
    # the app not using MSE. Retired rather than kept as dead cross-layer surface.

    logger.info("✅ All routers configured and registered")
