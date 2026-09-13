#!/usr/bin/env python3

"""
Auralis Web Backend

Modern FastAPI backend for Auralis audio processing and library management.
Replaces the Tkinter GUI with a professional web interface.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette import status
from starlette.routing import Match

# NOTE: logging.basicConfig was removed (#3537 / BE-NEW-79). uvicorn.run()
# installs its own logging configuration with handlers on the root logger,
# and basicConfig added a second handler so every line emitted by a
# `logger = logging.getLogger(__name__)` propagated to both — duplicate log
# lines in stdout and in the Electron-captured log file. Letting Uvicorn
# own the root logger config eliminates the duplication.
logger = logging.getLogger(__name__)

# Add parent directory to path for Auralis imports
# Detect execution context and set appropriate path
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as PyInstaller bundle (Electron AppImage)
    # Use working directory to find resources (cwd is set to resources/backend by Electron)
    auralis_parent = Path(os.getcwd()).parent
    logger.info("Running as PyInstaller bundle")
    # Absolute path (embeds OS username + install layout) kept at DEBUG only;
    # INFO stays free of it so it's safe to paste into a public bug report
    # (#4366 / #4778).
    logger.debug(f"Adding to sys.path: {auralis_parent}")
    # Append (not insert) so that _MEIPASS — which PyInstaller puts at the front — takes
    # priority over the external resources/auralis/ copy, preventing a stale copy from
    # shadowing the auralis version that was bundled into this executable.
    sys.path.append(str(auralis_parent))
elif os.environ.get('ELECTRON_MODE') == '1':
    # Running in Electron but not frozen (shouldn't happen in production)
    auralis_parent = Path(__file__).parent.parent
    logger.info("Running in Electron mode (unfrozen)")
    logger.debug(f"Adding to sys.path: {auralis_parent}")
    sys.path.insert(0, str(auralis_parent))
else:
    # Running in development - auralis package is in ../../..
    auralis_parent = Path(__file__).parent.parent.parent
    logger.info("Running in development mode")
    logger.debug(f"Adding to sys.path: {auralis_parent}")
    sys.path.insert(0, str(auralis_parent))

# Import configuration modules
from config.app import create_app, is_dev_mode as _is_dev_mode
from config.globals import ConnectionManager, set_component_registry
from config.middleware import setup_middleware
from config.routes import setup_routers
from config.startup import create_lifespan
from core.env_config import get_int_env

# Import state management
from player_state import create_track_info

# Check feature availability via real import probes (fixes #3534 /
# BE-NEW-76 — prior code had empty try: pass blocks that could never
# flip the flag, so HAS_STREAMLINED_CACHE / HAS_SIMILARITY were
# hard-coded True regardless of actual import success).
HAS_AURALIS = True
HAS_PROCESSING = True
HAS_STREAMLINED_CACHE = True
HAS_SIMILARITY = True

# #4684: HAS_AURALIS was the one flag #3534 left as a bare literal, so the
# demo-mode branch it gates in startup could never run. Probe it like the
# others. Note this only answers "is the engine importable"; whether it came up
# is a runtime question /api/health now answers from the live component.
try:
    import auralis as _auralis_probe  # noqa: F401
except ImportError:
    HAS_AURALIS = False
    logger.warning("⚠️  Auralis engine not available")

# Import core components for router setup
ProcessingEngine: Any = None
ChunkedAudioProcessor: Any = None
try:
    from core.chunked_processor import ChunkedAudioProcessor
except ImportError:
    HAS_PROCESSING = False
    logger.warning("⚠️  Processing components not available")

try:
    import cache as _cache_probe  # noqa: F401
except ImportError:
    HAS_STREAMLINED_CACHE = False
    logger.warning("⚠️  Streamlined cache not available")

try:
    from auralis.analysis.fingerprint import FingerprintSimilarity as _similarity_probe  # noqa: F401
except ImportError:
    HAS_SIMILARITY = False
    logger.warning("⚠️  Similarity system not available")

# Create global state dictionary with all dependencies
manager = ConnectionManager()
globals_dict = {
    # Components (initialized during startup)
    'library_database': None,
    'repository_factory': None,  # Phase 2: RepositoryFactory for DI
    'settings_repository': None,
    'audio_player': None,
    'player_state_manager': None,
    'processing_engine': None,
    'streamlined_cache': None,
    'streamlined_worker': None,
    'similarity_system': None,
    'graph_builder': None,
    # Configuration
    'enhancement_settings': {
        "enabled": True,
        "preset": "adaptive",
        "intensity": 1.0
    },
}

# Register this dict as THE process-wide component registry (#4578). Modules
# that cannot be passed dependencies explicitly — notably
# ChunkedAudioProcessor's Tier-1 fingerprint accessor — resolve components
# through config.globals.get_component_registry(). Startup mutates this same
# object in place, so those readers observe components as they are populated.
set_component_registry(globals_dict)

# Prepare dependencies dictionary for startup and routers
deps = {
    'HAS_AURALIS': HAS_AURALIS,
    'HAS_PROCESSING': HAS_PROCESSING,
    'HAS_STREAMLINED_CACHE': HAS_STREAMLINED_CACHE,
    'HAS_SIMILARITY': HAS_SIMILARITY,
    'manager': manager,
    'globals': globals_dict,
    'enhancement_settings': globals_dict['enhancement_settings'],
    'chunked_audio_processor_class': ChunkedAudioProcessor,
    'create_track_info_fn': create_track_info,
}

# Create lifespan context manager for startup/shutdown (populates globals_dict)
lifespan = create_lifespan(deps)

# Create FastAPI application with lifespan
app = create_app(lifespan=lifespan)

# Setup middleware
setup_middleware(app)

# Setup routers (registers all routes with app)
setup_routers(app, deps)

# Frontend static file serving
if os.environ.get('ELECTRON_MODE') == '1':
    # Running inside Electron (frozen binary or script)
    # Frontend is in resources/frontend alongside resources/backend
    frontend_path = Path(os.getcwd()).parent / "frontend"
    if not frontend_path.exists():
        # Fallback: resolve relative to this script
        frontend_path = Path(__file__).resolve().parent.parent / "frontend"
    logger.info("Electron mode: looking for frontend")
elif hasattr(sys, 'frozen') and hasattr(sys, '_MEIPASS'):
    # PyInstaller bundle but not Electron - frontend might be bundled
    meipass = getattr(sys, '_MEIPASS')
    frontend_path = Path(meipass) / "frontend"
    logger.info("PyInstaller mode: frontend bundled with _MEIPASS")
else:
    # Development mode - look in regular location
    frontend_path = Path(__file__).parent.parent / "frontend" / "dist"

# Absolute path (embeds OS username + install layout) kept at DEBUG only;
# INFO stays free of it so it's safe to paste into a public bug report (#4366).
logger.debug(f"Looking for frontend at: {frontend_path}")

# Body for the `/` fallback when the built frontend is missing.
#
# Deliberately free of `frontend_path` (#4351). The old version interpolated it
# — HTML-escaped, so never an XSS vector, but it handed any client that can
# reach `/` the install layout: the PyInstaller _MEIPASS temp dir, or
# /home/<user>/... which also discloses the OS username. A stray browser tab or
# a DNS-rebound page could read it.
#
# The path is still available for diagnostics from the logger.debug calls, which
# is where #4366 put absolute paths so INFO-level output stays safe to paste
# into a public bug report. Kept a module-level constant rather than inlined so
# the branch is reachable from a test: which `/` handler gets registered is
# decided at import time by conditions a test cannot easily force.
FRONTEND_MISSING_HTML = """
        <html>
            <head><title>Auralis Web</title></head>
            <body>
                <h1>🎵 Auralis Web Backend</h1>
                <p>FastAPI backend is running!</p>
                <p>Frontend assets not found — the installation looks incomplete.
                   Reinstall Auralis, or run the backend with --dev and serve the
                   frontend with Vite.</p>
                <p><a href="/api/docs">View API Documentation</a></p>
            </body>
        </html>
        """

# Catch-all for unregistered WebSocket paths (#4800). Starlette's Mount
# matches websocket scopes just like http ones (it only special-cases the
# path, not scope["type"]), so without this a client that upgrades to any
# unregistered /ws* path in production falls through the single real /ws
# route (registered above, via setup_routers) all the way to the
# StaticFiles Mount below, whose __call__ asserts scope["type"] == "http"
# and raises an unhandled AssertionError instead of a clean close.
# Registered here — after setup_routers's real /ws route, before the Mount
# — so /ws itself still matches first (Starlette checks routes in
# registration order) and every other path gets a clean policy-violation
# close instead.
@app.websocket("/{path:path}")
async def _unregistered_websocket(websocket: WebSocket) -> None:
    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)


# Named so _methods_allowed_for can skip them. Both match *every* path, so
# leaving them in would report each path as allowing every method: the
# catch-all matches its own request, and the SPA Mount is what shadowed the 405
# in the first place.
_CATCH_ALL_ROUTE_NAMES = {"_unmatched_api_path", "frontend"}

_PROBE_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def _methods_allowed_for(request: Request) -> set[str]:
    """Methods some real route accepts for this request's path.

    Empty when no route matches the path under any method — a genuine 404
    rather than a method mismatch.

    Probes each method through Starlette's own matcher rather than reading
    `route.methods`: `app.include_router()` leaves each router as a single
    nested `_IncludedRouter` entry in `app.routes` (children are not flattened
    up), and those expose no `.methods` — so a direct attribute read finds no
    API routes at all and every path looks like a 404. Asking `matches()` for a
    FULL match per candidate method delegates to the same resolution the real
    request would use, at any nesting depth.
    """
    allowed: set[str] = set()
    probe = dict(request.scope)
    for method in _PROBE_METHODS:
        probe["method"] = method
        for route in request.app.routes:
            if getattr(route, "name", None) in _CATCH_ALL_ROUTE_NAMES:
                continue
            if route.matches(probe)[0] is Match.FULL:
                allowed.add(method)
                break
    return allowed


# Catch-all for unmatched /api paths (#5090). Same placement rationale as the
# WebSocket catch-all above: after setup_routers's real routes, before the
# StaticFiles Mount.
#
# Starlette returns 405 for a method mismatch only when NO route fully matches
# the request. The SPA Mount at "/" below fully matches *every* path, so once it
# is registered it beats the partial (path-matched, method-mismatched) match on
# every real API route — `GET /api/files/upload` reached StaticFiles, found no
# such file on disk, and became a 404 instead of the 405 the route shape
# implies. That also meant every unknown /api path returned StaticFiles' error
# rather than the app's JSON error shape.
#
# Because the mount only exists in production with a built frontend, and
# `auralis-web/frontend/dist/` is gitignored and never built by
# `backend-tests.yml`, this silently made behaviour differ between CI (no
# mount, 405) and a developer machine that had run a frontend build (mount,
# 404) — the same environment-dependence that makes worktree baselines report
# phantom diffs. Handling /api explicitly here makes the response identical in
# both, and keeps API errors in the API's own format.
@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def _unmatched_api_path(request: Request, path: str) -> JSONResponse:
    allowed = _methods_allowed_for(request)
    if allowed:
        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={"detail": "Method Not Allowed"},
            headers={"Allow": ", ".join(sorted(allowed))},
        )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "Not Found"},
    )


# Only mount static files in production (when not running --dev)
# In development, Vite serves the frontend and proxies API requests
# StaticFiles mount at "/" interferes with WebSocket routes, so we must avoid it in dev mode
is_dev_mode = _is_dev_mode()

if not is_dev_mode and frontend_path.exists():
    logger.info("✅ Serving frontend (production mode)")
    logger.debug(f"Serving frontend from: {frontend_path}")
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
elif is_dev_mode:
    logger.info("ℹ️  Development mode: Vite serves frontend, StaticFiles NOT mounted (preserves WebSocket routes)")

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse("""
        <html>
            <head><title>Auralis Web</title></head>
            <body>
                <h1>🎵 Auralis Web Backend</h1>
                <p>Backend API is running!</p>
                <p>Frontend served by Vite on http://localhost:3000+</p>
                <p><a href="/api/docs">View API Documentation</a></p>
            </body>
        </html>
        """)
else:
    logger.warning("⚠️  Frontend not found — check installation")
    logger.debug(f"Frontend not found at: {frontend_path}")

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(FRONTEND_MISSING_HTML)


if __name__ == "__main__":
    # AURALIS_PORT (#4805): launch-auralis-web.py accepts --port and threads
    # it through via this same env var, mirroring AURALIS_DEV_MODE two lines
    # below. Without this, --port silently no-op'd -- the launcher printed
    # the requested port while the backend always bound 8765.
    _port = get_int_env("AURALIS_PORT", 8765)
    print(f"🚀 Starting Auralis Web Backend on port {_port}...", flush=True)

    uvicorn.run(
        app,  # Pass app directly instead of "main:app" to avoid module duplication
        host="127.0.0.1",
        port=_port,
        log_level="info"
    )
