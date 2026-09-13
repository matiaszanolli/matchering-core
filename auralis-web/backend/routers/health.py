"""
Health Router
~~~~~~~~~~~~~

System health and version endpoints, extracted from the WebSocket system router
to keep infrastructure routes separate from real-time communication.

Endpoints:
- GET /api/health   - Liveness check
- GET /api/version  - Detailed version information

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter
from schemas import HealthResponse, VersionInfoResponse

logger = logging.getLogger(__name__)


def create_health_router(
    HAS_AURALIS: bool,
    get_library_database: Callable[[], Any] | None = None,
) -> APIRouter:
    """Factory: health and version routes."""
    router = APIRouter(tags=["system"])

    @router.get("/api/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Liveness check, plus whether the engine is actually usable.

        #4684: ``auralis_available`` used to echo the import-time HAS_AURALIS
        flag, which was a hardcoded True — so a backend whose Auralis init had
        failed and rolled every component back to None (#3812) still reported
        itself available while every data route returned 503. It now also
        requires the live library database to exist.

        ``status`` stays "healthy" with HTTP 200 on purpose: this remains the
        liveness probe. desktop/main.js waits on a 200 from here before showing
        the window, so turning a failed init into a non-200 would hang the
        launcher instead of surfacing the degraded state.
        """
        engine_ready = HAS_AURALIS and (
            get_library_database is not None and get_library_database() is not None
        )
        return HealthResponse(status="healthy", auralis_available=engine_ready)

    @router.get("/api/version", response_model=VersionInfoResponse)
    async def get_version() -> VersionInfoResponse:
        """Get version information.

        Returns detailed version info including semantic version components,
        build date, API version, and database schema version.
        """
        try:
            from auralis.version import get_version_info
            return VersionInfoResponse(**get_version_info())
        except ImportError:
            logger.warning("auralis.version not available, using fallback")
            # Derived fallback for degraded builds that cannot import the core.
            # sync_version.py keeps this aligned with auralis/version.py.
            # db_schema_version is sourced directly from auralis/__version__.py
            # (which has no heavy transitive deps) rather than hardcoded, so this
            # can never drift from the live value again (#4053, #5072).
            from auralis.__version__ import __db_schema_version__

            return VersionInfoResponse(
                version="1.5.1",
                major=1,
                minor=5,
                patch=1,
                prerelease="",
                build="",
                build_date="2026-07-24",
                git_commit="",
                api_version="v1",
                db_schema_version=__db_schema_version__,
                display="Auralis v1.5.1",
            )

    return router
