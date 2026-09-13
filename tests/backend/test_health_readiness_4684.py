"""/api/health reports whether the engine is actually up, not just importable (#4684).

HAS_AURALIS was a bare `True`, so `auralis_available` was true unconditionally —
including after a failed Auralis init rolled every component back to None, with
every data route returning 503. And the demo-mode branch it gates in startup
could never run.
"""

import asyncio
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))

from routers.health import create_health_router  # noqa: E402


def _client(**kwargs) -> TestClient:
    app = FastAPI()
    app.include_router(create_health_router(**kwargs))
    return TestClient(app)


def test_rolled_back_startup_is_not_reported_available():
    resp = _client(HAS_AURALIS=True, get_library_database=lambda: None).get("/api/health")
    assert resp.status_code == 200  # still the liveness probe desktop/main.js waits on
    assert resp.json() == {"status": "healthy", "auralis_available": False}


def test_fully_initialised_backend_is_healthy_and_available():
    resp = _client(HAS_AURALIS=True, get_library_database=lambda: object()).get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "auralis_available": True}


def test_engine_not_importable_is_not_available_even_with_a_database():
    resp = _client(HAS_AURALIS=False, get_library_database=lambda: object()).get("/api/health")
    assert resp.json()["auralis_available"] is False


def test_the_database_is_read_live_on_each_request():
    state = {"db": None}
    client = _client(HAS_AURALIS=True, get_library_database=lambda: state["db"])
    assert client.get("/api/health").json()["auralis_available"] is False
    state["db"] = object()
    assert client.get("/api/health").json()["auralis_available"] is True


def test_has_auralis_is_probed_not_hardcoded():
    source = (Path(__file__).parent.parent.parent / "auralis-web" / "backend" / "main.py").read_text()
    assert "import auralis as _auralis_probe" in source


def test_demo_mode_branch_is_reachable(caplog):
    from config.startup import _init_auralis_components

    globals_dict: dict = {}
    with caplog.at_level(logging.WARNING, logger="config.startup"):
        asyncio.run(_init_auralis_components(False, False, None, globals_dict))

    assert any("demo mode" in r.message for r in caplog.records)
    assert globals_dict == {}  # nothing initialised
