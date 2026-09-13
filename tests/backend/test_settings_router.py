"""
Settings router tests
~~~~~~~~~~~~~~~~~~~~~~

Regression coverage for the typed ``PUT /api/settings`` body (#3837 / BE-SCH-2).

Before the fix, the endpoint accepted ``updates: dict[str, Any]`` so a misspelled
field name silently no-op'd through the SettingsRepository whitelist (200 OK, no
change applied) and OpenAPI advertised "any object". The endpoint now takes a
typed ``SettingsUpdateRequest`` with ``extra='forbid'`` so unknown/out-of-range
fields are rejected with HTTP 422, and only fields the client actually sent are
forwarded to the repository (``exclude_unset``).
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# conftest already inserts auralis-web/backend on sys.path; keep this defensive
# so the module imports standalone too.
_BACKEND = str(Path(__file__).resolve().parent.parent.parent / "auralis-web" / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from routers.settings import create_settings_router  # noqa: E402


_DEFAULT_SETTINGS = {
    "id": 1,
    "scan_folders": [],
    "file_types": ["mp3", "flac"],
    "auto_scan": False,
    "scan_interval": 3600,
    "crossfade_enabled": False,
    "crossfade_duration": 5.0,
    "gapless_enabled": True,
    "replay_gain_enabled": False,
    "volume": 0.8,
    "output_device": "default",
    "bit_depth": 16,
    "sample_rate": 44100,
    "theme": "dark",
    "language": "en",
    "show_visualizations": True,
    "mini_player_on_close": False,
    "default_preset": "adaptive",
    "auto_enhance": False,
    "enhancement_intensity": 1.0,
    "cache_size": 1024,
    "max_concurrent_scans": 4,
    "enable_analytics": False,
    "debug_mode": False,
    "created_at": None,
    "updated_at": None,
}


class _FakeSettings:
    def __init__(self, data: dict) -> None:
        self._data = data
        # Mirrors the real UserSettings ORM column: scan_folders is a raw
        # JSON string (or None) on the model itself, only parsed to a list by
        # to_dict(). The router's scan_folders diff (#4765) reads this
        # attribute directly, the same way it reads a real settings row.
        folders = data.get('scan_folders')
        self.scan_folders = json.dumps(folders) if folders else None

    def to_dict(self) -> dict:
        return dict(self._data)


class _FakeSettingsRepo:
    """Records what reached the repository so tests can assert the request was
    (or was not) forwarded after validation."""

    def __init__(self) -> None:
        self.updated_with: dict | None = None

    def get_settings(self) -> _FakeSettings:
        return _FakeSettings(_DEFAULT_SETTINGS)

    def update_settings(self, payload: dict) -> _FakeSettings:
        self.updated_with = payload
        merged = {**_DEFAULT_SETTINGS, **payload}
        return _FakeSettings(merged)


@pytest.fixture()
def client() -> TestClient:
    repo = _FakeSettingsRepo()
    app = FastAPI()
    app.include_router(create_settings_router(lambda: repo))
    tc = TestClient(app)
    tc._repo = repo  # type: ignore[attr-defined]  # expose for assertions
    return tc


def test_update_settings_rejects_unknown_field(client: TestClient) -> None:
    """A misspelled/unknown field must 422, not silently no-op (the #3837 bug)."""
    resp = client.put("/api/settings", json={"volumee": 0.5})
    assert resp.status_code == 422
    assert client._repo.updated_with is None  # type: ignore[attr-defined]


def test_update_settings_validates_field_ranges(client: TestClient) -> None:
    """Out-of-range values are rejected with 422 (volume must be in [0, 1])."""
    resp = client.put("/api/settings", json={"volume": 2.0})
    assert resp.status_code == 422
    assert client._repo.updated_with is None  # type: ignore[attr-defined]


def test_update_settings_partial_excludes_unset(client: TestClient) -> None:
    """Only the fields the client sent reach the repo (exclude_unset)."""
    resp = client.put("/api/settings", json={"theme": "light"})
    assert resp.status_code == 200
    assert client._repo.updated_with == {"theme": "light"}  # type: ignore[attr-defined]
    assert resp.json()["settings"]["theme"] == "light"


def test_update_settings_accepts_multiple_known_fields(client: TestClient, tmp_path) -> None:
    """A multi-field valid update passes through unchanged.

    scan_folders must be a real, existing directory — validate_user_chosen_directory
    (#4765) now runs on this route too, so a literal "/music" 400s regardless of
    the repo being mocked.
    """
    resolved = str(tmp_path.resolve())
    resp = client.put(
        "/api/settings",
        json={"volume": 0.3, "crossfade_enabled": True, "scan_folders": [str(tmp_path)]},
    )
    assert resp.status_code == 200
    assert client._repo.updated_with == {  # type: ignore[attr-defined]
        "volume": 0.3,
        "crossfade_enabled": True,
        "scan_folders": [resolved],
    }


def test_update_settings_scan_folders_rejects_traversal(client: TestClient) -> None:
    """PUT /api/settings 400s an invalid scan_folders entry, same as the
    dedicated POST /api/settings/scan-folders route (#4765) — not a generic
    422, since callers already branch on this route's 400 for bad paths."""
    resp = client.put("/api/settings", json={"scan_folders": ["../../etc"]})
    assert resp.status_code == 400
    assert client._repo.updated_with is None  # type: ignore[attr-defined]


def test_update_settings_scan_folders_rejects_nonexistent(client: TestClient) -> None:
    resp = client.put("/api/settings", json={"scan_folders": ["/definitely_does_not_exist_xyz"]})
    assert resp.status_code == 400
    assert client._repo.updated_with is None  # type: ignore[attr-defined]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX system directories")
@pytest.mark.parametrize("system_dir", ["/etc", "/root", "/var"])
def test_update_settings_scan_folders_rejects_system_directories(
    client: TestClient, system_dir: str
) -> None:
    resp = client.put("/api/settings", json={"scan_folders": [system_dir]})
    assert resp.status_code == 400
    assert client._repo.updated_with is None  # type: ignore[attr-defined]


def test_update_settings_rejects_invalid_preset(client: TestClient) -> None:
    """default_preset is constrained to the shared EnhancementPreset enum, so a
    bogus value 422s at the boundary instead of silently reaching the repo (#4424)."""
    resp = client.put("/api/settings", json={"default_preset": "bogus"})
    assert resp.status_code == 422
    assert client._repo.updated_with is None  # type: ignore[attr-defined]


def test_update_settings_accepts_valid_preset(client: TestClient) -> None:
    """A canonical preset from the shared enum passes validation (#4424)."""
    resp = client.put("/api/settings", json={"default_preset": "warm"})
    assert resp.status_code == 200
    assert client._repo.updated_with == {"default_preset": "warm"}  # type: ignore[attr-defined]


def test_get_settings_returns_typed_shape(client: TestClient) -> None:
    """GET still works and is shaped by SettingsResponse."""
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["theme"] == "dark"
    assert body["volume"] == 0.8


# ---------------------------------------------------------------------------
# #4647: the response side carries the same constraints as the request side
# ---------------------------------------------------------------------------


def _client_with_row(**overrides) -> TestClient:
    repo = _FakeSettingsRepo()
    row = {**_DEFAULT_SETTINGS, **overrides}
    repo.get_settings = lambda: _FakeSettings(row)  # type: ignore[method-assign]
    app = FastAPI()
    app.include_router(create_settings_router(lambda: repo))
    return TestClient(app)


def test_response_schema_advertises_the_preset_enum_and_intensity_bounds() -> None:
    app = FastAPI()
    app.include_router(create_settings_router(lambda: _FakeSettingsRepo()))
    props = app.openapi()["components"]["schemas"]["SettingsResponse"]["properties"]

    preset_enum = {
        v for option in props["default_preset"]["anyOf"] for v in option.get("enum", [])
    }
    assert preset_enum == {"adaptive", "gentle", "warm", "bright", "punchy"}

    bounded = [o for o in props["enhancement_intensity"]["anyOf"] if o.get("type") == "number"]
    assert bounded and bounded[0]["minimum"] == 0.0 and bounded[0]["maximum"] == 1.0


def test_canonical_stored_values_pass_through() -> None:
    body = _client_with_row(default_preset="warm", enhancement_intensity=0.4).get("/api/settings").json()
    assert body["default_preset"] == "warm"
    assert body["enhancement_intensity"] == 0.4


def test_a_legacy_off_list_preset_degrades_to_null_not_a_500(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        resp = _client_with_row(default_preset="vintage").get("/api/settings")

    assert resp.status_code == 200
    assert resp.json()["default_preset"] is None
    assert any("default_preset" in r.message for r in caplog.records)


@pytest.mark.parametrize("stored", [1.7, -0.1, float("nan"), "loud"])
def test_an_out_of_range_stored_intensity_degrades_to_null_not_a_500(stored) -> None:
    resp = _client_with_row(enhancement_intensity=stored).get("/api/settings")
    assert resp.status_code == 200
    assert resp.json()["enhancement_intensity"] is None

