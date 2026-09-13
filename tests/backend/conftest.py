"""
Backend Test Configuration

Provides fixtures and configuration for backend testing.
"""

import sys
from pathlib import Path

import pytest

# Add backend to Python path for imports
backend_path = Path(__file__).parent.parent.parent / 'auralis-web' / 'backend'
sys.path.insert(0, str(backend_path))


@pytest.fixture(autouse=True)
def _bypass_streaming_path_validation(monkeypatch):
    """Stand in for validate_file_path()'s allowed-directory/is_file/readable
    checks in streaming tests, while still honoring Path.exists() (#4345).

    stream_normal.py / stream_enhanced.py / stream_seek.py now validate
    track.filepath via validate_file_path() before any file I/O. Existing
    tests simulate a present/missing file with fabricated `/tmp/...` paths
    plus `patch.object(Path, "exists", return_value=True/False)`, which
    doesn't satisfy validate_file_path's stricter is_file/readable/
    allowed-directory checks. Those tests exercise message ordering / error
    handling / lifecycle, not the validation boundary itself, so this stand-in
    reproduces the pre-#4345 existence-only check — still driven by the same
    Path.exists() mock each test already sets up. Tests that DO exercise path
    validation (test_streaming_path_validation.py) restore the real function
    explicitly.
    """
    from security.path_security import PathValidationError

    def _existence_only_check(filepath: str, context: str | None = None) -> Path:
        # `context` mirrors the keyword the real validator gained in #4925 so
        # it can log a rejection once, with the caller's context, instead of
        # each caller remembering to. The stand-in accepts and ignores it —
        # without the parameter every streaming test that goes through this
        # fixture dies with "unexpected keyword argument 'context'".
        p = Path(filepath)
        if not p.exists():
            raise PathValidationError(f"File does not exist: {filepath}")
        return p

    monkeypatch.setattr("core.stream_normal.validate_file_path", _existence_only_check)
    monkeypatch.setattr("core.stream_enhanced.validate_file_path", _existence_only_check)
    monkeypatch.setattr("core.stream_seek.validate_file_path", _existence_only_check)


@pytest.fixture
def mock_track():
    """Create a mock track object"""
    from unittest.mock import Mock

    track = Mock()
    track.id = 1
    track.filepath = str(Path(__file__).parent / "fixtures" / "test_audio.mp3")
    track.duration = 238.5
    track.title = "Test Track"
    track.artist = "Test Artist"
    track.album = "Test Album"
    track.album_id = 1

    return track


@pytest.fixture
def sample_audio():
    """
    Generate sample audio data for testing.

    Returns:
        tuple: (audio_data, sample_rate)
            - audio_data: numpy array (samples, channels) for stereo
            - sample_rate: int (44100)
    """
    import numpy as np

    sample_rate = 44100
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)

    # Generate 440 Hz sine wave (A4 note)
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t)

    # Make stereo
    audio_stereo = np.column_stack([audio, audio])

    return audio_stereo, sample_rate


@pytest.fixture
def sample_audio_long():
    """
    Generate longer sample audio (30 seconds) for chunk testing.

    Returns:
        tuple: (audio_data, sample_rate)
    """
    import numpy as np

    sample_rate = 44100
    duration = 30.0  # 30 seconds
    samples = int(sample_rate * duration)

    # Generate 440 Hz sine wave
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t)

    # Make stereo
    audio_stereo = np.column_stack([audio, audio])

    return audio_stereo, sample_rate


def _clear_rate_limit_windows() -> None:
    """Clear RateLimitMiddleware's sliding windows (#5089).

    OriginCheckMiddleware is registered outside RateLimitMiddleware precisely
    so a rejected CSRF-shaped request never consumes rate budget. While the
    test client sent no Origin, every state-changing /api request was rejected
    at the origin check and the rate limiter never saw it. Now that the client
    sends a trusted Origin, those requests reach the limiter for real -- and
    the limits are tight (``/api/library/scan`` allows 2 per 60s, uploads 5),
    so either a class with more than two scan tests, or a single test that
    posts more than two payloads to the same endpoint (e.g. a security test
    looping over several malicious inputs), exhausts the budget and gets a 429
    that has nothing to do with what it actually asserts.

    ``main.app`` is a module-level singleton shared by every test in this
    directory, so its middleware instances -- and their windows -- persist for
    the whole session. Clearing them restores isolation without weakening the
    limiter itself, which keeps its own dedicated coverage (see
    test_rate_limit_*.py / test_middleware_429_security_headers.py).
    """
    try:
        import main
    except ImportError:
        return

    stack = getattr(main.app, "middleware_stack", None)
    if stack is None:
        # No request has been served yet, so the stack was never built.
        return

    from config.middleware import RateLimitMiddleware

    node, depth = stack, 0
    while node is not None and depth < 30:
        if isinstance(node, RateLimitMiddleware):
            node._windows.clear()
            node._request_count = 0
            return
        node = getattr(node, "app", None)
        depth += 1


@pytest.fixture(autouse=True)
def _reset_rate_limit_windows():
    """Clear rate-limit state after every test -- see _clear_rate_limit_windows()."""
    yield
    _clear_rate_limit_windows()


@pytest.fixture
def reset_rate_limits():
    """Callable fixture for a test body to reset rate-limit budget mid-test.

    For a single test that deliberately sends more than a rate-limited
    endpoint's per-window allowance -- e.g. looping several malicious payloads
    through the same POST route -- rather than testing the rate limit itself.
    """
    return _clear_rate_limit_windows


@pytest.fixture
def client():
    """
    FastAPI test client for API endpoint testing.

    Returns:
        TestClient: Configured test client with all routes
    """
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient

    # Import main app
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'auralis-web' / 'backend'))
    try:
        from main import app
        with TestClient(app) as test_client:
            # Starlette TestClient WebSocket connections use 'testclient' as
            # the host, which the origin-check fix in #3845 correctly rejects
            # (non-loopback empty-Origin).  Patch websocket_connect here so
            # every backend test automatically sends a valid listed Origin
            # without requiring each test to specify it explicitly.
            #
            # #4781: this used to default to http://localhost:3000 (a Vite
            # dev-server origin), which build_ws_origins() only allows when
            # is_dev_mode() is true (#4350) -- and ALLOWED_WS_ORIGINS is
            # frozen once at config.globals import time, so no per-test env
            # var/monkeypatch can fix it up after the fact. Since pytest never
            # runs with --dev/AURALIS_DEV_MODE (#4802), every WebSocket-connecting test using
            # this fixture was rejected with "untrusted origin", not hanging
            # as such but failing before any real assertion ran. Port 8765
            # (the backend's own origin) is unconditionally allowlisted
            # regardless of dev/prod mode, so it works no matter when
            # config.globals first got imported this session.
            _orig_ws_connect = test_client.websocket_connect

            def _ws_connect_with_origin(url: str, **kwargs):  # type: ignore[override]
                headers = dict(kwargs.pop("headers", {}))
                headers.setdefault("origin", "http://localhost:8765")
                return _orig_ws_connect(url, headers=headers, **kwargs)

            test_client.websocket_connect = _ws_connect_with_origin  # type: ignore[method-assign]

            # #5089: the same reasoning applies to plain HTTP verbs.
            # OriginCheckMiddleware rejects any state-changing /api request
            # that arrives with an empty Origin from a non-loopback host, and
            # Starlette's TestClient reports its host as 'testclient'. Every
            # POST/PUT/DELETE/PATCH built on this fixture therefore got a 403
            # before the route, dependency graph, or Pydantic validator ever
            # ran -- 131 tests across 10 files asserting against a rejection
            # they never intended to exercise. Setting a default Origin on the
            # client (httpx merges these into every request) restores the
            # write-path coverage. The middleware's own enforcement stays
            # covered by test_origin_check_middleware.py, which builds its own
            # app and client rather than using this fixture.
            test_client.headers.update({"origin": "http://localhost:8765"})

            yield test_client
    except ImportError:
        # If backend not available, skip fixture
        pytest.skip("Backend main app not available")


@pytest.fixture
def sample_track_ids(client, mock_track):
    """
    Create sample tracks in the database for testing.

    Returns:
        list: List of track IDs [1, 2, 3]
    """
    # For now, return mock track IDs
    # In a real implementation, this would create tracks in test database
    return [1, 2, 3]


@pytest.fixture
def fingerprint_count():
    """
    Return the number of fingerprints in test database.

    Returns:
        int: Number of fingerprints (mock value for testing)
    """
    # For now, return a mock value
    # In a real implementation, this would query the test database
    return 100


# ============================================================
# Phase 5C: Mock Repository Fixtures for Dual-Mode Testing
# ============================================================

@pytest.fixture
def mock_library_database():
    """
    Create a mock LibraryManager for testing routers.

    This mock provides all necessary repository interfaces
    without requiring a real database connection.
    """
    from unittest.mock import MagicMock, Mock

    manager = Mock()

    # Mock repositories
    manager.tracks = Mock()
    manager.albums = Mock()
    manager.artists = Mock()
    manager.genres = Mock()
    manager.playlists = Mock()
    manager.fingerprints = Mock()
    manager.stats = Mock()
    manager.settings = Mock()

    # Mock get_all returns (list, total)
    manager.tracks.get_all = Mock(return_value=([], 0))
    manager.albums.get_all = Mock(return_value=([], 0))
    manager.artists.get_all = Mock(return_value=([], 0))
    manager.genres.get_all = Mock(return_value=([], 0))
    manager.playlists.get_all = Mock(return_value=([], 0))

    # Mock search returns (list, total)
    manager.tracks.search = Mock(return_value=([], 0))
    manager.artists.search = Mock(return_value=([], 0))
    manager.albums.search = Mock(return_value=([], 0))

    # Mock individual get_by_id methods
    manager.tracks.get_by_id = Mock(return_value=None)
    manager.albums.get_by_id = Mock(return_value=None)
    manager.artists.get_by_id = Mock(return_value=None)
    manager.genres.get_by_id = Mock(return_value=None)
    manager.playlists.get_by_id = Mock(return_value=None)

    # Mock CRUD operations
    manager.tracks.create = Mock(return_value=Mock(id=1))
    manager.tracks.update = Mock(return_value=Mock(id=1))
    manager.tracks.delete = Mock(return_value=True)
    manager.albums.create = Mock(return_value=Mock(id=1))
    manager.artists.create = Mock(return_value=Mock(id=1))

    # Mock cache operations
    manager.clear_cache = Mock()
    manager.get_cache_stats = Mock(return_value={"hits": 0, "misses": 0})

    return manager


@pytest.fixture
def mock_repository_factory():
    """
    Create a mock RepositoryFactory for testing routers.

    This mock provides the repository pattern interface
    without requiring a real database connection.

    Phase 5C: Enables testing endpoints with RepositoryFactory
    while maintaining backward compatibility with LibraryManager tests.
    """
    from unittest.mock import MagicMock, Mock

    factory = Mock()

    # Mock all repositories
    factory.tracks = Mock()
    factory.albums = Mock()
    factory.artists = Mock()
    factory.genres = Mock()
    factory.playlists = Mock()
    factory.fingerprints = Mock()
    factory.stats = Mock()
    factory.settings = Mock()
    factory.queue = Mock()
    factory.queue_history = Mock()

    # Mock get_all returns (list, total)
    factory.tracks.get_all = Mock(return_value=([], 0))
    factory.albums.get_all = Mock(return_value=([], 0))
    factory.artists.get_all = Mock(return_value=([], 0))
    factory.genres.get_all = Mock(return_value=([], 0))
    factory.playlists.get_all = Mock(return_value=([], 0))

    # Mock search returns (list, total)
    factory.tracks.search = Mock(return_value=([], 0))
    factory.artists.search = Mock(return_value=([], 0))
    factory.albums.search = Mock(return_value=([], 0))

    # Mock individual get_by_id methods
    factory.tracks.get_by_id = Mock(return_value=None)
    factory.albums.get_by_id = Mock(return_value=None)
    factory.artists.get_by_id = Mock(return_value=None)
    factory.genres.get_by_id = Mock(return_value=None)
    factory.playlists.get_by_id = Mock(return_value=None)
    factory.fingerprints.get_by_id = Mock(return_value=None)

    # Mock CRUD operations
    factory.tracks.create = Mock(return_value=Mock(id=1))
    factory.tracks.create_batch = Mock(return_value=[Mock(id=i) for i in range(1, 4)])
    factory.tracks.update = Mock(return_value=Mock(id=1))
    factory.tracks.delete = Mock(return_value=True)
    factory.tracks.update_metadata = Mock(return_value=Mock(id=1))
    factory.albums.create = Mock(return_value=Mock(id=1))
    factory.artists.create = Mock(return_value=Mock(id=1))
    factory.genres.create = Mock(return_value=Mock(id=1))

    # Mock query operations
    factory.tracks.get_by_artist = Mock(return_value=[])
    factory.tracks.get_by_album = Mock(return_value=[])
    factory.albums.get_by_artist = Mock(return_value=[])

    # Mock stats/fingerprint operations
    factory.fingerprints.get_fingerprint_stats = Mock(
        return_value={
            'total': 0,
            'fingerprinted': 0,
            'pending': 0,
            'progress_percent': 0
        }
    )
    factory.fingerprints.get_fingerprint_status = Mock(return_value=None)
    factory.fingerprints.cleanup_incomplete_fingerprints = Mock(return_value=0)

    # Mock settings operations
    factory.settings.get_all = Mock(return_value={})
    factory.settings.get = Mock(return_value=None)
    factory.settings.set = Mock(return_value=True)

    return factory


@pytest.fixture
def mock_repository_factory_callable(mock_repository_factory):
    """Return a callable that provides mock RepositoryFactory (for DI pattern).

    This fixture enables dependency injection pattern for testing router endpoints
    with mock repositories. Use this to test components that accept a callable
    returning a RepositoryFactory.

    Phase 5A.2: Mock variant for backend API router testing.

    Args:
        mock_repository_factory: Mock RepositoryFactory fixture

    Returns:
        Callable: Function that returns the mock RepositoryFactory instance

    Example:
        def test_get_artists(mock_repository_factory_callable):
            from auralis.player.queue_controller import QueueController
            controller = QueueController(mock_repository_factory_callable)
            # controller now has access to mocked repositories
    """
    def _get_factory():
        """Get mock repository factory instance."""
        return mock_repository_factory

    return _get_factory


@pytest.fixture(params=["library_database", "repository_factory"])
def mock_data_source(request, mock_library_database, mock_repository_factory):
    """
    Parametrized fixture for dual-mode backend testing.

    Automatically provides both LibraryManager and RepositoryFactory mocks,
    allowing router tests to validate both access patterns work correctly.

    Phase 5C: Enables automatic dual-mode validation of all router endpoints.

    Example:
        def test_get_artists_both_modes(client, mock_data_source):
            # Test automatically runs with both patterns
            mode, source = mock_data_source
            # Can check which mode and handle accordingly
    """
    if request.param == "library_database":
        return ("library_database", mock_library_database)
    else:
        return ("repository_factory", mock_repository_factory)


@pytest.fixture
def mock_track_repository():
    """Get mock TrackRepository for testing."""
    from unittest.mock import Mock
    repo = Mock()
    repo.get_all = Mock(return_value=([], 0))
    repo.get_by_id = Mock(return_value=None)
    repo.search = Mock(return_value=([], 0))
    repo.create = Mock(return_value=Mock(id=1))
    repo.update = Mock(return_value=Mock(id=1))
    repo.delete = Mock(return_value=True)
    return repo


@pytest.fixture
def mock_album_repository():
    """Get mock AlbumRepository for testing."""
    from unittest.mock import Mock
    repo = Mock()
    repo.get_all = Mock(return_value=([], 0))
    repo.get_by_id = Mock(return_value=None)
    repo.search = Mock(return_value=([], 0))
    repo.create = Mock(return_value=Mock(id=1))
    repo.update = Mock(return_value=Mock(id=1))
    repo.delete = Mock(return_value=True)
    return repo


@pytest.fixture
def mock_artist_repository():
    """Get mock ArtistRepository for testing."""
    from unittest.mock import Mock
    repo = Mock()
    repo.get_all = Mock(return_value=([], 0))
    repo.get_by_id = Mock(return_value=None)
    repo.search = Mock(return_value=([], 0))
    repo.create = Mock(return_value=Mock(id=1))
    repo.update = Mock(return_value=Mock(id=1))
    repo.delete = Mock(return_value=True)
    return repo


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with asyncio settings"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an asyncio test"
    )
    config.addinivalue_line(
        "markers", "phase5c: Phase 5C dual-mode backend tests"
    )
