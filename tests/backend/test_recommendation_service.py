"""
Unit tests for RecommendationService (#3860 / BE-TC-5)

RecommendationService is wired into routers/player.py and called on track
load to broadcast mastering recommendations, but had no dedicated tests.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))

from cache import StreamlinedCacheManager  # noqa: E402
from services.recommendation_service import RecommendationService  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service():
    connection_manager = MagicMock()
    connection_manager.broadcast = AsyncMock()
    cache_manager = StreamlinedCacheManager()
    return (
        RecommendationService(
            connection_manager=connection_manager,
            cache_manager=cache_manager,
        ),
        connection_manager,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecommendationServiceInit:
    def test_stores_connection_manager(self):
        service, conn_mgr = _make_service()
        assert service.connection_manager is conn_mgr


class TestGenerateAndBroadcastRecommendation:
    @pytest.mark.asyncio
    async def test_broadcasts_recommendation_when_analysis_succeeds(self):
        service, conn_mgr = _make_service()

        rec = {"preset": "adaptive", "confidence": 0.85, "track_id": 1}

        with patch.object(service, "generate_and_broadcast_recommendation") as mock_method:
            mock_method.return_value = rec
            result = await service.generate_and_broadcast_recommendation(1, "/music/track.mp3")

        assert result == rec

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_analysis_returns_none(self):
        """When _analyze() returns None (low confidence), service returns {} without broadcasting."""
        service, conn_mgr = _make_service()

        # Patch asyncio.to_thread to return None (simulate low confidence)
        with patch("services.recommendation_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = None
            result = await service.generate_and_broadcast_recommendation(1, "/music/track.mp3")

        assert result == {}
        conn_mgr.broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_analysis_raises(self):
        """Exceptions during analysis are swallowed — recommendations are optional."""
        service, conn_mgr = _make_service()

        with patch("services.recommendation_service.asyncio.to_thread", side_effect=RuntimeError("analysis failed")):
            result = await service.generate_and_broadcast_recommendation(1, "/music/track.mp3")

        assert result == {}
        conn_mgr.broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_broadcasts_mastering_recommendation_message(self):
        """When analysis returns a dict, it must be broadcast with type mastering_recommendation."""
        service, conn_mgr = _make_service()

        rec_dict = {"preset": "warm", "confidence": 0.9, "track_id": 42}

        with patch("services.recommendation_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = rec_dict
            await service.generate_and_broadcast_recommendation(42, "/music/track.mp3")

        conn_mgr.broadcast.assert_awaited_once()
        payload = conn_mgr.broadcast.call_args[0][0]
        assert payload["type"] == "mastering_recommendation"
        assert payload["data"] is rec_dict


class TestGetRecommendationForTrack:
    @pytest.mark.asyncio
    async def test_returns_recommendation_dict_on_success(self):
        service, conn_mgr = _make_service()
        rec_dict = {"preset": "adaptive", "confidence": 0.75, "track_id": 7}

        with patch("services.recommendation_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = rec_dict
            result = await service.get_recommendation_for_track(7, "/music/track.mp3")

        assert result == rec_dict
        conn_mgr.broadcast.assert_not_awaited()  # no broadcast in this path

    @pytest.mark.asyncio
    async def test_returns_none_when_analysis_returns_none(self):
        service, _ = _make_service()

        with patch("services.recommendation_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = None
            result = await service.get_recommendation_for_track(7, "/music/track.mp3")

        assert result is None

    @pytest.mark.asyncio
    async def test_second_call_hits_shared_cache_without_reanalysis(self):
        service, _ = _make_service()
        rec_dict = {"preset": "adaptive", "confidence": 0.75, "track_id": 7}

        with patch(
            "services.recommendation_service.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=rec_dict,
        ) as mock_thread:
            first = await service.get_recommendation_for_track(7, "/music/track.mp3")
            second = await service.get_recommendation_for_track(7, "/music/track.mp3")

        assert first is rec_dict
        assert second is rec_dict
        mock_thread.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_and_query_paths_share_cached_result(self):
        service, conn_mgr = _make_service()
        rec_dict = {"preset": "warm", "confidence": 0.9, "track_id": 42}

        with patch(
            "services.recommendation_service.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=rec_dict,
        ) as mock_thread:
            broadcast_result = await service.generate_and_broadcast_recommendation(
                42, "/music/track.mp3"
            )
            query_result = await service.get_recommendation_for_track(
                42, "/music/track.mp3"
            )

        assert broadcast_result is rec_dict
        assert query_result is rec_dict
        mock_thread.assert_awaited_once()
        conn_mgr.broadcast.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_on_exception(self):
        """get_recommendation_for_track propagates exceptions (not swallowed)."""
        service, _ = _make_service()

        with patch("services.recommendation_service.asyncio.to_thread", side_effect=ValueError("bad")):
            with pytest.raises(ValueError, match="bad"):
                await service.get_recommendation_for_track(7, "/music/track.mp3")


class TestAnalysisTimeout:
    """#5248: both call sites must bound the analysis in
    asyncio.wait_for(..., timeout=CHUNK_PROCESS_TIMEOUT), mirroring every
    streaming entry point's ChunkedAudioProcessor construction — otherwise a
    corrupt-header file (sf.info() has no timeout of its own for
    natively-decodable formats) can hang a shared IO_EXECUTOR thread forever.
    """

    @pytest.mark.asyncio
    async def test_generate_and_broadcast_returns_empty_dict_on_timeout(self):
        """A hung analysis must degrade like any other analysis failure.

        Mocking `to_thread` (not `wait_for`) to raise TimeoutError directly
        is functionally equivalent to a real wait_for timeout from the
        caller's point of view (both raise inside the same try block, hit
        the same except clause) — and, unlike mocking `wait_for`, it never
        constructs a real `_analyze()` coroutine that would be left
        unawaited when the mock intercepts it.
        """
        service, conn_mgr = _make_service()

        with patch(
            "services.recommendation_service.asyncio.to_thread",
            side_effect=TimeoutError,
        ):
            result = await service.generate_and_broadcast_recommendation(1, "/music/track.mp3")

        assert result == {}
        conn_mgr.broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_recommendation_raises_timeout_error(self):
        """Unlike the broadcast path, this one preserves its documented
        "raises on failure" contract — a timeout must still propagate."""
        service, _ = _make_service()

        with patch(
            "services.recommendation_service.asyncio.to_thread",
            side_effect=TimeoutError,
        ):
            with pytest.raises(TimeoutError):
                await service.get_recommendation_for_track(7, "/music/track.mp3")

    @pytest.mark.asyncio
    async def test_generate_and_broadcast_uses_chunk_process_timeout(self):
        """The bound must match the constant every streaming entry point
        uses (core.audio_stream_controller.CHUNK_PROCESS_TIMEOUT), not an
        ad-hoc value that could drift from it."""
        from core import audio_stream_controller as _asc

        service, _ = _make_service()

        with patch(
            "services.recommendation_service.asyncio.wait_for",
            new_callable=AsyncMock,
        ) as mock_wait_for:
            mock_wait_for.return_value = None
            await service.generate_and_broadcast_recommendation(1, "/music/track.mp3")

        assert mock_wait_for.await_args.kwargs["timeout"] == _asc.CHUNK_PROCESS_TIMEOUT
        # The mock intercepted wait_for before it could await its argument —
        # close the real (never-run) `asyncio.to_thread(_analyze)` coroutine
        # it was handed, so it isn't left dangling for the GC to warn about.
        mock_wait_for.await_args.args[0].close()

    @pytest.mark.asyncio
    async def test_get_recommendation_uses_chunk_process_timeout(self):
        from core import audio_stream_controller as _asc

        service, _ = _make_service()

        with patch(
            "services.recommendation_service.asyncio.wait_for",
            new_callable=AsyncMock,
        ) as mock_wait_for:
            mock_wait_for.return_value = None
            await service.get_recommendation_for_track(7, "/music/track.mp3")

        assert mock_wait_for.await_args.kwargs["timeout"] == _asc.CHUNK_PROCESS_TIMEOUT
        mock_wait_for.await_args.args[0].close()


class TestAnalyzeDoesNotMutateSysPath:
    """#4745: both _analyze() closures used to unconditionally
    sys.path.insert(0, ...) on every call with no removal — an unbounded,
    unnecessary mutation of process-global sys.path. core.chunked_processor
    already ensures the backend dir is importable (idempotently), and other
    callers (proactive_buffer.py, streamlined_worker.py, chunk_mastering.py,
    audio_stream_controller.py) already do a bare import with no path
    juggling — so the import must keep resolving with no insert at all.

    Unlike the other test classes here, this does NOT mock
    asyncio.to_thread — it must actually run the real _analyze() closure so
    the `from core.chunked_processor import ChunkedAudioProcessor` line
    executes for real.
    """

    @pytest.mark.asyncio
    async def test_generate_and_broadcast_does_not_grow_sys_path(self):
        service, _ = _make_service()

        with patch("core.chunked_processor.ChunkedAudioProcessor") as mock_cls:
            mock_cls.return_value.get_mastering_recommendation.return_value = None

            before = list(sys.path)
            for _ in range(3):
                result = await service.generate_and_broadcast_recommendation(1, "/music/track.mp3")
            after = list(sys.path)

        assert result == {}
        assert after == before, "sys.path must be unchanged after repeated calls (#4745)"

    @pytest.mark.asyncio
    async def test_get_recommendation_does_not_grow_sys_path(self):
        service, _ = _make_service()

        with patch("core.chunked_processor.ChunkedAudioProcessor") as mock_cls:
            mock_cls.return_value.get_mastering_recommendation.return_value = None

            before = list(sys.path)
            for _ in range(3):
                result = await service.get_recommendation_for_track(1, "/music/track.mp3")
            after = list(sys.path)

        assert result is None
        assert after == before, "sys.path must be unchanged after repeated calls (#4745)"

# The recommendation/pre-warm paths re-check a DB filepath with
# validate_file_path before any file I/O (#4817/#4818). These tests exercise
# what happens *after* that check with fabricated paths like /music/x.flac, so
# stand the check in with a pass-through; the guard itself is covered by the
# tests below that restore the real validator.
@pytest.fixture(autouse=True)
def _accept_fabricated_track_paths(monkeypatch):
    from pathlib import Path as _Path
    monkeypatch.setattr("services.recommendation_service.validate_file_path", lambda filepath, *a, **k: _Path(filepath))



@pytest.mark.asyncio
async def test_a_stored_path_outside_allowed_directories_is_not_analysed(monkeypatch):
    """#4817: POST /api/player/load feeds the DB filepath straight in; both
    public methods funnel through _get_or_analyze, which now re-validates."""
    from security.path_security import validate_file_path as real_validate

    monkeypatch.setattr("services.recommendation_service.validate_file_path", real_validate)
    service, conn_mgr = _make_service()

    with patch("services.recommendation_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        result = await service.get_recommendation_for_track(7, "/definitely/not/allowed/track.mp3")

    assert result is None
    mock_thread.assert_not_awaited()

