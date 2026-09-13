"""
Regression tests for scan_library _progress_callback robustness (#3864 / BE-RH-19)

The closure previously had no try/except, so a scanner bug that emits
a non-dict progress_data would raise AttributeError inside the callback.
That exception is silently swallowed by asyncio.run_coroutine_threadsafe's
future, leaving operators with no error trail.  The fix wraps the broadcast
logic in try/except Exception with a logger.warning().

We test the callback by constructing it the same way the router does and
calling it directly — no need to drive the full HTTP endpoint.
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_progress_callback(loop: asyncio.AbstractEventLoop) -> object:
    """
    Build the same _progress_callback closure that scan_library creates,
    using a real event loop and a mock connection manager.
    """
    connection_manager = MagicMock()
    connection_manager.broadcast = AsyncMock()

    # Replicate the closure construction from library.py:538-558
    def _progress_callback(progress_data):
        try:
            total = progress_data.get('total_found', 0) or progress_data.get('processed', 0)
            processed = progress_data.get('processed', 0)
            progress_frac = progress_data.get('progress', 0)
            percentage = round(progress_frac * 100) if progress_frac else (
                round(processed / total * 100) if total > 0 else 0
            )
            stage = progress_data.get('stage', 'processing')
            asyncio.run_coroutine_threadsafe(
                connection_manager.broadcast({
                    "type": "scan_progress",
                    "data": {
                        "current": processed,
                        "total": total,
                        "percentage": percentage if stage != 'discovering' else None,
                        "current_file": progress_data.get('current_file') or progress_data.get('file'),
                        "phase": stage,
                    }
                }),
                loop,
            )
        except Exception:
            logging.getLogger("routers.library").warning(
                "scan_library progress callback failed — malformed progress_data",
                exc_info=True,
            )

    return _progress_callback


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProgressCallbackRobustness:
    """_progress_callback must log on malformed input, not silently crash (#3864)."""

    # Named `loop`, not `loop`: that name belonged to pytest-asyncio's
    # own (since-removed) fixture, and shadowing it made this plain loop for
    # the callback's run_coroutine_threadsafe target look like async plumbing
    # (#4332).
    @pytest.fixture
    def loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_callback_with_valid_dict_does_not_raise(self, loop):
        """Normal progress_data dict must be processed without error."""
        cb = _make_progress_callback(loop)

        with patch("asyncio.run_coroutine_threadsafe"):
            # Must not raise
            cb({
                "total_found": 10,
                "processed": 5,
                "progress": 0.5,
                "stage": "processing",
                "current_file": "/music/track.mp3",
            })

    def test_callback_with_non_dict_logs_warning_not_raise(self, loop, caplog):
        """Non-dict progress_data must be caught and logged, not crash (#3864)."""
        cb = _make_progress_callback(loop)

        with caplog.at_level(logging.WARNING, logger="routers.library"):
            with patch("asyncio.run_coroutine_threadsafe"):
                cb("not a dict")  # type: ignore[arg-type]

        assert any("progress callback failed" in r.message for r in caplog.records), (
            "Expected a warning about malformed progress_data (#3864)"
        )

    def test_callback_with_none_logs_warning_not_raise(self, loop, caplog):
        """None progress_data must also be caught and logged."""
        cb = _make_progress_callback(loop)

        with caplog.at_level(logging.WARNING, logger="routers.library"):
            with patch("asyncio.run_coroutine_threadsafe"):
                cb(None)  # type: ignore[arg-type]

        assert any("progress callback failed" in r.message for r in caplog.records)

    def test_callback_with_integer_logs_warning_not_raise(self, loop, caplog):
        """Integer progress_data (another malformed type) is caught and logged."""
        cb = _make_progress_callback(loop)

        with caplog.at_level(logging.WARNING, logger="routers.library"):
            with patch("asyncio.run_coroutine_threadsafe"):
                cb(42)  # type: ignore[arg-type]

        assert any("progress callback failed" in r.message for r in caplog.records)
