"""ArtworkDownloader bounds its network calls (#4686).

The shared aiohttp session had no ClientTimeout, so every request inherited
aiohttp's 300 s default, and download_artwork walks up to four remote calls in
sequence — one black-holing host could stall a worker ~20 minutes, silently.
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "auralis-web" / "backend"))

import services.artwork_downloader as mod
from services.artwork_downloader import ArtworkDownloader


class _TimingOutGet:
    async def __aenter__(self):
        raise TimeoutError()

    async def __aexit__(self, *args):
        return False


class _TimingOutSession:
    closed = False

    def get(self, *args, **kwargs):
        return _TimingOutGet()


@pytest.mark.asyncio
async def test_shared_session_carries_a_bounded_timeout(tmp_path):
    downloader = ArtworkDownloader(cache_dir=str(tmp_path))
    try:
        session = downloader._get_session()
        assert session.timeout.total == 15
        assert session.timeout.connect == 5
    finally:
        await downloader.close()


@pytest.mark.asyncio
async def test_whole_lookup_is_bounded_and_logged(tmp_path, monkeypatch, caplog):
    downloader = ArtworkDownloader(cache_dir=str(tmp_path))
    monkeypatch.setattr(mod, "_ARTWORK_LOOKUP_BUDGET_S", 0.05)

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    with patch.object(downloader, "_try_musicbrainz", _hang), \
         caplog.at_level(logging.WARNING, logger=mod.logger.name):
        result = await asyncio.wait_for(downloader.download_artwork("A", "B", 1), timeout=2)

    assert result is None
    assert any("timed out" in r.message for r in caplog.records)
    await downloader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["_try_musicbrainz", "_try_itunes"])
async def test_a_source_timeout_is_logged_at_warning(tmp_path, caplog, source):
    downloader = ArtworkDownloader(cache_dir=str(tmp_path))
    with patch.object(downloader, "_get_session", return_value=_TimingOutSession()), \
         caplog.at_level(logging.WARNING, logger=mod.logger.name):
        result = await getattr(downloader, source)("Artist", "Album", 1)

    assert result is None
    assert any("timed out" in r.message and r.levelno == logging.WARNING for r in caplog.records)
