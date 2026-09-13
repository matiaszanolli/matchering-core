"""
Artwork Downloader Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically fetches album artwork from online sources.

Features:
- MusicBrainz Cover Art Archive (primary)
- iTunes Search API (fallback)
- Last.fm API (fallback)
- Image caching and optimization

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import asyncio
import hashlib
import logging
from pathlib import Path

import aiohttp

from auralis.utils.artwork_security import (
    MAX_ARTWORK_PAYLOAD_BYTES as _MAX_ARTWORK_BYTES,
)
from auralis.utils.artwork_security import validate_artwork_url as _validate_artwork_url
from auralis.utils.logging import sanitize_log_value

logger = logging.getLogger(__name__)


def _detect_image_extension(data: bytes, default: str = "jpg") -> str:
    """Pick a file extension from an image's magic bytes.

    Cover Art Archive / iTunes can return PNG or WebP even when we requested
    a JPEG, and the GET endpoint infers Content-Type from the extension, so a
    PNG saved as .jpg is served image/jpeg (#4419). Mirrors the embedded
    extractor in auralis/library/artwork.py. Falls back to ``default`` for
    unrecognised bytes.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return default


# #4686: aiohttp's default ClientTimeout is total=300s, and the downloader
# walks up to four remote calls per album in sequence, so one host that accepts
# the connection and never answers could hold a worker for ~20 minutes with
# nothing logged. Bound every request, and bound the album's whole chain too, so
# adding another fallback source cannot multiply the worst case again.
_ARTWORK_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15, connect=5)
_ARTWORK_LOOKUP_BUDGET_S = 45.0


class ArtworkDownloader:
    """
    Service for downloading album artwork from online sources.

    Uses multiple sources with fallback:
    1. MusicBrainz Cover Art Archive (open source, no API key needed)
    2. iTunes Search API (free, no API key needed)
    3. Last.fm API (requires API key)
    """

    def __init__(self, cache_dir: str = "~/.auralis/artwork"):
        """
        Initialize artwork downloader.

        Args:
            cache_dir: Directory to cache downloaded artwork. Defaults to the
                single served directory ``~/.auralis/artwork`` — the same one
                the embedded extractor (``auralis/library/artwork.py``) writes
                to and the only one the GET endpoint
                (``routers/artwork.py``) will serve. A sibling
                ``~/.auralis/artwork_cache`` used to be the default, so every
                downloaded image failed the serving guard with 403 (#4408).
                Both writers share the ``album_{id}_{hash}.{ext}`` naming, so
                they coexist in this directory without collision.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.musicbrainz_api = "https://musicbrainz.org/ws/2"
        self.coverart_api = "https://coverartarchive.org"
        self.itunes_api = "https://itunes.apple.com/search"

        # Shared session (fixes #3915 regression of #3558): reused across
        # calls with a small connection pool so bulk artwork backfills reuse
        # keep-alive connections instead of a fresh TCP/TLS handshake per
        # request. Created lazily on first use, not in __init__, because it
        # must be opened from within the event loop it will be used on.
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared HTTP session, creating it on first use."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=4, ttl_dns_cache=300),
                timeout=_ARTWORK_REQUEST_TIMEOUT,
            )
        return self._session

    async def close(self) -> None:
        """Close the shared HTTP session, releasing pooled connections.

        Call from application shutdown (see close_artwork_downloader()).
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def download_artwork(
        self,
        artist: str,
        album: str,
        album_id: int
    ) -> str | None:
        """
        Download artwork for an album from online sources.

        Args:
            artist: Artist name
            album: Album name
            album_id: Album ID (for unique cache naming)

        Returns:
            str: Path to downloaded artwork file, or None if not found
        """
        try:
            return await asyncio.wait_for(
                self._lookup(artist, album, album_id),
                timeout=_ARTWORK_LOOKUP_BUDGET_S,
            )
        except TimeoutError:
            logger.warning(
                f"Artwork lookup for '{sanitize_log_value(album)}' by "
                f"'{sanitize_log_value(artist)}' timed out after "
                f"{_ARTWORK_LOOKUP_BUDGET_S:.0f}s"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to download artwork: {e}")
            return None

    async def _lookup(self, artist: str, album: str, album_id: int) -> str | None:
        """Walk the source chain; bounded as a whole by download_artwork."""
        # Try MusicBrainz first (best quality, open source)
        artwork_path = await self._try_musicbrainz(artist, album, album_id)
        if artwork_path:
            logger.info(f"Downloaded artwork from MusicBrainz for '{sanitize_log_value(album)}' by '{sanitize_log_value(artist)}'")
            return artwork_path

        # Fallback to iTunes
        artwork_path = await self._try_itunes(artist, album, album_id)
        if artwork_path:
            logger.info(f"Downloaded artwork from iTunes for '{sanitize_log_value(album)}' by '{sanitize_log_value(artist)}'")
            return artwork_path

        logger.warning(f"No artwork found for '{sanitize_log_value(album)}' by '{sanitize_log_value(artist)}'")
        return None

    async def _try_musicbrainz(
        self,
        artist: str,
        album: str,
        album_id: int
    ) -> str | None:
        """
        Try to download artwork from MusicBrainz Cover Art Archive.

        Args:
            artist: Artist name
            album: Album name
            album_id: Album ID

        Returns:
            str: Path to downloaded artwork, or None if not found
        """
        try:
            session = self._get_session()
            # Search for release
            search_url = f"{self.musicbrainz_api}/release/"
            params = {
                "query": f'artist:"{artist}" AND release:"{album}"',
                "fmt": "json",
                "limit": 1
            }
            headers = {
                "User-Agent": "Auralis/1.0 (https://github.com/matiaszanolli/Auralis)"
            }

            async with session.get(search_url, params=params, headers=headers) as resp:  # type: ignore[arg-type]
                if resp.status != 200:
                    return None

                data = await resp.json()
                releases = data.get("releases", [])

                if not releases:
                    return None

                release_id = releases[0]["id"]

            # Get cover art
            coverart_url = f"{self.coverart_api}/release/{release_id}/front"

            async with session.get(coverart_url, headers=headers) as resp:
                if resp.status != 200:
                    return None

                # Validate final URL after redirects (SSRF mitigation #2576)
                if not _validate_artwork_url(str(resp.url)):
                    logger.warning(f"Rejecting untrusted MusicBrainz redirect: {resp.url!r}")
                    return None

                # Size-limited read to prevent memory exhaustion (#2576)
                content_length = resp.content_length or 0
                if content_length > _MAX_ARTWORK_BYTES:
                    logger.warning(f"MusicBrainz artwork too large: {content_length} bytes")
                    return None
                artwork_data = await resp.content.read(_MAX_ARTWORK_BYTES + 1)
                if len(artwork_data) > _MAX_ARTWORK_BYTES:
                    logger.warning(f"MusicBrainz artwork exceeded {_MAX_ARTWORK_BYTES} byte limit")
                    return None
                return await self._save_artwork(artwork_data, album_id, "jpg")

        except TimeoutError:
            # #4686: hit _ARTWORK_REQUEST_TIMEOUT. WARNING, not the debug level
            # other lookup failures use, so a black-holing host is visible.
            logger.warning("MusicBrainz artwork request timed out")
            return None
        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed: {e}")
            return None

    async def _try_itunes(
        self,
        artist: str,
        album: str,
        album_id: int
    ) -> str | None:
        """
        Try to download artwork from iTunes Search API.

        Args:
            artist: Artist name
            album: Album name
            album_id: Album ID

        Returns:
            str: Path to downloaded artwork, or None if not found
        """
        try:
            session = self._get_session()
            # Search iTunes
            params = {
                "term": f"{artist} {album}",
                "media": "music",
                "entity": "album",
                "limit": 1
            }

            async with session.get(self.itunes_api, params=params) as resp:  # type: ignore[arg-type]
                if resp.status != 200:
                    return None

                data = await resp.json()
                results = data.get("results", [])

                if not results:
                    return None

                # Get high-res artwork URL (replace 100x100 with larger size)
                artwork_url = results[0].get("artworkUrl100", "")
                if not artwork_url:
                    return None

                # Request larger artwork (600x600)
                artwork_url = artwork_url.replace("100x100", "600x600")

                # Validate artwork URL against trusted domains (fixes #2416: SSRF mitigation)
                if not _validate_artwork_url(artwork_url):
                    logger.warning(f"Rejecting untrusted artwork URL: {artwork_url!r}")
                    return None

            # Download artwork (size-limited, #2576)
            async with session.get(artwork_url) as resp:
                if resp.status != 200:
                    return None

                # Re-check the final URL after aiohttp follows redirects.
                if not _validate_artwork_url(str(resp.url)):
                    logger.warning(f"Rejecting untrusted iTunes redirect: {resp.url!r}")
                    return None

                content_length = resp.content_length or 0
                if content_length > _MAX_ARTWORK_BYTES:
                    logger.warning(f"iTunes artwork too large: {content_length} bytes")
                    return None
                artwork_data = await resp.content.read(_MAX_ARTWORK_BYTES + 1)
                if len(artwork_data) > _MAX_ARTWORK_BYTES:
                    logger.warning(f"iTunes artwork exceeded {_MAX_ARTWORK_BYTES} byte limit")
                    return None
                return await self._save_artwork(artwork_data, album_id, "jpg")

        except TimeoutError:
            # #4686: hit _ARTWORK_REQUEST_TIMEOUT. WARNING, not the debug level
            # other lookup failures use, so a black-holing host is visible.
            logger.warning("iTunes artwork request timed out")
            return None
        except Exception as e:
            logger.debug(f"iTunes lookup failed: {e}")
            return None

    async def _save_artwork(self, data: bytes, album_id: int, ext: str = "jpg") -> str:
        """
        Save artwork data to cache directory.

        Args:
            data: Image data bytes
            album_id: Album ID
            ext: Fallback extension used only when the bytes are unrecognised;
                the real extension is sniffed from magic bytes (#4419).

        Returns:
            str: Path to saved artwork file
        """
        # Sniff the true format from magic bytes so a downloaded PNG/WebP is not
        # mislabelled .jpg and later served with the wrong Content-Type (#4419).
        ext = _detect_image_extension(data, default=ext)

        # Create unique filename based on album ID and data hash
        data_hash = hashlib.md5(data).hexdigest()[:8]
        filename = f"album_{album_id}_{data_hash}.{ext}"
        filepath = self.cache_dir / filename

        # Offload the blocking write so the event loop isn't stalled during
        # bulk artwork backfills (#3915).
        await asyncio.to_thread(filepath.write_bytes, data)

        return str(filepath)

    def clear_cache(self) -> None:
        """Clear all cached artwork files."""
        try:
            for file in self.cache_dir.glob("album_*.*"):
                file.unlink()
            logger.info("Artwork cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear artwork cache: {e}")


# Global instance
_artwork_downloader = None


def get_artwork_downloader() -> ArtworkDownloader:
    """Get or create global artwork downloader instance."""
    global _artwork_downloader
    if _artwork_downloader is None:
        _artwork_downloader = ArtworkDownloader()
    return _artwork_downloader


async def close_artwork_downloader() -> None:
    """Close the global artwork downloader's HTTP session, if one was ever
    created. Call from application shutdown (#3915)."""
    if _artwork_downloader is not None:
        await _artwork_downloader.close()
