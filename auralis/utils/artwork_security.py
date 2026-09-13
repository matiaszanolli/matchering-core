"""Shared safety limits for external artwork URLs and payloads."""

from urllib.parse import urlparse

# Keep external artwork responses bounded before parsing or persisting them.
MAX_ARTWORK_PAYLOAD_BYTES = 5 * 1024 * 1024


# Domains used by the artwork sources supported by ArtworkService and
# ArtworkDownloader. Exact hosts and their subdomains are accepted.
TRUSTED_ARTWORK_DOMAINS = frozenset({
    "mzstatic.com",
    "coverartarchive.org",
    "archive.org",
    "lastfm.freetls.fastly.net",
    "lastfm-img2.akamaized.net",
    "i.discogs.com",
    "img.discogs.com",
    "upload.wikimedia.org",
    "commons.wikimedia.org",
})


def validate_artwork_url(url: str) -> bool:
    """Return whether *url* uses HTTP(S) on a trusted artwork host."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            return False

        hostname = parsed.hostname.lower()
        return any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in TRUSTED_ARTWORK_DOMAINS
        )
    except (TypeError, ValueError):
        return False


def detect_image_extension(data: bytes, default: str = "jpg") -> str:
    """Pick a file extension (without a leading dot) from an image's magic bytes.

    Both artwork write paths — online downloads (``artwork_downloader.py``)
    and embedded/folder extraction (``library/artwork.py``) — used to trust a
    caller-supplied label (a requested format, or a tag's declared MIME) when
    naming the saved file. Cover Art Archive/iTunes can return PNG/WebP for a
    JPEG request, and ID3/FLAC picture tags can declare GIF/WebP MIME types;
    either way the GET endpoint infers ``Content-Type`` from the extension, so
    a mislabelled file is served with the wrong Content-Type (#4419, #4849).
    Sniffing the real bytes here is shared so both write paths stay in sync.
    Falls back to ``default`` for unrecognised bytes.
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
