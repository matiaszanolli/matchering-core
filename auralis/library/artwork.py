"""
Auralis Artwork Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Album artwork extraction and management for music library

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import hashlib
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.flac import FLAC, Picture
from mutagen.id3 import ID3
from mutagen.mp4 import MP4, MP4Cover

from ..utils.artwork_security import detect_image_extension
from ..utils.logging import debug, error, info

# Cap the largest side of stored artwork (#4439). Embedded/folder covers can be
# arbitrarily large (e.g. 6000x6000); bounding the source here keeps disk usage
# and client-side decode cost in check while retaining more than enough
# resolution for the largest served thumbnail bucket (1024px) and full-size
# display. Matches the GET endpoint's maximum requestable size (le=2048).
_MAX_ARTWORK_DIMENSION = 2048


class ArtworkExtractor:
    """
    Extract and manage album artwork from audio files

    Features:
    - Extract embedded artwork from MP3, FLAC, M4A, OGG
    - Save artwork to organized directory structure
    - Generate unique filenames to avoid collisions
    - Support multiple image formats (JPEG, PNG)
    """

    def __init__(self, artwork_directory: str):
        """
        Initialize artwork extractor

        Args:
            artwork_directory: Base directory for saving artwork
        """
        self.artwork_dir = Path(artwork_directory)
        self.artwork_dir.mkdir(parents=True, exist_ok=True)

    def extract_artwork(self, audio_filepath: str, album_id: int) -> str | None:
        """
        Extract artwork from audio file and save to artwork directory

        Prioritizes embedded artwork in audio files, then falls back to
        folder.jpg/folder.png if no embedded artwork is found.

        Args:
            audio_filepath: Path to audio file
            album_id: Album ID for organizing artwork

        Returns:
            Path to saved artwork file, or None if no artwork found
        """
        try:
            # First priority: Extract embedded artwork from audio file
            audio_file = MutagenFile(audio_filepath)
            if audio_file:
                # Extract artwork data based on file type
                artwork_data = None
                mime_type = None

                if isinstance(audio_file, ID3) or hasattr(audio_file, 'tags') and isinstance(audio_file.tags, ID3):
                    # MP3 files with ID3 tags
                    artwork_data, mime_type = self._extract_from_id3(audio_file)
                elif isinstance(audio_file, MP4):
                    # M4A/MP4 files
                    artwork_data, mime_type = self._extract_from_mp4(audio_file)
                elif isinstance(audio_file, FLAC):
                    # FLAC files
                    artwork_data, mime_type = self._extract_from_flac(audio_file)
                elif hasattr(audio_file, 'tags') and audio_file.tags:
                    # OGG and other formats with generic tags
                    artwork_data, mime_type = self._extract_from_generic(audio_file)

                if artwork_data:
                    # Save artwork to file
                    return self._save_artwork(artwork_data, album_id, mime_type)

            # Fallback: Look for folder artwork if no embedded artwork found
            folder_artwork = self._find_folder_artwork(audio_filepath, album_id)
            if folder_artwork:
                return folder_artwork

            debug(f"No artwork found in {audio_filepath}")
            return None

        except Exception as e:
            debug(f"Failed to extract artwork from {audio_filepath}: {e}")
            return None

    def _find_folder_artwork(self, audio_filepath: str, album_id: int) -> str | None:
        """
        Look for folder.jpg/folder.png or cover.jpg/cover.png in the album directory

        Supports multiple naming conventions:
        - folder.jpg, cover.jpg (standard)
        - front.jpg (common in digital music collections)
        - AlbumArtSmall.jpg (Windows Media Player)
        - Various case variations

        Args:
            audio_filepath: Path to audio file (to determine album directory)
            album_id: Album ID for organizing artwork

        Returns:
            Path to saved artwork file, or None if not found
        """
        try:
            # Get the directory containing the audio file (album folder)
            album_dir = Path(audio_filepath).parent

            # Priority order: check most common artwork filenames first
            # Standard naming (highest priority)
            artwork_names = [
                'folder.jpg', 'folder.jpeg', 'folder.png',
                'Folder.jpg', 'Folder.jpeg', 'Folder.png',
                'cover.jpg', 'cover.jpeg', 'cover.png',
                'Cover.jpg', 'Cover.jpeg', 'Cover.png',
            ]

            # Front artwork (very common in many music libraries)
            artwork_names.extend([
                'front.jpg', 'front.jpeg', 'front.png',
                'Front.jpg', 'Front.jpeg', 'Front.png',
                'FRONT.jpg', 'FRONT.jpeg', 'FRONT.png',
            ])

            # Windows Media Player and other naming conventions
            artwork_names.extend([
                'AlbumArtSmall.jpg', 'AlbumArt.jpg',
                'albumart.jpg', 'albumartsmall.jpg',
            ])

            for artwork_name in artwork_names:
                artwork_file = album_dir / artwork_name
                if artwork_file.exists():
                    # Read the artwork file
                    with open(artwork_file, 'rb') as f:
                        artwork_data = f.read()

                    # Determine MIME type from extension
                    mime_type = 'image/png' if artwork_file.suffix.lower() in ['.png'] else 'image/jpeg'

                    # Save to artwork directory
                    saved_path = self._save_artwork(artwork_data, album_id, mime_type)
                    if saved_path:
                        info(f"Found folder artwork: {artwork_file}")
                        return saved_path

        except Exception as e:
            debug(f"Failed to check for folder artwork: {e}")

        return None

    def _extract_from_id3(self, audio_file: Any) -> tuple[bytes | None, str | None]:
        """Extract artwork from ID3 tags (MP3)"""
        try:
            tags = audio_file if isinstance(audio_file, ID3) else audio_file.tags
            if not tags:
                return None, None

            # Look for APIC (Attached Picture) frames
            for key in tags.keys():
                if key.startswith('APIC'):
                    apic = tags[key]
                    return apic.data, apic.mime

        except Exception as e:
            debug(f"Failed to extract from ID3: {e}")

        return None, None

    def _extract_from_mp4(self, audio_file: MP4) -> tuple[bytes | None, str | None]:
        """Extract artwork from MP4/M4A files"""
        try:
            if audio_file.tags and 'covr' in audio_file.tags:  # type: ignore[unreachable]
                cover = audio_file.tags['covr'][0]  # type: ignore[unreachable]
                # MP4Cover has imageformat attribute
                if cover.imageformat == MP4Cover.FORMAT_JPEG:
                    mime_type = 'image/jpeg'
                elif cover.imageformat == MP4Cover.FORMAT_PNG:
                    mime_type = 'image/png'
                else:
                    mime_type = 'image/jpeg'  # Default
                return bytes(cover), mime_type

        except Exception as e:
            debug(f"Failed to extract from MP4: {e}")

        return None, None

    def _extract_from_flac(self, audio_file: FLAC) -> tuple[bytes | None, str | None]:
        """Extract artwork from FLAC files"""
        try:
            if audio_file.pictures:
                picture = audio_file.pictures[0]  # Get first picture
                return picture.data, picture.mime

        except Exception as e:
            debug(f"Failed to extract from FLAC: {e}")

        return None, None

    def _extract_from_generic(self, audio_file: Any) -> tuple[bytes | None, str | None]:
        """Extract artwork from generic tag formats (OGG, etc.)"""
        try:
            tags = audio_file.tags

            # Try common artwork tag names
            artwork_keys = ['COVERART', 'COVER', 'METADATA_BLOCK_PICTURE']

            import base64

            for key in artwork_keys:
                if key in tags:
                    # Handle base64-encoded artwork (Vorbis comments)
                    if key == 'METADATA_BLOCK_PICTURE':
                        picture_data = base64.b64decode(tags[key][0])
                        # Parse FLAC Picture block
                        picture = Picture(picture_data)  # type: ignore[no-untyped-call]
                        return picture.data, picture.mime
                    else:
                        # Legacy COVERART/COVER store the image as a base64 STRING;
                        # decode to bytes so _save_artwork (hashlib.md5 / binary
                        # write) doesn't TypeError and silently drop the art (#4121).
                        return base64.b64decode(tags[key][0]), 'image/jpeg'

        except Exception as e:
            debug(f"Failed to extract from generic tags: {e}")

        return None, None

    def _bound_dimensions(self, artwork_data: bytes) -> bytes:
        """Downscale artwork whose largest side exceeds ``_MAX_ARTWORK_DIMENSION``.

        Bounds arbitrarily large embedded/folder covers (#4439) so oversized
        images are neither stored nor later streamed at full resolution.
        Best-effort: any failure (Pillow missing, unsupported/corrupt image)
        returns the original bytes unchanged so artwork is never dropped.

        Args:
            artwork_data: Raw image bytes.

        Returns:
            Downscaled image bytes, or the original bytes if no resize was
            needed or possible.
        """
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(artwork_data)) as image:
                if max(image.size) <= _MAX_ARTWORK_DIMENSION:
                    return artwork_data
                # thumbnail() preserves aspect ratio and only ever downsizes.
                image.thumbnail((_MAX_ARTWORK_DIMENSION, _MAX_ARTWORK_DIMENSION))
                # Re-encode in the format Pillow itself detected rather than a
                # caller-supplied extension (#4849) -- forcing every non-PNG
                # format to JPEG here silently produced JPEG bytes under a
                # .webp/.gif filename whenever a large WebP/GIF cover needed
                # downscaling, reproducing this same issue's Content-Type
                # mismatch for exactly the oversized case.
                fmt = image.format or 'JPEG'
                if fmt == 'JPEG' and image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                buffer = io.BytesIO()
                image.save(buffer, format=fmt)
                debug(f"Downscaled oversized artwork to <= {_MAX_ARTWORK_DIMENSION}px")
                return buffer.getvalue()
        except Exception as e:
            debug(f"Artwork dimension-bounding skipped: {e}")
            return artwork_data

    def _save_artwork(self, artwork_data: bytes, album_id: int, mime_type: str | None) -> str | None:
        """
        Save artwork data to file

        Args:
            artwork_data: Raw image data
            album_id: Album ID for filename
            mime_type: MIME type of image (e.g., 'image/jpeg')

        Returns:
            Path to saved artwork file
        """
        try:
            # Bound the stored image to a sane maximum dimension (#4439) so an
            # oversized embedded/folder cover isn't persisted and later streamed
            # at full resolution. Best-effort — falls back to the original bytes.
            artwork_data = self._bound_dimensions(artwork_data)

            # Sniff the real extension from the (possibly re-encoded) bytes
            # rather than trusting the tag's declared mime_type (#4849) —
            # mirrors services/artwork_downloader.py's #4419 fix. Sniffing
            # AFTER bounding means the extension always matches what actually
            # gets written to disk, even when an oversized non-JPEG/PNG image
            # was re-encoded above. mime_type is kept only as the last-resort
            # default for bytes neither sniff recognises.
            default_ext = 'png' if mime_type and 'png' in mime_type.lower() else 'jpg'
            ext = '.' + detect_image_extension(artwork_data, default=default_ext)

            # Generate unique filename using album ID and content hash
            content_hash = hashlib.md5(artwork_data).hexdigest()[:8]
            filename = f"album_{album_id}_{content_hash}{ext}"

            artwork_path = self.artwork_dir / filename

            # Save artwork data
            with open(artwork_path, 'wb') as f:
                f.write(artwork_data)

            info(f"Saved artwork to {artwork_path}")
            return str(artwork_path)

        except Exception as e:
            error(f"Failed to save artwork: {e}")
            return None

    def get_artwork_path(self, album_id: int) -> str | None:
        """
        Get artwork path for album ID

        Args:
            album_id: Album ID

        Returns:
            Path to artwork file if exists, None otherwise
        """
        # Search for artwork files matching the album ID
        pattern = f"album_{album_id}_*"
        matches = list(self.artwork_dir.glob(pattern))

        if matches:
            return str(matches[0])

        return None

    def delete_artwork(self, artwork_path: str) -> bool:
        """
        Delete artwork file

        Args:
            artwork_path: Path to artwork file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            path = Path(artwork_path)
            if path.exists():
                path.unlink()
                info(f"Deleted artwork: {artwork_path}")
                return True
        except Exception as e:
            error(f"Failed to delete artwork {artwork_path}: {e}")

        return False


def create_artwork_extractor(artwork_directory: str) -> ArtworkExtractor:
    """
    Factory function to create artwork extractor

    Args:
        artwork_directory: Base directory for artwork storage

    Returns:
        ArtworkExtractor instance
    """
    return ArtworkExtractor(artwork_directory)
