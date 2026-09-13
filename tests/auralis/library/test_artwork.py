"""
Tests for Artwork Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests the album artwork extraction and management system.

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add auralis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from auralis.library.artwork import ArtworkExtractor, create_artwork_extractor


class TestArtworkExtractorInit:
    """Tests for ArtworkExtractor initialization"""

    def test_initialization(self):
        """Test basic initialization with temp directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            assert extractor.artwork_dir == Path(tmpdir)
            assert extractor.artwork_dir.exists()

    def test_initialization_creates_directory(self):
        """Test that initialization creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            artwork_dir = os.path.join(tmpdir, 'nested', 'artwork')

            extractor = ArtworkExtractor(artwork_dir)

            assert extractor.artwork_dir.exists()
            assert extractor.artwork_dir == Path(artwork_dir)


class TestExtractArtwork:
    """Tests for extract_artwork method"""

    def test_extract_artwork_no_file(self):
        """Test extracting from non-existent file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            with patch('auralis.library.artwork.MutagenFile', return_value=None):
                result = extractor.extract_artwork('/path/to/nonexistent.mp3', album_id=1)

                assert result is None

    def test_extract_artwork_exception_handling(self):
        """Test that exceptions are handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            with patch('auralis.library.artwork.MutagenFile', side_effect=Exception("Test error")):
                result = extractor.extract_artwork('/path/to/test.mp3', album_id=1)

                assert result is None


class TestExtractFromID3:
    """Tests for _extract_from_id3 method"""

    def test_extract_from_id3_no_apic(self):
        """Test extracting from ID3 without APIC frame"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            mock_tags = MagicMock()
            mock_tags.keys.return_value = ['TIT2', 'TPE1']

            data, mime = extractor._extract_from_id3(mock_tags)

            assert data is None
            assert mime is None

    def test_extract_from_id3_exception(self):
        """Test _extract_from_id3 handles exceptions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            mock_tags = MagicMock()
            mock_tags.keys.side_effect = Exception("Test error")

            data, mime = extractor._extract_from_id3(mock_tags)

            assert data is None
            assert mime is None


class TestExtractFromMP4:
    """Tests for _extract_from_mp4 method"""

    def test_extract_from_mp4_jpeg(self):
        """Test extracting JPEG artwork from MP4"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            # Create mock cover that returns bytes when bytes() is called
            mock_cover = b'jpeg_data'

            # Mock the imageformat attribute
            mock_cover_obj = MagicMock()
            mock_cover_obj.imageformat = 13  # MP4Cover.FORMAT_JPEG
            # Make bytes(mock_cover_obj) work
            mock_cover_obj.__class__.__bytes__ = lambda self: mock_cover

            mock_audio = MagicMock()
            mock_audio.tags = {'covr': [mock_cover_obj]}

            with patch('auralis.library.artwork.MP4Cover.FORMAT_JPEG', 13):
                data, mime = extractor._extract_from_mp4(mock_audio)

                assert data == b'jpeg_data'
                assert mime == 'image/jpeg'

    def test_extract_from_mp4_png(self):
        """Test extracting PNG artwork from MP4"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            # Create mock cover that returns bytes when bytes() is called
            mock_cover = b'png_data'

            # Mock the imageformat attribute
            mock_cover_obj = MagicMock()
            mock_cover_obj.imageformat = 14  # MP4Cover.FORMAT_PNG
            # Make bytes(mock_cover_obj) work
            mock_cover_obj.__class__.__bytes__ = lambda self: mock_cover

            mock_audio = MagicMock()
            mock_audio.tags = {'covr': [mock_cover_obj]}

            with patch('auralis.library.artwork.MP4Cover.FORMAT_PNG', 14):
                data, mime = extractor._extract_from_mp4(mock_audio)

                assert data == b'png_data'
                assert mime == 'image/png'

    def test_extract_from_mp4_no_cover(self):
        """Test extracting from MP4 without cover art"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            mock_audio = MagicMock()
            mock_audio.tags = {}

            data, mime = extractor._extract_from_mp4(mock_audio)

            assert data is None
            assert mime is None


class TestExtractFromFLAC:
    """Tests for _extract_from_flac method"""

    def test_extract_from_flac_with_pictures(self):
        """Test extracting from FLAC with embedded pictures"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            mock_picture = MagicMock()
            mock_picture.data = b'flac_image_data'
            mock_picture.mime = 'image/jpeg'

            mock_audio = MagicMock()
            mock_audio.pictures = [mock_picture]

            data, mime = extractor._extract_from_flac(mock_audio)

            assert data == b'flac_image_data'
            assert mime == 'image/jpeg'

    def test_extract_from_flac_no_pictures(self):
        """Test extracting from FLAC without pictures"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            mock_audio = MagicMock()
            mock_audio.pictures = []

            data, mime = extractor._extract_from_flac(mock_audio)

            assert data is None
            assert mime is None


class TestSaveArtwork:
    """Tests for _save_artwork method"""

    def test_save_artwork_jpeg(self):
        """Test saving JPEG artwork"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            artwork_data = b'jpeg_image_data'
            album_id = 123

            result = extractor._save_artwork(artwork_data, album_id, 'image/jpeg')

            assert result is not None
            assert 'album_123_' in result
            assert result.endswith('.jpg')
            assert Path(result).exists()

    def test_save_artwork_png(self):
        """Test saving PNG artwork"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            artwork_data = b'png_image_data'
            album_id = 456

            result = extractor._save_artwork(artwork_data, album_id, 'image/png')

            assert result is not None
            assert 'album_456_' in result
            assert result.endswith('.png')

    def test_save_artwork_default_extension(self):
        """Test saving artwork with unknown MIME type defaults to JPG"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            artwork_data = b'unknown_image_data'
            album_id = 789

            result = extractor._save_artwork(artwork_data, album_id, 'image/unknown')

            assert result is not None
            assert result.endswith('.jpg')

    def test_save_artwork_same_content_same_hash(self):
        """Test that same content produces same hash in filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            artwork_data = b'identical_data'
            album_id = 100

            result1 = extractor._save_artwork(artwork_data, album_id, 'image/jpeg')
            result2 = extractor._save_artwork(artwork_data, album_id, 'image/jpeg')

            # Same album + same content = same filename
            assert Path(result1).name == Path(result2).name


class TestBoundDimensions:
    """Tests for extract-time artwork dimension bounding (#4439)"""

    @staticmethod
    def _encode(width: int, height: int, fmt: str = "JPEG") -> bytes:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format=fmt)
        return buffer.getvalue()

    def _dimensions(self, data: bytes) -> tuple[int, int]:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return image.size

    def test_oversized_image_is_downscaled(self):
        """A cover larger than the cap is downscaled so its largest side fits."""
        from auralis.library.artwork import _MAX_ARTWORK_DIMENSION

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)
            oversized = self._encode(4000, 3000)

            saved = extractor._save_artwork(oversized, album_id=1, mime_type="image/jpeg")

            assert saved is not None
            width, height = self._dimensions(Path(saved).read_bytes())
            assert max(width, height) <= _MAX_ARTWORK_DIMENSION
            # Aspect ratio preserved (4:3)
            assert width > height

    def test_small_image_is_left_untouched(self):
        """An image already within the cap is stored byte-for-byte unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)
            small = self._encode(300, 300)

            saved = extractor._save_artwork(small, album_id=2, mime_type="image/jpeg")

            assert saved is not None
            assert Path(saved).read_bytes() == small

    def test_non_image_bytes_are_preserved(self):
        """Best-effort: undecodable bytes fall through unchanged (artwork never dropped)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)
            junk = b"not-a-real-image"

            saved = extractor._save_artwork(junk, album_id=3, mime_type="image/jpeg")

            assert saved is not None
            assert Path(saved).read_bytes() == junk


class TestSaveArtworkExtensionSniffing:
    """The saved extension matches the real bytes, not the declared mime_type (#4849).

    #4419 fixed this for online-downloaded artwork (services/artwork_downloader.py);
    the embedded/folder extractor here had the identical defect and only ever
    special-cased 'png', defaulting GIF/WebP (and anything else) to '.jpg'.
    """

    def test_gif_bytes_saved_with_gif_extension_despite_mime_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)
            gif_data = b"GIF89a" + b"\x00" * 32

            result = extractor._save_artwork(gif_data, album_id=1, mime_type="image/gif")

            assert result is not None
            assert result.endswith(".gif")

    def test_webp_bytes_saved_with_webp_extension_even_when_mime_disagrees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)
            # A tag whose declared MIME disagrees with the actual bytes --
            # exactly what the old mime_type-trusting code got wrong.
            webp_data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 8

            result = extractor._save_artwork(webp_data, album_id=2, mime_type="image/jpeg")

            assert result is not None
            assert result.endswith(".webp")

    def test_oversized_webp_stays_webp_after_downscaling(self):
        """The re-encode-on-downscale path must not silently convert to JPEG.

        _bound_dimensions used to force every non-PNG format to JPEG when
        resizing, which would have reproduced this exact bug for an oversized
        WebP/GIF cover: correct extension chosen up front, then overwritten
        with JPEG bytes during the resize.
        """
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (4000, 3000), (5, 5, 5)).save(buffer, format="WEBP")
        oversized_webp = buffer.getvalue()

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            result = extractor._save_artwork(oversized_webp, album_id=3, mime_type="image/webp")

            assert result is not None
            assert result.endswith(".webp")
            from auralis.library.artwork import _MAX_ARTWORK_DIMENSION
            with Image.open(result) as saved:
                assert saved.format == "WEBP"
                assert max(saved.size) <= _MAX_ARTWORK_DIMENSION


class TestGetArtworkPath:
    """Tests for get_artwork_path method"""

    def test_get_artwork_path_exists(self):
        """Test getting path for existing artwork"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            # Create a test artwork file
            artwork_data = b'test_data'
            album_id = 42
            saved_path = extractor._save_artwork(artwork_data, album_id, 'image/jpeg')

            # Now retrieve it
            retrieved_path = extractor.get_artwork_path(album_id)

            assert retrieved_path is not None
            assert retrieved_path == saved_path

    def test_get_artwork_path_not_exists(self):
        """Test getting path for non-existent artwork"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            result = extractor.get_artwork_path(album_id=999)

            assert result is None


class TestDeleteArtwork:
    """Tests for delete_artwork method"""

    def test_delete_artwork_exists(self):
        """Test deleting existing artwork"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            # Create artwork file
            artwork_data = b'test_data'
            saved_path = extractor._save_artwork(artwork_data, 1, 'image/jpeg')

            # Delete it
            result = extractor.delete_artwork(saved_path)

            assert result is True
            assert not Path(saved_path).exists()

    def test_delete_artwork_not_exists(self):
        """Test deleting non-existent artwork"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            result = extractor.delete_artwork('/path/to/nonexistent.jpg')

            assert result is False

    def test_delete_artwork_exception(self):
        """Test delete handles exceptions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            with patch('pathlib.Path.unlink', side_effect=PermissionError("Access denied")):
                # Create a file to attempt deletion
                test_file = Path(tmpdir) / 'test.jpg'
                test_file.write_bytes(b'data')

                result = extractor.delete_artwork(str(test_file))

                assert result is False


class TestFactoryFunction:
    """Tests for create_artwork_extractor factory"""

    def test_create_artwork_extractor(self):
        """Test factory function creates ArtworkExtractor instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = create_artwork_extractor(tmpdir)

            assert isinstance(extractor, ArtworkExtractor)
            assert extractor.artwork_dir == Path(tmpdir)


class TestExtractFromGeneric:
    """Tests for _extract_from_generic (OGG/Vorbis comment artwork)"""

    def test_legacy_coverart_returns_decoded_bytes(self):
        """Legacy COVERART tags store base64 text; the extractor must return
        decoded bytes (not the str) so _save_artwork's md5/binary-write work (#4121)."""
        import base64

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)

            image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
            mock_audio = MagicMock()
            mock_audio.tags = {"COVERART": [base64.b64encode(image_bytes).decode("ascii")]}

            data, mime = extractor._extract_from_generic(mock_audio)

            assert isinstance(data, bytes)  # not the raw base64 str
            assert data == image_bytes
            assert mime == "image/jpeg"

    def test_no_artwork_keys_returns_none(self):
        """No artwork tag → (None, None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = ArtworkExtractor(tmpdir)
            mock_audio = MagicMock()
            mock_audio.tags = {"TITLE": ["Song"]}

            data, mime = extractor._extract_from_generic(mock_audio)

            assert data is None
            assert mime is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
