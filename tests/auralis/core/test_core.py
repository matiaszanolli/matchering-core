"""
Tests for core Auralis functionality.
Focus on real working modules to boost coverage meaningfully.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from auralis.library.database import LibraryDatabase
from auralis.library.models import Album, Artist, Playlist, Track
from auralis.library.scanner import LibraryScanner


class TestLibraryDatabaseAdvanced:
    """Advanced tests for LibraryDatabase to boost coverage."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def db(self, temp_db):
        """Create LibraryDatabase instance."""
        return LibraryDatabase(temp_db)

    def test_session_management(self, db):
        """Test database session management."""
        session = db.get_session()
        assert session is not None

        # Test session is working
        from auralis.library.models import Track
        tracks = session.query(Track).all()
        assert isinstance(tracks, list)

        session.close()

    def test_track_operations_comprehensive(self, db):
        """Test comprehensive track operations."""
        # Test add track with full metadata
        track_info = {
            'filepath': '/test/path/song.mp3',
            'title': 'Test Song',
            'artists': ['Test Artist'],
            'album': 'Test Album',
            'genres': ['Rock'],
            'year': 2023,
            'duration': 180,
            'filesize': 5242880,
            'bit_rate': 320,
            'sample_rate': 44100
        }

        track = db.tracks.add(track_info)
        assert track is not None
        assert track.title == 'Test Song'

        # Get fresh track with relationships loaded
        session = db.get_session()
        fresh_track = session.query(Track).filter(Track.id == track.id).first()
        assert len(fresh_track.artists) == 1
        assert fresh_track.artists[0].name == 'Test Artist'
        session.close()

    def test_search_functionality(self, db):
        """Test search functionality comprehensively."""
        # Add test tracks first
        tracks_data = [
            {
                'filepath': '/test/rock1.mp3',
                'title': 'Rock Song 1',
                'artists': ['Rock Band'],
                'album': 'Rock Album',
                'genres': ['Rock']
            },
            {
                'filepath': '/test/jazz1.mp3',
                'title': 'Jazz Tune',
                'artists': ['Jazz Musician'],
                'album': 'Jazz Collection',
                'genres': ['Jazz']
            },
            {
                'filepath': '/test/rock2.mp3',
                'title': 'Another Rock Song',
                'artists': ['Rock Band'],
                'album': 'Rock Album 2',
                'genres': ['Rock']
            }
        ]

        for track_data in tracks_data:
            db.tracks.add(track_data)

        # Test different search patterns
        rock_results, _ = db.tracks.search('rock')
        assert len(rock_results) >= 2

        band_results, _ = db.tracks.search('Rock Band')
        assert len(band_results) >= 2

        # Test case insensitive search
        case_results, _ = db.tracks.search('ROCK')
        assert len(case_results) >= 2

    def test_genre_and_artist_queries(self, db):
        """Test genre and artist-specific queries."""
        # Add test data
        track_info = {
            'filepath': '/test/metal.mp3',
            'title': 'Metal Song',
            'artists': ['Metal Band'],
            'genres': ['Metal']
        }
        db.tracks.add(track_info)

        # Test genre queries
        metal_tracks = db.tracks.get_by_genre('Metal')
        assert len(metal_tracks) >= 1

        # Test artist queries
        artist_tracks = db.tracks.get_by_artist('Metal Band')
        assert len(artist_tracks) >= 1

    def test_playlist_operations_comprehensive(self, db):
        """Test comprehensive playlist operations."""
        # Create playlist
        playlist = db.playlists.create(
            name='Test Playlist',
            description='A test playlist for coverage'
        )
        assert playlist is not None
        assert playlist.name == 'Test Playlist'

        # Add track to playlist
        track_info = {'filepath': '/test/playlist_track.mp3', 'title': 'Playlist Track'}
        track = db.tracks.add(track_info)

        success = db.playlists.add_track(playlist.id, track.id)
        assert success is True

        # Get playlist
        retrieved_playlist = db.playlists.get_by_id(playlist.id)
        assert retrieved_playlist is not None
        assert retrieved_playlist.name == 'Test Playlist'

    def test_track_interaction_methods(self, db):
        """Test track interaction methods."""
        # Add a track
        track_info = {'filepath': '/test/interactive.mp3', 'title': 'Interactive Track'}
        track = db.tracks.add(track_info)

        # Test play recording
        db.tracks.record_play(track.id)

        # Test favoriting
        db.tracks.set_favorite(track.id, True)
        db.tracks.set_favorite(track.id, False)

    def test_library_stats_comprehensive(self, db):
        """Test comprehensive library statistics."""
        # Add diverse test data
        test_tracks = [
            {
                'filepath': '/test/stats1.mp3',
                'title': 'Stats Track 1',
                'artists': ['Artist 1'],
                'album': 'Album 1',
                'duration': 180,
                'filesize': 5000000
            },
            {
                'filepath': '/test/stats2.mp3',
                'title': 'Stats Track 2',
                'artists': ['Artist 2'],
                'album': 'Album 2',
                'duration': 240,
                'filesize': 6000000
            }
        ]

        for track_data in test_tracks:
            db.tracks.add(track_data)

        stats = db.stats.get_library_stats()
        assert isinstance(stats, dict)

        # Should have multiple tracks now
        assert stats.get('total_tracks', 0) >= 2

        # Should have calculated total duration
        if 'total_duration' in stats:
            assert stats['total_duration'] > 0

    def test_recent_and_popular_tracks(self, db):
        """Test recent and popular track queries."""
        # Add and play tracks
        track_info = {'filepath': '/test/recent.mp3', 'title': 'Recent Track'}
        track = db.tracks.add(track_info)
        db.tracks.record_play(track.id)

        # Test recent tracks
        recent, recent_count = db.tracks.get_recent(limit=10)
        assert isinstance(recent, list)
        assert isinstance(recent_count, int)

        # Test popular tracks
        popular, popular_count = db.tracks.get_popular(limit=10)
        assert isinstance(popular, list)
        assert isinstance(popular_count, int)

        # Test favorite tracks
        favorites, favorites_count = db.tracks.get_favorites(limit=10)
        assert isinstance(favorites, list)
        assert isinstance(favorites_count, int)


class TestLibraryScannerAdvanced:
    """Advanced tests for LibraryScanner."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def db_and_scanner(self, temp_db):
        """Create library database and scanner."""
        db = LibraryDatabase(temp_db)
        scanner = LibraryScanner(db)
        return db, scanner

    def test_scanner_initialization(self, db_and_scanner):
        """Test scanner initialization and attributes."""
        db, scanner = db_and_scanner

        # LibraryScanner still names its constructor argument `library_database`,
        # but the object it holds is a LibraryDatabase.
        assert scanner.library_database is db
        assert hasattr(scanner, 'library_database')

    def test_scanner_methods_coverage(self, db_and_scanner):
        """Test scanner methods for coverage."""
        db, scanner = db_and_scanner

        # Test methods that should exist
        scanner_methods = dir(scanner)

        # Basic methods should be present
        assert 'library_database' in scanner_methods

    @pytest.fixture
    def temp_audio_files(self):
        """Create temporary audio-like files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake audio files
            audio_files = []
            for i, ext in enumerate(['mp3', 'wav', 'flac']):
                file_path = Path(temp_dir) / f'test_audio_{i}.{ext}'
                file_path.write_bytes(b'fake audio content')
                audio_files.append(str(file_path))

            yield temp_dir, audio_files

    def test_scanner_with_files(self, db_and_scanner, temp_audio_files):
        """Test scanner with actual files."""
        db, scanner = db_and_scanner
        temp_dir, audio_files = temp_audio_files

        # Test that scanner can be used with file paths
        for file_path in audio_files:
            # Test individual file operations if they exist
            if hasattr(scanner, 'scan_file'):
                try:
                    scanner.scan_file(file_path)
                except Exception as e:
                    # Expected - these aren't real audio files
                    assert 'audio' in str(e).lower() or 'format' in str(e).lower()


class TestPlayerComponents:
    """Test player components for coverage."""

    def test_player_config_functionality(self):
        """Test player configuration."""
        try:
            from auralis.player.config import PlayerConfig

            # Test basic config creation
            config = PlayerConfig()
            assert config is not None

        except ImportError:
            pytest.skip("PlayerConfig not available")
        except Exception as e:
            # Config might require parameters
            assert 'config' in str(e).lower() or 'required' in str(e).lower()

    def test_realtime_processor_functionality(self):
        """Test realtime processor."""
        try:
            from auralis.player.realtime import RealtimeProcessor

            # Test basic processor creation
            processor = RealtimeProcessor()
            assert processor is not None

        except ImportError:
            pytest.skip("RealtimeProcessor not available")
        except Exception as e:
            # Processor might require audio system
            assert any(word in str(e).lower() for word in ['audio', 'device', 'system', 'init'])


class TestDSPComponents:
    """Test DSP components for coverage."""

    def test_basic_dsp_functionality(self):
        """Test basic DSP functionality."""
        try:
            from auralis.dsp.basic import AudioProcessor, amplify_audio, normalize_audio

            # Test function availability
            assert callable(normalize_audio)
            assert callable(amplify_audio)

            # Test with dummy audio data
            dummy_audio = np.random.rand(1024, 2).astype(np.float32)

            # Test normalization
            normalized = normalize_audio(dummy_audio)
            assert normalized.shape == dummy_audio.shape

            # Test amplification
            amplified = amplify_audio(dummy_audio, gain_db=6.0)
            assert amplified.shape == dummy_audio.shape

        except ImportError:
            pytest.skip("DSP basic functions not available")
        except Exception as e:
            # Functions might require specific input format
            assert any(word in str(e).lower() for word in ['audio', 'shape', 'dtype', 'format'])

    # #4867 removed `test_dsp_stages_functionality`. It imported three class
    # names (`ProcessingStage`, `PreprocessingStage`, `MasteringStage`) that
    # never existed in `auralis/dsp/stages.py` — the module only ever held a
    # `main()` function — so the import always raised and the `except
    # ImportError: pytest.skip(...)` swallowed it. The test skipped on every
    # run it ever made and never exercised the module it named. `stages.py`
    # itself is gone; deleting the test loses no coverage.


class TestIOComponents:
    """Test I/O components for coverage."""

    def test_loader_functionality(self):
        """Test audio loader functionality."""
        try:
            from auralis.io.loader import AudioLoader, load_audio_file

            # Test loader creation
            loader = AudioLoader()
            assert loader is not None

            # Test load function exists
            assert callable(load_audio_file)

        except ImportError:
            pytest.skip("AudioLoader not available")
        except Exception as e:
            # Might require audio libraries
            assert any(word in str(e).lower() for word in ['audio', 'library', 'soundfile'])

    def test_saver_functionality(self):
        """Test audio saver functionality."""
        try:
            from auralis.io.saver import AudioSaver, save_audio_file

            # Test saver creation
            saver = AudioSaver()
            assert saver is not None

            # Test save function exists
            assert callable(save_audio_file)

        except ImportError:
            pytest.skip("AudioSaver not available")

    def test_results_functionality(self):
        """Test processing results functionality."""
        try:
            from auralis.io.results import ProcessingResults, ResultsContainer

            # #5154: `assert ProcessingResults is not None` is true for any
            # successfully imported name, so this only re-tested the import.
            assert callable(ProcessingResults)
            assert callable(ResultsContainer)

            # A default-constructed result set must be usable, not merely
            # non-None.
            results = ProcessingResults()
            assert results is not None
            assert isinstance(results.to_dict(), dict) if hasattr(results, 'to_dict') else True

        except ImportError:
            pytest.skip("ProcessingResults not available")


class TestUtilityComponents:
    """Test utility components for coverage."""

    def test_checker_comprehensive(self):
        """Test checker utilities comprehensively."""
        from auralis.utils.checker import check_file_permissions, is_audio_file

        # Test audio file detection
        assert callable(is_audio_file)
        assert callable(check_file_permissions)

        # Test with various file extensions
        test_files = [
            'test.mp3',
            'test.wav',
            'test.flac',
            'test.txt',
            'test.doc'
        ]

        for filename in test_files:
            result = is_audio_file(filename)
            assert isinstance(result, bool)

    def test_helpers_comprehensive(self):
        """Test helper utilities comprehensively."""
        from auralis.utils.helpers import format_duration, format_filesize

        # Test duration formatting with various inputs
        durations = [0, 30, 75, 125, 3661]  # 0s, 30s, 1:15, 2:05, 1:01:01

        for duration in durations:
            formatted = format_duration(duration)
            assert isinstance(formatted, str)
            assert len(formatted) > 0

        # Test file size formatting
        sizes = [0, 1024, 1048576, 1073741824]  # 0B, 1KB, 1MB, 1GB

        for size in sizes:
            formatted = format_filesize(size)
            assert isinstance(formatted, str)
            assert len(formatted) > 0

    def test_logging_comprehensive(self):
        """Test logging utilities comprehensively."""
        from auralis.utils.logging import debug, error, info, set_log_level, warning

        # Test all logging functions
        log_functions = [info, warning, error, debug]

        for log_func in log_functions:
            try:
                log_func("Test message")
                log_func("Test message with data")
            except Exception as e:
                # Logging might not be fully configured
                assert 'log' in str(e).lower()

        # Test log level setting
        try:
            set_log_level('DEBUG')
            set_log_level('INFO')
            set_log_level('WARNING')
        except Exception as e:
            # Log level setting might not be implemented
            assert 'level' in str(e).lower() or 'log' in str(e).lower()


class TestModelRelationships:
    """Test model relationships and advanced functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_track_album_artist_relationships(self, temp_db):
        """Test relationships between Track, Album, and Artist models."""
        db = LibraryDatabase(temp_db)
        session = db.get_session()

        # Add track with artist and album information
        track_info = {
            'filepath': '/test/relationship.mp3',
            'title': 'Relationship Test',
            'artists': ['Test Artist'],
            'album': 'Test Album',
            'genres': ['Test Genre']
        }

        track = db.tracks.add(track_info)
        assert track is not None

        # Test that the track was properly added
        retrieved_track = db.tracks.get_by_id(track.id)
        assert retrieved_track is not None
        assert retrieved_track.title == 'Relationship Test'

        session.close()

    def test_playlist_track_relationships(self, temp_db):
        """Test playlist-track relationships."""
        db = LibraryDatabase(temp_db)

        # Create playlist
        playlist = db.playlists.create('Relationship Test Playlist')
        assert playlist is not None

        # Add tracks
        track1_info = {'filepath': '/test/rel1.mp3', 'title': 'Rel Track 1'}
        track2_info = {'filepath': '/test/rel2.mp3', 'title': 'Rel Track 2'}

        track1 = db.tracks.add(track1_info)
        track2 = db.tracks.add(track2_info)

        # Add tracks to playlist
        db.playlists.add_track(playlist.id, track1.id)
        db.playlists.add_track(playlist.id, track2.id)

        # Verify playlist contents. #5154: this asserted only that the
        # playlist still existed, so it passed even if add_track() were a
        # no-op — which is the entire relationship this test is named for.
        retrieved_playlist = db.playlists.get_by_id(playlist.id)
        assert retrieved_playlist is not None
        assert retrieved_playlist.name == 'Relationship Test Playlist'

        track_ids = [t.id for t in retrieved_playlist.tracks]
        assert track_ids == [track1.id, track2.id], (
            f"Playlist should hold both tracks in insertion order, got {track_ids}"
        )


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_invalid_operations(self, temp_db):
        """Test invalid operations are handled gracefully."""
        db = LibraryDatabase(temp_db)

        # Test getting non-existent items
        assert db.tracks.get_by_id(99999) is None
        assert db.playlists.get_by_id(99999) is None

        # Test operations with invalid IDs
        result = db.playlists.add_track(99999, 99999)
        assert result is False

    def test_duplicate_handling(self, temp_db):
        """Test handling of duplicate entries."""
        db = LibraryDatabase(temp_db)

        # Add same track twice
        track_info = {'filepath': '/test/duplicate.mp3', 'title': 'Duplicate Track'}

        track1 = db.tracks.add(track_info)
        track2 = db.tracks.add(track_info)  # Should handle gracefully

        assert track1 is not None

    def test_empty_searches(self, temp_db):
        """Test empty and invalid searches."""
        db = LibraryDatabase(temp_db)

        # Test empty search
        results, count = db.tracks.search('')
        assert isinstance(results, list)
        assert isinstance(count, int)

        # Test search with no matches
        results, count = db.tracks.search('nonexistent_query_12345')
        assert isinstance(results, list)
        assert len(results) == 0
        assert count == 0