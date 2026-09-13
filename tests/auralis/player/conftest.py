"""
Player Component Test Configuration

Provides fixtures for player component testing with RepositoryFactory pattern.
Used in Phase 5A.3 test suite migration.
"""

from typing import Any, Callable

import pytest


@pytest.fixture
def queue_controller(get_repository_factory_callable):
    """Create QueueController with RepositoryFactory.

    This fixture provides a QueueController instance properly configured
    with the RepositoryFactory dependency injection pattern.

    Phase 5A.3: Player component fixtures for test suite migration.

    Args:
        get_repository_factory_callable: Callable that returns RepositoryFactory

    Returns:
        QueueController: Configured controller instance

    Example:
        def test_queue_operations(queue_controller):
            queue_controller.add_track({"title": "Test", "filepath": "/tmp/test.wav"})
            assert queue_controller.get_track_count() == 1
    """
    from auralis.player.queue_controller import QueueController

    return QueueController(get_repository_factory_callable)


@pytest.fixture
def playback_controller():
    """Create PlaybackController for testing.

    Provides a simple playback state machine for player component tests.

    Returns:
        PlaybackController: Configured controller instance
    """
    from auralis.player.playback_controller import PlaybackController

    return PlaybackController()


@pytest.fixture
def audio_file_manager():
    """Create AudioFileManager for testing.

    Provides file I/O handling for player component tests.

    Returns:
        AudioFileManager: Configured manager instance
    """
    from auralis.player.audio_file_manager import AudioFileManager

    return AudioFileManager(sample_rate=44100)


@pytest.fixture
def realtime_processor():
    """Create RealtimeProcessor for testing.

    Provides DSP processing for player component tests.

    Returns:
        RealtimeProcessor: Configured processor instance
    """
    from auralis.player.config import PlayerConfig
    from auralis.player.realtime import RealtimeProcessor

    config = PlayerConfig()
    return RealtimeProcessor(config)


@pytest.fixture
def gapless_playback_engine(audio_file_manager, queue_controller):
    """Create GaplessPlaybackEngine for testing.

    Provides gapless playback handling with prebuffering.

    Args:
        audio_file_manager: AudioFileManager fixture
        queue_controller: QueueController fixture

    Returns:
        GaplessPlaybackEngine: Configured engine instance
    """
    from auralis.player.gapless_playback_engine import GaplessPlaybackEngine

    return GaplessPlaybackEngine(audio_file_manager, queue_controller)


@pytest.fixture
def integration_manager(
    playback_controller,
    audio_file_manager,
    queue_controller,
    realtime_processor,
    get_repository_factory_callable
):
    """Create IntegrationManager with RepositoryFactory.

    Provides library integration and callback coordination for player tests.

    Phase 5A.3: Player component fixture with dependency injection.

    Args:
        playback_controller: PlaybackController fixture
        audio_file_manager: AudioFileManager fixture
        queue_controller: QueueController fixture
        realtime_processor: RealtimeProcessor fixture
        get_repository_factory_callable: Callable that returns RepositoryFactory

    Returns:
        IntegrationManager: Configured manager instance

    Example:
        def test_track_loading(integration_manager):
            success = integration_manager.load_track_from_library(track_id=1)
            assert isinstance(success, bool)
    """
    from auralis.player.integration_manager import IntegrationManager

    return IntegrationManager(
        playback_controller,
        audio_file_manager,
        queue_controller,
        realtime_processor,
        get_repository_factory_callable
    )


@pytest.fixture
def enhanced_player(get_repository_factory_callable):
    """Create AudioPlayer with RepositoryFactory.

    This is the main fixture for testing the enhanced player facade.
    It provides a fully configured player with all components properly injected.

    Phase 5A.3: Player component fixture for comprehensive player testing.

    Args:
        get_repository_factory_callable: Callable that returns RepositoryFactory

    Returns:
        AudioPlayer: Configured player instance

    Example:
        def test_player_playback(enhanced_player):
            assert enhanced_player.load_file("/path/to/audio.wav")
            assert enhanced_player.play()
            assert enhanced_player.is_playing()
    """
    from auralis.player.config import PlayerConfig
    from auralis.player.enhanced_audio_player import AudioPlayer

    config = PlayerConfig()
    return AudioPlayer(
        config=config,
        get_repository_factory=get_repository_factory_callable
    )


@pytest.fixture
def player_config():
    """Create PlayerConfig for testing.

    Provides standard configuration for player component tests.

    Returns:
        PlayerConfig: Standard player configuration
    """
    from auralis.player.config import PlayerConfig

    return PlayerConfig()
