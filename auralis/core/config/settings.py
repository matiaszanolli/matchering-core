"""
Configuration Settings
~~~~~~~~~~~~~~~~~~~~~

Configuration dataclasses for audio processing

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LimiterConfig:
    """Configuration for the audio limiter"""
    attack: float = 1.0
    hold: float = 1.0
    release: float = 3000.0
    attack_filter_coefficient: float = -2.0
    hold_filter_order: int = 1
    hold_filter_coefficient: float = 7.0
    release_filter_order: int = 1
    release_filter_coefficient: float = 800.0

    def __post_init__(self) -> None:
        assert self.attack > 0, "Attack time must be positive"
        assert self.hold > 0, "Hold time must be positive"
        assert self.release > 0, "Release time must be positive"
        assert self.hold_filter_order > 0, "Hold filter order must be positive"
        assert self.release_filter_order > 0, "Release filter order must be positive"


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive processing features"""
    # Processing mode
    mode: Literal["reference", "adaptive", "hybrid"] = "adaptive"

    # Content analysis settings
    enable_genre_detection: bool = True
    enable_tempo_analysis: bool = True
    enable_energy_analysis: bool = True

    # Adaptation settings
    adaptation_strength: float = 0.8  # 0.0 = no adaptation, 1.0 = full adaptation
    parameter_smoothing: float = 0.1  # Smoothing factor for parameter changes

    # Real-time processing
    chunk_size_ms: float = 20.0  # Processing chunk size in milliseconds
    latency_budget_ms: float = 20.0  # Maximum allowed latency

    # Quality control
    enable_quality_monitoring: bool = True
    auto_quality_adjustment: bool = True
    min_quality_level: Literal["basic", "medium", "high", "maximum"] = "medium"

    # User learning
    enable_user_learning: bool = True
    learning_rate: float = 0.05  # How quickly to adapt to user preferences

    # Psychoacoustic modeling
    enable_psychoacoustic_eq: bool = True
    # No `critical_bands` knob (#4613): the EQ's band layout is the fixed
    # 25-band Bark-scale table in auralis/dsp/eq/critical_bands.py, which
    # PsychoacousticEQ builds with no arguments. A validated-but-ignored field
    # (default 26, range 8-64) used to sit here and looked tunable.

    def __post_init__(self) -> None:
        assert 0.0 <= self.adaptation_strength <= 1.0, "Adaptation strength must be 0-1"
        assert 0.0 <= self.parameter_smoothing <= 1.0, "Parameter smoothing must be 0-1"
        assert 5.0 <= self.chunk_size_ms <= 100.0, "Chunk size must be 5-100ms"
        assert 10.0 <= self.latency_budget_ms <= 100.0, "Latency budget must be 10-100ms"
        assert 0.0 <= self.learning_rate <= 1.0, "Learning rate must be 0-1"


@dataclass
class GenreProfile:
    """Processing profile for a specific genre"""
    name: str
    target_lufs: float
    bass_boost_db: float
    midrange_clarity_db: float
    treble_enhancement_db: float
    compression_ratio: float
    stereo_width: float
    mastering_intensity: float

    def __post_init__(self) -> None:
        assert -30.0 <= self.target_lufs <= 0.0, "Target LUFS must be -30 to 0"
        assert -12.0 <= self.bass_boost_db <= 12.0, "Bass boost must be -12 to +12 dB"
        assert -12.0 <= self.midrange_clarity_db <= 12.0, "Midrange clarity must be -12 to +12 dB"
        assert -12.0 <= self.treble_enhancement_db <= 12.0, "Treble enhancement must be -12 to +12 dB"
        assert 1.0 <= self.compression_ratio <= 10.0, "Compression ratio must be 1-10"
        assert 0.0 <= self.stereo_width <= 2.0, "Stereo width must be 0-2"
        assert 0.0 <= self.mastering_intensity <= 1.0, "Mastering intensity must be 0-1"
