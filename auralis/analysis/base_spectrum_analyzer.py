"""
Base Spectrum Analyzer
~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for spectrum analysis implementations.

Defines the common interface and shared functionality for all spectrum
analyzer implementations (sequential, parallel, real-time, etc.).

:copyright: (C) 2024 Auralis Team
:license: GPLv3, see LICENSE for more details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fingerprint.metrics import AggregationUtils
from .spectrum_operations import SpectrumOperations


@dataclass
class SpectrumSettings:
    """Configuration for spectrum analysis"""
    fft_size: int = 4096
    window_type: str = 'hann'
    overlap: float = 0.75
    sample_rate: int = 44100
    frequency_bands: int = 64
    frequency_weighting: str = 'A'  # 'A', 'C', or 'Z' (none)
    smoothing_factor: float = 0.8

    # Frequency range
    min_frequency: float = 20.0
    max_frequency: float = 20000.0


class BaseSpectrumAnalyzer(ABC):
    """Abstract base class for spectrum analyzers"""

    def __init__(self, settings: SpectrumSettings | None = None) -> None:
        """
        Initialize base spectrum analyzer

        Args:
            settings: SpectrumSettings configuration object
        """
        self.settings = settings or SpectrumSettings()

        # Create frequency bins
        self.frequency_bins = SpectrumOperations.create_frequency_bins(
            min_freq=self.settings.min_frequency,
            max_freq=self.settings.max_frequency,
            num_bands=self.settings.frequency_bands
        )

        # Smoothing buffer for real-time analysis
        self.smoothing_buffer: np.ndarray | None = None

        # Window function
        self.window = SpectrumOperations.create_window(
            self.settings.window_type,
            self.settings.fft_size
        )

        # Frequency weighting curve
        self.weighting_curve = SpectrumOperations.get_weighting_curve(
            self.frequency_bins,
            self.settings.frequency_weighting
        )

    @abstractmethod
    def analyze_chunk(self, audio_chunk: np.ndarray, channel: int = 0) -> dict[str, Any]:
        """
        Analyze a chunk of audio data

        Args:
            audio_chunk: Audio data (mono or stereo)
            channel: Channel to analyze (0=left, 1=right, -1=sum)

        Returns:
            Dictionary with analysis results including:
            - spectrum: Frequency spectrum values
            - frequency_bins: Frequency bin centers
            - peak_frequency: Frequency with max energy
            - spectral_centroid: Center of mass frequency
            - spectral_rolloff: 85th percentile frequency
            - total_energy: Total spectral energy
            - settings: Analysis settings used
        """

    @abstractmethod
    def analyze_file(self, audio_data: np.ndarray, sample_rate: int | None = None) -> dict[str, Any]:
        """
        Analyze an entire audio file

        Args:
            audio_data: Complete audio data
            sample_rate: Sample rate (if different from settings)

        Returns:
            Dictionary with comprehensive analysis including:
            - spectrum: Aggregated frequency spectrum
            - frequency_bins: Frequency bin centers
            - peak_frequency: Mean peak frequency
            - spectral_centroid: Mean spectral centroid
            - spectral_rolloff: Mean spectral rolloff
            - total_energy: Mean total energy
            - num_chunks_analyzed: Number of chunks processed
            - analysis_duration: Duration analyzed in seconds
            - settings: Analysis settings used
        """

    def _create_chunk_result(self,
                            audio_chunk: np.ndarray,
                            channel: int = 0,
                            sample_rate: int | None = None) -> dict[str, Any]:
        """
        Create analysis result for a single audio chunk

        This is a helper method for subclasses to use.

        Args:
            audio_chunk: Audio data to analyze
            channel: Channel to analyze
            sample_rate: Sample rate (uses self.settings.sample_rate if None)

        Returns:
            Analysis result dictionary
        """
        if sample_rate is None:
            sample_rate = self.settings.sample_rate

        # Compute FFT
        magnitude, fft_size = SpectrumOperations.compute_fft(
            audio_chunk, self.window, self.settings.fft_size, channel
        )

        # Get frequency array
        frequencies = np.fft.rfftfreq(self.settings.fft_size, 1 / sample_rate)

        # Map to frequency bands
        spectrum = SpectrumOperations.map_to_bands(
            frequencies, magnitude, self.frequency_bins, sample_rate
        )

        # Apply frequency weighting
        weighted_spectrum = spectrum + self.weighting_curve

        # Apply smoothing
        weighted_spectrum = SpectrumOperations.apply_smoothing(
            weighted_spectrum,
            self.smoothing_buffer,
            self.settings.smoothing_factor
        )

        self.smoothing_buffer = weighted_spectrum

        # Calculate metrics
        peak_frequency = self.frequency_bins[np.argmax(weighted_spectrum)]
        spectral_centroid = SpectrumOperations.calculate_spectral_centroid(
            self.frequency_bins, weighted_spectrum
        )
        spectral_rolloff = SpectrumOperations.calculate_spectral_rolloff(
            self.frequency_bins, weighted_spectrum, 0.85
        )

        return {
            'spectrum': weighted_spectrum.tolist(),
            'frequency_bins': self.frequency_bins.tolist(),
            'peak_frequency': float(peak_frequency),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'total_energy': float(np.sum(weighted_spectrum)),
            'settings': {
                'fft_size': self.settings.fft_size,
                'frequency_bands': self.settings.frequency_bands,
                'weighting': self.settings.frequency_weighting
            }
        }

    def get_frequency_band_names(self) -> list[str]:
        """
        Get human-readable frequency band names

        Returns:
            List of band name strings
        """
        return SpectrumOperations.get_band_names(self.frequency_bins)

    def reset_smoothing(self) -> None:
        """Reset smoothing buffer for new analysis session"""
        self.smoothing_buffer = None

    def update_settings(self, settings: SpectrumSettings) -> None:
        """
        Update analyzer settings and rebuild state

        Args:
            settings: New SpectrumSettings configuration
        """
        self.settings = settings

        # Rebuild frequency bins
        self.frequency_bins = SpectrumOperations.create_frequency_bins(
            min_freq=self.settings.min_frequency,
            max_freq=self.settings.max_frequency,
            num_bands=self.settings.frequency_bands
        )

        # Rebuild window
        self.window = SpectrumOperations.create_window(
            self.settings.window_type,
            self.settings.fft_size
        )

        # Rebuild weighting curve
        self.weighting_curve = SpectrumOperations.get_weighting_curve(
            self.frequency_bins,
            self.settings.frequency_weighting
        )

        # Reset smoothing buffer
        self.reset_smoothing()


class SpectrumAnalyzer(BaseSpectrumAnalyzer):
    """Spectrum analyzer with real-time and file analysis capabilities"""

    def analyze_chunk(self, audio_chunk: np.ndarray, channel: int = 0) -> dict[str, Any]:
        return self._create_chunk_result(audio_chunk, channel, self.settings.sample_rate)

    def analyze_file(self, audio_data: np.ndarray, sample_rate: int | None = None) -> dict[str, Any]:
        if sample_rate:
            self.settings.sample_rate = sample_rate

        # "Analyze a whole file" must not inherit smoothing state from the
        # previous file under any caller (#4539). QualityMetrics resets this
        # explicitly too, but doing it here makes the guarantee a property of
        # the analyzer rather than something every caller has to remember —
        # `smoothing_buffer` has leaked across boundaries three times already
        # (#3448, #3433, #2890).
        self.reset_smoothing()

        hop_size = int(self.settings.fft_size * (1 - self.settings.overlap))
        num_chunks = (len(audio_data) - self.settings.fft_size) // hop_size + 1
        chunk_results = []

        for i in range(num_chunks):
            start_idx = i * hop_size
            end_idx = start_idx + self.settings.fft_size
            if end_idx <= len(audio_data):
                chunk = audio_data[start_idx:end_idx]
                if audio_data.ndim == 2:
                    chunk = chunk.reshape(-1, audio_data.shape[1])
                chunk_results.append(self.analyze_chunk(chunk))

        if not chunk_results:
            return {}

        aggregated = SpectrumOperations.aggregate_analysis_results(chunk_results)
        return {
            'spectrum': aggregated['spectrum'].tolist(),
            'frequency_bins': self.frequency_bins.tolist(),
            'peak_frequency': float(AggregationUtils.aggregate_frames_to_track(
                np.array([r['peak_frequency'] for r in chunk_results]), method='mean'
            )),
            'spectral_centroid': float(AggregationUtils.aggregate_frames_to_track(
                np.array([r['spectral_centroid'] for r in chunk_results]), method='mean'
            )),
            'spectral_rolloff': float(AggregationUtils.aggregate_frames_to_track(
                np.array([r['spectral_rolloff'] for r in chunk_results]), method='mean'
            )),
            'total_energy': float(AggregationUtils.aggregate_frames_to_track(
                np.array([r['total_energy'] for r in chunk_results]), method='mean'
            )),
            'num_chunks_analyzed': len(chunk_results),
            'analysis_duration': len(audio_data) / self.settings.sample_rate,
            'settings': {
                'fft_size': self.settings.fft_size,
                'frequency_bands': self.settings.frequency_bands,
                'weighting': self.settings.frequency_weighting,
                'overlap': self.settings.overlap,
            },
        }
