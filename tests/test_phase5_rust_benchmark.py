"""
Phase 5 Benchmark: Rust DSP vs librosa Performance

Compares execution time for HPSS, YIN, and Chroma CQT implementations.

Test scenarios:
- Short audio (1 second)
- Medium audio (10 seconds)
- Long audio (60 seconds)

Metrics tracked:
- Execution time (Rust vs librosa)
- Speedup factor
- Memory usage (if available)
- Output shape and normalization verification
"""

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import auralis_dsp
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    auralis_dsp = None

# #4968: this used to auto-install a hardcoded wheel path
# (auralis_dsp-0.1.0-cp313-cp313-manylinux_2_35_x86_64.whl) whose interpreter
# tag is dead under requires-python >= 3.14 and whose platform tag never
# matched a local build's linux_x86_64 tag either way -- the .exists() check
# always failed, silently falling back to a multi-minute release compile on
# every run. Mirrors the skip-if-missing pattern
# tests/vendor/test_rust_panic_handler.py already uses correctly: skip rather
# than auto-install/auto-build, since a benchmark comparing Rust against
# librosa is meaningless without the real extension already built via
# `cd vendor/auralis-dsp && maturin develop`.
pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="auralis_dsp Rust extension not installed (run: cd vendor/auralis-dsp && maturin develop)"
)


class Phase5Benchmark:
    """Benchmark Rust implementations against librosa."""

    def __init__(self):
        self.results = {}
        self.sr = 44100

    def generate_test_audio(self, duration_s: float) -> np.ndarray:
        """Generate realistic test audio."""
        n_samples = int(self.sr * duration_s)
        t = np.arange(n_samples) / self.sr

        # Blend multiple frequencies for realistic content
        freqs = [60, 120, 440, 880, 2000]  # Sub-bass, bass, A4, A5, presence
        audio = np.zeros(n_samples)
        for freq in freqs:
            audio += np.sin(2 * np.pi * freq * t) / len(freqs)

        # Add some dynamics
        audio *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz amplitude modulation

        return audio.astype(np.float64)

    def benchmark_hpss(self, duration_s: float):
        """Benchmark HPSS implementation."""
        audio = self.generate_test_audio(duration_s)
        label = f"HPSS_{duration_s}s"

        print(f"\n{'=' * 60}")
        print(f"HPSS Benchmark - {duration_s} second audio")
        print(f"{'=' * 60}")

        # Rust implementation
        if RUST_AVAILABLE:
            start = time.perf_counter()
            harmonic_rust, percussive_rust = auralis_dsp.hpss(audio)
            rust_time = time.perf_counter() - start
            print(f"✅ Rust:    {rust_time:.4f}s")

            # Verify output
            assert harmonic_rust.shape == audio.shape
            assert percussive_rust.shape == audio.shape
            print(f"   Output shape: {harmonic_rust.shape}")

        else:
            rust_time = None
            harmonic_rust = percussive_rust = None
            print("⚠️  Rust not available")

        # librosa implementation
        start = time.perf_counter()
        harmonic_librosa, percussive_librosa = librosa.effects.hpss(audio)
        librosa_time = time.perf_counter() - start
        print(f"📚 librosa: {librosa_time:.4f}s")

        # Calculate speedup
        if rust_time is not None and rust_time > 0:
            speedup = librosa_time / rust_time
            print(f"🚀 Speedup: {speedup:.2f}x")

            # Verify similarity
            if harmonic_rust is not None:
                mae = np.mean(np.abs(harmonic_rust - harmonic_librosa))
                print(f"   Harmonic MAE: {mae:.6f}")

            self.results[label] = {
                "rust_time": rust_time,
                "librosa_time": librosa_time,
                "speedup": speedup,
            }
        else:
            self.results[label] = {
                "rust_time": None,
                "librosa_time": librosa_time,
                "speedup": None,
            }

    def benchmark_yin(self, duration_s: float):
        """Benchmark YIN implementation."""
        audio = self.generate_test_audio(duration_s)
        label = f"YIN_{duration_s}s"

        print(f"\n{'=' * 60}")
        print(f"YIN Benchmark - {duration_s} second audio")
        print(f"{'=' * 60}")

        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C7")

        # Rust implementation
        if RUST_AVAILABLE:
            start = time.perf_counter()
            f0_rust = auralis_dsp.yin(audio, sr=self.sr, fmin=fmin, fmax=fmax)
            rust_time = time.perf_counter() - start
            print(f"✅ Rust:    {rust_time:.4f}s")

            # Verify output
            assert f0_rust.ndim == 1
            print(f"   Output shape: {f0_rust.shape}")

        else:
            rust_time = None
            f0_rust = None
            print("⚠️  Rust not available")

        # librosa implementation
        start = time.perf_counter()
        f0_librosa = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=self.sr)
        librosa_time = time.perf_counter() - start
        print(f"📚 librosa: {librosa_time:.4f}s")

        # Calculate speedup
        if rust_time is not None and rust_time > 0:
            speedup = librosa_time / rust_time
            print(f"🚀 Speedup: {speedup:.2f}x")

            # Verify similarity (handle shape differences due to different hop lengths)
            if f0_rust is not None:
                # Both should have similar length, but may differ by 1-2 frames
                # (rust does not centre its frames; librosa does).
                n_frames = min(len(f0_rust), len(f0_librosa))
                mae = np.mean(np.abs(f0_rust[:n_frames] - f0_librosa[:n_frames]))
                print(f"   F0 MAE: {mae:.2f} Hz (over {n_frames} frames)")

                # #5169: this MAE was computed and printed but never asserted,
                # so the benchmark stayed green for ~9 months while the Rust
                # yin returned a near-constant ~1150-1660 Hz.
                #
                # Compare only frames the Rust side calls voiced. The two
                # implementations differ by convention, not correctness, on
                # this signal: it is a blend of 60/120/440/880/2000 Hz whose
                # common period is 20 Hz, below fmin (C2 = 65.4 Hz), so no
                # in-range lag is genuinely periodic. Rust reports 0.0
                # (unvoiced) for every frame; librosa.yin has no unvoiced
                # concept and always returns its best in-range guess (~111 Hz).
                # Differencing those directly would assert a convention.
                #
                # In the fixed state this leaves nothing to compare, which is
                # the correct outcome. It is still a real gate on the #5169
                # regression: the broken implementation called every frame
                # voiced at ~1150 Hz, giving an MAE around 1040 Hz here. The
                # pitch-accuracy contract proper lives in
                # tests/test_yin_rust_validation.py, whose assertions are live
                # (previously xfail) as of #5169.
                voiced = f0_rust[:n_frames] > 0.0
                print(f"   Voiced frames: {int(voiced.sum())}/{n_frames}")
                if voiced.any():
                    voiced_mae = np.mean(
                        np.abs(f0_rust[:n_frames][voiced] - f0_librosa[:n_frames][voiced])
                    )
                    assert voiced_mae < 50.0, (
                        f"Rust/librosa F0 MAE {voiced_mae:.2f} Hz over "
                        f"{int(voiced.sum())} voiced frames — the Rust yin is "
                        "not tracking pitch (#5169)"
                    )

            self.results[label] = {
                "rust_time": rust_time,
                "librosa_time": librosa_time,
                "speedup": speedup,
            }
        else:
            self.results[label] = {
                "rust_time": None,
                "librosa_time": librosa_time,
                "speedup": None,
            }

    def benchmark_chroma_cqt(self, duration_s: float):
        """Benchmark Chroma CQT implementation."""
        audio = self.generate_test_audio(duration_s)
        label = f"ChromaCQT_{duration_s}s"

        print(f"\n{'=' * 60}")
        print(f"Chroma CQT Benchmark - {duration_s} second audio")
        print(f"{'=' * 60}")

        # Rust implementation
        if RUST_AVAILABLE:
            start = time.perf_counter()
            chroma_rust = auralis_dsp.chroma_cqt(audio, sr=self.sr)
            rust_time = time.perf_counter() - start
            print(f"✅ Rust:    {rust_time:.4f}s")

            # Verify output
            assert chroma_rust.shape[0] == 12
            print(f"   Output shape: {chroma_rust.shape}")
            print(f"   Normalized: {np.allclose(chroma_rust.sum(axis=0), 1.0, atol=0.01)}")

        else:
            rust_time = None
            chroma_rust = None
            print("⚠️  Rust not available")

        # librosa implementation
        start = time.perf_counter()
        chroma_librosa = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
        librosa_time = time.perf_counter() - start
        print(f"📚 librosa: {librosa_time:.4f}s")

        # Calculate speedup
        if rust_time is not None and rust_time > 0:
            speedup = librosa_time / rust_time
            print(f"🚀 Speedup: {speedup:.2f}x")

            # Verify similarity (with tolerance for different algorithms)
            if chroma_rust is not None:
                # Resample to same shape if needed
                n_frames = min(chroma_rust.shape[1], chroma_librosa.shape[1])
                chroma_rust_resampled = chroma_rust[:, :n_frames]
                chroma_librosa_resampled = chroma_librosa[:, :n_frames]

                mae = np.mean(
                    np.abs(chroma_rust_resampled - chroma_librosa_resampled)
                )
                print(f"   Chroma MAE: {mae:.4f}")

            self.results[label] = {
                "rust_time": rust_time,
                "librosa_time": librosa_time,
                "speedup": speedup,
            }
        else:
            self.results[label] = {
                "rust_time": None,
                "librosa_time": librosa_time,
                "speedup": None,
            }

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'=' * 60}")
        print("PHASE 5 BENCHMARK SUMMARY")
        print(f"{'=' * 60}")

        # Group by algorithm
        algorithms = {}
        for label, times in self.results.items():
            algo = label.rsplit("_", 1)[0]
            if algo not in algorithms:
                algorithms[algo] = []
            algorithms[algo].append((label, times))

        # Print grouped results
        for algo in sorted(algorithms.keys()):
            print(f"\n{algo}:")
            print("-" * 60)

            total_speedup = []
            for label, times in algorithms[algo]:
                if times["speedup"] is not None:
                    duration = label.split("_")[-1]
                    print(
                        f"  {duration:>4}: Rust={times['rust_time']:.4f}s, "
                        f"librosa={times['librosa_time']:.4f}s, "
                        f"speedup={times['speedup']:.2f}x"
                    )
                    total_speedup.append(times["speedup"])

            if total_speedup:
                avg_speedup = np.mean(total_speedup)
                print(f"  Average speedup: {avg_speedup:.2f}x")

        print(f"\n{'=' * 60}")
        print("✅ BENCHMARK COMPLETE")
        print(f"{'=' * 60}")

    def run(self):
        """Run all benchmarks."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 12 + "PHASE 5: RUST vs LIBROSA BENCHMARK" + " " * 12 + "║")
        print("╚" + "═" * 58 + "╝")

        # Run benchmarks for different audio durations
        durations = [1, 10, 60]
        for duration in durations:
            self.benchmark_hpss(duration)
            self.benchmark_yin(duration)
            self.benchmark_chroma_cqt(duration)

        self.print_summary()


@pytest.mark.benchmark
def test_phase5_rust_vs_librosa():
    """Run Phase 5 benchmark tests."""
    bench = Phase5Benchmark()
    bench.run()

    # All benchmarks should complete without error
    assert len(bench.results) > 0, "No benchmark results recorded"


def main():
    """Run benchmark standalone."""
    bench = Phase5Benchmark()
    bench.run()


if __name__ == "__main__":
    main()
