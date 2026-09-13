# Rust DSP Library Integration Guide

**Status**: Phase 2 Complete - HPSS Fully Implemented & Validated
**Last Updated**: November 24, 2025

---

## 📍 Quick Start

### What Is This?

This directory contains Rust implementations of expensive librosa functions to solve the CPU bottleneck in Auralis audio fingerprinting:

- **HPSS** (Harmonic/Percussive Source Separation): ✅ Complete
- **YIN** (Pitch detection): ⏳ Week 2
- **Chroma CQT** (Constant-Q transform): ⏳ Week 3

### Where Is Everything?

```
vendor/auralis-dsp/
├── src/
│   ├── hpss.rs          ✅ Complete (364 lines, 10/10 tests pass)
│   ├── yin.rs           ⏳ Skeleton (to implement Week 2)
│   ├── chroma.rs        ⏳ Skeleton (to implement Week 3)
│   ├── lib.rs           ✅ Module root
│   ├── py_bindings.rs   ⏳ Disabled (enable in Phase 5)
│   └── median_filter.rs (internal utility)
├── Cargo.toml           ✅ Configured (PyO3 disabled for core validation)
└── target/             (build artifacts - git ignored)

tests/
├── test_hpss_rust_validation.py    ✅ Integration test suite
└── test_phase25_2b_real_audio_validation.py (Phase 2.5 real audio tests)

docs/
├── RUST_DSP_PROJECT_STATUS.md      ✅ Project timeline & progress
├── RUST_DSP_EXTRACTION_PLAN.md     ✅ Algorithm specifications
├── RUST_DSP_HPSS_IMPLEMENTATION_SUMMARY.md ✅ Technical details
└── PHASE2_COMPLETION_REPORT.md     ✅ What was accomplished
```

---

## 🚀 Building & Testing

### Prerequisites
- Rust 1.56+ (via rustup)
- Python 3.14+ with librosa
- 200MB disk space for build artifacts

### Build

```bash
cd vendor/auralis-dsp
cargo build --release     # Compile with optimizations
```

**Expected**: 5-6 seconds, ~30MB build artifacts

### Test

```bash
# Unit tests (all Rust modules)
cargo test --release

# Integration tests (compare against librosa)
python ../../tests/test_hpss_rust_validation.py

# PyTest suite
pytest ../../tests/test_hpss_rust_validation.py -v
```

**Expected Results**:
- ✅ 10/10 unit tests pass (0.01s)
- ✅ 3/3 real audio files process correctly
- ✅ All invariants validated

---

## 📊 Algorithm Status

### Phase 2: HPSS ✅ COMPLETE

**What It Does**: Separates audio into harmonic (sustained tones) and percussive (drums, attacks) components

**Algorithm**:
1. STFT (Short-Time Fourier Transform) with Hann windowing
2. 2D median filtering (frequency and time dimensions)
3. Wiener soft masking
4. ISTFT reconstruction

**Performance**: ~2-3 seconds per track (Python reference, before Rust speedup)
**Target speedup**: 5-10x with Rust

**Test Results**:
- Compilation: ✅ 0 errors
- Unit tests: ✅ 10/10 pass
- Real audio: ✅ 3/3 Blind Guardian tracks process correctly
- Output validation: ✅ All invariants pass (shape, NaN/Inf, energy)

---

### Phase 3: YIN ⏳ PENDING

**What It Does**: Fundamental frequency (pitch) detection using autocorrelation

**Algorithm**:
1. Frame-based autocorrelation difference function (ACDF)
2. Normalization to autocorrelation coefficient (AACF)
3. Period detection with trough threshold
4. Parabolic interpolation refinement
5. Frame-level parallelization

**Estimated speedup**: 10-20x single-threaded, 40-160x with parallelization

**Next**: Week 2 implementation (Days 8-14)

---

### Phase 4: Chroma CQT ⏳ PENDING

**What It Does**: Extract 12-dimensional chromagram (pitch distribution) from audio

**Decision Point**: Full Constant-Q Transform vs STFT-based alternative
- Full CQT: ~8-12x speedup
- STFT-based: ~2-3x speedup (simpler, proven in Phase 2.5.2B)

**Next**: Week 3 implementation (Days 15-21)

---

## 🔌 PyO3 Integration (Phase 5)

### Current Status
PyO3 and numpy dependencies are **disabled** in Cargo.toml during core validation phase.

### How to Enable (After HPSS validation complete)

1. **Uncomment in Cargo.toml**:
```toml
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"

[lib]
crate-type = ["cdylib"]  # Change from "lib"
```

2. **Implement PyO3 wrappers in py_bindings.rs**:
```rust
#[pymodule]
fn auralis_dsp(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_hpss, m)?)?;
    // ... bind other functions
    Ok(())
}

#[pyfunction]
fn py_hpss(audio: &PyArray1<f64>, sr: usize) -> PyResult<(PyArray1<f64>, PyArray1<f64>)> {
    let config = HpssConfig::default();
    let (harmonic, percussive) = hpss(audio, &config);
    // Convert to PyArray and return
}
```

3. **Build as Python extension**:
```bash
maturin develop
```

4. **Test Python import**:
```python
from auralis_dsp import hpss
harmonic, percussive = hpss(audio_array, sr=44100)
```

---

## 🧪 Testing Strategy

### Unit Tests (Rust)
- Located in each `src/*.rs` file
- Test individual functions (STFT, median filter, etc.)
- Run with: `cargo test --release`

### Integration Tests (Python)
- Located in `tests/test_hpss_rust_validation.py`
- Compare against librosa reference implementation
- Test on real Blind Guardian audio
- Run with: `python tests/test_hpss_rust_validation.py`

### Validation Tests (pytest)
- Same file, pytest-compatible
- Run with: `pytest tests/test_hpss_rust_validation.py -v`

### Performance Benchmarks (Future)
- Criterion.rs benchmarks (in `benches/`)
- Compare Rust vs Python execution time
- Measure FFI overhead

---

## 🛠️ Development Workflow

### Adding a New Function

1. **Create skeleton in `src/module.rs`**:
```rust
pub fn my_function(audio: &[f64]) -> Vec<f64> {
    // TODO: Implement
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert!(true);  // Add real tests
    }
}
```

2. **Implement algorithm**:
   - Follow librosa source for reference
   - Use ndarray for matrix operations
   - rustfft for FFT operations
   - rayon for parallelization (optional)

3. **Add unit tests**:
   - Test boundary cases
   - Test against librosa reference
   - Test on synthetic and real audio

4. **Document in module root** (`src/lib.rs`):
```rust
pub mod my_module;
pub use my_module::my_function;
```

5. **Benchmark & optimize** (Phase 5):
   - Profile with `cargo flamegraph`
   - Identify bottlenecks
   - Apply SIMD/parallelization if needed

---

## 📈 Performance Expectations

### Current (Phase 2.5.2B Baseline)
| Operation | Per-Track | Target System |
|-----------|-----------|---|
| HPSS | 2-3s | 44.1kHz, 5.7GHz |
| YIN | 1-2s | (Part of 10s total) |
| Chroma | 1-2s | (Part of 10s total) |
| **Total** | **10-15s** | Python baseline |

### After Rust Integration (Phase 5 Target)
| Operation | Single-Thread | 4-Core Parallel |
|-----------|---|---|
| HPSS | 300-600ms | 75-150ms |
| YIN | 100-200ms | 25-50ms |
| Chroma | 150-300ms | 40-75ms |
| **Total** | **600-1100ms** | **140-275ms** |

### Speedup Factors
- **HPSS**: 5-10x single-threaded
- **YIN**: 10-20x single-threaded (FFT-based)
- **Chroma**: 8-12x single-threaded
- **Overall**: 10-50x with parallelization

---

## ⚠️ Known Limitations

### Current (Phase 2)
- PyO3 bindings disabled (will enable in Phase 5)
- No SIMD optimizations yet
- Median filter uses O(n log n) sort (could optimize to O(n))

### FFI Overhead (Phase 5 consideration)
- Python ↔ Rust data marshaling ~1-5% overhead
- Not a bottleneck for per-track processing (seconds not milliseconds)

### Platform Support
- Linux: ✅ Tested and working
- macOS: ✅ Should work (not tested)
- Windows: ⚠️ Untested (MSVC compiler may require settings adjustment)

---

## 📚 References & Documentation

### In This Repository
- `RUST_DSP_PROJECT_STATUS.md` - Timeline, phases, progress
- `RUST_DSP_EXTRACTION_PLAN.md` - Algorithm specifications & performance targets
- `RUST_DSP_HPSS_IMPLEMENTATION_SUMMARY.md` - Detailed technical breakdown
- `PHASE2_COMPLETION_REPORT.md` - What was accomplished, lessons learned

### External
- librosa source: https://github.com/librosa/librosa
- Rust DSP book: https://rust-dsp.org/
- PyO3 guide: https://pyo3.rs/
- Criterion.rs (benchmarking): https://bheisler.github.io/criterion.rs/book/

---

## 🔧 Troubleshooting

### Build fails with "pyo3-ffi build script failed"
**Solution**: PyO3 is disabled in Cargo.toml during core validation. This is expected. Comment out PyO3 dependencies, or wait for Phase 5 when we re-enable them.

### Cargo complains about unused variables
**Expected**: YIN and Chroma modules are skeletons. Warnings are harmless. To silence: `#[allow(dead_code)]`

### Tests hang or run slowly
**Check**:
- Real audio tests require loading FLAC files from disk (~7-15s per file)
- Librosa STFT is still Python (slow reference)
- Expected: ~25 seconds for 3-file batch test

### Performance not improving after Rust compilation
**Reason**: Phase 5 includes FFI integration. Currently, code is Rust-only with no Python bridge.

---

## 🎯 Next Steps

### Immediate (Today/Tomorrow)
- ✅ Commit Phase 2 work to git
- ✅ Update project documentation
- ⏳ Start Phase 3 (YIN implementation)

### Short-term (Week 2)
- Implement YIN pitch detection
- Add frame-level parallelization with rayon
- Write integration tests for real audio

### Medium-term (Week 3)
- Implement Chroma CQT (or STFT-based alternative)
- Complete feature extraction coverage
- Begin performance benchmarking

### Long-term (Week 4)
- Enable PyO3 bindings
- Create Python wrappers
- Full validation on 72-track Blind Guardian collection
- Measure end-to-end performance improvement

---

## 📞 Questions?

Refer to:
1. `PHASE2_COMPLETION_REPORT.md` for context & decisions
2. `RUST_DSP_HPSS_IMPLEMENTATION_SUMMARY.md` for technical details
3. Algorithm papers in docs (Fitzgerald, Driedger, de Cheveigné, Brown)

---

**Status**: ✅ Phase 2 Complete | **Ready**: Phase 3 Implementation
