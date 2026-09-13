---
description: "Verify the 6 audio processing invariants across the DSP pipeline (read-only, no fixes)"
argument-hint: "[<file-path> | <glob> | <git-range>]"
---

# DSP Invariant Verification

Verify all 6 audio processing invariants across the DSP pipeline. This is a **read-only** audit — it reports violations but does NOT fix them.

**Input**: `$ARGUMENTS` = optional file path, glob pattern, or git range (default: all DSP files)

## Scope

Determine scope from arguments:

- **No arguments**: Scan ALL DSP files (full verification)
- **File path** (e.g. `auralis/core/simple_mastering.py`): Scan only that file
- **Glob pattern** (e.g. `auralis/dsp/*.py`): Scan matching files
- **Git range** (e.g. `HEAD~5..HEAD`): Scan only files changed in that range via `git diff <range> --name-only`

### Default full-scope paths

| Component | Path |
|-----------|------|
| Core Pipeline | `auralis/core/hybrid_processor.py`, `simple_mastering.py`, `processing/` |
| DSP Modules | `auralis/dsp/eq/psychoacoustic_eq.py`, `auralis/dsp/advanced_dynamics.py`, `auralis/dsp/basic.py` |
| Audio I/O | `auralis/io/unified_loader.py`, `results.py` |
| Player RT Processing | `auralis/player/realtime/` |
| Chunked Mastering | `auralis/core/mastering_chunk_loop.py`, `mastering_process_chunk.py` |
| Backend Chunking | `auralis-web/backend/core/chunked_processor.py` |
| Rust DSP Bindings | `vendor/auralis-dsp/src/*.rs` |

## The 6 Invariants

### INV-1: Sample Count Preservation

**Rule**: `len(output) == len(input)` at EVERY processing stage.

**How to verify**:
- Find every function that takes audio data as input and returns audio data
- Check that no operation changes the array length: no trimming, padding, resampling, or filtering that alters length
- Watch for: `np.pad()`, `np.trim_zeros()`, `signal.resample()`, array slicing that changes length, `np.concatenate()` with extra samples
- Parallel processing: verify `sum(chunk_lengths) == total_length` after reassembly

**Pass criteria**: Every audio processing function preserves sample count, or explicitly documents why it doesn't (e.g., resampling).

### INV-2: Copy Before Modify

**Rule**: `audio.copy()` before ANY in-place NumPy operation.

**How to verify**:
- Find every function that receives an audio array parameter
- Check that it creates a copy before modifying: `output = audio.copy()` or `result = np.copy(audio)`
- In-place operations to watch for: `audio *= gain`, `audio += offset`, `audio[i] = ...`, `np.multiply(a, b, out=a)`, `audio.clip(...)` without assignment
- Views are NOT copies: `audio[::2]` creates a view, modifying the view modifies the original

**Pass criteria**: No function modifies its input array. All modifications happen on explicit copies.

### INV-3: Dtype Preservation

**Rule**: Audio stays `float32` or `float64` throughout — no silent casts.

**How to verify**:
- Check return types of all processing functions
- Watch for operations that change dtype: integer division `//`, mixing float32 and float64 arrays, `np.int16` casts mid-pipeline
- Rust DSP boundary: verify PyO3 returns match expected Python dtype
- `results.py` output formatting: verify pcm16/pcm24 conversion only happens at final output

**Pass criteria**: Audio dtype is preserved or explicitly converted with documented intent.

### INV-4: NaN/Inf Safety

**Rule**: Guards on division, log, sqrt — no NaN/Inf propagation between stages.

**How to verify**:
- Find all division operations: `/`, `np.divide()`, `1.0 / x`
- Find all log operations: `np.log()`, `np.log10()`, `np.log2()`
- Find all sqrt operations: `np.sqrt()`
- Check for guards: `np.maximum(x, epsilon)` before division, `np.clip(x, min_val, ...)` before log, `np.abs(x)` before sqrt
- Check for NaN propagation: does any stage check `np.isnan()` or `np.isinf()` in output?

**Pass criteria**: Every potentially NaN/Inf-producing operation has a guard. No stage can pass NaN/Inf to the next.

### INV-5: Clipping Prevention

**Rule**: Audio clamped to `[-1.0, 1.0]` before output.

**How to verify**:
- Find every function that returns audio destined for output (playback, streaming, file writing)
- Check for clipping: `np.clip(audio, -1.0, 1.0)`, `audio.clip(-1.0, 1.0)`
- Inter-stage clipping: check if intermediate stages can produce values > 1.0 that accumulate
- Watch for: gain application without subsequent limiting, additive mixing without normalization

**Pass criteria**: Final output is always clamped. Intermediate stages that can exceed [-1.0, 1.0] have downstream clamping.

### INV-6: Chunk Safety

**Rule**: True copies (not views), correct reassembly order, continuous carried state, equal-power crossfade at boundaries.

The engine-side parallel processor was deleted in #4565. Two live chunk paths remain — verify both:

**How to verify** — engine loop (`auralis/core/mastering_chunk_loop.py`, `auralis/core/mastering_process_chunk.py`):
- Chunks created with `.copy()`, not array slicing (views)
- Carried context (notch state, level/gain smoothing, limiter memory) evolves continuously and is not reset per chunk
- Final short chunk is neither zero-padded nor dropped

**How to verify** — backend processor (`auralis-web/backend/core/chunked_processor.py`, `auralis-web/backend/core/processor_pool.py`):
- Chunk reassembly: ordering is index-based, not arrival-order
- Boundary crossfade in `auralis-web/backend/core/chunk_crossfade.py`: overlapping regions use equal-power curves, not linear
- No shared mutable state between concurrent workers — mastering targets read by workers must be immutable
- Chunk geometry comes from `auralis-web/backend/core/chunk_boundaries.py`, never a literal

**Pass criteria**: Chunked processing produces the same result as whole-file processing (within floating-point tolerance), and `sum(chunk_lengths) == total_length`.

## Output Format

### Verification Report

```
| Invariant | Status | Files Checked | Violations |
|-----------|--------|---------------|------------|
| INV-1: Sample Count | PASS/FAIL | N | M |
| INV-2: Copy Before Modify | PASS/FAIL | N | M |
| INV-3: Dtype Preservation | PASS/FAIL | N | M |
| INV-4: NaN/Inf Safety | PASS/FAIL | N | M |
| INV-5: Clipping Prevention | PASS/FAIL | N | M |
| INV-6: Parallel Safety | PASS/FAIL | N | M |
```

### Per-Violation Detail

```
#### VIOLATION: <invariant> in <file>:<line-range>
- **Function**: `function_name()`
- **Issue**: <what is wrong>
- **Evidence**: <code snippet showing the violation>
- **Risk**: <what could go wrong — audio artifacts, corruption, crash>
- **Existing Issue?**: Check `gh issue list` for duplicates
```

### Summary

```
## Verdict
- Total files scanned: N
- Total violations: M
- CRITICAL violations: X (these MUST be fixed before release)
- Recommendation: PASS / NEEDS ATTENTION / FAIL
```

## Rules

- **Read-only**: Do NOT modify any files. Report only.
- **Evidence-based**: Every FAIL must include a code snippet proving the violation.
- **No false positives**: If unsure, re-read surrounding code. Only report confirmed violations.
- **Acknowledge intentional exceptions**: Some operations legitimately change sample count (resampling) or dtype (final output). Note these as "intentional, not a violation."
