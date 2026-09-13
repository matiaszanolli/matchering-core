---
description: "Deep audit of the core audio engine — DSP pipeline, player, chunked mastering, analysis, library"
argument-hint: "[--focus <dimensions>] [--depth shallow|deep] [--limit <N>]"
---

# Audio Engine Audit

Perform a deep audit of the Auralis core audio engine — DSP pipeline, player, analysis, library.

**Architecture**: This is an orchestrator. Each dimension runs as an Agent-tool subagent (`subagent_type: general-purpose`, `model: sonnet`). Max 3 run concurrently.

See `.claude/commands/_audit-common.md` for project layout, severity framework, methodology, context management rules, deduplication, and finding format.

## Parameters (from $ARGUMENTS)

- `--focus <dimensions>`: Comma-separated dimension numbers or names (e.g., `1,3` or `sample-integrity,player`). Default: all 7.
- `--depth shallow|deep`: `shallow` = check key patterns only; `deep` = trace full call graphs. Default: `deep`.
- `--limit <N>`: Stop after N findings (useful for time-boxed audits). Default: unlimited.

## Scope

| Component | Path | Key Files |
|-----------|------|-----------|
| Core Pipeline | `auralis/core/` | `hybrid_processor.py` + `hybrid/`, `simple_mastering.py` + `mastering_chunk_loop.py` / `mastering_prepare.py` / `mastering_process_chunk.py` / `mastering_notch_context.py` / `mastering_branches/` (continuous path only), `processing/`, `processors/`, `stages/` (13 named stages), `analysis/`, `dsp/`, `utils/` |
| Continuous Processing | `auralis/core/processing/` | `continuous_space.py` (`ProcessingCoordinates` — the 3D space replacing discrete presets), `continuous_mode.py`, `adaptive_mode.py`, `hybrid_mode.py`, `parameter_generator.py`, `target_derivation.py`, `delta_eq.py`, `cross_dimensional_guard.py`, `hf_aware_limiter.py`, `base/`. See the Retired Architecture table in `_audit-common.md` — categorical branch classification is gone; do not report its absence. |
| Core Config | `auralis/core/config/` | `unified_config.py` (UnifiedConfig), `factory.py`, `settings.py`, `preset_profiles.py`, `genre_profiles.py`. This package is now the *only* config layer — the same-named legacy module was deleted in #4918, so there is no second definition site to reconcile. |
| DSP Modules | `auralis/dsp/` | `basic.py`, `advanced_dynamics.py`, `eq/psychoacoustic_eq.py` + `eq/parallel_eq_processor/`, `dynamics/`, `utils/` |
| Player | `auralis/player/` | `enhanced_audio_player.py`, `gapless_playback_engine.py`, `queue_controller.py`, `playback_controller.py`, `realtime/`, `components/`, `audio_file_manager.py` |
| Audio I/O | `auralis/io/` | `unified_loader.py`, `loader.py`, `loaders/`, `formats.py`, `saver.py`, `results.py` |
| Chunked Mastering | `auralis/core/` | `mastering_chunk_loop.py`, `mastering_process_chunk.py`, `mastering_prepare.py`, `mastering_notch_context.py`, `mastering_diagnostics.py` — the engine-side chunk loop that replaced the deleted parallel processor (#4565) |
| Optimization (test-only) | `auralis/optimization/` | `acceleration/`, `caching/`, `memory/`, `profiling/`, `performance_optimizer.py`. **Not imported by any production code** — tests are the only importers. Findings here are tech debt, not engine defects; cap severity at LOW unless you can show a runtime call path. |
| Analysis | `auralis/analysis/` | `fingerprint/` (25D system), `ml/`, `quality/`, `quality_assessors/` |
| Library | `auralis/library/` | `database.py` (`LibraryDatabase` — the sole composition root: engine, pragmas, migration, sessions, scan slots, shutdown; the `LibraryManager` facade and its cache layer were deleted in #4915), `repositories/` (13 repos + `base.py` BaseRepository), `scanner/`, `models/`, `migrations/`, `migration_manager.py`, `sidecar_manager.py`, `resource_monitor.py`, `path_key.py`, `fingerprint_quantizer.py` |
| Services | `auralis/services/` | `artwork_service.py`, `fingerprint_extractor.py`, `fingerprint_queue.py`, `resizable_semaphore.py` |
| Rust DSP | `vendor/auralis-dsp/` | PyO3 bindings in `vendor/auralis-dsp/src/py_bindings.rs` — 11 exposed functions: hpss, yin, chroma_cqt, detect_tempo, envelope_follow, compress, limit, compute_fingerprint, apply_multiband_eq, detect_onsets, process_chunks. Rhythm/tempo/onset were ported in from the deleted standalone fingerprint server (#4533). |

Out of scope: React frontend, FastAPI backend (routing, WebSocket layer), Electron desktop. DO verify engine public API contracts.

## Severity Examples

| Severity | Engine-Specific Examples |
|----------|------------------------|
| **CRITICAL** | Sample count mismatch causing clicks/gaps, buffer corruption from missing copy, in-place modification of shared array, database corruption from concurrent writes |
| **HIGH** | Discontinuity at a mastering chunk seam, RLock deadlock in player, gapless transition audible gap, memory leak during extended playback |
| **MEDIUM** | Inconsistent dtype handling across stages, missing copy-before-modify in non-critical path, fingerprint accuracy degradation at edge cases |
| **LOW** | Redundant array copies hurting performance, sub-optimal FFT windowing, unused analysis metrics |

## Audit Dimensions

### Dimension 1: Sample Integrity

**Key files**: `auralis/core/hybrid_processor.py`, `auralis/core/simple_mastering.py`, all DSP modules

**Check**:
- [ ] `len(output) == len(input)` — verified at EVERY processing stage, not just the outer wrapper?
- [ ] `audio.copy()` before ANY in-place NumPy operation — no exceptions?
- [ ] dtype preservation — does audio stay `float32`/`float64` throughout? Any silent casts?
- [ ] Clipping prevention — is audio clamped to [-1.0, 1.0] before output? Inter-stage clipping?
- [ ] NaN/Inf propagation — can a NaN from one DSP stage corrupt the entire output?
- [ ] Mono/stereo handling — are mono files correctly handled through stereo pipelines?
- [ ] Bit depth output — does `auralis/io/results.py` (pcm16, pcm24) correctly scale and quantize?

### Dimension 2: DSP Pipeline Correctness

**Key files**: `auralis/core/hybrid_processor.py` + `auralis/core/hybrid/`, `auralis/core/simple_mastering.py` + `auralis/core/mastering_process_chunk.py` / `auralis/core/mastering_chunk_loop.py` / `auralis/core/mastering_branches/`, `auralis/core/processing/continuous_space.py`, `auralis/core/processing/continuous_mode.py`, `auralis/core/stages/`, `auralis/core/dsp/`, `auralis/dsp/eq/psychoacoustic_eq.py`, `auralis/dsp/advanced_dynamics.py`

**Check**:
- [ ] Processing chain order — is the sequence (EQ → dynamics → mastering) correct and documented?
- [ ] Stage independence — does each stage receive clean input, or can a failed stage leave dirty state?
- [ ] Parameter validation — are gain, frequency, Q factor ranges validated before DSP math?
- [ ] Windowing — are FFT windows applied correctly? Double-windowing removed? (Fix: `cca59d9c`)
- [ ] Spectral leakage — are FFT sizes appropriate for the sample rate?
- [ ] Phase coherence — does multi-band processing maintain phase relationships?
- [ ] Sub-bass parallel path — correctly mixed back in? (Fix: `8bc5b217`)
- [ ] EQ band mapping — is the psychoacoustic EQ curve mapped to bands **by frequency**, not by raw index? An out-of-range index used to fall back silently to the simple EQ (fix `2b3c5b35`).
- [ ] WOLA overlap — the psychoacoustic path uses a fixed 50% hop with a full-Hann synthesis window. Any configurable-overlap change must re-derive COLA; flag if one was introduced without it.
- [ ] Parameter-space continuity — mastering is driven by continuous `ProcessingCoordinates` (spectral_balance, dynamic_range, energy_level) from `auralis/core/processing/continuous_space.py`. Does any consumer reintroduce a categorical step — a hard `if coord > X` threshold, a clamp that flattens the range, or a lookup table with discrete buckets? Two fingerprints that differ slightly must not produce audibly different mastering.
- [ ] Monotonicity — as one coordinate increases with the others fixed, do the derived parameters move monotonically, without plateaus or clipped ranges? Non-monotonic parameter generation makes mastering unpredictable across a catalog.
- [ ] Coordinate derivation — are the 3 axes computed from the 25D fingerprint with bounded, finite math? `_smooth_unit()` uses `tanh` to map unbounded measurements into (0, 1) — check every axis actually routes through a bounded mapping and cannot emit NaN/Inf from a degenerate fingerprint.
- [ ] Config reachability — a parameter defined in `auralis/core/config/unified_config.py` must actually be read on the path that claims to honor it. Trace from the definition to the DSP call site; a parameter with no reader is dead config, and a DSP stage with a hardcoded value that shadows a config field is the inverse bug. (The old "legacy `config.py` vs package" duality is gone — that module was deleted in #4918.)
- [ ] Rust DSP boundary — do PyO3 calls handle errors and return correct formats?
- [ ] GIL handling — does Rust code release the GIL during compute? Can concurrent calls corrupt state?

### Dimension 3: Player State Machine

**Key files**: `auralis/player/enhanced_audio_player.py`, `auralis/player/gapless_playback_engine.py`, `auralis/player/queue_controller.py`, `auralis/player/playback_controller.py`, `auralis/player/realtime/`, `auralis/player/components/`, `auralis/player/audio_file_manager.py`

**Check**:
- [ ] State transitions — are play/pause/stop/seek transitions atomic under RLock?
- [ ] Position invariant — can `position > duration` ever occur? During seek?
- [ ] Queue bounds — can index go out of bounds during skip/previous/remove?
- [ ] Gapless transitions — is the next track pre-loaded? Race between load and play?
- [ ] Callback safety — are callbacks invoked outside the lock to prevent deadlock?
- [ ] Resource cleanup — does stop() release file handles, audio buffers, threads?
- [ ] Real-time processor lifecycle — started/stopped atomically with playback?
- [ ] Can `stop()` race with `play()` leaving player in undefined state?

### Dimension 4: Audio I/O

**Key files**: `auralis/io/unified_loader.py`, `auralis/io/results.py`

**Check**:
- [ ] Format coverage — MP3, FLAC, WAV, AAC, OGG, OPUS, M4A all tested?
- [ ] Corrupt file handling — crash vs meaningful error on corrupt header?
- [ ] Large file handling — files > 1GB without OOM?
- [ ] Sample rate detection — always from metadata, never assumed?
- [ ] Channel handling — files with > 2 channels downmixed correctly?
- [ ] FFmpeg subprocess — properly terminated on cancellation? Zombie risk?
- [ ] File path safety — paths validated before passing to FFmpeg?
- [ ] Metadata extraction — ID3/Vorbis/FLAC tags parsed robustly? Malformed tags?

### Dimension 5: Chunked Mastering Loop

**Key files**: `auralis/core/mastering_chunk_loop.py`, `auralis/core/mastering_process_chunk.py`, `auralis/core/mastering_prepare.py`, `auralis/core/mastering_notch_context.py`, `auralis/core/simple_mastering.py`, `vendor/auralis-dsp/src/py_bindings.rs` (`process_chunks`)

This dimension replaced the old "Parallel Processing" one: *auralis/optimization/parallel_processor.py* and its *parallel/* package were deleted as unreachable in #4565. The chunk model now lives in the engine's sequential-with-carried-context mastering loop and in the backend's concurrent chunk processor (the latter belongs to `/audit-backend`, not here). Do not report the deleted processor's absence.

**Check**:
- [ ] Chunk independence — does each iteration get a true copy, or a view into the caller's buffer that a later stage mutates in place?
- [ ] Carried context — state threaded between chunks (notch context, level/gain smoothing, limiter memory) must evolve continuously. Does a reset mid-stream produce an audible step at a chunk boundary?
- [ ] Reassembly — `sum(chunk_lengths) == total_length`, and chunks are concatenated in index order, never arrival order?
- [ ] Boundary continuity — is the seam between consecutive chunks free of discontinuity? Two identical inputs mastered whole vs chunked should match within float tolerance.
- [ ] Partial failure — one chunk raising must not silently drop audio or leave the carried context poisoned for the remainder.
- [ ] Analysis scope — are per-chunk measurements (loudness, crest, spectral balance) computed against the whole-track target, not re-derived per chunk? Re-deriving makes mastering drift across a long track.
- [ ] Rust `process_chunks` boundary — does the PyO3 path agree with the Python path on chunk length, dtype, and channel layout? Does it release the GIL?
- [ ] Last chunk — a short final chunk must not be zero-padded into the output (added samples) or dropped (missing samples).

### Dimension 6: Analysis Subsystem

**Key files**: `auralis/analysis/fingerprint/`, `auralis/analysis/ml/`, `auralis/analysis/quality/`, `auralis/analysis/quality_assessors/`

**Check**:
- [ ] Fingerprint determinism — same file always produces same fingerprint?
- [ ] Resource bounds — pathological files (silence, noise, 6hr podcast) bounded in CPU/memory?
- [ ] Batch vs streaming — both analysis paths produce identical results?
- [ ] ML model lifecycle — loaded once and reused, not reloaded per-track?
- [ ] Quality metrics — LUFS, dynamic range, distortion correctly computed?
- [ ] Thread safety — concurrent analysis tasks don't interfere?
- [ ] KeyboardInterrupt — analysis can be interrupted cleanly? (Fix: `53cef6b4`)

### Dimension 7: Library & Database

**Key files**: `auralis/library/database.py`, `auralis/library/repositories/`, `auralis/library/models/`, `auralis/library/scanner/`, `auralis/library/migrations/`, `auralis/library/migration_manager.py`, `auralis/library/sidecar_manager.py`, `auralis/library/resource_monitor.py`

**Check**:
- [ ] Repository pattern — ALL database access via the 13 repository classes? No raw SQL?
- [ ] Detached ORM instances — repositories `expunge()` what they return, so any relationship a query did not eager-load raises `DetachedInstanceError` when `to_dict()` touches it. Do read paths carry `selectinload()`, and does `to_dict()` go through `_safe_collection()` / `_safe_scalar()` in `auralis/library/models/core.py`? `refresh()` expires without re-applying query options — post-commit paths must touch the relationship while still attached.
- [ ] `BaseRepository._session_scope()` — the context manager exists in `auralis/library/repositories/base.py`, but most call sites still hand-roll session lifecycle. Flag leaks/missing rollbacks in the hand-rolled ones (the bulk migration itself is tracked debt, not a new finding).
- [ ] SQLite config — `check_same_thread=False`, `pool_pre_ping=True` set?
- [ ] N+1 queries — list operations use `selectinload()`?
- [ ] Scanner robustness — symlinks, permission errors, Unicode filenames handled?
- [ ] Migration safety — can run while app is serving requests? (Uses file locking)
- [ ] Concurrent scans — two scan operations can't conflict?
- [ ] Cleanup — `cleanup_missing_files` handles large libraries without OOM? (Fix: `bd94fd59`)
- [ ] Engine disposal — SQLAlchemy engine disposed on close? (Fix: `8adb8d0a`)

## Phase 1: Setup

1. Parse `$ARGUMENTS` for `--focus`, `--depth`, `--limit`
2. `mkdir -p /tmp/audit/engine`
3. Fetch dedup baseline: `gh issue list --limit 200 --json number,title,state,labels > /tmp/audit/engine/issues.json`
4. Scan `docs/audits/` for prior engine audit reports

## Phase 2: Launch Dimension Agents

Launch one Agent-tool subagent per dimension (max 3 concurrent). Each agent writes its output to `/tmp/audit/engine/dim_<N>.md`.

Every agent prompt MUST include:
- The project root is `/mnt/data/src/matchering`
- The depth parameter value
- The limit parameter value (if set)
- Reference to dedup file: `/tmp/audit/engine/issues.json`
- The context management rules from `_audit-common.md`
- The per-finding format below

### Per-Finding Format

```
### <ID>: <Short Title>
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Dimension**: Sample Integrity | DSP Pipeline | Player State | Audio I/O | Chunked Mastering | Analysis | Library & Database
- **Location**: `<file-path>:<line-range>`
- **Status**: NEW | Existing: #NNN | Regression of #NNN
- **Description**: What is wrong and why
- **Evidence**: Code snippet or exact call path
- **Impact**: What breaks — audio artifacts, crashes, data loss
- **Suggested Fix**: Brief direction (1-3 sentences)
```

Dimension → Output mapping:
- Dimension 1 (Sample Integrity) → `/tmp/audit/engine/dim_1.md`
- Dimension 2 (DSP Pipeline) → `/tmp/audit/engine/dim_2.md`
- Dimension 3 (Player State) → `/tmp/audit/engine/dim_3.md`
- Dimension 4 (Audio I/O) → `/tmp/audit/engine/dim_4.md`
- Dimension 5 (Chunked Mastering) → `/tmp/audit/engine/dim_5.md`
- Dimension 6 (Analysis) → `/tmp/audit/engine/dim_6.md`
- Dimension 7 (Library & Database) → `/tmp/audit/engine/dim_7.md`

## Phase 3: Merge

1. Read all `/tmp/audit/engine/dim_*.md` files
2. Combine into `docs/audits/AUDIT_ENGINE_<TODAY>.md` with structure:
   - **Executive Summary** — Total findings by severity, key themes, most impactful issues
   - **Findings** — Grouped by severity (CRITICAL first), deduplicated across dimensions
   - **Relationships** — How findings interact, shared root causes
   - **Prioritized Fix Order** — What to fix first and why
3. Remove cross-dimension duplicates (same file:line found by multiple dimensions)

## Phase 4: Cleanup

1. `rm -rf /tmp/audit/engine`
2. Inform user the report is ready
3. Suggest: `/audit-publish docs/audits/AUDIT_ENGINE_<TODAY>.md`

## Labels

Use labels when publishing: severity label + domain labels (`audio-integrity`, `dsp`, `player`, `library`, `fingerprint`, `performance`) + `bug`
