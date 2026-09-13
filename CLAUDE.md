# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) — and, via the
pointer in `AGENTS.md`, other agent tooling — when working with code in this
repository. This is the single hand-maintained source of truth; `AGENTS.md`
is a thin stub that points here rather than an independent copy (#5027).

**Project**: Auralis — Music player with real-time audio enhancement
**Version**: 1.5.1 recovery milestone (`auralis/version.py` is source of truth; not tagged)
**Python**: 3.14+ | **Node**: 24+ | **Rust**: Required (PyO3 DSP module)
**License**: AGPL-3.0 (dual-licensed, see COMMERCIAL_LICENSE.md)

## Commands

```bash
# Run verified recovery components (the root/Electron launcher is a known blocker)
# `--python-preference only-managed` or uv silently picks a stale pyenv shim
uv venv --python-preference only-managed && source .venv/bin/activate
uv pip install -r requirements.txt
cd auralis-web/backend && python main.py --dev             # Backend :8765
cd auralis-web/frontend && pnpm install && pnpm run dev   # pnpm is the only supported JS package manager (#4357)

# Test — scope first, widen once. `-q` keeps passes as dots; `-v` dumps every
# test name into context for ~6,552 tests.
python -m pytest -q -m "not slow" tests/auralis/dsp     # A domain (the normal inner loop)
python -m pytest tests/path.py::test_name -vv -s        # Single test
cd auralis-web/frontend && pnpm run test:memory         # Frontend (2GB heap; OOMs without it)

# Whole backend suite. NOT "~2-3 min" — it is tens of minutes, and these two
# files HANG when run as whole files, so exclude them exactly like CI does:
python -m pytest -q -m "not slow" \
  --ignore=tests/backend/test_system_api.py \
  --ignore=tests/concurrency/test_thread_safety.py

# Type check
mypy auralis/ auralis-web/backend/ --ignore-missing-imports
cd auralis-web/frontend && pnpm run type-check

# Build Rust DSP (required before first run)
cd vendor/auralis-dsp && maturin develop
```

## CI gates and the failure baselines

Neither test suite is green, so both CI gates are **ratchets**: a run is judged
against a checked-in list of known failures and fails only on failures *not* in
that list. The list may shrink, never grow.

| Workflow | Runs | Baseline | Gate script |
|---|---|---|---|
| `frontend-test.yml` | `pnpm run test:ci` | `auralis-web/frontend/test-baseline.json` | `scripts/check-test-baseline.mjs` |
| `backend-tests.yml` | `pytest --junitxml` | `pytest-baseline.json` | `scripts/check_pytest_baseline.py` |
| `frontend-typecheck.yml` | `pnpm run type-check:prod` | — (must be clean) | — |

The test step itself always exits 0; **the baseline step is what decides the
job**. Both gates also fail on a missing report or 0 collected tests, so a
crashed runner cannot pass as green.

**When you fix a failing test, regenerate the baseline** — otherwise the entry
lingers and silently re-permits that exact failure:

```bash
cd auralis-web/frontend && pnpm run test:ci && pnpm run test:baseline:update
```

Generate a baseline from a **CI artifact**, not a local run — a baseline built
against a different interpreter or dependency set reports spurious new failures.

`pytest-baseline.json` **is tracked** (216 entries, regenerated 2026-08-19 in
`7c03249e`) — the "it does not exist yet" note that stood here was true when
written and is not any more (#4974). `backend-tests.yml` still fails, but on the
ratchet doing its job rather than on a missing file: the ratchet rejects
failures absent from the list. Read the failing run's *baseline* step, not the
pytest step, to see which. #5091 (69 entries whose tests now pass, silently
re-permitted) is now CLOSED — `check_pytest_baseline.py --strict-stale` is wired
into `backend-tests.yml` and fails the job on any stale baseline entry too, not
just on new unlisted failures.

Do not add a `version:` input to `pnpm/action-setup`: every `package.json` here
declares `packageManager`, and supplying both makes the action hard-error before
it installs anything. Backend CI must stay on Python 3.14 — on 3.13 the
codebase's PEP 649 deferred annotations fail at import and the suite collects
zero tests.

## Codebase Map

```
auralis/                          Core Python audio engine
├── core/                         Processing pipeline
│   ├── hybrid_processor.py         HybridProcessor — main DSP pipeline
│   ├── simple_mastering.py         SimpleMastering algorithm
│   ├── processing/                 Mode processors (adaptive, continuous, hybrid, realtime pipeline)
│   ├── config/                     Processing configuration (UnifiedConfig)
│   └── recording_type_detector.py  Content type detection
├── dsp/                          Signal processing
│   ├── basic.py                    DSP primitives
│   ├── advanced_dynamics.py        Dynamics control
│   ├── eq/                         Psychoacoustic EQ (psychoacoustic_eq.py)
├── analysis/                     Audio analysis (largest module, 57 files)
│   ├── fingerprint/                25D fingerprinting system
│   │   ├── analyzers/                Batch & streaming analyzers
│   │   ├── metrics/                  Spectral, harmonic, temporal
│   │   └── utilities/                DSP ops, backend selection
│   ├── content/                    Content-aware analysis
│   ├── ml/                         Genre classification (neural nets)
│   └── quality/                    Quality assessment (loudness, distortion, DR)
├── player/                       Playback engine
│   ├── enhanced_audio_player.py    Main player with adaptive DSP
│   ├── gapless_playback_engine.py  Gapless playback
│   ├── queue_controller.py         Queue management
│   └── realtime/                   Real-time processing (RealtimeProcessor, AutoMasterProcessor)
├── library/                      SQLite library (~/.auralis/library.db)
│   ├── database.py                 LibraryDatabase (engine, migration, sessions, scan slots)
│                                     Startup stores it under the globals key `library_database`
│                                     (renamed in #5162; it used to name the class #4915 deleted)
│   ├── repositories/               13 repos + base.py (BaseRepository) + factory.py
│   │                                 (track, album, artist, playlist, genre, stats,
│   │                                 fingerprint, fingerprint_scheduler, fingerprint_stats,
│   │                                 queue, queue_history, settings, similarity_graph)
│   ├── scanner/                    Folder scanning (a package, not a module)
│   └── migration_manager.py        DB migrations (schema v18)
├── io/                           Audio I/O
│   ├── unified_loader.py           Unified loading (FFmpeg, SoundFile)
│   └── results.py                  Output formats (pcm16, pcm24)
├── optimization/                 Performance — LIVE, not test-only (#5142)
│   ├── performance_optimizer.py    get_performance_optimizer(); applied at import
│   │                                 time by core/hybrid_processor.py
│   ├── config.py
│   └── acceleration/, caching/, memory/, profiling/
│                                   (parallel/ + parallel_processor.py deleted #4565;
│                                    NO rust_integration.py — see #5168)
├── services/                     Background services (fingerprint, artwork)
├── learning/                     Preference engine, reference analysis
└── utils/                        Logging, helpers, preview creator

auralis-web/
├── backend/                      FastAPI REST + WebSocket (:8765)
│   ├── main.py                     App entry point
│   ├── routers/                    20 registered routers (player, library, albums,
│   │                                 artists, playlists, enhancement, metadata,
│   │                                 artwork, system, similarity, streaming...)
│   ├── core/                       Engine-facing layer. NOT top-level (#4627):
│   │   ├── processing_engine.py      Audio processing orchestration
│   │   ├── chunked_processor.py      15s chunks rendered w/ context, emitted as
│   │   │                               10s NON-overlapping segments (no crossfade)
│   │   ├── audio_stream_controller.py  WebSocket audio streaming
│   │   ├── stream_*.py               normal/enhanced/seek streaming paths
│   │   ├── chunk_boundaries.py       Sole chunk-geometry authority
│   │   └── executors.py              Streaming + I/O thread pools (#5086)
│   ├── schemas.py                  Request/response schemas
│   └── services/, config/          Service layer, startup/config
└── frontend/                     React 18 + TypeScript + Vite + Redux
    └── src/
        ├── components/               UI components
        ├── hooks/                    Domain hooks (player, library, enhancement,
        │                               websocket, api, app, fingerprint, shared)
        ├── store/                    Redux state management
        ├── design-system/            Design tokens (single source of truth)
        ├── services/                 API clients
        └── test/                     Test utilities

vendor/auralis-dsp/               Rust DSP via PyO3 (HPSS, YIN, Chroma)
desktop/                          Electron wrapper
tests/                            ~6,552 test functions (576 files) across 18 subdirs (auralis, backend,
                                    integration, boundaries, concurrency, security, load_stress, regression...)
docs/                             18 topic dirs (development, features, frontend...)
```

Structural counts above (analysis file count, router count, test file/function
counts, docs topic-dir count) and the matching table in
`.claude/commands/_audit-common.md` are two hand-maintained copies of the same
numbers and will drift apart again if only one is edited. Run
`python scripts/check_doc_counts.py` to recompute both from the live tree
before updating either file, and update both together (#4982).

## Architecture Flow

```
User → FastAPI (REST + WebSocket :8765) → Backend Services
         → LibraryDatabase (SQLite) → HybridProcessor (DSP pipeline)
         → ChunkedProcessor (15s w/ context → 10s non-overlapping) → WS → React (Redux)
```

## Critical Invariants

**Audio processing** — sample count preservation is critical for gapless playback:
```python
assert len(output) == len(input)              # Never change sample count
assert isinstance(output, np.ndarray)         # Always NumPy, never lists
assert output.dtype in [np.float32, np.float64]
output = audio.copy()                         # Never modify in-place
```

**Player state**: position ≤ duration, queue index valid, state changes atomic (RLock).
**Database**: thread-safe pooling (`pool_pre_ping=True`), no N+1 (`selectinload()`), all access via repositories.

**Detached ORM instances** — repositories `expunge()` everything they return, so
any relationship a query did not eager-load raises `DetachedInstanceError` when
`to_dict()` touches it. Two rules, both required:
- the repository's read paths carry `selectinload(Model.rel)` (define the option
  tuple once at module scope so a new read path cannot silently omit it);
- `to_dict()` reads relationships through `_safe_collection()` / `_safe_scalar()`
  in `library/models/core.py`, which degrade to `[]` / `None` and log a WARNING
  naming the missing eager-load.

`refresh()` expires an instance without re-applying query options, so
post-commit paths must touch the relationship while still attached.

**Chunk geometry**: `CHUNK_DURATION` / `INTERVAL` / `OVERLAP` / `CONTEXT` come
only from `backend/core/chunk_boundaries.py`; `content_chunk_count()` is the sole
chunk-counting authority (overlap-aware — not `ceil(duration / CHUNK_DURATION)`).
Cached chunk files are 16-bit PCM WAV, not float32.

## Patterns

**Python backend**: Routers auto-included via `include_router()`. All handlers `async def`. Errors via `HTTPException`. Shared state protected with `threading.RLock()`.

**React frontend**: `@/` absolute imports only. Colors via `import { tokens } from '@/design-system'`. Components < 300 lines. Tests use `vi.*` (Vitest), `render` from `@/test/test-utils`.

**Audio DSP**: Load metadata (sample rate, channels) BEFORE processing. Vectorize with NumPy (chunks, not samples). Copy before modify.

## Principles

1. **DRY** — Improve existing code, never duplicate. Use utilities for shared logic.
2. **Modular** — < 300 lines per module, single responsibility.
3. **No variants** — No "Enhanced"/"V2"/"Advanced" copies. Refactor in-place.
4. **Repository pattern** — All DB via `auralis/library/repositories/`, never raw SQL.
5. **Deferrals are greppable** — Write a deferred decision as `# TODO(#NNNN):`
   with a real issue number, not as prose ("For now, …", "Temporarily …").
   Prose deferrals are invisible to any marker sweep, so `/audit-tech-debt` has
   to fall back to a high-recall prose grep that cannot distinguish a deferral
   from an ordinary sentence (#4564). Genuine marker debt in shipped code —
   `auralis/`, `auralis-web/`, `vendor/` — is **1**: `core/hybrid_processor.py`
   carries `TODO(#5295)` for the retirement of `DynamicsProcessor.process()`.
   It was 0 until #5152, which is worth reading as the rule working rather than
   breaking: that marker replaced the prose "Retiring it is tracked separately",
   so the count going 0 → 1 records debt that already existed and was
   unsweepable. `tests/` holds **4**, each citing an OPEN issue (#5172 ×2,
   #5173, #5174) — #5171 was the fifth until it was fixed and closed. Keep both
   figures honest by linking the issue instead of leaving a bare `TODO`.
   The scope matters: that "0" was quoted repo-wide for weeks while every
   genuine marker in the tree sat in `tests/`, uncounted (#5143), so
   `/audit-tech-debt` now reports the two censuses as separate lines.

## Git

Branch from `master`. Prefixes: `feature/`, `fix/`, `refactor/`, `docs/`. Commit:
`type: description`. Before PR, run the **scoped** tests for what you touched plus
`mypy` / `type-check` — the full suite is a CI job, not an inner loop.

Commits to `master` do not auto-push. `git log` / `git status` can shift mid-session:
work is done in this repo from parallel terminals, so re-check rather than assuming
staleness. Commit each logical unit as soon as its scoped tests pass.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 8765 in use | `lsof -ti:8765 \| xargs kill -9` |
| Frontend tests OOM | `pnpm run test:memory` (2GB heap) |
| Database locked | Kill python, delete `~/.auralis/library.db` |
| Rust module missing | `cd vendor/auralis-dsp && maturin develop` |
| pytest run never finishes | You included `test_system_api.py` / `test_thread_safety.py` — scope or `--ignore` them |
| CI red in <20s | Setup failure, not tests — check `pnpm/action-setup` / Python version before reading test output |

## Reference Docs

- [TESTING_GUIDELINES.md](docs/development/TESTING_GUIDELINES.md)
- [WEBSOCKET_API.md](auralis-web/backend/WEBSOCKET_API.md)
- [FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md)
