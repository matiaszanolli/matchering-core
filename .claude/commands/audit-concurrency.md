---
description: "Audit for race conditions, missing locks, thread-safety violations, state-machine bugs, unsafe concurrent access"
argument-hint: "[--focus <dimensions>] [--depth shallow|deep] [--limit <N>]"
---

# Concurrency and State Integrity Audit

Perform a deep audit of Auralis for race conditions, missing locks, thread safety violations, state machine bugs, and unsafe concurrent access.

**Architecture**: This is an orchestrator. Each dimension runs as an Agent-tool subagent (`subagent_type: general-purpose`, `model: sonnet`). Max 3 run concurrently.

See `.claude/commands/_audit-common.md` for project layout, severity framework, methodology, context management rules, deduplication, and finding format.

## Parameters (from $ARGUMENTS)

- `--focus <dimensions>`: Comma-separated dimension numbers or names (e.g., `1,3` or `player,pipeline,backend`). Default: all 5.
- `--depth shallow|deep`: `shallow` = check key patterns only; `deep` = trace full execution paths. Default: `deep`.
- `--limit <N>`: Stop after N findings. Default: unlimited.

## Severity Examples

| Severity | Concurrency-Specific Examples |
|----------|------------------------------|
| **CRITICAL** | Concurrent writes to audio buffer, race between player state and output, database corruption from parallel access |
| **HIGH** | Seek during gapless transition, queue modification during playback, parallel DSP chunk processing errors |
| **MEDIUM** | Unprotected shared state, missing copy-before-modify, in-memory state without persistence |
| **LOW** | Coarse-grained locks, unnecessary serialization, redundant locking |

## Audit Dimensions

### Dimension 1: Player Thread Safety

**Key files**: `auralis/player/enhanced_audio_player.py`, `auralis/player/gapless_playback_engine.py`, `auralis/player/queue_controller.py`, `auralis/player/realtime/`

**Check**:
- [ ] RLock usage — is every shared state access protected? Are locks properly scoped?
- [ ] State transitions (play/pause/stop/seek) — are they atomic? Can a seek overlap with a gapless transition?
- [ ] Queue modifications during playback — is the queue protected while iterating?
- [ ] Position/duration invariant (`position <= duration`) — can a race violate this?
- [ ] Audio buffer access — can the playback thread and processing thread access the same buffer simultaneously?
- [ ] Callback safety — are player callbacks invoked with or without locks held?
- [ ] Can `stop()` race with `play()` leaving the player in an undefined state?

### Dimension 2: Audio Processing Pipeline

**Key files**: `auralis/core/hybrid_processor.py`, `auralis/core/simple_mastering.py`, `auralis/core/mastering_chunk_loop.py`, `auralis/core/mastering_process_chunk.py`, `vendor/auralis-dsp/`

The engine-side parallel processor (*auralis/optimization/parallel_processor.py* + *parallel/*) was deleted as unreachable in #4565 — audit the engine's sequential chunk loop here, and the backend's concurrent chunk processor in Dimension 3. Do not report the deleted module.

**Check**:
- [ ] Chunk loop — are audio chunks independently copied before processing? Can adjacent chunks interfere through carried context (notch state, level smoothing, limiter memory)?
- [ ] Shared mastering targets — targets derived once per track and read by concurrent chunk workers must be immutable. Is any target dict/array mutated in place by a worker?
- [ ] Copy-before-modify pattern — is `audio.copy()` consistently used before in-place NumPy operations?
- [ ] Rust DSP boundary — does PyO3 correctly handle GIL release during processing? Can concurrent calls corrupt shared state?
- [ ] HybridProcessor chain — if one stage fails mid-array, is the pipeline state consistent?
- [ ] Sample count preservation — can the chunked and whole-file paths produce different lengths?
- [ ] Crossfade between chunks (recent fix `0a5df7a3`) — is the crossfade buffer shared or independent?

### Dimension 3: Backend WebSocket & Streaming

**Key files**: `auralis-web/backend/core/audio_stream_controller.py`, `auralis-web/backend/core/chunked_processor.py`, `auralis-web/backend/core/processing_engine.py`, `auralis-web/backend/core/processor_pool.py`, `auralis-web/backend/core/job_worker.py`, `auralis-web/backend/core/state_manager.py`, `auralis-web/backend/core/proactive_buffer.py`, `auralis-web/backend/core/chunk_cache_manager.py`, `auralis-web/backend/ws_handlers/`, `auralis-web/backend/config/background_workers.py`

**Check**:
- [ ] Multiple WebSocket clients — can two clients request different tracks simultaneously?
- [ ] Chunked processor state — is it per-request or shared? Can concurrent requests corrupt chunk state?
- [ ] Processor pool — is checkout/return in `auralis-web/backend/core/processor_pool.py` leak-free on every early-exit and exception path?
- [ ] Job worker lifecycle — can `auralis-web/backend/core/job_worker.py` die silently and leave the queue stalled? Is there a watchdog?
- [ ] Chunk cache — concurrent writers to the same cache key: torn/partial files, or last-writer-wins? Are writes atomic (temp + rename via `auralis-web/backend/core/encoding/atomic_io.py`), and does the same reasoning hold for `auralis-web/backend/core/thumbnail_cache.py` and `auralis-web/backend/cache/manager.py`?
- [ ] Seek vs in-flight work — a seek arriving mid-stream must cancel prefetch (`auralis-web/backend/core/stream_prefetch.py`) and drain `auralis-web/backend/core/proactive_buffer.py`. Can a cancelled task still deliver a chunk into the post-seek stream, or write a stale entry into the cache?
- [ ] Controller state via helpers — `auralis-web/backend/core/stream_chunk_ops.py` and `stream_fingerprint.py` take the controller instance and read/write its attributes. Two concurrent streams on one controller: is that state per-connection or shared?
- [ ] Streaming semaphores — `stream_enhanced.py` / `stream_normal.py` must release in `finally`; are all early exits accounted for?
- [ ] Processing engine — shared or per-request instances? Thread safety?
- [ ] FastAPI async handlers calling sync audio code — are they using `run_in_executor` / `asyncio.to_thread`? Can blocking calls starve the event loop?
- [ ] Background workers started in the lifespan — are they cancelled and awaited on shutdown?
- [ ] WebSocket disconnect during processing — is cleanup atomic? Resource leaks?

### Dimension 4: Library & Database

**Key files**: `auralis/library/database.py`, `auralis/library/repositories/`, `auralis/library/scanner/`, `auralis/library/migration_manager.py`, `auralis/library/resource_monitor.py`

**Check**:
- [ ] SQLite thread safety — `check_same_thread=False`? Connection pooling config?
- [ ] `pool_pre_ping=True` — is it actually set?
- [ ] Concurrent scans — can two scan operations run simultaneously and cause conflicts? Scan slots are owned by `auralis/library/database.py`; verify acquisition and release are symmetric on every exception path.
- [ ] Detached-instance races — repositories `expunge()` what they return. A post-commit `refresh()` expires the instance *without* re-applying `selectinload()` options, so another thread touching a relationship then raises `DetachedInstanceError`. Are relationships touched while still attached?
- [ ] Repository pattern — any raw SQL bypassing the ORM?
- [ ] Library writes during playback reads — can a scan update a track that's currently playing?
- [ ] Migration execution — safe to run while the app is serving requests? Migrations use inter-process file locking (`fcntl`/`msvcrt`) plus a same-process `threading.Lock`; the file lock alone does NOT serialize threads in one process — verify both are still present.

### Dimension 5: Frontend State Consistency

**Key files**: `auralis-web/frontend/src/store/`, `auralis-web/frontend/src/hooks/`, `auralis-web/frontend/src/services/`

**Check**:
- [ ] Redux dispatch ordering — can WebSocket messages arrive and dispatch out of order?
- [ ] Stale closure bugs in hooks — do effect cleanup functions race with new connections?
- [ ] Optimistic updates — are they reconciled correctly when the backend responds?
- [ ] WebSocket reconnection — does the frontend correctly re-sync state after reconnect?
- [ ] Multiple rapid user actions (skip, skip, skip) — does the frontend handle rapid state changes?

## Phase 1: Setup

1. Parse `$ARGUMENTS` for `--focus`, `--depth`, `--limit`
2. `mkdir -p /tmp/audit/concurrency`
3. Fetch dedup baseline: `gh issue list --limit 200 --json number,title,state,labels > /tmp/audit/concurrency/issues.json`
4. Scan `docs/audits/` for prior concurrency audit reports

## Phase 2: Launch Dimension Agents

Launch one Agent-tool subagent per dimension (max 3 concurrent). Each agent writes its output to `/tmp/audit/concurrency/dim_<N>.md`.

Every agent prompt MUST include:
- The project root is `/mnt/data/src/matchering`
- The depth parameter value
- The limit parameter value (if set)
- Reference to dedup file: `/tmp/audit/concurrency/issues.json`
- The context management rules from `_audit-common.md`
- The per-finding format below

### Per-Finding Format

```
### <ID>: <Short Title>
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Dimension**: Player Thread Safety | Audio Processing | Backend Streaming | Library & Database | Frontend State
- **Location**: `<file-path>:<line-range>`
- **Status**: NEW | Existing: #NNN | Regression of #NNN
- **Trigger Conditions**: Exact timing/concurrency scenario needed
- **Evidence**: Code snippet showing the unsafe pattern
- **Impact**: What state gets corrupted, what audio artifacts occur
- **Suggested Fix**: Lock type, copy pattern, or state machine change needed
```

Dimension → Output mapping:
- Dimension 1 (Player Thread Safety) → `/tmp/audit/concurrency/dim_1.md`
- Dimension 2 (Audio Processing) → `/tmp/audit/concurrency/dim_2.md`
- Dimension 3 (Backend Streaming) → `/tmp/audit/concurrency/dim_3.md`
- Dimension 4 (Library & Database) → `/tmp/audit/concurrency/dim_4.md`
- Dimension 5 (Frontend State) → `/tmp/audit/concurrency/dim_5.md`

## Phase 3: Merge

1. Read all `/tmp/audit/concurrency/dim_*.md` files
2. Combine into `docs/audits/AUDIT_CONCURRENCY_<TODAY>.md` with structure:
   - **Executive Summary** — Total findings by severity, key themes, most impactful races
   - **Concurrency Matrix** — Components, lock types, thread-safety status
   - **Findings** — Grouped by severity (CRITICAL first), deduplicated across dimensions
   - **Relationships** — Shared root causes, compound race conditions
   - **Prioritized Fix Order** — What to fix first and why
3. Remove cross-dimension duplicates (same file:line found by multiple dimensions)

## Phase 4: Cleanup

1. `rm -rf /tmp/audit/concurrency`
2. Inform user the report is ready
3. Suggest: `/audit-publish docs/audits/AUDIT_CONCURRENCY_<TODAY>.md`

## Labels

Use labels when publishing: severity label + `concurrency` + `bug`
