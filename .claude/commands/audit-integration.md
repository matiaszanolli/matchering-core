---
description: "Trace 9 critical data flows across audio engine, FastAPI backend, and React frontend"
argument-hint: "[--focus <flow-names>] [--depth shallow|deep] [--limit <N>]"
---

# Backend-Frontend-Engine Integration Audit

Trace the 9 critical data flows between the Auralis audio engine, FastAPI backend, and React frontend.

**Architecture**: This is an orchestrator. Each flow runs as an Agent-tool subagent (`subagent_type: general-purpose`, `model: sonnet`). Max 3 run concurrently.

See `.claude/commands/_audit-common.md` for project layout, severity framework, methodology, context management rules, deduplication, and finding format.

## Parameters (from $ARGUMENTS)

- `--flows <numbers>`: Comma-separated flow numbers (e.g., `1,3,5`). Default: all 9.
- `--depth shallow|deep`: `shallow` = check key patterns only; `deep` = trace full data paths. Default: `deep`.
- `--limit <N>`: Stop after N findings. Default: unlimited.

## Severity Examples

| Severity | Integration-Specific Examples |
|----------|------------------------------|
| **CRITICAL** | Dropped audio chunks across boundary, sample rate mismatch between engine and streaming, truncated playback |
| **HIGH** | WebSocket message schema mismatch, timeout causing playback stutter, unhandled error propagation across layers |
| **MEDIUM** | Different field names for same concept, inconsistent error codes, stale frontend state after backend update |
| **LOW** | Missing API schema docs, undocumented WebSocket message types, unused response fields |

## Flows to Trace

### Flow 1: Track Playback (Full Pipeline)

| Step | Layer | File |
|------|-------|------|
| User clicks play | Frontend | `src/hooks/player/` hooks, `src/store/` player slice |
| REST request | Frontend | `src/services/` API client |
| Play endpoint | Backend | `auralis-web/backend/routers/player.py` |
| Audio loading | Engine | `auralis/io/unified_loader.py` |
| Processing | Engine | `auralis/core/hybrid_processor.py` → `simple_mastering.py` |
| Chunking | Backend | `chunked_processor.py` (15s chunks, 10s interval, 5s overlap crossfade) |
| WebSocket stream | Backend | `audio_stream_controller.py` |
| Audio playback | Frontend | WebSocket hook → Web Audio API |

**Check**: Sample rate consistency from file to playback. Chunk boundaries — do crossfades preserve continuity? WebSocket binary frame format — does the frontend decode it correctly? What happens when processing is slower than playback?

### Flow 2: Library Browsing

| Step | Layer | File |
|------|-------|------|
| User navigates | Frontend | `src/hooks/library/`, `src/store/` library slice |
| REST requests | Frontend | `src/services/` API client |
| Library endpoints | Backend | `routers/` (albums, artists, playlists, tracks) |
| Database queries | Engine | `auralis/library/database.py` (`LibraryDatabase`) → `repositories/` |
| Response format | Backend | `schemas.py` |

**Check**: Pagination — consistent between frontend expectations and backend response? Field naming — camelCase (frontend) vs snake_case (backend)? Null handling — what happens when optional metadata is missing? Large libraries — does the frontend handle 100k+ tracks?

### Flow 3: Audio Enhancement

| Step | Layer | File |
|------|-------|------|
| User adjusts settings | Frontend | `useEnhancementControl()` local state — the live source of truth (`playerSlice.preset`/`intensity` are dead) |
| Settings API call | Frontend | `src/services/` |
| Enhancement endpoint | Backend | `auralis-web/backend/routers/enhancement.py` |
| Runtime settings | Backend | shared `enhancement_settings` dict (mutated in place, seeded at startup from UserSettings) |
| Processing config | Engine | `auralis/core/config/unified_config.py` (UnifiedConfig) — the only config layer since #4918 |
| DSP pipeline | Engine | `auralis/core/hybrid_processor.py` → DSP modules |
| Real-time application | Engine | `auralis/player/realtime/` |

**Check**: Config format — do frontend slider values map correctly to engine parameters? Does the transport actually pass the *current* preset/intensity through to playback, or re-read stale Redux state? Range validation — can the frontend send out-of-range values? Real-time vs offline — is the same config used for both paths? Do the enhancement settings participate in the chunk cache key, or can a settings change serve cached audio mastered with the *previous* settings? Latency — does enhancement cause audible gaps?

### Flow 4: Library Scanning

| Step | Layer | File |
|------|-------|------|
| User adds folder | Frontend | Library management hooks |
| Scan request | Backend | `auralis-web/backend/routers/library_scan.py` (browse endpoints live in `auralis-web/backend/routers/library.py`) |
| Auto-scan | Backend | `auralis-web/backend/services/library_auto_scanner.py` (background folder watcher) |
| Filesystem scan | Engine | `auralis/library/scanner/` |
| Metadata extraction | Engine | `auralis/io/unified_loader.py` |
| Database insert | Engine | `auralis/library/database.py` → repositories |
| Progress updates | Backend | WebSocket or polling |
| UI update | Frontend | Redux store |

**Check**: Progress reporting — does the frontend receive scan progress? Error handling — what happens when a file can't be read? Duplicate detection — does rescan handle already-imported tracks? Large folders — timeout or chunking?

### Flow 5: WebSocket Lifecycle

| Step | Layer | File |
|------|-------|------|
| Connection init | Frontend | `src/hooks/websocket/`, `src/contexts/WebSocketContext.tsx` |
| WS accept + checks | Backend | `auralis-web/backend/ws_handlers/connection.py`, `auralis-web/backend/websocket/websocket_security.py` |
| Message routing | Backend | `auralis-web/backend/ws_handlers/messages.py`, `playback_commands.py`, `playback_control.py` |
| Protocol contract | Backend | `auralis-web/backend/websocket/websocket_protocol.py`, `auralis-web/backend/core/stream_protocol.py`, `auralis-web/backend/core/stream_messages.py` |
| Binary audio frames | Backend | `auralis-web/backend/core/encoding/wav_encoder.py`, `auralis-web/backend/core/audio_stream_controller.py` |
| Frame decode | Frontend | WebSocket hook → Web Audio API |

**Check**: Connection establishment — handshake protocol? Message types — are all types documented and handled on both sides? Binary vs text frames — consistent usage? Reconnection — does the frontend re-establish state after disconnect? Backpressure — what happens when the frontend can't consume frames fast enough?

### Flow 6: Fingerprint & Similarity

| Step | Layer | File |
|------|-------|------|
| Similarity request | Frontend | Fingerprint/similarity hooks |
| Similarity endpoints | Backend | `auralis-web/backend/routers/similarity.py`, `auralis-web/backend/routers/similarity_graph.py` (shared helpers in `auralis-web/backend/routers/similarity_common.py`) |
| Fingerprint queue/status | Backend | `auralis-web/backend/routers/fingerprint_queue.py`, `auralis-web/backend/routers/fingerprint_status.py`, `auralis-web/backend/analysis/fingerprint_generator.py` |
| Fingerprint engine | Engine | `auralis/analysis/fingerprint/` |
| Database lookup | Engine | `auralis/library/repositories/` (fingerprint, similarity repos) |
| Results format | Backend | `schemas.py` |

**Check**: Fingerprint format — is it consistent between compute and lookup? Similarity scores — range and precision? Missing fingerprints — graceful fallback? Batch vs single — consistent API?

### Flow 7: Artwork

| Step | Layer | File |
|------|-------|------|
| Artwork request | Frontend | Image components, hooks |
| Artwork endpoint | Backend | `auralis-web/backend/routers/artwork.py` |
| Artwork extraction | Engine | `auralis/services/artwork_service.py`, `auralis/library/artwork.py` |
| Thumbnail cache | Backend | `auralis-web/backend/core/thumbnail_cache.py` — on-disk, content-addressed on source mtime/size (#4447) |
| Image serving | Backend | Static file or stream response |

**Check**: Image formats — are all common formats handled? Cache headers — are they set correctly, and do they agree with the on-disk thumbnail cache's own invalidation? Does the thumbnail key's size bucket match the sizes the frontend actually requests, or does every request miss? Missing artwork — fallback behavior? Large images — resizing on backend or frontend?

### Flow 8: Seek & Rebuffer

| Step | Layer | File |
|------|-------|------|
| User drags the scrubber | Frontend | `auralis-web/frontend/src/hooks/player/`, player slice in `auralis-web/frontend/src/store/` |
| Seek command | Frontend → Backend | WebSocket message → `auralis-web/backend/ws_handlers/playback_commands.py` |
| Seek stream path | Backend | `auralis-web/backend/core/stream_seek.py`, `auralis-web/backend/core/stream_chunk_ops.py` |
| Chunk index math | Backend | `auralis-web/backend/core/chunk_boundaries.py` (`content_chunk_count()`, overlap-aware) |
| Source seek | Backend | `auralis-web/backend/core/seekable_source.py` (converts non-seekable sources once, #4737) |
| Cache lookup | Backend | `auralis-web/backend/core/chunk_cache.py` + `file_signature.py` |
| Prefetch / buffer reset | Backend | `auralis-web/backend/core/stream_prefetch.py`, `auralis-web/backend/core/proactive_buffer.py` |
| Level continuity | Backend | `auralis-web/backend/core/level_manager.py` |
| Resume playback | Frontend | WebSocket hook → Web Audio API |

**Check**: Position units — does the frontend send seconds where the backend expects samples (or ms)? Does the reported position after seek match what was requested, or drift by an overlap width? Are stale pre-seek chunks drained from both the backend buffer and the frontend audio queue, or does the user hear pre-seek audio after the jump? Rapid consecutive seeks — does the last one win, or do two streams interleave? Seek past end / to 0 — clamped consistently on both sides? Does the first post-seek chunk arrive at the right level, given it has no predecessor context?

### Flow 9: Queue & Playback State

| Step | Layer | File |
|------|-------|------|
| User reorders / adds / removes | Frontend | Queue components, `auralis-web/frontend/src/store/` |
| Queue request | Backend | `auralis-web/backend/routers/player.py` |
| Queue service | Backend | `auralis-web/backend/services/queue_service.py`, `queue_enrichment.py`, `queue_protocols.py` |
| Engine queue | Engine | `auralis/player/queue_controller.py` (authoritative order) |
| Persisted queue | Engine | `auralis/library/repositories/` (queue, queue_history) |
| State broadcast | Backend | `auralis-web/backend/core/state_manager.py` → WebSocket |
| UI update | Frontend | Redux store |

**Check**: The engine queue and the backend state manager hold *different shapes* of the same queue — the engine's is filepath-only and authoritative on order, the state manager's carries rich track info and goes stale after add/remove/move (#4374 bridged them via enrichment in `queue_service.py`). Does every mutation path re-enrich, or do some return the stale copy? Is `filepath` ever leaked to the client (it must not be — #3205; clients get `format`)? After a reorder, does the frontend's optimistic order match what the engine actually plays next? Does queue persistence survive a restart, and does the persisted index still point at the same track after a library rescan?

## Per-Flow Verification Checklist

For EVERY flow, systematically check:
- [ ] **Schema match**: Request fields sent == fields expected. Response fields returned == fields consumed.
- [ ] **Error handling**: Does the frontend handle 4xx and 5xx responses gracefully?
- [ ] **Timeouts**: Frontend timeout vs backend processing time — compatible?
- [ ] **Data types**: Sample rates, bit depths, timestamps, durations — consistent units and precision?
- [ ] **Null handling**: What happens when optional fields are missing on either side?
- [ ] **Case conversion**: camelCase ↔ snake_case at the API boundary?

## Phase 1: Setup

1. Parse `$ARGUMENTS` for `--flows`, `--depth`, `--limit`
2. `mkdir -p /tmp/audit/integration`
3. Fetch dedup baseline: `gh issue list --limit 200 --json number,title,state,labels > /tmp/audit/integration/issues.json`
4. Scan `docs/audits/` for prior integration audit reports

## Phase 2: Launch Flow Agents

Launch one Agent-tool subagent per flow (max 3 concurrent). Each agent writes its output to `/tmp/audit/integration/flow_<N>.md`.

Every agent prompt MUST include:
- The project root is `/mnt/data/src/matchering`
- The depth parameter value
- The limit parameter value (if set)
- The per-flow verification checklist from this file
- Reference to dedup file: `/tmp/audit/integration/issues.json`
- The context management rules from `_audit-common.md`
- The per-finding format below

### Per-Finding Format

```
### <ID>: <Short Title>
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Flow**: <which of the 9 flows>
- **Boundary**: <sender layer> → <receiver layer>
- **Location**: `<sender-file>:<line>` → `<receiver-file>:<line>`
- **Status**: NEW | Existing: #NNN | Regression of #NNN
- **Description**: What is wrong at this boundary
- **Evidence**: Code from both sides showing the mismatch
- **Impact**: What breaks and when
- **Suggested Fix**: Which side should change and how
```

Flow → Output mapping:
- Flow 1 (Track Playback) → `/tmp/audit/integration/flow_1.md`
- Flow 2 (Library Browsing) → `/tmp/audit/integration/flow_2.md`
- Flow 3 (Audio Enhancement) → `/tmp/audit/integration/flow_3.md`
- Flow 4 (Library Scanning) → `/tmp/audit/integration/flow_4.md`
- Flow 5 (WebSocket Lifecycle) → `/tmp/audit/integration/flow_5.md`
- Flow 6 (Fingerprint & Similarity) → `/tmp/audit/integration/flow_6.md`
- Flow 7 (Artwork) → `/tmp/audit/integration/flow_7.md`
- Flow 8 (Seek & Rebuffer) → `/tmp/audit/integration/flow_8.md`
- Flow 9 (Queue & Playback State) → `/tmp/audit/integration/flow_9.md`

## Phase 3: Merge

1. Read all `/tmp/audit/integration/flow_*.md` files
2. Combine into `docs/audits/AUDIT_INTEGRATION_<TODAY>.md` with structure:
   - **Executive Summary** — Total findings by severity, key themes, most impactful boundary mismatches
   - **Flow Coverage Matrix** — Table of all 9 flows with boundary-check status
   - **Findings** — Grouped by severity (CRITICAL first), deduplicated across flows
   - **Relationships** — Shared root causes, cross-flow boundary issues
   - **Prioritized Fix Order** — What to fix first and why
3. Remove cross-flow duplicates (same boundary issue found by multiple flows)

## Phase 4: Cleanup

1. `rm -rf /tmp/audit/integration`
2. Inform user the report is ready
3. Suggest: `/audit-publish docs/audits/AUDIT_INTEGRATION_<TODAY>.md`

## Labels

Use labels when publishing: severity label + domain labels (`backend`, `frontend`, `audio-integrity`, `websocket`, `streaming`, `library`, `fingerprint`) + `bug`
