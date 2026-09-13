/**
 * Queue State Slice
 * ~~~~~~~~~~~~~~~~~
 *
 * Redux slice for managing audio queue state including:
 * - Queue tracks
 * - Current queue position
 * - Queue operations (add, remove, reorder)
 *
 * Phase C.3: Component Testing & Integration
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3, see LICENSE for more details
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { QueueTrack as Track } from '@/types/domain';

export type RepeatMode = 'off' | 'all' | 'one';

/** Allowlist for repeat_mode coming off the wire (#4159). An invalid value
 *  would enter queue.repeatMode and silently halt next/previous advancement. */
export const VALID_REPEAT_MODES: readonly RepeatMode[] = ['off', 'all', 'one'];

export const isRepeatMode = (value: unknown): value is RepeatMode =>
  typeof value === 'string' && (VALID_REPEAT_MODES as readonly string[]).includes(value);

export interface QueueState {
  tracks: Track[];
  currentIndex: number;
  isShuffled: boolean;
  repeatMode: RepeatMode;
  isLoading: boolean;
  error: string | null;
  lastUpdated: number;
}

const initialState: QueueState = {
  tracks: [],
  currentIndex: 0,
  isShuffled: false,
  repeatMode: 'off',
  isLoading: false,
  error: null,
  lastUpdated: 0,
};

const queueSlice = createSlice({
  name: 'queue',
  initialState,
  reducers: {
    /**
     * Add track to queue, optionally at a specific position.
     * If position is omitted or out of range the track is appended.
     */
    addTrack: {
      reducer(state, action: PayloadAction<Track, string, { timestamp: number; position?: number }>) {
        const pos = action.meta.position;
        if (pos !== undefined && pos >= 0 && pos <= state.tracks.length) {
          state.tracks.splice(pos, 0, action.payload);
          // Mirror reorderTrack/removeTrack (#4927): everything from `pos`
          // onward just shifted right one slot, so currentIndex must too or
          // it silently points at the newly inserted track instead of the
          // track that was actually playing.
          if (pos <= state.currentIndex) {
            state.currentIndex += 1;
          }
        } else {
          state.tracks.push(action.payload);
        }
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(track: Track, position?: number) {
        return { payload: track, meta: { timestamp: Date.now(), position } };
      },
    },

    /**
     * Add multiple tracks to queue
     */
    addTracks: {
      reducer(state, action: PayloadAction<Track[], string, { timestamp: number }>) {
        state.tracks.push(...action.payload);
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(tracks: Track[]) {
        return { payload: tracks, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Remove track from queue by index
     */
    removeTrack: {
      reducer(state, action: PayloadAction<number, string, { timestamp: number }>) {
        const index = action.payload;
        if (index >= 0 && index < state.tracks.length) {
          state.tracks.splice(index, 1);
          // Adjust currentIndex if needed
          if (index < state.currentIndex) {
            state.currentIndex = Math.max(0, state.currentIndex - 1);
          } else if (index === state.currentIndex && state.currentIndex >= state.tracks.length) {
            state.currentIndex = Math.max(0, state.currentIndex - 1);
          }
          state.lastUpdated = action.meta.timestamp;
        }
      },
      prepare(index: number) {
        return { payload: index, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Reorder track in queue
     */
    reorderTrack: {
      reducer(
        state,
        action: PayloadAction<{ fromIndex: number; toIndex: number }, string, { timestamp: number }>
      ) {
        const { fromIndex, toIndex } = action.payload;
        // Validate both indices like every sibling index-taking reducer (#4432);
        // an out-of-range fromIndex would splice out nothing and insert a literal
        // undefined into tracks.
        const len = state.tracks.length;
        if (
          fromIndex < 0 || fromIndex >= len ||
          toIndex < 0 || toIndex >= len
        ) {
          return;
        }
        if (fromIndex === toIndex) return;

        const [movedTrack] = state.tracks.splice(fromIndex, 1);
        state.tracks.splice(toIndex, 0, movedTrack);

        // Update currentIndex
        if (state.currentIndex === fromIndex) {
          state.currentIndex = toIndex;
        } else if (fromIndex < state.currentIndex && toIndex >= state.currentIndex) {
          state.currentIndex = Math.max(0, state.currentIndex - 1);
        } else if (fromIndex > state.currentIndex && toIndex <= state.currentIndex) {
          state.currentIndex = Math.min(state.tracks.length - 1, state.currentIndex + 1);
        }

        state.lastUpdated = action.meta.timestamp;
      },
      prepare(payload: { fromIndex: number; toIndex: number }) {
        return { payload, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Clear entire queue
     */
    clearQueue: {
      reducer(state, action: PayloadAction<void, string, { timestamp: number }>) {
        state.tracks = [];
        state.currentIndex = 0;
        state.lastUpdated = action.meta.timestamp;
      },
      prepare() {
        return { payload: undefined, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Set entire queue
     */
    setQueue: {
      reducer(state, action: PayloadAction<Track[], string, { timestamp: number }>) {
        state.tracks = action.payload;
        state.currentIndex = Math.max(0, Math.min(state.currentIndex, state.tracks.length - 1));
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(tracks: Track[]) {
        return { payload: tracks, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Patch the stored record(s) for a track, by track id (#4580).
     *
     * The app keeps two copies of the current track — `player.currentTrack`
     * and `queue.tracks[currentIndex]` — with structurally identical types.
     * Corrections that arrive for one (a re-analysed duration) used to reach
     * only the player copy, so queue-derived selectors kept showing the
     * pre-correction value indefinitely.
     *
     * Patches every entry with a matching id, not just `currentIndex`: the
     * same track can legitimately sit in the queue more than once, and a
     * corrected duration is a property of the track, not of the slot.
     */
    updateTrackById: {
      reducer(
        state,
        action: PayloadAction<
          { id: number; changes: Partial<Track> },
          string,
          { timestamp: number }
        >
      ) {
        const { id, changes } = action.payload;
        let patched = false;
        for (const track of state.tracks) {
          if (track.id === id) {
            Object.assign(track, changes);
            patched = true;
          }
        }
        if (patched) {
          state.lastUpdated = action.meta.timestamp;
        }
      },
      prepare(payload: { id: number; changes: Partial<Track> }) {
        return { payload, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Set current queue index
     */
    setCurrentIndex: {
      reducer(state, action: PayloadAction<number, string, { timestamp: number }>) {
        const index = action.payload;
        if (index >= 0 && index < state.tracks.length) {
          state.currentIndex = index;
          state.lastUpdated = action.meta.timestamp;
        }
      },
      prepare(index: number) {
        return { payload: index, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Set loading state
     */
    setIsLoading: {
      reducer(state, action: PayloadAction<boolean, string, { timestamp: number }>) {
        state.isLoading = action.payload;
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(isLoading: boolean) {
        return { payload: isLoading, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Set error message
     */
    setError: {
      reducer(state, action: PayloadAction<string | null, string, { timestamp: number }>) {
        state.error = action.payload;
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(error: string | null) {
        return { payload: error, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Clear error
     *
     * No production dispatch sites (#4921) for THIS slice; cacheSlice's and
     * connectionSlice's same-named actions ARE dispatched, from useReduxState.ts — kept as an idiomatic
     * Redux action. Live sync uses field-level dispatches; see the note on
     * resetPlayer in playerSlice.ts for why the bulk-update siblings were
     * deleted rather than documented.
     */
    clearError(state) {
      state.error = null;
    },

    /**
     * Set shuffle state
     */
    setIsShuffled: {
      reducer(state, action: PayloadAction<boolean, string, { timestamp: number }>) {
        state.isShuffled = action.payload;
        // Bump lastUpdated like every other queue mutator (#4192) so
        // lastUpdated-based memoization/sync sees the change.
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(isShuffled: boolean) {
        return { payload: isShuffled, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Set repeat mode
     */
    setRepeatMode: {
      reducer(state, action: PayloadAction<RepeatMode, string, { timestamp: number }>) {
        state.repeatMode = action.payload;
        state.lastUpdated = action.meta.timestamp;
      },
      prepare(mode: RepeatMode) {
        return { payload: mode, meta: { timestamp: Date.now() } };
      },
    },

    /**
     * Reset queue state
     *
     * No production dispatch sites (#4921) — kept as an idiomatic
     * Redux action. Live sync uses field-level dispatches; see the note on
     * resetPlayer in playerSlice.ts for why the bulk-update siblings were
     * deleted rather than documented.
     */
    resetQueue(state) {
      Object.assign(state, initialState);
    },
  },
});

export const {
  addTrack,
  addTracks,
  removeTrack,
  reorderTrack,
  clearQueue,
  setQueue,
  updateTrackById,
  setCurrentIndex,
  setIsLoading,
  setError,
  clearError,
  setIsShuffled,
  setRepeatMode,
  resetQueue,
} = queueSlice.actions;

// Selectors
export const selectQueueTracks = (state: { queue: QueueState }) => state.queue.tracks;
export const selectCurrentIndex = (state: { queue: QueueState }) => state.queue.currentIndex;
export const selectIsShuffled = (state: { queue: QueueState }) => state.queue.isShuffled;
export const selectRepeatMode = (state: { queue: QueueState }) => state.queue.repeatMode;
// selectCurrentQueueTrack: use the memoized version from store/selectors/index.ts (#3382)
export const selectQueueLength = (state: { queue: QueueState }) => state.queue.tracks.length;
export const selectIsLoading = (state: { queue: QueueState }) => state.queue.isLoading;
export const selectError = (state: { queue: QueueState }) => state.queue.error;
export const selectLastUpdated = (state: { queue: QueueState }) => state.queue.lastUpdated;
// selectQueueState is exported from store/selectors/index.ts (memoized).

// Derived selectors (selectRemainingTime, selectTotalQueueTime) are memoized
// via createSelector in store/selectors/index.ts — import from there.

export default queueSlice.reducer;
