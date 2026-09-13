/**
 * Redux Selector Tests
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * Verifies memoization correctness for all consolidated selectors:
 * - Same reference returned on repeated calls with unchanged state
 * - New reference returned when relevant state changes
 * - Derived values are correct
 * - All selector imports come from a single file (store/selectors/index.ts)
 *
 * Issue #2222: consolidated from store/selectors/advanced.ts into index.ts
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { configureStore } from '@reduxjs/toolkit';
import playerReducer, {
  setCurrentTime,
  setDuration,
  setIsPlaying,
} from '@/store/slices/playerSlice';
import queueReducer, { setQueue, setCurrentIndex } from '@/store/slices/queueSlice';
import cacheReducer from '@/store/slices/cacheSlice';
import connectionReducer, { setWSConnected, setAPIConnected } from '@/store/slices/connectionSlice';
import {
  // Simple selectors
  playerSelectors,
  queueSelectors,
  // Memoized derived selectors
  selectPlaybackProgress,
  selectFormattedTime,
  selectCurrentQueueTrack,
  selectRemainingTime,
  selectTotalQueueTime,
  selectFormattedQueueTime,
  selectQueueStats,
  selectConnectionStatus,
  selectPlaybackState,
  selectQueueState,
  selectAppSnapshot,
  // Factory selectors
  makeSelectTrackAtIndex,
  makeSelectTracksInRange,
  makeSelectFilteredTracks,
  makeSelectTracksByDuration,
  optimizedSelectors,
} from '../index';
import type { RootState } from '@/store/index';

// ============================================================================
// Test store helpers
// ============================================================================

function createTestStore() {
  return configureStore({
    reducer: {
      player: playerReducer,
      queue: queueReducer,
      cache: cacheReducer,
      connection: connectionReducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({ serializableCheck: false }),
  });
}

type TestStore = ReturnType<typeof createTestStore>;

const mockTrack = (id: number, duration = 180) => ({
  id,
  title: `Track ${id}`,
  artist: 'Artist',
  album: 'Album',
  duration,
});

// ============================================================================
// Single import-source contract
// ============================================================================

describe('Single import source', () => {
  it('exports all selectors from store/selectors/index.ts (no advanced.ts)', () => {
    // All critical exports exist on the consolidated module
    expect(typeof selectPlaybackProgress).toBe('function');
    expect(typeof selectFormattedTime).toBe('function');
    expect(typeof selectCurrentQueueTrack).toBe('function');
    expect(typeof selectRemainingTime).toBe('function');
    expect(typeof selectConnectionStatus).toBe('function');
    expect(typeof selectPlaybackState).toBe('function');
    expect(typeof selectQueueState).toBe('function');
    expect(typeof optimizedSelectors).toBe('object');
  });
});

// ============================================================================
// Simple (pass-through) selectors
// ============================================================================

describe('playerSelectors (simple)', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('selectIsPlaying returns false by default', () => {
    expect(playerSelectors.selectIsPlaying(store.getState() as RootState)).toBe(false);
  });

  it('selectCurrentTrack returns null by default', () => {
    expect(playerSelectors.selectCurrentTrack(store.getState() as RootState)).toBeNull();
  });

  it('selectVolume returns the initial volume (80, per #2251)', () => {
    expect(playerSelectors.selectVolume(store.getState() as RootState)).toBe(80);
  });

  it('reflects dispatched state changes', () => {
    store.dispatch(setIsPlaying(true));
    expect(playerSelectors.selectIsPlaying(store.getState() as RootState)).toBe(true);
  });
});

describe('queueSelectors (simple)', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('selectQueueTracks returns empty array by default', () => {
    expect(queueSelectors.selectQueueTracks(store.getState() as RootState)).toEqual([]);
  });

  it('selectCurrentIndex returns 0 by default', () => {
    expect(queueSelectors.selectCurrentIndex(store.getState() as RootState)).toBe(0);
  });
});

// ============================================================================
// Memoization: same reference on unchanged state
// ============================================================================

describe('Memoization – same reference on unchanged state', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setDuration(300));
    store.dispatch(setCurrentTime(60));
    store.dispatch(setQueue([mockTrack(1), mockTrack(2), mockTrack(3)]));
  });

  it('selectPlaybackProgress returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectPlaybackProgress(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectPlaybackProgress(state as RootState)).toBe(first);
    }
  });

  it('selectFormattedTime returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectFormattedTime(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectFormattedTime(state as RootState)).toBe(first);
    }
  });

  it('selectCurrentQueueTrack returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectCurrentQueueTrack(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectCurrentQueueTrack(state as RootState)).toBe(first);
    }
  });

  it('selectRemainingTime returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectRemainingTime(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectRemainingTime(state as RootState)).toBe(first);
    }
  });

  it('selectQueueStats returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectQueueStats(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectQueueStats(state as RootState)).toBe(first);
    }
  });

  it('selectConnectionStatus returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectConnectionStatus(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectConnectionStatus(state as RootState)).toBe(first);
    }
  });

  it('selectPlaybackState returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectPlaybackState(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectPlaybackState(state as RootState)).toBe(first);
    }
  });

  it('selectQueueState returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectQueueState(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectQueueState(state as RootState)).toBe(first);
    }
  });

  it('selectAppSnapshot returns same reference across 10 calls', () => {
    const state = store.getState();
    const first = selectAppSnapshot(state as RootState);
    for (let i = 0; i < 9; i++) {
      expect(selectAppSnapshot(state as RootState)).toBe(first);
    }
  });
});

// ============================================================================
// Memoization: new reference when relevant state changes
// ============================================================================

describe('Memoization – new reference when state changes', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setDuration(300));
    store.dispatch(setCurrentTime(60));
    store.dispatch(setQueue([mockTrack(1), mockTrack(2)]));
  });

  it('selectPlaybackProgress: new value when currentTime changes', () => {
    const before = selectPlaybackProgress(store.getState() as RootState);
    store.dispatch(setCurrentTime(120));
    const after = selectPlaybackProgress(store.getState() as RootState);
    expect(after).not.toBe(before);
    expect(after).toBeCloseTo(0.4);
  });

  it('selectFormattedTime: new object when duration changes', () => {
    const before = selectFormattedTime(store.getState() as RootState);
    store.dispatch(setDuration(600));
    const after = selectFormattedTime(store.getState() as RootState);
    expect(after).not.toBe(before);
    expect(after.total).toBe('10:00');
  });

  it('selectCurrentQueueTrack: new object when currentIndex changes', () => {
    const before = selectCurrentQueueTrack(store.getState() as RootState);
    store.dispatch(setCurrentIndex(1));
    const after = selectCurrentQueueTrack(store.getState() as RootState);
    expect(after).not.toBe(before);
    expect(after?.id).toBe(2);
  });

  it('selectQueueStats: new object when queue changes', () => {
    const before = selectQueueStats(store.getState() as RootState);
    store.dispatch(setQueue([mockTrack(1), mockTrack(2), mockTrack(3)]));
    const after = selectQueueStats(store.getState() as RootState);
    expect(after).not.toBe(before);
    expect(after.length).toBe(3);
  });

  it('selectConnectionStatus: new object when wsConnected changes', () => {
    const before = selectConnectionStatus(store.getState() as RootState);
    store.dispatch(setWSConnected(true));
    const after = selectConnectionStatus(store.getState() as RootState);
    expect(after).not.toBe(before);
  });
});

// ============================================================================
// Derived value correctness
// ============================================================================

describe('selectPlaybackProgress – correctness', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('returns 0 when duration is 0 (avoids division by zero)', () => {
    expect(selectPlaybackProgress(store.getState() as RootState)).toBe(0);
  });

  it('returns correct fractional progress', () => {
    store.dispatch(setDuration(200));
    store.dispatch(setCurrentTime(50));
    expect(selectPlaybackProgress(store.getState() as RootState)).toBeCloseTo(0.25);
  });

  it('returns 1 at full duration', () => {
    store.dispatch(setDuration(100));
    store.dispatch(setCurrentTime(100));
    expect(selectPlaybackProgress(store.getState() as RootState)).toBeCloseTo(1);
  });
});

describe('selectFormattedTime – correctness', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('formats minutes:seconds correctly', () => {
    store.dispatch(setDuration(185)); // 3:05
    store.dispatch(setCurrentTime(65));  // 1:05
    const { current, total } = selectFormattedTime(store.getState() as RootState);
    expect(current).toBe('1:05');
    expect(total).toBe('3:05');
  });

  it('formats hours:mm:ss when duration exceeds 1 hour', () => {
    store.dispatch(setDuration(3661)); // 1:01:01
    const { total } = selectFormattedTime(store.getState() as RootState);
    expect(total).toBe('1:01:01');
  });
});

describe('selectCurrentQueueTrack – correctness', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('returns null when queue is empty', () => {
    expect(selectCurrentQueueTrack(store.getState() as RootState)).toBeNull();
  });

  it('returns the track at currentIndex', () => {
    store.dispatch(setQueue([mockTrack(10), mockTrack(20)]));
    store.dispatch(setCurrentIndex(1));
    expect(selectCurrentQueueTrack(store.getState() as RootState)?.id).toBe(20);
  });
});

describe('selectRemainingTime – correctness', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('returns 0 when queue is empty', () => {
    expect(selectRemainingTime(store.getState() as RootState)).toBe(0);
  });

  it('sums durations of tracks after currentIndex', () => {
    store.dispatch(setQueue([mockTrack(1, 60), mockTrack(2, 90), mockTrack(3, 120)]));
    store.dispatch(setCurrentIndex(0));
    expect(selectRemainingTime(store.getState() as RootState)).toBe(210); // 90+120
  });

  it('returns 0 when at last track', () => {
    store.dispatch(setQueue([mockTrack(1, 60), mockTrack(2, 90)]));
    store.dispatch(setCurrentIndex(1));
    expect(selectRemainingTime(store.getState() as RootState)).toBe(0);
  });
});

describe('selectTotalQueueTime / selectFormattedQueueTime – correctness', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setQueue([mockTrack(1, 60), mockTrack(2, 60), mockTrack(3, 60)]));
  });

  it('selectTotalQueueTime returns sum of all durations', () => {
    expect(selectTotalQueueTime(store.getState() as RootState)).toBe(180);
  });

  it('selectFormattedQueueTime returns "3m" for 180 seconds', () => {
    expect(selectFormattedQueueTime(store.getState() as RootState)).toBe('3m');
  });

  it('selectFormattedQueueTime includes hours when >= 3600s', () => {
    store.dispatch(setQueue([mockTrack(1, 3600), mockTrack(2, 600)]));
    expect(selectFormattedQueueTime(store.getState() as RootState)).toBe('1h 10m');
  });
});

describe('selectConnectionStatus – correctness', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
  });

  it('health is "disconnected" when neither WS nor API connected', () => {
    const status = selectConnectionStatus(store.getState() as RootState);
    expect(status.health).toBe('disconnected');
    expect(status.connected).toBe(false);
  });

  it('health is "degraded" when only WS connected', () => {
    store.dispatch(setWSConnected(true));
    expect(selectConnectionStatus(store.getState() as RootState).health).toBe('degraded');
  });

  it('health is "healthy" when both connected', () => {
    store.dispatch(setWSConnected(true));
    store.dispatch(setAPIConnected(true));
    expect(selectConnectionStatus(store.getState() as RootState).health).toBe('healthy');
    expect(selectConnectionStatus(store.getState() as RootState).connected).toBe(true);
  });
});

// ============================================================================
// Factory selectors
// ============================================================================

describe('Factory selectors – makeSelectTrackAtIndex', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setQueue([mockTrack(1), mockTrack(2), mockTrack(3)]));
  });

  it('returns the track at the given index', () => {
    const selector = makeSelectTrackAtIndex(1);
    expect(selector(store.getState() as RootState)?.id).toBe(2);
  });

  it('returns null for out-of-bounds index', () => {
    const selector = makeSelectTrackAtIndex(99);
    expect(selector(store.getState() as RootState)).toBeNull();
  });

  it('memoizes when wrapped in a stable selector instance', () => {
    const selector = makeSelectTrackAtIndex(0);
    const state = store.getState() as RootState;
    const a = selector(state);
    const b = selector(state);
    expect(a).toBe(b); // same Reselect instance, same reference
  });
});

describe('Factory selectors – makeSelectTracksInRange', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setQueue([mockTrack(1), mockTrack(2), mockTrack(3), mockTrack(4)]));
  });

  it('returns the correct slice', () => {
    const selector = makeSelectTracksInRange(1, 3);
    const result = selector(store.getState() as RootState);
    expect(result.map((t) => t.id)).toEqual([2, 3]);
  });
});

describe('Factory selectors – makeSelectFilteredTracks', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setQueue([mockTrack(1, 60), mockTrack(2, 300), mockTrack(3, 120)]));
  });

  it('filters tracks by predicate', () => {
    const selector = makeSelectFilteredTracks((t) => t.duration > 100);
    const result = selector(store.getState() as RootState);
    expect(result.map((t) => t.id)).toEqual([2, 3]);
  });
});

describe('Factory selectors – makeSelectTracksByDuration', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setQueue([mockTrack(1, 60), mockTrack(2, 180), mockTrack(3, 300)]));
  });

  it('returns tracks within duration range', () => {
    const selector = makeSelectTracksByDuration(100, 250);
    const result = selector(store.getState() as RootState);
    expect(result.map((t) => t.id)).toEqual([2]);
  });
});

// ============================================================================
// optimizedSelectors aggregate
// ============================================================================

describe('optimizedSelectors aggregate', () => {
  let store: TestStore;

  beforeEach(() => {
    store = createTestStore();
    store.dispatch(setDuration(300));
    store.dispatch(setCurrentTime(60));
  });

  it('player.selectPlaybackProgress computes correctly', () => {
    expect(optimizedSelectors.player.selectPlaybackProgress(store.getState() as RootState))
      .toBeCloseTo(0.2);
  });

  it('cache.selectCacheMetrics returns zero stats when cache empty', () => {
    const metrics = optimizedSelectors.cache.selectCacheMetrics(store.getState() as RootState);
    expect(metrics.hitRate).toBe(0);
    expect(metrics.totalChunks).toBe(0);
  });

  it('appSnapshot exposes its real fields, not loading and error-free on a fresh store (#4485)', () => {
    // The previous name promised an `isReady` field that selectAppSnapshot has
    // never had, and the body only asserted `typeof snap === 'object'`.
    const snap = optimizedSelectors.appSnapshot(store.getState() as RootState);

    expect(Object.keys(snap).sort()).toEqual(
      ['cache', 'connection', 'hasErrors', 'isLoading', 'playback', 'queue']
    );
    expect(snap).not.toHaveProperty('isReady');
    expect(snap.isLoading).toBe(false);
    expect(snap.hasErrors).toBe(false);
    // It is the same memoized selector exported from combined.ts, not a copy.
    expect(optimizedSelectors.appSnapshot).toBe(selectAppSnapshot);
  });
});
