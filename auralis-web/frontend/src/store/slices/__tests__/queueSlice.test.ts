/**
 * queueSlice reducer unit tests (#2815)
 *
 * Covers all action creators, index adjustment logic, and selectors.
 */

import reducer, {
  addTrack,
  addTracks,
  removeTrack,
  reorderTrack,
  clearQueue,
  setQueue,
  setCurrentIndex,
  nextTrack,
  previousTrack,
  setIsLoading,
  setError,
  clearError,
  setIsShuffled,
  setRepeatMode,
  resetQueue,
  selectQueueTracks,
  selectCurrentIndex,
  selectQueueLength,
  selectIsLoading,
  selectError,
} from '../queueSlice';
import { selectCurrentQueueTrack } from '@/store/selectors';
import type { QueueState } from '../queueSlice';
import type { QueueTrack } from '@/types/domain';

const initialState: QueueState = {
  tracks: [],
  currentIndex: 0,
  isLoading: false,
  error: null,
  lastUpdated: 0,
  isShuffled: false,
  repeatMode: 'off',
};

const mockTrack = (id: number): QueueTrack => ({
  id,
  title: `Track ${id}`,
  artist: `Artist ${id}`,
  album: `Album ${id}`,
  duration: 200 + id,
  artworkUrl: null,
});

describe('queueSlice', () => {
  it('should return initial state', () => {
    const state = reducer(undefined, { type: 'unknown' });
    expect(state.tracks).toEqual([]);
    expect(state.currentIndex).toBe(0);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
  });

  // ─── Add/Remove ───────────────────────────────────────────────

  it('addTrack appends track and sets timestamp', () => {
    const state = reducer(initialState, addTrack(mockTrack(1)));
    expect(state.tracks).toHaveLength(1);
    expect(state.tracks[0].id).toBe(1);
    expect(state.lastUpdated).toBeGreaterThan(0);
  });

  it('addTrack shifts currentIndex when inserting at or before it (#4927)', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 1 }; // playing track 2
    state = reducer(state, addTrack(mockTrack(99), 0));
    expect(state.currentIndex).toBe(2);
    expect(selectCurrentQueueTrack({ queue: state } as never)?.id).toBe(2);
  });

  it('addTrack leaves currentIndex unchanged when inserting after it (#4927)', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 1 };
    state = reducer(state, addTrack(mockTrack(99), 5));
    expect(state.currentIndex).toBe(1);
  });

  it('addTrack inserts mid-queue in order and bumps lastUpdated (#4483)', () => {
    // The #4927 pair above does not pin this: its "after it" case inserts at
    // position 5 into a 3-track queue, which is out of range and takes the
    // append branch, and neither case asserts the resulting order.
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 0, lastUpdated: 0 };

    state = reducer(state, addTrack(mockTrack(99), 1));

    expect(state.tracks.map((t) => t.id)).toEqual([1, 99, 2, 3]);
    expect(state.currentIndex).toBe(0); // inserted after the playing track
    expect(state.lastUpdated).toBeGreaterThan(0);
  });

  it('addTrack at position === length appends via the splice branch (#4483)', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = reducer(state, addTrack(mockTrack(99), 2));
    expect(state.tracks.map((t) => t.id)).toEqual([1, 2, 99]);
  });

  it('addTracks appends multiple tracks', () => {
    const state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    expect(state.tracks).toHaveLength(2);
  });

  it('removeTrack removes by index', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = reducer(state, removeTrack(1));
    expect(state.tracks).toHaveLength(2);
    expect(state.tracks[0].id).toBe(1);
    expect(state.tracks[1].id).toBe(3);
  });

  it('removeTrack adjusts currentIndex when removing before it', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 2 };
    state = reducer(state, removeTrack(0));
    expect(state.currentIndex).toBe(1);
  });

  it('removeTrack adjusts currentIndex when removing at end', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = { ...state, currentIndex: 1 };
    state = reducer(state, removeTrack(1));
    expect(state.currentIndex).toBe(0);
  });

  it('removeTrack ignores out-of-range index', () => {
    let state = reducer(initialState, addTrack(mockTrack(1)));
    state = reducer(state, removeTrack(5));
    expect(state.tracks).toHaveLength(1);
  });

  // ─── Reorder ──────────────────────────────────────────────────

  it('reorderTrack moves track forward', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = reducer(state, reorderTrack({ fromIndex: 0, toIndex: 2 }));
    expect(state.tracks.map((t) => t.id)).toEqual([2, 3, 1]);
  });

  it('reorderTrack moves track backward', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = reducer(state, reorderTrack({ fromIndex: 2, toIndex: 0 }));
    expect(state.tracks.map((t) => t.id)).toEqual([3, 1, 2]);
  });

  it('reorderTrack updates currentIndex when current track is moved', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 0 };
    state = reducer(state, reorderTrack({ fromIndex: 0, toIndex: 2 }));
    expect(state.currentIndex).toBe(2);
  });

  it('reorderTrack is no-op when fromIndex equals toIndex', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    const before = state.lastUpdated;
    state = reducer(state, reorderTrack({ fromIndex: 0, toIndex: 0 }));
    expect(state.lastUpdated).toBe(before);
  });

  it('reorderTrack is a no-op for an out-of-range fromIndex (never inserts undefined) (#4432)', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = reducer(state, reorderTrack({ fromIndex: 99, toIndex: 0 }));
    expect(state.tracks.map((t) => t.id)).toEqual([1, 2, 3]);
    expect(state.tracks).not.toContain(undefined);
  });

  it('reorderTrack is a no-op for an out-of-range toIndex (#4432)', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = reducer(state, reorderTrack({ fromIndex: 0, toIndex: 99 }));
    expect(state.tracks.map((t) => t.id)).toEqual([1, 2, 3]);
    expect(state.tracks).not.toContain(undefined);
  });

  it('reorderTrack is a no-op for negative indices (#4432)', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = reducer(state, reorderTrack({ fromIndex: -1, toIndex: 0 }));
    expect(state.tracks.map((t) => t.id)).toEqual([1, 2, 3]);
    expect(state.tracks).not.toContain(undefined);
  });

  // ─── Queue operations ────────────────────────────────────────

  it('clearQueue empties tracks and resets index', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = { ...state, currentIndex: 1 };
    state = reducer(state, clearQueue());
    expect(state.tracks).toEqual([]);
    expect(state.currentIndex).toBe(0);
  });

  it('setQueue replaces entire queue and clamps index', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 2 };
    state = reducer(state, setQueue([mockTrack(4)]));
    expect(state.tracks).toHaveLength(1);
    expect(state.currentIndex).toBe(0); // clamped
  });

  // ─── Navigation ───────────────────────────────────────────────

  it('setCurrentIndex sets valid index', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = reducer(state, setCurrentIndex(1));
    expect(state.currentIndex).toBe(1);
  });

  it('setCurrentIndex ignores out-of-range index', () => {
    let state = reducer(initialState, addTrack(mockTrack(1)));
    state = reducer(state, setCurrentIndex(5));
    expect(state.currentIndex).toBe(0);
  });

  it('nextTrack increments index', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = reducer(state, nextTrack());
    expect(state.currentIndex).toBe(1);
  });

  it('nextTrack does not go past last track', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = { ...state, currentIndex: 1 };
    state = reducer(state, nextTrack());
    expect(state.currentIndex).toBe(1);
  });

  it('previousTrack decrements index', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = { ...state, currentIndex: 1 };
    state = reducer(state, previousTrack());
    expect(state.currentIndex).toBe(0);
  });

  it('previousTrack does not go below zero', () => {
    let state = reducer(initialState, addTrack(mockTrack(1)));
    state = reducer(state, previousTrack());
    expect(state.currentIndex).toBe(0);
  });

  // ─── Repeat mode: 'one' (stay on current track) ────────────────

  it('nextTrack stays on current track when repeatMode is "one"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 1, repeatMode: 'one' };
    state = reducer(state, nextTrack());
    expect(state.currentIndex).toBe(1);
  });

  it('previousTrack stays on current track when repeatMode is "one"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 1, repeatMode: 'one' };
    state = reducer(state, previousTrack());
    expect(state.currentIndex).toBe(1);
  });

  it('nextTrack does not bump lastUpdated when repeatMode is "one"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = { ...state, currentIndex: 0, repeatMode: 'one', lastUpdated: 42 };
    state = reducer(state, nextTrack());
    expect(state.lastUpdated).toBe(42);
  });

  // ─── Repeat mode: 'all' (wrap around) ───────────────────────────

  it('nextTrack wraps to first track when repeatMode is "all"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 2, repeatMode: 'all' };
    state = reducer(state, nextTrack());
    expect(state.currentIndex).toBe(0);
  });

  it('nextTrack still advances normally mid-queue when repeatMode is "all"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 0, repeatMode: 'all' };
    state = reducer(state, nextTrack());
    expect(state.currentIndex).toBe(1);
  });

  it('previousTrack wraps to last track when repeatMode is "all"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 0, repeatMode: 'all' };
    state = reducer(state, previousTrack());
    expect(state.currentIndex).toBe(2);
  });

  it('previousTrack still decrements normally mid-queue when repeatMode is "all"', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 2, repeatMode: 'all' };
    state = reducer(state, previousTrack());
    expect(state.currentIndex).toBe(1);
  });

  it('previousTrack on an empty queue with repeatMode "all" leaves currentIndex at 0, not -1 (#4457)', () => {
    // tracks.length - 1 is -1 on an empty queue, violating the
    // 0 <= currentIndex invariant — Math.max(0, ...) must clamp it.
    const state = reducer(
      { ...initialState, currentIndex: 0, repeatMode: 'all' },
      previousTrack()
    );
    expect(state.currentIndex).toBe(0);
  });

  // ─── Repeat mode: 'off' at boundaries (explicit, mirrors 'all') ─

  it('nextTrack does not wrap when repeatMode is "off" and at the last track', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 2, repeatMode: 'off' };
    state = reducer(state, nextTrack());
    expect(state.currentIndex).toBe(2);
  });

  it('previousTrack does not wrap when repeatMode is "off" and at the first track', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2), mockTrack(3)]));
    state = { ...state, currentIndex: 0, repeatMode: 'off' };
    state = reducer(state, previousTrack());
    expect(state.currentIndex).toBe(0);
  });

  // ─── Loading/Error ────────────────────────────────────────────

  it('setIsLoading sets loading state', () => {
    const state = reducer(initialState, setIsLoading(true));
    expect(state.isLoading).toBe(true);
  });

  it('setError sets error message', () => {
    const state = reducer(initialState, setError('fail'));
    expect(state.error).toBe('fail');
  });

  it('clearError clears error', () => {
    let state = reducer(initialState, setError('fail'));
    state = reducer(state, clearError());
    expect(state.error).toBeNull();
  });

  // ─── Reset ────────────────────────────────────────────────────

  it('resetQueue returns to initial state', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = reducer(state, setError('fail'));
    state = reducer(state, resetQueue());
    expect(state.tracks).toEqual([]);
    expect(state.currentIndex).toBe(0);
    expect(state.error).toBeNull();
  });

  // ─── Selectors ────────────────────────────────────────────────

  it('selectors return correct values', () => {
    let state = reducer(initialState, addTracks([mockTrack(1), mockTrack(2)]));
    state = { ...state, currentIndex: 1 };
    const root = { queue: state };

    expect(selectQueueTracks(root)).toHaveLength(2);
    expect(selectCurrentIndex(root)).toBe(1);
    expect(selectCurrentQueueTrack(root as any)?.id).toBe(2);
    expect(selectQueueLength(root)).toBe(2);
    expect(selectIsLoading(root)).toBe(false);
    expect(selectError(root)).toBeNull();
  });

  it('selectCurrentQueueTrack returns null for empty queue', () => {
    const root = { queue: initialState };
    expect(selectCurrentQueueTrack(root as any)).toBeNull();
  });

  it('setIsShuffled bumps lastUpdated (#4192)', () => {
    const state = reducer(initialState, setIsShuffled(true));
    expect(state.isShuffled).toBe(true);
    expect(state.lastUpdated).toBeGreaterThan(0);
  });

  it('setRepeatMode bumps lastUpdated (#4192)', () => {
    const state = reducer(initialState, setRepeatMode('all'));
    expect(state.repeatMode).toBe('all');
    expect(state.lastUpdated).toBeGreaterThan(0);
  });
});
