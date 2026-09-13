/**
 * Redux State Hooks
 * ~~~~~~~~~~~~~~~~
 *
 * Convenience hooks for accessing Redux state and dispatching actions.
 * Provides a cleaner API for components instead of raw useSelector/useDispatch.
 *
 * Features:
 * - Memoized selectors to prevent unnecessary re-renders
 * - Type-safe actions
 * - Batch updates for related state
 * - Loading and error state helpers
 *
 * Phase C.4a: Redux Integration
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3, see LICENSE for more details
 */

import { useSelector, useDispatch, useStore, shallowEqual } from 'react-redux';
import { useCallback, useMemo } from 'react';
import type { RootState, AppDispatch } from '@/store';
import type { PlayerTrack, QueueTrack } from '@/types/domain';
import * as playerActions from '@/store/slices/playerSlice';
import * as queueActions from '@/store/slices/queueSlice';
import * as cacheActions from '@/store/slices/cacheSlice';
import type { CacheStats, CacheHealth } from '@/services/api/standardizedAPIClient';
import * as connectionActions from '@/store/slices/connectionSlice';
import { selectFormattedRemainingTime } from '@/store/selectors';

// ============================================================================
// Player State Hooks
// ============================================================================

/**
 * Access entire player state
 */
export const usePlayerState = () => {
  return useSelector((state: RootState) => state.player, shallowEqual);
};

/**
 * Access ONLY the player action callbacks — no state selectors.
 *
 * The returned object is stable across renders (every callback depends only on
 * the stable `dispatch`/`store` references), so action-only consumers don't
 * re-render on state changes. usePlayer() bundles `currentTime`, so it returns a
 * new object on every 1Hz position tick during playback; consumers that only
 * need callbacks (e.g. a context menu's "play") should use this instead to avoid
 * 1Hz render churn (#4176).
 */
export const usePlayerActions = () => {
  const dispatch = useDispatch<AppDispatch>();
  // store.getState() lets togglePlay read the live isPlaying at call time without
  // subscribing to it (which would re-render on every play/pause).
  const store = useStore<RootState>();

  const play = useCallback(() => dispatch(playerActions.setIsPlaying(true)), [dispatch]);
  const pause = useCallback(() => dispatch(playerActions.setIsPlaying(false)), [dispatch]);
  const togglePlay = useCallback(
    () => dispatch(playerActions.setIsPlaying(!store.getState().player.isPlaying)),
    [dispatch, store]
  );
  const seek = useCallback(
    (time: number) => dispatch(playerActions.setCurrentTime(time)),
    [dispatch]
  );
  const setVolume = useCallback(
    (vol: number) => dispatch(playerActions.setVolume(vol)),
    [dispatch]
  );
  const setMuted = useCallback(
    (muted: boolean) => dispatch(playerActions.setMuted(muted)),
    [dispatch]
  );
  const toggleMute = useCallback(() => dispatch(playerActions.toggleMute()), [dispatch]);
  const setPreset = useCallback(
    (p: playerActions.PresetName) => dispatch(playerActions.setPreset(p)),
    [dispatch]
  );
  const setTrack = useCallback(
    (track: PlayerTrack | null) => dispatch(playerActions.setCurrentTrack(track)),
    [dispatch]
  );

  return useMemo(() => ({
    play, pause, togglePlay, seek, setVolume, setMuted, toggleMute, setPreset, setTrack,
  }), [play, pause, togglePlay, seek, setVolume, setMuted, toggleMute, setPreset, setTrack]);
};

/**
 * Access player playback controls and state
 */
export const usePlayer = () => {
  // Reuse the stable action callbacks (#4176) so they're defined in one place.
  const actions = usePlayerActions();

  // Granular selectors — each only triggers a re-render when its specific field
  // changes, preventing the full player slice subscription from causing cascading
  // re-renders on every WebSocket position update (fixes #2537).
  const isPlaying = useSelector((state: RootState) => state.player.isPlaying);
  const currentTrack = useSelector((state: RootState) => state.player.currentTrack);
  const currentTime = useSelector((state: RootState) => state.player.currentTime);
  const duration = useSelector((state: RootState) => state.player.duration);
  const volume = useSelector((state: RootState) => state.player.volume);
  const isMuted = useSelector((state: RootState) => state.player.isMuted);
  const preset = useSelector((state: RootState) => state.player.preset);
  const isLoading = useSelector((state: RootState) => state.player.isLoading);
  const error = useSelector((state: RootState) => state.player.error);

  // Memoize the returned object so consumers only re-render when state they
  // actually use changes, not on every render of this hook (fixes #2537).
  return useMemo(() => ({
    isPlaying,
    currentTrack,
    currentTime,
    duration,
    volume,
    isMuted,
    preset,
    isLoading,
    error,
    ...actions,
  }), [
    isPlaying, currentTrack, currentTime, duration, volume, isMuted, preset,
    isLoading, error, actions,
  ]);
};

// ============================================================================
// Queue State Hooks
// ============================================================================

/**
 * Access entire queue state
 */
export const useQueueState = () => {
  return useSelector((state: RootState) => state.queue, shallowEqual);
};

/**
 * Access queue management controls
 */
export const useQueue = () => {
  const dispatch = useDispatch<AppDispatch>();
  const state = useSelector((state: RootState) => state.queue, shallowEqual);

  // Memoize O(n) queue duration computations — only recalculate when tracks or
  // currentIndex change, not on every 10Hz position_changed re-render (fixes #2545).
  const remainingTime = useMemo(
    () => state.tracks.slice(state.currentIndex + 1).reduce((sum, t) => sum + t.duration, 0),
    [state.tracks, state.currentIndex]
  );
  const totalTime = useMemo(
    () => state.tracks.reduce((sum, t) => sum + t.duration, 0),
    [state.tracks]
  );

  const add = useCallback(
    (track: QueueTrack, position?: number) => dispatch(queueActions.addTrack(track, position)),
    [dispatch]
  );
  const addMany = useCallback(
    (tracks: QueueTrack[]) => dispatch(queueActions.addTracks(tracks)),
    [dispatch]
  );
  const remove = useCallback(
    (index: number) => dispatch(queueActions.removeTrack(index)),
    [dispatch]
  );
  const reorder = useCallback(
    (fromIndex: number, toIndex: number) =>
      dispatch(queueActions.reorderTrack({ fromIndex, toIndex })),
    [dispatch]
  );
  const setCurrentIndex = useCallback(
    (index: number) => dispatch(queueActions.setCurrentIndex(index)),
    [dispatch]
  );
  // #4660: `next`/`previous` used to live here, dispatching queueSlice's
  // nextTrack/previousTrack — reducers that moved currentIndex client-side
  // with no API call and no track_changed round-trip, so the UI could show a
  // different track than the engine was streaming. The real skip path is
  // usePlaybackControl().next()/.previous(); use that.
  const clear = useCallback(() => dispatch(queueActions.clearQueue()), [dispatch]);
  const setQueue = useCallback(
    (tracks: QueueTrack[]) => dispatch(queueActions.setQueue(tracks)),
    [dispatch]
  );

  // #3619: stable object identity across renders when nothing relevant
  // changed — matches the usePlayer pattern from #2537.
  return useMemo(
    () => ({
      tracks: state.tracks,
      currentIndex: state.currentIndex,
      currentTrack: state.tracks[state.currentIndex] || null,
      queueLength: state.tracks.length,
      isLoading: state.isLoading,
      error: state.error,
      remainingTime,
      totalTime,
      add,
      addMany,
      remove,
      reorder,
      setCurrentIndex,
      clear,
      setQueue,
    }),
    [
      state.tracks,
      state.currentIndex,
      state.isLoading,
      state.error,
      remainingTime,
      totalTime,
      add,
      addMany,
      remove,
      reorder,
      setCurrentIndex,
      clear,
      setQueue,
    ]
  );
};

// ============================================================================
// Cache State Hooks
// ============================================================================

/**
 * Access entire cache state
 */
export const useCacheState = () => {
  return useSelector((state: RootState) => state.cache, shallowEqual);
};

/**
 * Access cache monitoring and management
 */
export const useCache = () => {
  const dispatch = useDispatch<AppDispatch>();
  const state = useSelector((state: RootState) => state.cache, shallowEqual);

  const setStats = useCallback(
    (stats: CacheStats) => dispatch(cacheActions.setCacheStats(stats)),
    [dispatch]
  );
  const setHealth = useCallback(
    (health: CacheHealth) => dispatch(cacheActions.setCacheHealth(health)),
    [dispatch]
  );
  const clear = useCallback(
    () => dispatch(cacheActions.clearCacheLocal()),
    [dispatch]
  );
  const clearError = useCallback(
    () => dispatch(cacheActions.clearError()),
    [dispatch]
  );

  // #3619: stable object identity (matches usePlayer pattern).
  return useMemo(
    () => ({
      stats: state.stats,
      health: state.health,
      isLoading: state.isLoading,
      error: state.error,
      lastUpdated: state.lastUpdated,
      isHealthy: state.health?.healthy ?? false,
      hitRate: state.stats?.overall.overall_hit_rate ?? 0,
      totalSize: state.stats?.overall.total_size_mb ?? 0,
      totalChunks: state.stats?.overall.total_chunks ?? 0,
      tracksCached: state.stats?.overall.tracks_cached ?? 0,
      setStats,
      setHealth,
      clear,
      clearError,
    }),
    [state.stats, state.health, state.isLoading, state.error, state.lastUpdated, setStats, setHealth, clear, clearError]
  );
};

// ============================================================================
// Connection State Hooks
// ============================================================================

/**
 * Access entire connection state
 */
export const useConnectionState = () => {
  return useSelector((state: RootState) => state.connection, shallowEqual);
};

/**
 * Access connection status and management
 */
export const useConnection = () => {
  const dispatch = useDispatch<AppDispatch>();
  const state = useSelector((state: RootState) => state.connection, shallowEqual);

  const setWSConnected = useCallback(
    (connected: boolean) => dispatch(connectionActions.setWSConnected(connected)),
    [dispatch]
  );
  const setAPIConnected = useCallback(
    (connected: boolean) => dispatch(connectionActions.setAPIConnected(connected)),
    [dispatch]
  );
  const setLatency = useCallback(
    (latency: number) => dispatch(connectionActions.setLatency(latency)),
    [dispatch]
  );
  const incrementReconnectAttempts = useCallback(
    () => dispatch(connectionActions.incrementReconnectAttempts()),
    [dispatch]
  );
  const resetReconnectAttempts = useCallback(
    () => dispatch(connectionActions.resetReconnectAttempts()),
    [dispatch]
  );
  const clearError = useCallback(
    () => dispatch(connectionActions.clearError()),
    [dispatch]
  );

  // #3619: stable object identity (matches usePlayer pattern).
  return useMemo(
    () => ({
      wsConnected: state.wsConnected,
      apiConnected: state.apiConnected,
      latency: state.latency,
      reconnectAttempts: state.reconnectAttempts,
      maxReconnectAttempts: state.maxReconnectAttempts,
      lastError: state.lastError,
      isFullyConnected: state.wsConnected && state.apiConnected,
      canReconnect: state.reconnectAttempts < state.maxReconnectAttempts,
      connectionHealth: (state.wsConnected && state.apiConnected
        ? 'healthy'
        : state.wsConnected || state.apiConnected
          ? 'degraded'
          : 'disconnected') as 'healthy' | 'degraded' | 'disconnected',
      setWSConnected,
      setAPIConnected,
      setLatency,
      incrementReconnectAttempts,
      resetReconnectAttempts,
      clearError,
    }),
    [
      state.wsConnected,
      state.apiConnected,
      state.latency,
      state.reconnectAttempts,
      state.maxReconnectAttempts,
      state.lastError,
      setWSConnected,
      setAPIConnected,
      setLatency,
      incrementReconnectAttempts,
      resetReconnectAttempts,
      clearError,
    ]
  );
};

// ============================================================================
// Combined State Hook
// ============================================================================

/**
 * Access complete application state in one hook
 * Useful for complex components needing multiple slices
 */
export const useAppState = () => {
  const player = usePlayerState();
  const queue = useQueueState();
  const cache = useCacheState();
  const connection = useConnectionState();

  return {
    player,
    queue,
    cache,
    connection,
  };
};

// ============================================================================
// Convenience Hooks
// ============================================================================

/**
 * Check if any loading is in progress
 */
export const useIsLoading = () => {
  const player = useSelector((state: RootState) => state.player.isLoading);
  const queue = useSelector((state: RootState) => state.queue.isLoading);
  const cache = useSelector((state: RootState) => state.cache.isLoading);

  return player || queue || cache;
};

/**
 * Check if any errors exist
 */
export const useAppErrors = () => {
  const playerError = useSelector((state: RootState) => state.player.error);
  const queueError = useSelector((state: RootState) => state.queue.error);
  const cacheError = useSelector((state: RootState) => state.cache.error);
  const connectionError = useSelector((state: RootState) => state.connection.lastError);

  return {
    playerError,
    queueError,
    cacheError,
    connectionError,
    hasErrors: !!(playerError || queueError || cacheError || connectionError),
  };
};

/**
 * Check connection status with health information
 */
export const useConnectionHealth = () => {
  const connection = useConnectionState();

  return {
    connected: connection.wsConnected && connection.apiConnected,
    wsConnected: connection.wsConnected,
    apiConnected: connection.apiConnected,
    latency: connection.latency,
    attempting: connection.reconnectAttempts > 0 && connection.reconnectAttempts < connection.maxReconnectAttempts,
    failed: connection.reconnectAttempts >= connection.maxReconnectAttempts,
    health:
      connection.wsConnected && connection.apiConnected
        ? 'connected'
        : connection.wsConnected || connection.apiConnected
          ? 'partial'
          : 'disconnected',
  };
};

/**
 * Get current playback progress (0-1)
 */
export const usePlaybackProgress = () => {
  const currentTime = useSelector((state: RootState) => state.player.currentTime);
  const duration = useSelector((state: RootState) => state.player.duration);
  return duration > 0 ? currentTime / duration : 0;
};

/**
 * Format remaining time in queue (memoized via createSelector, #2812)
 */
export const useQueueTimeRemaining = () => {
  return useSelector(selectFormattedRemainingTime);
};
