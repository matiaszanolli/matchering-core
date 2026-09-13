/**
 * Redux State Hooks Tests
 * ~~~~~~~~~~~~~~~~~~~~~~
 *
 * Unit tests for Redux state and action hooks.
 *
 * Test Coverage:
 * - Player hooks (play, pause, seek, volume)
 * - Queue hooks (add, remove, reorder)
 * - Cache hooks (stats, health)
 * - Connection hooks (status, latency)
 * - Convenience hooks (loading, errors, health)
 *
 * Phase C.4a: Redux Hooks Integration
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3, see LICENSE for more details
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import playerReducer from '@/store/slices/playerSlice';
import queueReducer from '@/store/slices/queueSlice';
import cacheReducer from '@/store/slices/cacheSlice';
import connectionReducer from '@/store/slices/connectionSlice';
import {
  usePlayer,
  useQueue,
  useCache,
  useConnection,
  usePlayerState,
  useQueueState,
  useCacheState,
  useConnectionState,
  useAppState,
  useIsLoading,
  useAppErrors,
  useConnectionHealth,
  usePlaybackProgress,
} from '../useReduxState';
import type React from 'react';

describe('Redux State Hooks', () => {
  let store: any;
  let wrapper: (props: { children: React.ReactNode }) => React.ReactElement | null;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        player: playerReducer,
        queue: queueReducer,
        cache: cacheReducer,
        connection: connectionReducer,
      },
    });

    wrapper = ({ children }) => <Provider store={store}>{children}</Provider>;
  });

  // ============================================================================
  // Player Hooks Tests
  // ============================================================================

  describe('usePlayer', () => {
    it('should return player state and actions', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      expect(result.current.isPlaying).toBe(false);
      expect(result.current.volume).toBe(70);
      expect(result.current.preset).toBe('adaptive');
      expect(result.current.currentTrack).toBeNull();
    });

    it('should play track', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.play();
      });

      expect(result.current.isPlaying).toBe(true);
    });

    it('should pause track', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.play();
      });

      expect(result.current.isPlaying).toBe(true);

      act(() => {
        result.current.pause();
      });

      expect(result.current.isPlaying).toBe(false);
    });

    it('should toggle play', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.togglePlay();
      });

      expect(result.current.isPlaying).toBe(true);

      act(() => {
        result.current.togglePlay();
      });

      expect(result.current.isPlaying).toBe(false);
    });

    it('should seek to position', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.seek(120);
      });

      expect(result.current.currentTime).toBe(120);
    });

    it('should set volume', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.setVolume(50);
      });

      expect(result.current.volume).toBe(50);
    });

    it('should toggle mute', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.toggleMute();
      });

      expect(result.current.isMuted).toBe(true);

      act(() => {
        result.current.toggleMute();
      });

      expect(result.current.isMuted).toBe(false);
    });

    it('should set preset', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });

      act(() => {
        result.current.setPreset('warm');
      });

      expect(result.current.preset).toBe('warm');
    });

    it('should set track', () => {
      const { result } = renderHook(() => usePlayer(), { wrapper });
      const track = { id: 1, title: 'Test', artist: 'Artist', duration: 180 };

      act(() => {
        result.current.setTrack(track);
      });

      expect(result.current.currentTrack).toEqual(track);
    });
  });

  // ============================================================================
  // Queue Hooks Tests
  // ============================================================================

  describe('useQueue', () => {
    it('should return queue state and actions', () => {
      const { result } = renderHook(() => useQueue(), { wrapper });

      expect(result.current.tracks).toEqual([]);
      expect(result.current.queueLength).toBe(0);
      expect(result.current.currentIndex).toBe(0);
    });

    it('should add track to queue', () => {
      const { result } = renderHook(() => useQueue(), { wrapper });
      const track = { id: 1, title: 'Track', artist: 'Artist', duration: 180 };

      act(() => {
        result.current.add(track);
      });

      expect(result.current.queueLength).toBe(1);
      expect(result.current.tracks[0]).toEqual(track);
    });

    it('should add multiple tracks', () => {
      const { result } = renderHook(() => useQueue(), { wrapper });
      const tracks = [
        { id: 1, title: 'Track 1', artist: 'Artist', duration: 180 },
        { id: 2, title: 'Track 2', artist: 'Artist', duration: 200 },
      ];

      act(() => {
        result.current.addMany(tracks);
      });

      expect(result.current.queueLength).toBe(2);
    });

    it('should remove track from queue', () => {
      const { result } = renderHook(() => useQueue(), { wrapper });
      const tracks = [
        { id: 1, title: 'Track 1', artist: 'Artist', duration: 180 },
        { id: 2, title: 'Track 2', artist: 'Artist', duration: 200 },
      ];

      act(() => {
        result.current.addMany(tracks);
      });

      expect(result.current.queueLength).toBe(2);

      act(() => {
        result.current.remove(0);
      });

      expect(result.current.queueLength).toBe(1);
    });

    it('should calculate remaining time', () => {
      const { result } = renderHook(() => useQueue(), { wrapper });
      const tracks = [
        { id: 1, title: 'Track 1', artist: 'Artist', duration: 180 },
        { id: 2, title: 'Track 2', artist: 'Artist', duration: 200 },
        { id: 3, title: 'Track 3', artist: 'Artist', duration: 150 },
      ];

      act(() => {
        result.current.addMany(tracks);
        result.current.setCurrentIndex(0);
      });

      // Remaining time = track 2 + track 3 = 200 + 150 = 350
      expect(result.current.remainingTime).toBe(350);
    });

    it('should clear queue', () => {
      const { result } = renderHook(() => useQueue(), { wrapper });
      const track = { id: 1, title: 'Track', artist: 'Artist', duration: 180 };

      act(() => {
        result.current.add(track);
      });

      expect(result.current.queueLength).toBe(1);

      act(() => {
        result.current.clear();
      });

      expect(result.current.queueLength).toBe(0);
    });
  });

  // ============================================================================
  // Cache Hooks Tests
  // ============================================================================

  describe('useCache', () => {
    it('should return cache state and actions', () => {
      const { result } = renderHook(() => useCache(), { wrapper });

      expect(result.current.stats).toBeNull();
      expect(result.current.health).toBeNull();
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('should set cache stats', () => {
      const { result } = renderHook(() => useCache(), { wrapper });
      const stats = {
        tier1: { chunks: 4, size_mb: 6, hits: 100, misses: 5, hit_rate: 0.95 },
        tier2: { chunks: 8, size_mb: 120, hits: 50, misses: 50, hit_rate: 0.5 },
        overall: {
          total_chunks: 12,
          total_size_mb: 225,
          total_hits: 150,
          total_misses: 55,
          overall_hit_rate: 0.73,
          tracks_cached: 42,
        },
        tracks: {},
      };

      act(() => {
        result.current.setStats(stats);
      });

      expect(result.current.stats).toEqual(stats);
      expect(result.current.hitRate).toBe(0.73);
      expect(result.current.totalSize).toBe(225);
    });

    it('should clear cache', () => {
      const { result } = renderHook(() => useCache(), { wrapper });
      const stats = {
        tier1: { chunks: 4, size_mb: 6, hits: 100, misses: 5, hit_rate: 0.95 },
        tier2: { chunks: 8, size_mb: 120, hits: 50, misses: 50, hit_rate: 0.5 },
        overall: {
          total_chunks: 12,
          total_size_mb: 225,
          total_hits: 150,
          total_misses: 55,
          overall_hit_rate: 0.73,
          tracks_cached: 42,
        },
        tracks: {},
      };

      act(() => {
        result.current.setStats(stats);
      });

      expect(result.current.totalChunks).toBe(12);

      act(() => {
        result.current.clear();
      });

      expect(result.current.stats?.overall.total_chunks).toBe(0);
    });
  });

  // ============================================================================
  // Connection Hooks Tests
  // ============================================================================

  describe('useConnection', () => {
    it('should return connection state', () => {
      const { result } = renderHook(() => useConnection(), { wrapper });

      expect(result.current.wsConnected).toBe(false);
      expect(result.current.apiConnected).toBe(false);
      expect(result.current.isFullyConnected).toBe(false);
      expect(result.current.latency).toBe(0);
    });

    it('should update connection status', () => {
      const { result } = renderHook(() => useConnection(), { wrapper });

      act(() => {
        result.current.setWSConnected(true);
        result.current.setLatency(25);
      });

      expect(result.current.wsConnected).toBe(true);
      expect(result.current.latency).toBe(25);
    });

    it('should track connection health', () => {
      const { result } = renderHook(() => useConnection(), { wrapper });

      expect(result.current.connectionHealth).toBe('disconnected');

      act(() => {
        result.current.setWSConnected(true);
      });

      expect(result.current.connectionHealth).toBe('degraded');

      act(() => {
        result.current.setAPIConnected(true);
      });

      expect(result.current.connectionHealth).toBe('healthy');
    });
  });

  // ============================================================================
  // Convenience Hooks Tests
  // ============================================================================

  describe('Convenience Hooks', () => {
    it('useIsLoading should detect loading state', () => {
      const { result: loadingResult } = renderHook(() => useIsLoading(), {
        wrapper,
      });

      expect(loadingResult.current).toBe(false);
    });

    it('useAppErrors should detect errors', () => {
      const { result: errorResult } = renderHook(() => useAppErrors(), { wrapper });

      expect(errorResult.current.hasErrors).toBe(false);
    });

    it('useConnectionHealth should provide detailed status', () => {
      const { result } = renderHook(() => useConnectionHealth(), { wrapper });

      expect(result.current.connected).toBe(false);
      expect(result.current.health).toBe('disconnected');
    });

    it('usePlaybackProgress should calculate progress', () => {
      const { result: playerResult } = renderHook(() => usePlayer(), { wrapper });
      const { result: progressResult } = renderHook(() => usePlaybackProgress(), {
        wrapper,
      });

      act(() => {
        playerResult.current.setTrack({
          id: 1,
          title: 'Test',
          artist: 'Artist',
          duration: 180,
        });
        playerResult.current.seek(90);
      });

      expect(progressResult.current).toBe(0.5); // 90/180 = 0.5
    });
  });

  // ============================================================================
  // Combined State Tests
  // ============================================================================

  describe('useAppState', () => {
    it('should return all state slices', () => {
      const { result } = renderHook(() => useAppState(), { wrapper });

      expect(result.current.player).toBeDefined();
      expect(result.current.queue).toBeDefined();
      expect(result.current.cache).toBeDefined();
      expect(result.current.connection).toBeDefined();
    });

    it('should reflect changes across slices', () => {
      const { result: playerHookResult } = renderHook(() => usePlayer(), { wrapper });
      const { result: appStateResult } = renderHook(() => useAppState(), { wrapper });

      act(() => {
        playerHookResult.current.play();
      });

      expect(appStateResult.current.player.isPlaying).toBe(true);
    });
  });

  // ============================================================================
  // State Selector Tests
  // ============================================================================

  describe('State Selectors', () => {
    it('usePlayerState should return player slice', () => {
      const { result } = renderHook(() => usePlayerState(), { wrapper });

      expect(result.current.isPlaying).toBe(false);
      expect(result.current.volume).toBe(70);
    });

    it('useQueueState should return queue slice', () => {
      const { result } = renderHook(() => useQueueState(), { wrapper });

      expect(result.current.tracks).toEqual([]);
      expect(result.current.currentIndex).toBe(0);
    });

    it('useCacheState should return cache slice', () => {
      const { result } = renderHook(() => useCacheState(), { wrapper });

      expect(result.current.stats).toBeNull();
      expect(result.current.health).toBeNull();
    });

    it('useConnectionState should return connection slice', () => {
      const { result } = renderHook(() => useConnectionState(), { wrapper });

      expect(result.current.wsConnected).toBe(false);
      expect(result.current.apiConnected).toBe(false);
    });
  });
});
