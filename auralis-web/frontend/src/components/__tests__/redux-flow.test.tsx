/**
 * Redux Flow Tests
 * ~~~~~~~~~~~~~~~~
 *
 * Unit tests for Redux action/reducer flows (dispatch → state assertions).
 *
 * Test Scenarios:
 * 1. Complete Playback Flow
 *    - Connect to service
 *    - Load queue
 *    - Play track
 *    - Adjust volume
 *    - Seek position
 *    - Next/previous navigation
 *
 * 2. Queue Management Flow
 *    - Add tracks
 *    - Reorder tracks
 *    - Remove tracks
 *    - Clear queue
 *    - Play from queue
 *
 * 3. Cache Management Flow
 *    - View cache statistics
 *    - Monitor cache health
 *    - Clear cache
 *    - Track per-track cache
 *
 * 4. Connection Recovery Flow
 *    - Detect disconnection
 *    - Display reconnect prompt
 *    - Attempt reconnection
 *    - Restore state after reconnection
 *
 * 5. Error Recovery Flow
 *    - Handle playback errors
 *    - Handle API errors
 *    - Handle connection errors
 *    - Display error messages
 *    - Provide recovery options
 *
 * Phase C.3: Component Testing & Integration
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3, see LICENSE for more details
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { configureStore } from '@reduxjs/toolkit';
import playerReducer from '@/store/slices/playerSlice';
import queueReducer from '@/store/slices/queueSlice';
import cacheReducer from '@/store/slices/cacheSlice';
import connectionReducer from '@/store/slices/connectionSlice';
import * as playerActions from '@/store/slices/playerSlice';
import * as queueActions from '@/store/slices/queueSlice';
import * as cacheActions from '@/store/slices/cacheSlice';
import * as connectionActions from '@/store/slices/connectionSlice';

// Mock hooks
vi.mock('@/contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({
    isConnected: true,
    connectionStatus: 'connected',
    send: vi.fn(),
    subscribe: vi.fn(() => () => {}),
    subscribeAll: vi.fn(() => () => {}),
    connect: vi.fn(),
    disconnect: vi.fn(),
  }),
}));
vi.mock('@/hooks/useStandardizedAPI');

describe('End-to-End User Flows', () => {
  let store: any;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        player: playerReducer,
        queue: queueReducer,
        cache: cacheReducer,
        connection: connectionReducer,
      },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ============================================================================
  // Complete Playback Flow
  // ============================================================================

  describe('Complete Playback Flow', () => {
    it('should complete full playback workflow', async () => {
      const tracks = [
        { id: 1, title: 'Track 1', artist: 'Artist 1', duration: 180 },
        { id: 2, title: 'Track 2', artist: 'Artist 2', duration: 200 },
        { id: 3, title: 'Track 3', artist: 'Artist 3', duration: 150 },
      ];

      // 1. Add tracks to queue
      tracks.forEach((t) => store.dispatch(queueActions.addTrack(t)));

      let state = store.getState();
      expect(state.queue.tracks).toHaveLength(3);

      // 2. Load first track
      store.dispatch(playerActions.setCurrentTrack(tracks[0]));
      state = store.getState();
      expect(state.player.currentTrack?.id).toBe(1);

      // 3. Start playback
      store.dispatch(playerActions.setIsPlaying(true));
      state = store.getState();
      expect(state.player.isPlaying).toBe(true);

      // 4. Adjust volume
      store.dispatch(playerActions.setVolume(50));
      state = store.getState();
      expect(state.player.volume).toBe(50);

      // 5. Seek in track
      store.dispatch(playerActions.setCurrentTime(90));
      state = store.getState();
      expect(state.player.currentTime).toBe(90);

      // 6. Play next track
      store.dispatch(queueActions.setCurrentIndex(1));
      store.dispatch(playerActions.setCurrentTrack(tracks[1]));
      state = store.getState();
      expect(state.player.currentTrack?.id).toBe(2);

      // 7. Pause playback
      store.dispatch(playerActions.setIsPlaying(false));
      state = store.getState();
      expect(state.player.isPlaying).toBe(false);
    });

    it('should handle preset changes during playback', () => {
      const track = { id: 1, title: 'Test', artist: 'Artist', duration: 180 };

      store.dispatch(playerActions.setCurrentTrack(track));
      store.dispatch(playerActions.setIsPlaying(true));

      // Change preset
      store.dispatch(playerActions.setPreset('warm'));

      const state = store.getState();
      expect(state.player.preset).toBe('warm');
      expect(state.player.isPlaying).toBe(true);
    });

    it('should handle volume muting during playback', () => {
      store.dispatch(playerActions.setVolume(70));
      store.dispatch(playerActions.setIsPlaying(true));

      // Mute
      store.dispatch(playerActions.setMuted(true));

      let state = store.getState();
      expect(state.player.isMuted).toBe(true);
      expect(state.player.volume).toBe(70);
      expect(state.player.isPlaying).toBe(true);

      // Unmute
      store.dispatch(playerActions.setMuted(false));
      state = store.getState();
      expect(state.player.isMuted).toBe(false);
    });
  });

  // ============================================================================
  // Queue Management Flow
  // ============================================================================

  describe('Queue Management Flow', () => {
    it('should complete full queue management workflow', () => {
      const track1 = { id: 1, title: 'Track 1', artist: 'Artist 1', duration: 180 };
      const track2 = { id: 2, title: 'Track 2', artist: 'Artist 2', duration: 200 };
      const track3 = { id: 3, title: 'Track 3', artist: 'Artist 3', duration: 150 };

      // 1. Add tracks
      store.dispatch(queueActions.addTrack(track1));
      store.dispatch(queueActions.addTrack(track2));

      let state = store.getState();
      expect(state.queue.tracks).toHaveLength(2);

      // 2. Add another track
      store.dispatch(queueActions.addTrack(track3));
      state = store.getState();
      expect(state.queue.tracks).toHaveLength(3);

      // 3. Reorder tracks
      store.dispatch(queueActions.reorderTrack({ fromIndex: 0, toIndex: 2 }));
      state = store.getState();
      expect(state.queue.tracks[2].id).toBe(1);

      // 4. Remove a track
      store.dispatch(queueActions.removeTrack(1));
      state = store.getState();
      expect(state.queue.tracks).toHaveLength(2);

      // 5. Set current track
      store.dispatch(queueActions.setCurrentIndex(1));
      state = store.getState();
      expect(state.queue.currentIndex).toBe(1);

      // 6. Clear queue
      store.dispatch(queueActions.clearQueue());
      state = store.getState();
      expect(state.queue.tracks).toHaveLength(0);
      expect(state.queue.currentIndex).toBe(0);
    });

  });

  // ============================================================================
  // Cache Management Flow
  // ============================================================================

  describe('Cache Management Flow', () => {
    it('should complete cache management workflow', () => {
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

      // 1. Load cache stats
      store.dispatch(cacheActions.setCacheStats(stats));

      let state = store.getState();
      expect(state.cache.stats?.overall.total_chunks).toBe(12);
      expect(state.cache.stats?.overall.overall_hit_rate).toBe(0.73);

      // 2. Set cache health
      const health = {
        healthy: true,
        tier1_size_mb: 100,
        tier1_healthy: true,
        tier2_size_mb: 500,
        tier2_healthy: true,
        total_size_mb: 600,
        memory_healthy: true,
        tier1_hit_rate: 0.9,
        overall_hit_rate: 0.95,
        timestamp: new Date().toISOString(),
      };

      store.dispatch(cacheActions.setCacheHealth(health));
      state = store.getState();
      expect(state.cache.health?.healthy).toBe(true);

      // 3. Clear cache
      store.dispatch(cacheActions.clearCacheLocal());
      state = store.getState();
      expect(state.cache.stats?.overall.total_chunks).toBe(0);

      // 4. Update cache (simulated refresh) — setCacheStats is the real
      // production dispatch for this (useReduxState.ts); updateCache was
      // dead code with zero production callers, deleted in #5015.
      store.dispatch(cacheActions.setCacheStats({
        ...stats,
        overall: { ...stats.overall, total_chunks: 5 },
      }));

      state = store.getState();
      expect(state.cache.stats?.overall.total_chunks).toBe(5);
    });
  });

  // ============================================================================
  // Connection Recovery Flow
  // ============================================================================

  describe('Connection Recovery Flow', () => {
    it('should handle complete connection recovery', () => {
      // 1. Start connected
      store.dispatch(connectionActions.setWSConnected(true));
      store.dispatch(connectionActions.setLatency(25));

      let state = store.getState();
      expect(state.connection.wsConnected).toBe(true);

      // 2. Connection lost
      store.dispatch(connectionActions.setWSConnected(false));
      store.dispatch(connectionActions.setError('Connection lost'));

      state = store.getState();
      expect(state.connection.wsConnected).toBe(false);
      expect(state.connection.lastError).toBe('Connection lost');

      // 3. Attempt reconnection
      store.dispatch(connectionActions.resetReconnectAttempts());

      state = store.getState();
      expect(state.connection.reconnectAttempts).toBe(0);

      // 4. Reconnection attempts
      for (let i = 0; i < 2; i++) {
        store.dispatch(connectionActions.incrementReconnectAttempts());
      }

      state = store.getState();
      expect(state.connection.reconnectAttempts).toBe(2);

      // 5. Successful reconnection
      store.dispatch(connectionActions.setWSConnected(true));
      store.dispatch(connectionActions.setLatency(30));
      store.dispatch(connectionActions.clearError());
      store.dispatch(connectionActions.resetReconnectAttempts());

      state = store.getState();
      expect(state.connection.wsConnected).toBe(true);
      expect(state.connection.lastError).toBeNull();
      expect(state.connection.reconnectAttempts).toBe(0);
    });

    it('should handle API connectivity separately from WebSocket', () => {
      // WebSocket connected, API down
      store.dispatch(connectionActions.setWSConnected(true));
      store.dispatch(connectionActions.setAPIConnected(false));

      let state = store.getState();
      expect(state.connection.wsConnected).toBe(true);
      expect(state.connection.apiConnected).toBe(false);

      // Both connected
      store.dispatch(connectionActions.setAPIConnected(true));

      state = store.getState();
      expect(state.connection.wsConnected).toBe(true);
      expect(state.connection.apiConnected).toBe(true);
    });
  });

  // ============================================================================
  // Error Recovery Flow
  // ============================================================================

  describe('Error Recovery Flow', () => {
    it('should handle playback errors gracefully', () => {
      const track = { id: 1, title: 'Test', artist: 'Artist', duration: 180 };

      // Load and play track
      store.dispatch(playerActions.setCurrentTrack(track));
      store.dispatch(playerActions.setIsPlaying(true));

      // Simulate playback error
      store.dispatch(playerActions.setError('Audio playback failed'));

      let state = store.getState();
      expect(state.player.error).toBe('Audio playback failed');

      // Pause playback
      store.dispatch(playerActions.setIsPlaying(false));

      // Clear error
      store.dispatch(playerActions.clearError());

      state = store.getState();
      expect(state.player.error).toBeNull();
      expect(state.player.isPlaying).toBe(false);
    });

    it('should handle queue errors with recovery', () => {
      // Add tracks successfully
      const track1 = { id: 1, title: 'Track 1', artist: 'Artist 1', duration: 180 };
      store.dispatch(queueActions.addTrack(track1));

      // Simulate queue loading error
      store.dispatch(queueActions.setError('Failed to load queue'));

      let state = store.getState();
      expect(state.queue.error).toBe('Failed to load queue');

      // Attempt recovery - clear error and retry
      store.dispatch(queueActions.clearError());
      const track2 = { id: 2, title: 'Track 2', artist: 'Artist 2', duration: 200 };
      store.dispatch(queueActions.addTrack(track2));

      state = store.getState();
      expect(state.queue.error).toBeNull();
      expect(state.queue.tracks).toHaveLength(2);
    });

    it('should handle cache errors with graceful degradation', () => {
      // Simulate cache load error
      store.dispatch(cacheActions.setError('Cache stats unavailable'));

      let state = store.getState();
      expect(state.cache.error).toBe('Cache stats unavailable');
      expect(state.cache.stats).toBeNull();

      // Playback should still work
      const track = { id: 1, title: 'Test', artist: 'Artist', duration: 180 };
      store.dispatch(playerActions.setCurrentTrack(track));
      store.dispatch(playerActions.setIsPlaying(true));

      state = store.getState();
      expect(state.player.isPlaying).toBe(true);

      // Cache error doesn't affect playback
      expect(state.cache.error).toBe('Cache stats unavailable');
    });
  });

  // ============================================================================
  // Complex Multi-Feature Workflows
  // ============================================================================

  describe('Complex Multi-Feature Workflows', () => {
    it('should handle simultaneous playback, queue, and connection state changes', () => {
      const tracks = [
        { id: 1, title: 'Track 1', artist: 'Artist 1', duration: 180 },
        { id: 2, title: 'Track 2', artist: 'Artist 2', duration: 200 },
      ];

      // 1. Connect
      store.dispatch(connectionActions.setWSConnected(true));
      store.dispatch(connectionActions.setLatency(25));

      // 2. Load queue
      tracks.forEach((t) => store.dispatch(queueActions.addTrack(t)));

      // 3. Start playback
      store.dispatch(playerActions.setCurrentTrack(tracks[0]));
      store.dispatch(playerActions.setIsPlaying(true));

      // 4. Load cache stats
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

      store.dispatch(cacheActions.setCacheStats(stats));

      const state = store.getState();

      // Verify all subsystems working together
      expect(state.connection.wsConnected).toBe(true);
      expect(state.queue.tracks).toHaveLength(2);
      expect(state.player.isPlaying).toBe(true);
      expect(state.cache.stats?.overall.total_chunks).toBe(12);
    });

    it('should recover from disconnection while playing', () => {
      const track = { id: 1, title: 'Test', artist: 'Artist', duration: 180 };

      // 1. Connected and playing
      store.dispatch(connectionActions.setWSConnected(true));
      store.dispatch(playerActions.setCurrentTrack(track));
      store.dispatch(playerActions.setIsPlaying(true));

      let state = store.getState();
      expect(state.connection.wsConnected).toBe(true);
      expect(state.player.isPlaying).toBe(true);

      // 2. Disconnection occurs
      store.dispatch(connectionActions.setWSConnected(false));

      state = store.getState();
      expect(state.connection.wsConnected).toBe(false);
      // Playback may pause or continue depending on implementation
      // But track state should be preserved

      // 3. Reconnect
      store.dispatch(connectionActions.setWSConnected(true));

      state = store.getState();
      expect(state.connection.wsConnected).toBe(true);
      // Track state preserved
      expect(state.player.currentTrack?.id).toBe(1);
    });

    it('should handle queue changes while playing', () => {
      const tracks = [
        { id: 1, title: 'Track 1', artist: 'Artist 1', duration: 180 },
        { id: 2, title: 'Track 2', artist: 'Artist 2', duration: 200 },
        { id: 3, title: 'Track 3', artist: 'Artist 3', duration: 150 },
      ];

      // Start playing
      store.dispatch(playerActions.setCurrentTrack(tracks[0]));
      store.dispatch(playerActions.setIsPlaying(true));

      // Build queue
      tracks.forEach((t) => store.dispatch(queueActions.addTrack(t)));
      store.dispatch(queueActions.setCurrentIndex(0));

      // Modify queue while playing
      store.dispatch(queueActions.removeTrack(2)); // Remove last track

      let state = store.getState();
      expect(state.queue.tracks).toHaveLength(2);
      expect(state.player.isPlaying).toBe(true);

      // Move to next
      store.dispatch(queueActions.setCurrentIndex(1));
      store.dispatch(playerActions.setCurrentTrack(tracks[1]));

      state = store.getState();
      expect(state.queue.currentIndex).toBe(1);
      expect(state.player.currentTrack?.id).toBe(2);
    });
  });
});
