/**
 * Redux Test Fixtures Tests
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Unit tests for Redux test fixtures and utilities.
 *
 * Test Coverage:
 * - Mock data generation
 * - Store creation
 * - State verification
 * - State diffing
 * - Test scenarios
 *
 * Phase C.4d: Redux Test Utilities Testing
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3, see LICENSE for more details
 */

import { describe, it, expect } from 'vitest';
import {
  createMockTrack,
  createMockTracks,
  createMockCacheStats,
  createMockCacheHealth,
  emptyStoreState,
  connectedWithTrackState,
  playingWithQueueState,
  healthyCacheState,
  loadingState,
  errorState,
  offlineState,
  createTestStore,
  createStoreFromFixture,
  assertPlayerState,
  assertQueueState,
  assertNoErrors,
  assertConnected,
  getStateDiff,
  assertStateChanged,
  assertStateUnchanged,
  testScenarios,
} from './testFixtures';
import * as playerActions from '../slices/playerSlice';
import * as queueActions from '../slices/queueSlice';

describe('Redux Test Fixtures', () => {
  // ============================================================================
  // Mock Data Generation Tests
  // ============================================================================

  describe('Mock Data Generators', () => {
    it('should create mock track', () => {
      const track = createMockTrack();

      expect(track).toHaveProperty('id');
      expect(track).toHaveProperty('title');
      expect(track).toHaveProperty('artist');
      expect(track).toHaveProperty('duration');
      expect(typeof track.duration).toBe('number');
      expect(track.duration).toBeGreaterThan(0);
    });

    it('should create mock track with overrides', () => {
      const track = createMockTrack({
        id: 999,
        title: 'Custom Track',
        artist: 'Custom Artist',
      });

      expect(track.id).toBe(999);
      expect(track.title).toBe('Custom Track');
      expect(track.artist).toBe('Custom Artist');
    });

    it('should create multiple mock tracks', () => {
      const tracks = createMockTracks(10);

      expect(tracks).toHaveLength(10);
      expect(tracks.every((t) => t.title && t.artist)).toBe(true);
      expect(tracks.every((t) => typeof t.duration === 'number')).toBe(true);
    });

    it('should create mock cache stats', () => {
      const stats = createMockCacheStats();

      expect(stats).toHaveProperty('tier1');
      expect(stats).toHaveProperty('tier2');
      expect(stats).toHaveProperty('overall');
      expect(stats.overall.overall_hit_rate).toBeGreaterThanOrEqual(0);
      expect(stats.overall.overall_hit_rate).toBeLessThanOrEqual(1);
    });

    it('should create mock cache health', () => {
      const health = createMockCacheHealth();

      expect(health).toHaveProperty('healthy');
      expect(health).toHaveProperty('hit_rate');
      expect(health).toHaveProperty('status');
      expect(typeof health.healthy).toBe('boolean');
    });
  });

  // ============================================================================
  // Store Creation Tests
  // ============================================================================

  describe('Store Creation', () => {
    it('should create empty test store', () => {
      const store = createTestStore();

      expect(store).toBeDefined();
      expect(store.getState()).toBeDefined();
      const state = store.getState();
      expect(state.player).toBeDefined();
      expect(state.queue).toBeDefined();
    });

    it('should create store with initial state', () => {
      const store = createTestStore({
        player: {
          volume: 50,
          isPlaying: true,
        } as any,
      });

      const state = store.getState();
      expect(state.player.volume).toBe(50);
      expect(state.player.isPlaying).toBe(true);
    });

    it('should create store from fixture', () => {
      const store = createStoreFromFixture(playingWithQueueState);

      const state = store.getState();
      expect(state.player.isPlaying).toBe(true);
      expect(state.queue.tracks.length).toBeGreaterThan(0);
    });
  });

  // ============================================================================
  // State Fixtures Tests
  // ============================================================================

  describe('State Fixtures', () => {
    it('should have empty state fixture', () => {
      expect(emptyStoreState.player.isPlaying).toBe(false);
      expect(emptyStoreState.queue.tracks).toHaveLength(0);
      expect(emptyStoreState.connection.wsConnected).toBe(false);
    });

    it('should have connected state fixture', () => {
      expect(connectedWithTrackState.connection.wsConnected).toBe(true);
      expect(connectedWithTrackState.connection.apiConnected).toBe(true);
      expect(connectedWithTrackState.player.currentTrack).not.toBeNull();
    });

    it('should have playing state fixture', () => {
      expect(playingWithQueueState.player.isPlaying).toBe(true);
      expect(playingWithQueueState.queue.tracks.length).toBeGreaterThan(0);
    });

    it('should have healthy cache fixture', () => {
      expect(healthyCacheState.cache.stats).not.toBeNull();
      expect(healthyCacheState.cache.health?.healthy).toBe(true);
    });

    it('should have loading state fixture', () => {
      expect(loadingState.player.isLoading).toBe(true);
      expect(loadingState.queue.isLoading).toBe(true);
    });

    it('should have error state fixture', () => {
      expect(errorState.player.error).toBeTruthy();
      expect(errorState.queue.error).toBeTruthy();
      expect(errorState.connection.lastError).toBeTruthy();
    });

    it('should have offline state fixture', () => {
      expect(offlineState.connection.wsConnected).toBe(false);
      expect(offlineState.connection.apiConnected).toBe(false);
    });
  });

  // ============================================================================
  // State Verification Tests
  // ============================================================================

  describe('State Verification', () => {
    it('should verify player state', () => {
      expect(() => {
        assertPlayerState(playingWithQueueState, {
          isPlaying: true,
          volume: 70,
        });
      }).not.toThrow();
    });

    it('should fail on incorrect player state', () => {
      expect(() => {
        assertPlayerState(playingWithQueueState, {
          isPlaying: false,
        });
      }).toThrow();
    });

    it('should verify queue state', () => {
      expect(() => {
        assertQueueState(playingWithQueueState, {
          currentIndex: 0,
        });
      }).not.toThrow();
    });

    it('should verify no errors', () => {
      expect(() => {
        assertNoErrors(emptyStoreState);
      }).not.toThrow();

      expect(() => {
        assertNoErrors(errorState);
      }).toThrow();
    });

    it('should verify connected', () => {
      expect(() => {
        assertConnected(connectedWithTrackState);
      }).not.toThrow();

      expect(() => {
        assertConnected(offlineState);
      }).toThrow();
    });
  });

  // ============================================================================
  // State Comparison Tests
  // ============================================================================

  describe('State Comparison', () => {
    it('should get state diff', () => {
      const before = emptyStoreState;
      const after = connectedWithTrackState;

      const diff = getStateDiff(before, after);

      expect(Object.keys(diff).length).toBeGreaterThan(0);
      expect(diff.connection).toBeDefined();
    });

    it('should detect unchanged states', () => {
      expect(() => {
        assertStateUnchanged(emptyStoreState, emptyStoreState);
      }).not.toThrow();
    });

    it('should detect state changes', () => {
      expect(() => {
        assertStateChanged(emptyStoreState, connectedWithTrackState, ['connection']);
      }).not.toThrow();
    });

    it('should fail on unexpected changes', () => {
      expect(() => {
        assertStateChanged(emptyStoreState, emptyStoreState, ['connection']);
      }).toThrow();
    });
  });

  // ============================================================================
  // Test Scenarios Tests
  // ============================================================================

  describe('Test Scenarios', () => {
    it('should have all scenarios defined', () => {
      expect(testScenarios.empty).toBeDefined();
      expect(testScenarios.connectedWithTrack).toBeDefined();
      expect(testScenarios.playingWithQueue).toBeDefined();
      expect(testScenarios.healthyCache).toBeDefined();
      expect(testScenarios.loading).toBeDefined();
      expect(testScenarios.error).toBeDefined();
      expect(testScenarios.offline).toBeDefined();
      expect(testScenarios.reconnecting).toBeDefined();
    });

    it('should use empty scenario', () => {
      const store = createStoreFromFixture(testScenarios.empty);
      const state = store.getState();

      expect(state.queue.tracks).toHaveLength(0);
      expect(state.player.isPlaying).toBe(false);
    });

    it('should use playback scenario', () => {
      const store = createStoreFromFixture(testScenarios.playingWithQueue);
      const state = store.getState();

      expect(state.player.isPlaying).toBe(true);
      expect(state.queue.tracks.length).toBeGreaterThan(0);
    });
  });

  // ============================================================================
  // Integration Tests
  // ============================================================================

  describe('Integration', () => {
    it('should work with Redux actions', () => {
      const store = createTestStore();

      store.dispatch(playerActions.setVolume(50));

      const state = store.getState();
      expect(state.player.volume).toBe(50);
    });

    it('should track state changes after dispatch', () => {
      const store = createTestStore();
      const before = store.getState();

      store.dispatch(queueActions.addTrack(createMockTrack()));

      const after = store.getState();

      const diff = getStateDiff(before, after);
      expect(diff.queue).toBeDefined();
    });

    it('should enable testing workflows', () => {
      // 1. Start from fixture
      const store = createStoreFromFixture(playingWithQueueState);

      // 2. Make changes
      store.dispatch(playerActions.setVolume(40));
      store.dispatch(queueActions.setCurrentIndex(1));

      // 3. Verify results
      const state = store.getState();
      expect(state.player.volume).toBe(40);
      expect(state.queue.currentIndex).toBe(1);
    });
  });
});
