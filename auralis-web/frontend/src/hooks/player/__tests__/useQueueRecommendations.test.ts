/**
 * useQueueRecommendations Hook Tests
 *
 * Comprehensive test suite for recommendations hook.
 * Covers: calculations, memoization, filtering
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useQueueRecommendations } from '../useQueueRecommendations';
import type { Track } from '@/types/domain';

// Mock tracks for testing
const mockTracks: Track[] = [
  {
    id: 1,
    title: 'Song 1',
    artist: 'Artist A',
    album: 'Album 1',
    duration: 300,
    filepath: '/music/a1.mp3',
  },
  {
    id: 2,
    title: 'Song 2',
    artist: 'Artist A',
    album: 'Album 1',
    duration: 320,
    filepath: '/music/a2.mp3',
  },
  {
    id: 3,
    title: 'Song 3',
    artist: 'Artist A',
    album: 'Album 2',
    duration: 280,
    filepath: '/music/a3.mp3',
  },
  {
    id: 4,
    title: 'Song 4',
    artist: 'Artist B',
    album: 'Album 3',
    duration: 290,
    filepath: '/music/b1.mp3',
  },
  {
    id: 5,
    title: 'Song 5',
    artist: 'Artist B',
    album: 'Album 3',
    duration: 310,
    filepath: '/music/b2.flac',
  },
  {
    id: 6,
    title: 'Song 6',
    artist: 'Artist C',
    album: 'Album 4',
    duration: 330,
    filepath: '/music/c1.mp3',
  },
];

describe('useQueueRecommendations', () => {
  // =========================================================================
  // FOR YOU RECOMMENDATIONS
  // =========================================================================

  it('should provide for you recommendations', () => {
    const queue = [mockTracks[0], mockTracks[1], mockTracks[3]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.forYouRecommendations).toBeDefined();
    expect(Array.isArray(result.current.forYouRecommendations)).toBe(true);
  });

  it('should exclude queue tracks from for you recommendations', () => {
    const queue = [mockTracks[0], mockTracks[1]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    const queueIds = new Set(queue.map((t) => t.id));
    expect(
      result.current.forYouRecommendations.every(
        (r) => !queueIds.has(r.track.id)
      )
    ).toBe(true);
  });

  it('should return empty for you recommendations without enough data', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.forYouRecommendations.length).toBe(0);
  });

  // =========================================================================
  // SIMILAR TO CURRENT TRACK
  // =========================================================================

  it('should provide similar to current track recommendations', () => {
    const queue = [mockTracks[0], mockTracks[1]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.similarToCurrentTrack).toBeDefined();
    expect(Array.isArray(result.current.similarToCurrentTrack)).toBe(true);
  });

  it('should return empty similar recommendations without current track', () => {
    const queue = [mockTracks[0], mockTracks[1]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, null, mockTracks)
    );

    expect(result.current.similarToCurrentTrack).toHaveLength(0);
  });

  it('should not include current track in similar recommendations', () => {
    const queue = [mockTracks[0], mockTracks[1]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(
      result.current.similarToCurrentTrack.every(
        (r) => r.track.id !== mockTracks[0].id
      )
    ).toBe(true);
  });

  // =========================================================================
  // DISCOVERY PLAYLIST
  // =========================================================================

  it('should provide discovery playlist', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.discoveryPlaylist).toBeDefined();
    expect(Array.isArray(result.current.discoveryPlaylist)).toBe(true);
  });

  it('should return empty discovery playlist without available tracks', () => {
    const { result } = renderHook(() =>
      useQueueRecommendations([mockTracks[0]], mockTracks[0], [])
    );

    expect(result.current.discoveryPlaylist).toHaveLength(0);
  });

  // =========================================================================
  // NEW ARTISTS
  // =========================================================================

  it('should discover new artists', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.newArtists).toBeDefined();
    expect(Array.isArray(result.current.newArtists)).toBe(true);
  });

  it('should not include queue artists in new artists', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.newArtists.every((a) => a.artist !== 'Artist A')).toBe(
      true
    );
  });

  it('should return empty new artists without queue', () => {
    const { result } = renderHook(() =>
      useQueueRecommendations([], null, mockTracks)
    );

    expect(result.current.newArtists).toHaveLength(0);
  });

  // =========================================================================
  // RELATED ARTISTS
  // =========================================================================

  it('should find related artists', () => {
    const queue = [mockTracks[0], mockTracks[1]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.relatedArtists).toBeDefined();
    expect(Array.isArray(result.current.relatedArtists)).toBe(true);
  });

  it('should return empty related artists without current track', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, null, mockTracks)
    );

    expect(result.current.relatedArtists).toHaveLength(0);
  });

  // =========================================================================
  // UTILITIES
  // =========================================================================

  it('should provide getRecommendationsFor function', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    const recommendations = result.current.getRecommendationsFor(mockTracks[0], 5);
    expect(Array.isArray(recommendations)).toBe(true);
  });

  it('should provide getByArtist function', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    const tracks = result.current.getByArtist('Artist A', 10);
    expect(Array.isArray(tracks)).toBe(true);
    expect(tracks.every((t) => t.artist === 'Artist A')).toBe(true);
  });

  it('should provide getAlbumsByArtist function', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    const albums = result.current.getAlbumsByArtist('Artist A');
    expect(albums).toBeInstanceOf(Map);
    expect(albums.size).toBeGreaterThan(0);
  });

  // =========================================================================
  // HAS ENOUGH DATA
  // =========================================================================

  it('should indicate when data is insufficient', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.hasEnoughData).toBe(false);
  });

  it('should indicate when data is sufficient', () => {
    const queue = [mockTracks[0], mockTracks[1], mockTracks[3]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], mockTracks)
    );

    expect(result.current.hasEnoughData).toBe(true);
  });

  // =========================================================================
  // MEMOIZATION
  // =========================================================================

  it('should memoize for you recommendations', () => {
    const queue = [mockTracks[0], mockTracks[1], mockTracks[3]];
    const { result, rerender } = renderHook(
      ({ q, t, a }: { q: Track[]; t: Track | null; a: Track[] }) =>
        useQueueRecommendations(q, t, a),
      {
        initialProps: {
          q: queue,
          t: mockTracks[0],
          a: mockTracks,
        },
      }
    );

    const firstRecommendations = result.current.forYouRecommendations;

    // Same props, should return same reference
    rerender({
      q: queue,
      t: mockTracks[0],
      a: mockTracks,
    });

    const secondRecommendations = result.current.forYouRecommendations;

    expect(firstRecommendations).toBe(secondRecommendations);
  });

  it('should recalculate on queue change', () => {
    const queue1 = [mockTracks[0], mockTracks[1], mockTracks[3]];
    const queue2 = [mockTracks[0], mockTracks[4], mockTracks[5]];

    const { result, rerender } = renderHook(
      ({ q, t, a }: { q: Track[]; t: Track | null; a: Track[] }) =>
        useQueueRecommendations(q, t, a),
      {
        initialProps: {
          q: queue1,
          t: mockTracks[0],
          a: mockTracks,
        },
      }
    );

    const firstRecommendations = result.current.forYouRecommendations;

    rerender({
      q: queue2,
      t: mockTracks[0],
      a: mockTracks,
    });

    const secondRecommendations = result.current.forYouRecommendations;

    expect(firstRecommendations).not.toBe(secondRecommendations);
  });

  // =========================================================================
  // EDGE CASES
  // =========================================================================

  it('should handle empty queue', () => {
    const { result } = renderHook(() =>
      useQueueRecommendations([], null, mockTracks)
    );

    expect(result.current.forYouRecommendations).toHaveLength(0);
    expect(result.current.newArtists).toHaveLength(0);
    expect(result.current.relatedArtists).toHaveLength(0);
  });

  it('should handle empty available tracks', () => {
    const queue = [mockTracks[0]];
    const { result } = renderHook(() =>
      useQueueRecommendations(queue, mockTracks[0], [])
    );

    expect(result.current.discoveryPlaylist).toHaveLength(0);
    expect(result.current.similarToCurrentTrack).toHaveLength(0);
  });

  it('should handle queue equal to available tracks', () => {
    const { result } = renderHook(() =>
      useQueueRecommendations(mockTracks, mockTracks[0], mockTracks)
    );

    // Should still have recommendations, but might be limited
    expect(Array.isArray(result.current.forYouRecommendations)).toBe(true);
  });

  // =========================================================================
  // LARGE DATASETS
  // =========================================================================

  it('should handle large available track libraries', () => {
    const largeTracks: Track[] = Array.from({ length: 5000 }, (_, i) => ({
      id: i,
      title: `Track ${i}`,
      artist: `Artist ${i % 100}`,
      album: `Album ${i % 50}`,
      duration: 200 + (i % 200),
      filepath: `/music/track${i}.mp3`,
    }));

    const queue = largeTracks.slice(0, 10);

    const { result } = renderHook(() =>
      useQueueRecommendations(queue, largeTracks[0], largeTracks)
    );

    expect(result.current.forYouRecommendations.length).toBeGreaterThan(0);
    expect(result.current.discoveryPlaylist.length).toBeGreaterThan(0);
  });
});

describe('useQueueRecommendations oversized-queue warning (#4459)', () => {
  const bigQueue: Track[] = Array.from({ length: 1001 }, (_, i) => ({
    id: i + 1,
    title: `Song ${i + 1}`,
    artist: 'Artist',
    album: 'Album',
    duration: 200,
  })) as Track[];

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('warns once per hook instance in DEV, not on every render', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const { rerender } = renderHook(() => useQueueRecommendations(bigQueue, null, []));
    rerender();
    rerender();

    const hits = warn.mock.calls.filter((c) => String(c[0]).includes('useQueueRecommendations'));
    expect(hits).toHaveLength(1);
  });

  it('does not warn outside DEV', () => {
    vi.stubEnv('DEV', false);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    renderHook(() => useQueueRecommendations(bigQueue, null, []));

    const hits = warn.mock.calls.filter((c) => String(c[0]).includes('useQueueRecommendations'));
    expect(hits).toHaveLength(0);
  });

  it('does not warn for a queue at the limit', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    renderHook(() => useQueueRecommendations(bigQueue.slice(0, 1000), null, []));

    const hits = warn.mock.calls.filter((c) => String(c[0]).includes('useQueueRecommendations'));
    expect(hits).toHaveLength(0);
  });
});
