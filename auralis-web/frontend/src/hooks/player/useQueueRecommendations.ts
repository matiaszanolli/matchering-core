/**
 * useQueueRecommendations Hook
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Provides track recommendations based on queue.
 * Suggests similar tracks, discovers new artists, and generates playlists.
 *
 * Usage:
 * ```typescript
 * const {
 *   forYouRecommendations,
 *   similarToCurrentTrack,
 *   discoveryPlaylist,
 *   newArtists,
 *   relatedArtists
 * } = useQueueRecommendations(queue, currentTrack, availableTracks);
 * ```
 *
 * Features:
 * - For You recommendations (collaborative filtering)
 * - Similar to current track
 * - Artist-based recommendations
 * - New artist discovery
 * - Related artist suggestions
 * - Discovery playlist generation
 * - Configurable similarity threshold
 *
 * @module hooks/player/useQueueRecommendations
 */

import { useCallback, useMemo, useRef } from 'react';
import { QueueRecommender, type TrackRecommendation } from '@/utils/queue/queue_recommender';
import type { Track, QueueTrack } from '@/types/domain';

/**
 * Discovery artist with sample tracks
 */
export interface DiscoveryArtist {
  artist: string;
  trackCount: number;
  tracks: Track[];
}

/**
 * Related artist with similarity score
 */
export interface RelatedArtist {
  artist: string;
  similarity: number; // 0-1
  commonTracks: number;
}

/**
 * Return type for useQueueRecommendations hook
 */
export interface QueueRecommendationsActions {
  /** Recommendations based on entire queue (For You) */
  forYouRecommendations: TrackRecommendation[];

  /** Recommendations similar to current track */
  similarToCurrentTrack: TrackRecommendation[];

  /** Discovery playlist (diverse selection) */
  discoveryPlaylist: Track[];

  /** New artists to explore */
  newArtists: DiscoveryArtist[];

  /** Artists related to current artist */
  relatedArtists: RelatedArtist[];

  /** Get recommendations for specific track */
  getRecommendationsFor: (track: Track, count?: number) => TrackRecommendation[];

  /** Get recommendations by artist */
  getByArtist: (artist: string, count?: number) => Track[];

  /** Get albums by artist */
  getAlbumsByArtist: (artist: string) => Map<string, Track[]>;

  /** Whether enough data for recommendations */
  hasEnoughData: boolean;
}

/**
 * Hook for getting queue recommendations
 *
 * ⚠️ **IMPORTANT: Queue Size Constraints**
 * This hook is optimized for playback queues (100-500 tracks).
 * DO NOT use with entire music library (will crash).
 *
 * Safe ranges:
 * - Optimal: 100-500 tracks for queue, 1000+ for availableTracks
 * - Maximum queue: 1000 tracks (risky)
 * - Never queue: 10K+ tracks (will crash)
 *
 * @param queue Current queue tracks (max 500 recommended)
 * @param currentTrack Currently playing track (if any)
 * @param availableTracks All available tracks to recommend from
 * @returns Recommendations and utility functions
 *
 * @example
 * ```typescript
 * const { queue, currentTrack } = usePlaybackQueue();
 * const { library } = useLibrary();
 *
 * const {
 *   forYouRecommendations,
 *   similarToCurrentTrack,
 *   newArtists
 * } = useQueueRecommendations(queue, currentTrack, library);
 *
 * // Show "For You" section
 * forYouRecommendations.slice(0, 5).map(rec => rec.track.title)
 * ```
 */
export function useQueueRecommendations(
  queue: (Track | QueueTrack)[],
  currentTrack: Track | QueueTrack | null,
  availableTracks: Track[]
): QueueRecommendationsActions {
  // Guard: warn once per hook instance, DEV only — the same guard its siblings
  // useQueueStatistics (#3974) and useQueueSearch (#4194) got. This third
  // sibling was missed, so it warned on every render and in production
  // (#4459). Per-instance rather than module-level so HMR and tests reset.
  const _warnedRef = useRef(false);
  if (import.meta.env.DEV && queue.length > 1000 && !_warnedRef.current) {
    _warnedRef.current = true;
    console.warn(
      `⚠️ useQueueRecommendations: Queue size (${queue.length}) exceeds safe limit (1000). ` +
      `This hook is designed for playback queues only (100-500 tracks), not entire libraries. ` +
      `Using with large datasets will cause severe performance degradation or crashes. ` +
      `See: PHASE_7_ARCHITECTURAL_FIX.md for guidance.`
    );
  }
  // Check if we have enough data
  const hasEnoughData = queue.length >= 3 && availableTracks.length > queue.length;

  // For You recommendations (based on entire queue)
  const forYouRecommendations = useMemo(() => {
    if (!hasEnoughData) return [];

    return QueueRecommender.recommendForYou(
      queue as Track[],
      availableTracks,
      10,
      {
        excludeQueue: true,
        minScore: 0.2,
      }
    );
  }, [queue, availableTracks, hasEnoughData]);

  // Similar to current track
  const similarToCurrentTrack = useMemo(() => {
    if (!currentTrack || availableTracks.length === 0) return [];

    return QueueRecommender.recommendSimilarTracks(
      currentTrack as Track,
      availableTracks,
      8,
      {
        excludeQueue: true,
        minScore: 0.25,
      }
    );
  }, [currentTrack, availableTracks]);

  // Discovery playlist
  const discoveryPlaylist = useMemo(() => {
    if (availableTracks.length === 0) return [];

    return QueueRecommender.getDiscoveryPlaylist(availableTracks, 20);
  }, [availableTracks]);

  // New artists to discover
  const newArtists = useMemo(() => {
    if (queue.length === 0 || availableTracks.length === 0) return [];

    return QueueRecommender.discoverNewArtists(queue as Track[], availableTracks, 5);
  }, [queue, availableTracks]);

  // Related artists to current playing
  const relatedArtists = useMemo(() => {
    if (!currentTrack || availableTracks.length === 0) return [];

    return QueueRecommender.findRelatedArtists(
      currentTrack.artist,
      queue as Track[],
      availableTracks,
      5
    );
  }, [currentTrack, queue, availableTracks]);

  // Utility: Get recommendations for specific track
  const getRecommendationsFor = useCallback((track: Track, count: number = 5) => {
    return QueueRecommender.recommendSimilarTracks(
      track,
      availableTracks,
      count,
      {
        excludeQueue: true,
      }
    );
  }, [availableTracks]);

  // Utility: Get tracks by artist
  const getByArtist = useCallback((artist: string, count: number = 10) => {
    return QueueRecommender.getByArtist(artist, availableTracks, count, queue as Track[]);
  }, [availableTracks, queue]);

  // Utility: Get albums by artist
  const getAlbumsByArtist = useCallback((artist: string) => {
    return QueueRecommender.getAlbumsByArtist(artist, availableTracks);
  }, [availableTracks]);

  return {
    forYouRecommendations,
    similarToCurrentTrack,
    discoveryPlaylist,
    newArtists,
    relatedArtists,
    getRecommendationsFor,
    getByArtist,
    getAlbumsByArtist,
    hasEnoughData,
  };
}
