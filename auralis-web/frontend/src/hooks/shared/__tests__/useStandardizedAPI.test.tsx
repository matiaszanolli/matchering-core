/**
 * Cache telemetry hook tests — `useCacheStats` / `useCacheHealth`
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Written for #4693, which retires `StandardizedAPIClient` and moves these two
 * endpoints onto `utils/apiRequest.ts`. #4486 had recorded that this path was
 * untested at both layers, and it was: the only test touching these hooks
 * (`CacheStatsDashboard.test.tsx`) mocks `useCacheStats` wholesale, so nothing
 * exercised the request, the shape guard or the error path.
 *
 * These are characterization tests. They were written and made green against
 * the OLD `CacheAwareAPIClient` implementation *before* the transport swap, so
 * that "the same tests still pass" is real evidence the swap preserved
 * behaviour rather than an assertion about the new code only.
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3, see LICENSE for more details
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

import { server } from '@/test/mocks/server';
import { AllProviders } from '@/test/test-utils';
import {
  useCacheStats,
  useCacheHealth,
  CACHE_STATS_REFRESH_INTERVAL_MS,
} from '../useStandardizedAPI';
import { mockCacheStats } from '@/components/shared/__tests__/test-utils';

const mockCacheHealth = {
  healthy: true,
  total_size_mb: 225.0,
  max_size_mb: 500.0,
  usage_percent: 45.0,
};

describe('useCacheStats / useCacheHealth (#4693)', () => {
  beforeEach(() => {
    server.use(
      http.get('/api/cache/stats', () => HttpResponse.json(mockCacheStats)),
      http.get('/api/cache/health', () => HttpResponse.json(mockCacheHealth))
    );
  });

  describe('useCacheStats', () => {
    it('resolves the bare payload the backend actually returns', async () => {
      // The endpoint returns CacheStats directly, with no {status,data}
      // envelope. #4440 fixed a version that gated on the envelope and so
      // resolved null on every 200 OK — this pins the bare shape.
      const { result } = renderHook(() => useCacheStats(), { wrapper: AllProviders });

      await waitFor(() => expect(result.current.loading).toBe(false));

      expect(result.current.data).toEqual(mockCacheStats);
      expect(result.current.error).toBeNull();
    });

    it('reports an error rather than null data when the shape is wrong', async () => {
      server.use(
        http.get('/api/cache/stats', () => HttpResponse.json({ nonsense: true }))
      );

      const { result } = renderHook(() => useCacheStats(), { wrapper: AllProviders });

      await waitFor(() => expect(result.current.error).not.toBeNull());
      expect(result.current.data).toBeNull();
    });

    it('reports an error on an HTTP failure', async () => {
      server.use(
        http.get('/api/cache/stats', () =>
          HttpResponse.json({ detail: 'cache unavailable' }, { status: 503 })
        )
      );

      const { result } = renderHook(() => useCacheStats(), { wrapper: AllProviders });

      await waitFor(() => expect(result.current.error).not.toBeNull());
      expect(result.current.data).toBeNull();
    });

    it('exposes a stable refetch across renders', async () => {
      const { result, rerender } = renderHook(() => useCacheStats(), {
        wrapper: AllProviders,
      });

      await waitFor(() => expect(result.current.loading).toBe(false));
      const first = result.current.refetch;
      rerender();

      expect(result.current.refetch).toBe(first);
    });
  });

  describe('useCacheHealth', () => {
    it('resolves the bare payload and derives the health flags', async () => {
      const { result } = renderHook(() => useCacheHealth(), { wrapper: AllProviders });

      await waitFor(() => expect(result.current.loading).toBe(false));

      expect(result.current.data).toEqual(mockCacheHealth);
      expect(result.current.isHealthy).toBe(true);
      expect(result.current.healthStatus).toBe('healthy');
    });

    it('reports critical when the backend says unhealthy', async () => {
      server.use(
        http.get('/api/cache/health', () =>
          HttpResponse.json({ ...mockCacheHealth, healthy: false })
        )
      );

      const { result } = renderHook(() => useCacheHealth(), { wrapper: AllProviders });

      await waitFor(() => expect(result.current.loading).toBe(false));

      expect(result.current.isHealthy).toBe(false);
      expect(result.current.healthStatus).toBe('critical');
    });

    it('reports critical, not healthy, while the request is failing', async () => {
      // `healthy` defaults to false when data is null, so a failed request
      // must not read as a healthy cache.
      server.use(
        http.get('/api/cache/health', () => HttpResponse.json({ bad: 'shape' }))
      );

      const { result } = renderHook(() => useCacheHealth(), { wrapper: AllProviders });

      await waitFor(() => expect(result.current.error).not.toBeNull());

      expect(result.current.isHealthy).toBe(false);
      expect(result.current.healthStatus).toBe('critical');
    });
  });
});

// #4486: the hooks own the dashboards' auto-refresh, but the only component
// test that could have covered it mocks the hooks wholesale (and is skipped,
// #4264). Drive the real hooks against MSW and count requests instead.
//
// Deliberately no `waitFor` here: under vitest's fake timers Testing Library
// mis-detects Jest and calls a `jest.advanceTimersByTime` that does not exist.
// `advanceUntil` steps the fake clock itself, flushing React between steps.
describe('cache telemetry polling (#4486)', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  function countingHandler(path: string, body: unknown) {
    const hits = { count: 0 };
    server.use(
      http.get(path, () => {
        hits.count += 1;
        return HttpResponse.json(body);
      })
    );
    return hits;
  }

  async function advanceUntil(done: () => boolean, maxMs: number, stepMs = 25) {
    for (let elapsed = 0; elapsed < maxMs && !done(); elapsed += stepMs) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(stepMs);
      });
    }
  }

  async function advance(ms: number, stepMs = 25) {
    for (let elapsed = 0; elapsed < ms; elapsed += stepMs) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(stepMs);
      });
    }
  }

  it('useCacheStats refetches on CACHE_STATS_REFRESH_INTERVAL_MS and stops after unmount', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const hits = countingHandler('/api/cache/stats', mockCacheStats);

    const { result, unmount } = renderHook(() => useCacheStats(), { wrapper: AllProviders });
    await advanceUntil(() => result.current.data !== null, 1000);
    expect(result.current.data).not.toBeNull();
    const afterInitial = hits.count;

    await advanceUntil(() => hits.count > afterInitial, CACHE_STATS_REFRESH_INTERVAL_MS + 1000);
    expect(hits.count).toBeGreaterThan(afterInitial);

    unmount();
    const atUnmount = hits.count;
    await advance(CACHE_STATS_REFRESH_INTERVAL_MS * 3, 250);
    expect(hits.count).toBe(atUnmount);
  });

  it('useCacheHealth polls on the interval it is given, not the default', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const hits = countingHandler('/api/cache/health', mockCacheHealth);

    const { result, unmount } = renderHook(() => useCacheHealth(1000), { wrapper: AllProviders });
    await advanceUntil(() => result.current.data !== null, 1000);
    expect(result.current.data).not.toBeNull();
    const afterInitial = hits.count;

    // Well short of the 10 s default, so a refetch here can only come from
    // the custom interval being honoured.
    await advanceUntil(() => hits.count > afterInitial, 2000);
    expect(hits.count).toBeGreaterThan(afterInitial);

    unmount();
    const atUnmount = hits.count;
    await advance(5000, 250);
    expect(hits.count).toBe(atUnmount);
  });
});
