/**
 * Error Handling API Integration Tests — SKIPPED (#5119)
 *
 * These tests exercise `TestAPIComponent` / `TestConcurrentComponent` / `TestRetryComponent`, defined entirely within this file. No
 * production module is imported: the only non-test-infrastructure imports are
 * vitest, @testing-library, msw and react. The suite could therefore only ever
 * validate its own fixture, never regress with production code — deleting the
 * feature it names would leave every test in this file green.
 *
 * That is the same false-green defect #3935 fixed in
 * `src/tests/integration/streaming-audio/streaming-mse.test.tsx`, which was
 * skipped rather than deleted for the same reason. The fix was applied to that
 * one file and never swept across this directory (#5119).
 *
 * Real coverage for this area lives in `src/utils/__tests__/httpError` and the per-hook error-path specs.
 *
 * Kept (skipped, not deleted) as a starting point should these be rewritten to
 * render the production component/hook they claim to cover — see
 * `library-management.test.tsx` and `playlist-management.test.tsx` in this
 * directory for the pattern that does it correctly.
 */

import { useEffect, useState } from 'react';
import { describe, it, expect, afterEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { render } from '@/test/test-utils';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';

// Test component that makes API calls
const TestAPIComponent = ({ endpoint, onError }: { endpoint: string; onError?: (error: Error) => void }) => {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8765${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const json = await response.json();
      setData(json);
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [endpoint]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (data) return <div>Data: {JSON.stringify(data)}</div>;
  return <div>No data</div>;
};

// Test component for concurrent requests
const TestConcurrentComponent = () => {
  const [results, setResults] = useState<string[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  const makeConcurrentRequests = async () => {
    const endpoints = [
      '/api/player/state',
      '/api/library/tracks?limit=10',
      '/api/player/enhancement/status',
      '/api/albums?limit=10',
    ];

    try {
      const responses = await Promise.all(
        endpoints.map(endpoint =>
          fetch(`http://localhost:8765${endpoint}`)
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`${endpoint} failed`)))
        )
      );
      setResults(responses.map((_, i) => `Request ${i + 1} success`));
    } catch (err) {
      setErrors(prev => [...prev, (err as Error).message]);
    }
  };

  useEffect(() => {
    makeConcurrentRequests();
  }, []);

  return (
    <div>
      <div data-testid="results">{results.length} successful requests</div>
      <div data-testid="errors">{errors.length} failed requests</div>
    </div>
  );
};

// Test component for retry logic with rate limiting
const TestRetryComponent = () => {
  const [attempts, setAttempts] = useState(0);
  const [retryAfter, setRetryAfter] = useState<number | null>(null);
  const [status, setStatus] = useState<string>('idle');

  const fetchWithRetry = async () => {
    setAttempts(prev => prev + 1);
    setStatus('fetching');

    try {
      const response = await fetch('http://localhost:8765/api/test/rate-limit');

      if (response.status === 429) {
        const retryAfterValue = response.headers.get('Retry-After');
        setRetryAfter(retryAfterValue ? parseInt(retryAfterValue) : null);
        setStatus('rate-limited');
        return;
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setStatus('success');
    } catch (err) {
      setStatus('error');
    }
  };

  useEffect(() => {
    fetchWithRetry();
  }, []);

  return (
    <div>
      <div data-testid="attempts">Attempts: {attempts}</div>
      <div data-testid="status">Status: {status}</div>
      {retryAfter && <div data-testid="retry-after">Retry after: {retryAfter}s</div>}
    </div>
  );
};

describe.skip('Error Handling API Integration Tests', () => {
  // SKIPPED: Memory-intensive test (688 lines). Run separately with increased heap.
  // Reset handlers after each test
  afterEach(() => {
    server.resetHandlers();
  });

  // ==========================================
  // 1. HTTP Error Responses (6 tests)
  // ==========================================

  describe('HTTP Error Responses', () => {
    it('should handle 400 Bad Request error', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/bad-request', () => {
          return HttpResponse.json(
            { error: 'Bad Request', detail: 'Invalid parameters' },
            { status: 400 }
          );
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/bad-request" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: HTTP 400/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalledWith(expect.any(Error));
      });
    });

    it('should handle 401 Unauthorized error', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/unauthorized', () => {
          return HttpResponse.json(
            { error: 'Unauthorized', detail: 'Authentication required' },
            { status: 401 }
          );
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/unauthorized" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: HTTP 401/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle 403 Forbidden error', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/forbidden', () => {
          return HttpResponse.json(
            { error: 'Forbidden', detail: 'Access denied' },
            { status: 403 }
          );
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/forbidden" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: HTTP 403/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle 404 Not Found error', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/not-found', () => {
          return HttpResponse.json(
            { error: 'Not Found', detail: 'Resource does not exist' },
            { status: 404 }
          );
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/not-found" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: HTTP 404/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle 500 Internal Server Error', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/server-error', () => {
          return HttpResponse.json(
            { error: 'Internal Server Error', detail: 'An unexpected error occurred' },
            { status: 500 }
          );
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/server-error" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: HTTP 500/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle 503 Service Unavailable error', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/service-unavailable', () => {
          return HttpResponse.json(
            { error: 'Service Unavailable', detail: 'Server is temporarily unavailable' },
            { status: 503 }
          );
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/service-unavailable" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: HTTP 503/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });
  });

  // ==========================================
  // 2. Network Errors (4 tests)
  // ==========================================

  describe('Network Errors', () => {
    it('should handle network timeout', async () => {
      // Arrange — use request.signal so the timer is cancelled when the
      // request is aborted (prevents dangling 10s setTimeout, fixes #3064).
      server.use(
        http.get('http://localhost:8765/api/test/timeout', async ({ request }) => {
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(resolve, 10000);
            request.signal.addEventListener('abort', () => {
              clearTimeout(timer);
              reject(request.signal.reason);
            });
          });
          return HttpResponse.json({ data: 'too late' });
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/timeout" onError={onError} />);

      // Assert - Component should show loading state (since request hangs)
      await waitFor(() => {
        expect(screen.getByText(/Loading/i)).toBeInTheDocument();
      }, { timeout: 1000 });
    });

    it('should handle connection refused error', async () => {
      // Arrange - Use network error simulation
      server.use(
        http.get('http://localhost:8765/api/test/connection-refused', () => {
          return HttpResponse.error();
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/connection-refused" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error:/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle DNS resolution failure', async () => {
      // Arrange - Network error simulates DNS failure
      server.use(
        http.get('http://localhost:8765/api/test/dns-failure', () => {
          return HttpResponse.error();
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/dns-failure" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error:/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle aborted request', async () => {
      // Arrange
      const TestAbortComponent = () => {
        const [error, setError] = useState<string | null>(null);

        useEffect(() => {
          const controller = new AbortController();

          fetch('http://localhost:8765/api/player/state', { signal: controller.signal })
            .catch(err => setError(err.message));

          // Abort immediately
          controller.abort();
        }, []);

        return <div>{error ? `Error: ${error}` : 'No error'}</div>;
      };

      // Act
      render(<TestAbortComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error:/i)).toBeInTheDocument();
      });
    });
  });

  // ==========================================
  // 3. Malformed Responses (3 tests)
  // ==========================================

  describe('Malformed Responses', () => {
    it('should handle invalid JSON response', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/invalid-json', () => {
          return new HttpResponse('This is not JSON', {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
            },
          });
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/invalid-json" onError={onError} />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error:/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle missing required fields in response', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/missing-fields', () => {
          return HttpResponse.json({
            // Missing required 'tracks' field
            total: 100,
          });
        })
      );

      const TestMissingFieldsComponent = () => {
        const [error, setError] = useState<string | null>(null);

        useEffect(() => {
          fetch('http://localhost:8765/api/test/missing-fields')
            .then(r => r.json())
            .then(data => {
              if (!data.tracks) {
                setError('Missing required field: tracks');
              }
            })
            .catch(err => setError(err.message));
        }, []);

        return <div>{error ? `Error: ${error}` : 'No error'}</div>;
      };

      // Act
      render(<TestMissingFieldsComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: Missing required field/i)).toBeInTheDocument();
      });
    });

    it('should handle type mismatch in response data', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/type-mismatch', () => {
          return HttpResponse.json({
            tracks: 'should be array', // Wrong type
            total: 'should be number', // Wrong type
          });
        })
      );

      const TestTypeMismatchComponent = () => {
        const [error, setError] = useState<string | null>(null);

        useEffect(() => {
          fetch('http://localhost:8765/api/test/type-mismatch')
            .then(r => r.json())
            .then(data => {
              if (!Array.isArray(data.tracks)) {
                setError('Type mismatch: tracks should be array');
              } else if (typeof data.total !== 'number') {
                setError('Type mismatch: total should be number');
              }
            })
            .catch(err => setError(err.message));
        }, []);

        return <div>{error ? `Error: ${error}` : 'No error'}</div>;
      };

      // Act
      render(<TestTypeMismatchComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Error: Type mismatch/i)).toBeInTheDocument();
      });
    });
  });

  // ==========================================
  // 4. Missing Data Scenarios (3 tests)
  // ==========================================

  describe('Missing Data Scenarios', () => {
    it('should handle empty response body', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/empty-body', () => {
          return new HttpResponse(null, { status: 200 });
        })
      );

      const onError = vi.fn();

      // Act
      render(<TestAPIComponent endpoint="/api/test/empty-body" onError={onError} />);

      // Assert
      await waitFor(() => {
        // Empty body causes JSON parse error
        expect(screen.getByText(/Error:/i)).toBeInTheDocument();
        expect(onError).toHaveBeenCalled();
      });
    });

    it('should handle null data fields', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/null-fields', () => {
          return HttpResponse.json({
            tracks: null,
            total: null,
            has_more: null,
          });
        })
      );

      const TestNullFieldsComponent = () => {
        const [hasNulls, setHasNulls] = useState(false);

        useEffect(() => {
          // #3652: .catch so a malformed response (or MSW handler throwing)
          // surfaces as a real test failure instead of being swallowed.
          fetch('http://localhost:8765/api/test/null-fields')
            .then(r => r.json())
            .then(data => {
              if (data.tracks === null || data.total === null) {
                setHasNulls(true);
              }
            })
            .catch(err => {
              console.error('TestNullFieldsComponent fetch failed:', err);
            });
        }, []);

        return <div>{hasNulls ? 'Has null fields' : 'No null fields'}</div>;
      };

      // Act
      render(<TestNullFieldsComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Has null fields/i)).toBeInTheDocument();
      });
    });

    it('should handle undefined properties', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/undefined-props', () => {
          return HttpResponse.json({
            // Missing properties are undefined in JSON
            some_field: 'value',
          });
        })
      );

      const TestUndefinedPropsComponent = () => {
        const [hasUndefined, setHasUndefined] = useState(false);

        useEffect(() => {
          // #3652: .catch so failures surface instead of being swallowed.
          fetch('http://localhost:8765/api/test/undefined-props')
            .then(r => r.json())
            .then(data => {
              if (data.tracks === undefined) {
                setHasUndefined(true);
              }
            })
            .catch(err => {
              console.error('TestUndefinedPropsComponent fetch failed:', err);
            });
        }, []);

        return <div>{hasUndefined ? 'Has undefined properties' : 'No undefined'}</div>;
      };

      // Act
      render(<TestUndefinedPropsComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByText(/Has undefined properties/i)).toBeInTheDocument();
      });
    });
  });

  // ==========================================
  // 5. Concurrent Requests (2 tests)
  // ==========================================

  describe('Concurrent Requests', () => {
    it('should handle multiple simultaneous requests successfully', async () => {
      // Arrange - Default handlers from server should work

      // Act
      render(<TestConcurrentComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByTestId('results')).toHaveTextContent('4 successful requests');
        expect(screen.getByTestId('errors')).toHaveTextContent('0 failed requests');
      }, { timeout: 3000 });
    });

    it('should handle race condition with partial failures', async () => {
      // Arrange - Make one endpoint fail
      server.use(
        http.get('http://localhost:8765/api/library/tracks', () => {
          return HttpResponse.json(
            { error: 'Server error' },
            { status: 500 }
          );
        })
      );

      // Act
      render(<TestConcurrentComponent />);

      // Assert
      await waitFor(() => {
        const errors = screen.getByTestId('errors');
        expect(errors).toHaveTextContent(/failed request/i);
      }, { timeout: 3000 });
    });
  });

  // ==========================================
  // 6. Rate Limiting (2 tests)
  // ==========================================

  describe('Rate Limiting', () => {
    it('should handle 429 Too Many Requests status', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/rate-limit', () => {
          return HttpResponse.json(
            { error: 'Too Many Requests', detail: 'Rate limit exceeded' },
            {
              status: 429,
              headers: {
                'Retry-After': '60',
              }
            }
          );
        })
      );

      // Act
      render(<TestRetryComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByTestId('status')).toHaveTextContent('rate-limited');
        expect(screen.getByTestId('attempts')).toHaveTextContent('1');
      });
    });

    it('should respect Retry-After header', async () => {
      // Arrange
      server.use(
        http.get('http://localhost:8765/api/test/rate-limit', () => {
          return HttpResponse.json(
            { error: 'Too Many Requests' },
            {
              status: 429,
              headers: {
                'Retry-After': '120',
              }
            }
          );
        })
      );

      // Act
      render(<TestRetryComponent />);

      // Assert
      await waitFor(() => {
        expect(screen.getByTestId('retry-after')).toHaveTextContent('120s');
      });
    });
  });
});
