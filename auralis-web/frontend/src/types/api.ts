/**
 * REST API Error Types
 *
 * #4398 deleted 33 request/response interfaces (player/library/metadata/
 * enhancement/playlist/artwork/search/fingerprint/similarity/health/cache/
 * streaming) plus `buildQueryParams` and `QueryParamValue` from this file —
 * every one had zero runtime consumers (`services/*.ts` use ad-hoc inline
 * shapes instead; several were already partially cleaned up in #4372/#3894/
 * #4674, but this was the block those passes left behind). What survives is
 * the one part of the original "REST API Types" contract that real code
 * actually imports: the generic error shape and its helpers, used by
 * useRestAPI.ts and the hooks/components built on it.
 */

export interface ApiError {
  status: number;
  message: string;
  /**
   * Optional domain-specific error code (e.g. 'QUEUE_SET_ERROR', 'PLAY_ERROR').
   * Callers use this to disambiguate which user-facing flow failed.
   */
  code?: string;
  details?: Record<string, unknown>;
}

/** Type guard for ApiError — checks structural shape since ApiError is an interface */
export function isApiError(err: unknown): err is ApiError {
  return (
    typeof err === 'object' &&
    err !== null &&
    'status' in err &&
    'message' in err &&
    typeof (err as ApiError).status === 'number' &&
    typeof (err as ApiError).message === 'string'
  );
}

export class ApiErrorHandler {
  static parse(error: unknown): ApiError {
    if (error instanceof Error) {
      // useRestAPI attaches the real status as `.status` so the message can be
      // the backend's `detail` text instead of 'HTTP nnn: ...' (#4831).
      // `utils/apiRequest.ts`'s APIRequestError carries the identical
      // information under `.statusCode` instead (#4643) — a naming mismatch
      // between the app's two HTTP transports that meant this method quietly
      // fell through to the 500 default for every APIRequestError, regardless
      // of the real status, until this branch checked for it explicitly.
      // Fall back to scraping the legacy generic message for any thrower that
      // uses neither, so callers keep seeing the actual code rather than
      // always 500 (#2361).
      const attached = (error as { status?: unknown; statusCode?: unknown }).status
        ?? (error as { statusCode?: unknown }).statusCode;
      if (typeof attached === 'number') {
        return { status: attached, message: error.message };
      }
      const httpMatch = error.message.match(/^HTTP (\d{3}):/);
      return {
        status: httpMatch ? parseInt(httpMatch[1], 10) : 500,
        message: error.message,
      };
    }

    if (isApiError(error)) {
      return error;
    }

    return {
      status: 500,
      message: 'Unknown error',
      details: { raw: String(error) },
    };
  }

  /**
   * #3594: like parse() but lets the caller stamp on a domain-specific
   * `code` so existing per-hook code-strings stay observable.
   */
  static parseWithCode(error: unknown, code: string): ApiError {
    return { ...ApiErrorHandler.parse(error), code };
  }

  static isNetworkError(error: ApiError): boolean {
    return error.status === 0 || error.status >= 500;
  }

  static isClientError(error: ApiError): boolean {
    return error.status >= 400 && error.status < 500;
  }

  static isNotFound(error: ApiError): boolean {
    return error.status === 404;
  }

  static isUnauthorized(error: ApiError): boolean {
    return error.status === 401 || error.status === 403;
  }

  static isValidationError(error: ApiError): boolean {
    return error.status === 422;
  }
}
