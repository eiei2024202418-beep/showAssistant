/**
 * Rate-limit, retry-delay parsing, and per-model request pacing utilities
 * for Gemini API calls on Free Tier / High-Demand models.
 */

export interface ApiErrorClassification {
  isRateLimit: boolean;
  isDailyQuotaExhausted: boolean;
  isTransientUnavailable: boolean;
  retryDelayMs: number;
}

/**
 * Extracts Google API's requested backoff duration from error messages such as:
 * - "Please retry in 57.573842028s."
 * - `"retryDelay":"44s"`
 */
export function parseRetryDelayMs(errMsg: string): number {
  if (!errMsg) return 0;

  const retryInMatch = errMsg.match(/retry in\s+([\d.]+)\s*s/i);
  if (retryInMatch) {
    const secs = parseFloat(retryInMatch[1]);
    if (Number.isFinite(secs) && secs > 0) {
      return Math.min(120_000, Math.ceil(secs * 1000) + 1500);
    }
  }

  const retryDelayMatch = errMsg.match(/"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"/i);
  if (retryDelayMatch) {
    const secs = parseFloat(retryDelayMatch[1]);
    if (Number.isFinite(secs) && secs > 0) {
      return Math.min(120_000, Math.ceil(secs * 1000) + 1500);
    }
  }

  return 0;
}

/**
 * Classifies Gemini API errors into RPM rate limits, daily quota exhaustion,
 * or temporary 503/timeout/unavailable spikes.
 */
export function classifyApiError(errMsg: string): ApiErrorClassification {
  const isRateLimit = /429|RESOURCE_EXHAUSTED|rate[- ]?limit|quota/i.test(errMsg);
  const isDailyQuotaExhausted = /RequestsPerDay|PerDay|daily/i.test(errMsg);
  const isTransientUnavailable = /503|UNAVAILABLE|high demand|overloaded|502|504|timed out|ECONNRESET|ETIMEDOUT/i.test(
    errMsg
  );
  const retryDelayMs = parseRetryDelayMs(errMsg);

  return {
    isRateLimit,
    isDailyQuotaExhausted,
    isTransientUnavailable,
    retryDelayMs,
  };
}

const DEFAULT_MIN_INTERVAL_MS: Record<string, number> = {
  'gemini-3.1-flash-lite': 4500, // ~13.3 RPM max, safely under Free Tier 15 RPM cap
  'gemini-3.8-flash': 3500,      // ~17 RPM max, reduces 503 / 429 bursts
  'gemini-4-hydrogen': 1200,     // smooth spacing between concurrent batches
};

/**
 * Serializes slot acquisition per model so concurrent workers respect minimum intervals
 * and shared 429 cooldown timers without stampeding the same rate-limited model.
 */
export class ModelRateLimiter {
  private cooldownUntil = new Map<string, number>();
  private dailyExhausted = new Set<string>();
  private lastScheduledAt = new Map<string, number>();
  private modelMutex = new Map<string, Promise<void>>();
  private customIntervals: Record<string, number>;

  constructor(customIntervals: Record<string, number> = {}) {
    this.customIntervals = { ...DEFAULT_MIN_INTERVAL_MS, ...customIntervals };
  }

  isDailyExhausted(model: string): boolean {
    return this.dailyExhausted.has(model);
  }

  getRemainingCooldownMs(model: string): number {
    const until = this.cooldownUntil.get(model) ?? 0;
    return Math.max(0, until - Date.now());
  }

  recordError(model: string, errMsg: string): ApiErrorClassification {
    const classified = classifyApiError(errMsg);
    const now = Date.now();

    if (classified.isDailyQuotaExhausted) {
      this.dailyExhausted.add(model);
      this.cooldownUntil.set(model, Math.max(this.cooldownUntil.get(model) ?? 0, now + 600_000));
    } else if (classified.isRateLimit) {
      const waitMs = classified.retryDelayMs > 0 ? classified.retryDelayMs : 15_000;
      this.cooldownUntil.set(model, Math.max(this.cooldownUntil.get(model) ?? 0, now + waitMs));
    } else if (classified.isTransientUnavailable) {
      this.cooldownUntil.set(model, Math.max(this.cooldownUntil.get(model) ?? 0, now + 6000));
    }

    return classified;
  }

  async acquireSlot(
    model: string,
    onWait?: (waitMs: number, reason: string) => void
  ): Promise<void> {
    const minInterval = this.customIntervals[model] ?? 1200;

    // Chain onto per-model mutex so concurrent workers get strictly serialized slots
    const prevLock = this.modelMutex.get(model) ?? Promise.resolve();
    let releaseLock!: () => void;
    const nextLock = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });
    this.modelMutex.set(model, nextLock);

    await prevLock;

    try {
      const now = Date.now();
      const cooldownTarget = this.cooldownUntil.get(model) ?? 0;
      const intervalTarget = (this.lastScheduledAt.get(model) ?? 0) + minInterval;
      const targetTime = Math.max(now, cooldownTarget, intervalTarget);
      const waitMs = targetTime - now;

      if (waitMs > 0) {
        if (cooldownTarget > now && cooldownTarget >= intervalTarget) {
          onWait?.(waitMs, `Rate-limit cooldown on ${model}`);
        } else if (waitMs >= 2000) {
          onWait?.(waitMs, `Pacing requests on ${model}`);
        }
        await new Promise((resolve) => setTimeout(resolve, waitMs));
      }

      this.lastScheduledAt.set(model, Date.now());
    } finally {
      releaseLock();
    }
  }
}
