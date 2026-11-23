import { classifyApiError } from './rateLimiter.js';
import { SafetyBlockError } from './safetyStepDown.js';

export interface SplitBatchOptions<TItem, TResult> {
  batch: TItem[];
  translateFn: (subBatch: TItem[]) => Promise<TResult[]>;
  minSplitSize?: number;
  onSplit?: (originalLen: number, firstLen: number, secondLen: number, reason: string) => void;
}

/**
 * Returns true if the error is a transient overload/timeout/omission failure that benefits from smaller chunks.
 */
export function isTransientBatchFailure(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  const classified = classifyApiError(msg);
  if (classified.isDailyQuotaExhausted) return false;
  return (
    classified.isTransientUnavailable ||
    classified.isRateLimit ||
    /timed out|omitted index|empty or invalid array/i.test(msg)
  );
}

/**
 * When a large batch (>= minSplitSize * 2) fails across >= 2 distinct models due to timeouts or 503 overload,
 * fast-failing out of the batch retry loop lets executeBatchWithAutoSplit halve it immediately instead of stalling.
 */
export function shouldFastSplitLargeBatch(
  batchLength: number,
  failedDistinctModels: number,
  lastErr: unknown,
  minSplitSize = 5
): boolean {
  if (batchLength < minSplitSize * 2) return false;
  if (failedDistinctModels < 2) return false;
  return isTransientBatchFailure(lastErr);
}

/**
 * Determines whether a failed batch error is worth retrying by splitting the batch
 * into smaller halves (e.g. timeouts, 503 overload, truncated JSON, omitted index),
 * vs immediately propagating non-splittable errors (Daily Quota Exceeded or Safety Block).
 */
export function shouldAutoSplitOnBatchError(err: unknown, batchLength: number, minSplitSize = 8): boolean {
  if (batchLength < minSplitSize * 2) return false;
  if (err instanceof SafetyBlockError) return false;

  const msg = err instanceof Error ? err.message : String(err);
  const classified = classifyApiError(msg);
  if (classified.isDailyQuotaExhausted) return false;

  return true;
}

/**
 * Executes `translateFn` on a batch and, if a large batch fails due to timeout/overload/truncation,
 * automatically splits it into two balanced half-batches (up to `minSplitSize`) so dense anime
 * subtitle chunks never crash the whole translation stream.
 */
export async function executeBatchWithAutoSplit<TItem, TResult>({
  batch,
  translateFn,
  minSplitSize = 8,
  onSplit,
}: SplitBatchOptions<TItem, TResult>): Promise<TResult[]> {
  try {
    return await translateFn(batch);
  } catch (err) {
    if (!shouldAutoSplitOnBatchError(err, batch.length, minSplitSize)) {
      throw err;
    }

    const mid = Math.ceil(batch.length / 2);
    const firstHalf = batch.slice(0, mid);
    const secondHalf = batch.slice(mid);
    const reason = err instanceof Error ? err.message : String(err);

    onSplit?.(batch.length, firstHalf.length, secondHalf.length, reason);

    const firstResults = await executeBatchWithAutoSplit({
      batch: firstHalf,
      translateFn,
      minSplitSize,
      onSplit,
    });
    const secondResults = await executeBatchWithAutoSplit({
      batch: secondHalf,
      translateFn,
      minSplitSize,
      onSplit,
    });

    return [...firstResults, ...secondResults];
  }
}
