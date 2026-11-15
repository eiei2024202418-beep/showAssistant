import test from 'node:test';
import assert from 'node:assert/strict';
import { executeBatchWithAutoSplit, shouldAutoSplitOnBatchError } from './batchResilience';
import { SafetyBlockError } from './safetyStepDown';

test('shouldAutoSplitOnBatchError allows split on timeout/overload for large batches and blocks split on daily quota or safety block', () => {
  assert.equal(shouldAutoSplitOnBatchError(new Error('Model gemini-3.1-flash-lite timed out after 45s'), 28), true);
  assert.equal(shouldAutoSplitOnBatchError(new Error('503 UNAVAILABLE'), 16), true);
  assert.equal(shouldAutoSplitOnBatchError(new Error('Model omitted index 5'), 20), true);

  // Too small to split below minSplitSize * 2 (default 16)
  assert.equal(shouldAutoSplitOnBatchError(new Error('timed out'), 10), false);

  // Never split on Daily Quota Exhausted or SafetyBlockError
  assert.equal(
    shouldAutoSplitOnBatchError(new Error('429 GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded'), 28),
    false
  );
  assert.equal(shouldAutoSplitOnBatchError(new SafetyBlockError('Blocked', 'SAFETY', 3), 28), false);
});

test('executeBatchWithAutoSplit splits timed-out 28-line batch into 14+14 halves and succeeds', async () => {
  const input = Array.from({ length: 28 }, (_, i) => i + 1);
  const callSizes: number[] = [];
  const splitLogs: string[] = [];

  const result = await executeBatchWithAutoSplit<number, number>({
    batch: input,
    minSplitSize: 8,
    onSplit: (orig, f, s) => splitLogs.push(`${orig}->${f}+${s}`),
    translateFn: async (sub) => {
      callSizes.push(sub.length);
      if (sub.length > 14) {
        throw new Error('Model gemini-3.1-flash-lite timed out after 90s');
      }
      return sub.map((n) => n * 10);
    },
  });

  assert.deepEqual(callSizes, [28, 14, 14]);
  assert.deepEqual(splitLogs, ['28->14+14']);
  assert.equal(result.length, 28);
  assert.equal(result[0], 10);
  assert.equal(result[27], 280);
});

test('executeBatchWithAutoSplit immediately propagates daily quota error without splitting', async () => {
  const input = Array.from({ length: 28 }, (_, i) => i + 1);
  let calls = 0;

  await assert.rejects(
    () =>
      executeBatchWithAutoSplit<number, number>({
        batch: input,
        translateFn: async () => {
          calls++;
          throw new Error('429 RESOURCE_EXHAUSTED RequestsPerDay exceeded');
        },
      }),
    /RequestsPerDay/
  );

  assert.equal(calls, 1);
});
