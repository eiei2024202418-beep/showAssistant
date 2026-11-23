import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseRetryDelayMs,
  classifyApiError,
  ModelRateLimiter,
} from './rateLimiter';

test('parseRetryDelayMs extracts retry seconds from Gemini 429 error payloads', () => {
  const err57 =
    '{"error":{"code":429,"message":"Quota exceeded... Please retry in 57.573842028s.","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"57s"}]}}';
  const delay57 = parseRetryDelayMs(err57);
  assert.ok(delay57 >= 58000 && delay57 <= 60000, `Expected ~59000ms, got ${delay57}`);

  const errJsonOnly = '{"details":[{"@type":"RetryInfo","retryDelay":"44s"}]}';
  const delay44 = parseRetryDelayMs(errJsonOnly);
  assert.equal(delay44, 45500);

  assert.equal(parseRetryDelayMs('Normal network error'), 0);
});

test('classifyApiError identifies Free Tier RPM 429, daily quota exhaustion, and 503 unavailable', () => {
  const rpmErr = classifyApiError(
    '{"error":{"code":429,"message":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier exceeded. Please retry in 44.2s","status":"RESOURCE_EXHAUSTED"}}'
  );
  assert.equal(rpmErr.isRateLimit, true);
  assert.equal(rpmErr.isDailyQuotaExhausted, false);
  assert.ok(rpmErr.retryDelayMs > 44000);

  const dailyErr = classifyApiError(
    '429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel limit 20 exceeded'
  );
  assert.equal(dailyErr.isRateLimit, true);
  assert.equal(dailyErr.isDailyQuotaExhausted, true);

  const unavailableErr = classifyApiError(
    '{"error":{"code":503,"message":"This model is currently experiencing high demand. Spikes in demand are usually temporary.","status":"UNAVAILABLE"}}'
  );
  assert.equal(unavailableErr.isRateLimit, false);
  assert.equal(unavailableErr.isTransientUnavailable, true);
});

test('ModelRateLimiter enforces cooldown and daily exhaustion tracking', async () => {
  const limiter = new ModelRateLimiter({ 'test-model': 50 });

  limiter.recordError('test-model', '429 RESOURCE_EXHAUSTED retry in 0.1s');
  assert.ok(limiter.getRemainingCooldownMs('test-model') > 0);
  assert.equal(limiter.isDailyExhausted('test-model'), false);

  limiter.recordError('daily-model', '429 RequestsPerDay exceeded');
  assert.equal(limiter.isDailyExhausted('daily-model'), true);

  // Escalating 503 high-demand cooldowns (5s -> 10s -> 15s) and reset on recordSuccess
  limiter.recordError('spike-model', '503 UNAVAILABLE high demand');
  const firstWait = limiter.getRemainingCooldownMs('spike-model');
  assert.ok(firstWait > 3500 && firstWait <= 5000);

  limiter.recordError('spike-model', '503 UNAVAILABLE high demand');
  const secondWait = limiter.getRemainingCooldownMs('spike-model');
  assert.ok(secondWait > 8000 && secondWait <= 10000);

  limiter.recordSuccess('spike-model');
  limiter.recordError('spike-model', '503 UNAVAILABLE high demand');
  // Since cooldownUntil is still max'd until time passes, verify fresh model after success resets streak
  const limiter2 = new ModelRateLimiter({ 'fresh-model': 50 });
  limiter2.recordError('fresh-model', '503 UNAVAILABLE');
  limiter2.recordSuccess('fresh-model');
  limiter2.recordError('fresh-model', '503 UNAVAILABLE');
  const resetWait = limiter2.getRemainingCooldownMs('fresh-model');
  assert.ok(resetWait > 3500 && resetWait <= 5000);

  const realDailyPayload =
    '{"error":{"code":429,"message":"You exceeded your current quota... Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, model: gemini-4-hydrogen, description: Generate content requests per day limit exceeded on Free Tier. Please retry in 8h20m58.577536482s.","status":"RESOURCE_EXHAUSTED"}}';
  const classifiedRealDaily = classifyApiError(realDailyPayload);
  assert.equal(classifiedRealDaily.isRateLimit, true);
  assert.equal(classifiedRealDaily.isDailyQuotaExhausted, true);

  limiter2.recordError('gemini-4-hydrogen', realDailyPayload);
  assert.equal(limiter2.isDailyExhausted('gemini-4-hydrogen'), true);
  assert.equal(limiter2.getLastDailyQuotaError(), realDailyPayload);
});
