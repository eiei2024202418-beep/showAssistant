import test from 'node:test';
import assert from 'node:assert/strict';
import { HarmBlockThreshold } from '@google/genai';
import {
  SAFETY_STEP_DOWN_LEVELS,
  SafetyBlockError,
  buildSafetySettings,
  detectSafetyBlockFromResponse,
  executeWithSafetyStepDown,
} from './safetyStepDown';

test('SAFETY_STEP_DOWN_LEVELS starts at Block most and steps down cleanly to Block none', () => {
  assert.deepEqual(
    SAFETY_STEP_DOWN_LEVELS.map((l) => [l.label, l.threshold]),
    [
      ['Block most', HarmBlockThreshold.BLOCK_LOW_AND_ABOVE],
      ['Block some', HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE],
      ['Block few', HarmBlockThreshold.BLOCK_ONLY_HIGH],
      ['Block none', HarmBlockThreshold.BLOCK_NONE],
    ]
  );

  const settings = buildSafetySettings(HarmBlockThreshold.BLOCK_LOW_AND_ABOVE);
  assert.equal(settings.length, 4);
  assert.ok(settings.every((s) => s.threshold === HarmBlockThreshold.BLOCK_LOW_AND_ABOVE));
});

test('detectSafetyBlockFromResponse detects promptFeedback, candidate finishReason, and throwing text getter', () => {
  assert.equal(
    detectSafetyBlockFromResponse({ candidates: [{ finishReason: 'STOP' }], text: 'ok' }).blocked,
    false
  );

  const finishSafety = detectSafetyBlockFromResponse({ candidates: [{ finishReason: 'SAFETY' }] });
  assert.equal(finishSafety.blocked, true);
  assert.equal(finishSafety.source, 'finishReason');
  assert.equal(finishSafety.reason, 'SAFETY');

  const promptBlocked = detectSafetyBlockFromResponse({ promptFeedback: { blockReason: 'SAFETY' } });
  assert.equal(promptBlocked.blocked, true);
  assert.equal(promptBlocked.source, 'promptFeedback');

  const getterBlocked = detectSafetyBlockFromResponse({
    candidates: [{ finishReason: 'STOP' }],
    get text(): string {
      throw new Error('Response candidate was blocked due to SAFETY');
    },
  });
  assert.equal(getterBlocked.blocked, true);
  assert.equal(getterBlocked.source, 'textGetter');
});

test('executeWithSafetyStepDown succeeds on Block most without stepping down when clean', async () => {
  const attemptedThresholds: HarmBlockThreshold[] = [];
  const transitions: string[] = [];

  const result = await executeWithSafetyStepDown({
    delayMs: 0,
    onStepDown: (t) => transitions.push(`${t.fromLevel.label}->${t.toLevel.label}`),
    runAttempt: async (settings) => {
      attemptedThresholds.push(settings[0].threshold);
      return { candidates: [{ finishReason: 'STOP' }], text: '[{"index":0,"translated":"ဟုတ်"}]' };
    },
  });

  assert.deepEqual(attemptedThresholds, [HarmBlockThreshold.BLOCK_LOW_AND_ABOVE]);
  assert.deepEqual(transitions, []);
  assert.equal(result.usedLevel.label, 'Block most');
  assert.equal(result.didStepDown, false);
  assert.equal(result.stepIndex, 0);
});

test('executeWithSafetyStepDown automatically steps down from Block most -> Block some -> Block few on safety blocks', async () => {
  const attemptedThresholds: HarmBlockThreshold[] = [];
  const transitions: string[] = [];

  const result = await executeWithSafetyStepDown({
    delayMs: 0,
    onStepDown: (t) => transitions.push(`${t.fromLevel.label}->${t.toLevel.label}`),
    runAttempt: async (settings, level) => {
      attemptedThresholds.push(settings[0].threshold);
      if (level.id === 'block_most') {
        return { candidates: [{ finishReason: 'SAFETY' }] };
      }
      if (level.id === 'block_some') {
        return { promptFeedback: { blockReason: 'SAFETY' } };
      }
      return { candidates: [{ finishReason: 'STOP' }], text: '[{"index":0,"translated":"အဆင်ပြေတယ်"}]' };
    },
  });

  assert.deepEqual(attemptedThresholds, [
    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmBlockThreshold.BLOCK_ONLY_HIGH,
  ]);
  assert.deepEqual(transitions, ['Block most->Block some', 'Block some->Block few']);
  assert.equal(result.usedLevel.label, 'Block few');
  assert.equal(result.didStepDown, true);
  assert.equal(result.stepIndex, 2);
});

test('executeWithSafetyStepDown immediately rethrows non-safety errors (429/503/timeout) without stepping down', async () => {
  const attemptedThresholds: HarmBlockThreshold[] = [];

  await assert.rejects(
    () =>
      executeWithSafetyStepDown({
        delayMs: 0,
        runAttempt: async (settings) => {
          attemptedThresholds.push(settings[0].threshold);
          throw new Error('503 UNAVAILABLE high demand');
        },
      }),
    /503 UNAVAILABLE/
  );

  assert.deepEqual(attemptedThresholds, [HarmBlockThreshold.BLOCK_LOW_AND_ABOVE]);
});

test('executeWithSafetyStepDown throws SafetyBlockError after exhausting all 4 safety levels down to Block none', async () => {
  const attemptedThresholds: HarmBlockThreshold[] = [];

  await assert.rejects(
    () =>
      executeWithSafetyStepDown({
        delayMs: 0,
        runAttempt: async (settings) => {
          attemptedThresholds.push(settings[0].threshold);
          return { candidates: [{ finishReason: 'SAFETY' }] };
        },
      }),
    (err: unknown) => err instanceof SafetyBlockError && err.finalStepIndex === 3
  );

  assert.deepEqual(attemptedThresholds, [
    HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmBlockThreshold.BLOCK_NONE,
  ]);
});
