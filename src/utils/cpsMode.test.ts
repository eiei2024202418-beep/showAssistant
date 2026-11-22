import test from 'node:test';
import assert from 'node:assert/strict';
import {
  normalizeCpsMode,
  getCpsModeConfig,
  getMaxCharsForMode,
  shouldUseLocalContraction,
  shouldTriggerPass1,
  shouldTriggerPass2,
  isCpsCompliantForMode,
  BALANCED_COMPRESSION_PRESERVE_RULE,
} from './cpsMode';

test('normalizeCpsMode defaults to balanced_natural (Level 2) for missing or invalid values', () => {
  assert.equal(normalizeCpsMode(undefined), 'balanced_natural');
  assert.equal(normalizeCpsMode(null), 'balanced_natural');
  assert.equal(normalizeCpsMode('unknown'), 'balanced_natural');
  assert.equal(normalizeCpsMode('strict_compact'), 'strict_compact');
  assert.equal(normalizeCpsMode('balanced_natural'), 'balanced_natural');
  assert.equal(normalizeCpsMode('natural_unlimited'), 'natural_unlimited');
});

test('Level 1 (strict_compact) preserves exact CPS < 10 behavior', () => {
  const cfg = getCpsModeConfig('strict_compact');
  assert.equal(cfg.initialTargetCps, 8.8);
  assert.equal(cfg.minCharFloor, 4);
  assert.equal(getMaxCharsForMode(0.4, 'strict_compact'), 4);
  assert.equal(getMaxCharsForMode(2.0, 'strict_compact'), 17);
  assert.equal(shouldUseLocalContraction('strict_compact'), true);
  assert.equal(shouldTriggerPass1(9.3, 'strict_compact'), false);
  assert.equal(shouldTriggerPass1(9.4, 'strict_compact'), true);
  assert.equal(shouldTriggerPass2(9.4, 'strict_compact'), false);
  assert.equal(shouldTriggerPass2(9.5, 'strict_compact'), true);
  assert.equal(isCpsCompliantForMode(10.0, 'strict_compact'), true);
  assert.equal(isCpsCompliantForMode(10.1, 'strict_compact'), false);
});

test('Level 2 (balanced_natural) enforces Target CPS 12.0 with floor 10, disables contraction & Pass 2, and triggers Pass 1 only above 14.5', () => {
  const cfg = getCpsModeConfig('balanced_natural');
  assert.equal(cfg.initialTargetCps, 12.0);
  assert.equal(cfg.minCharFloor, 10);

  // Short <1s lines get floor of 10 so natural sentences aren't clipped awkwardly
  assert.equal(getMaxCharsForMode(0.5, 'balanced_natural'), 10);
  assert.equal(getMaxCharsForMode(0.8, 'balanced_natural'), 10);
  assert.equal(getMaxCharsForMode(1.0, 'balanced_natural'), 12);
  assert.equal(getMaxCharsForMode(2.0, 'balanced_natural'), 24);

  // Regex contraction & strict Pass 2 must be completely disabled
  assert.equal(shouldUseLocalContraction('balanced_natural'), false);
  assert.equal(shouldTriggerPass2(18.0, 'balanced_natural'), false);

  // Pass 1 must leave 10-14.5 CPS untouched and trigger ONLY when cps > 14.5
  assert.equal(shouldTriggerPass1(12.0, 'balanced_natural'), false);
  assert.equal(shouldTriggerPass1(14.5, 'balanced_natural'), false);
  assert.equal(shouldTriggerPass1(14.6, 'balanced_natural'), true);

  // Viewer compliance checks <= 14.0
  assert.equal(isCpsCompliantForMode(14.0, 'balanced_natural'), true);
  assert.equal(isCpsCompliantForMode(14.1, 'balanced_natural'), false);

  assert.ok(BALANCED_COMPRESSION_PRESERVE_RULE.includes('ကိုယ်/မင်း'));
  assert.ok(BALANCED_COMPRESSION_PRESERVE_RULE.includes('ဘူး/တယ်/လား/ရှင့်'));
});

test('Level 3 (natural_unlimited) disables all CPS limits, contractions, and compression passes', () => {
  assert.equal(getMaxCharsForMode(1.5, 'natural_unlimited'), undefined);
  assert.equal(shouldUseLocalContraction('natural_unlimited'), false);
  assert.equal(shouldTriggerPass1(30.0, 'natural_unlimited'), false);
  assert.equal(shouldTriggerPass2(30.0, 'natural_unlimited'), false);
  assert.equal(isCpsCompliantForMode(25.0, 'natural_unlimited'), true);
});
