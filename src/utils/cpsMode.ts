/**
 * Dialogue CPS / Pacing Mode configuration and pure rules for Anime Subtitle Translator.
 * Supports 3 selectable pacing modes:
 * - Level 1 ('strict_compact'): Strict CPS < 10 (target 8.8, floor 4, local contraction, Pass 1 >= 9.4, Strict Pass 2)
 * - Level 2 ('balanced_natural', DEFAULT): Balanced Natural CPS ~12-14 (target 12.0, floor 10, no regex contraction, Pass 1 > 14.5 only, no Pass 2)
 * - Level 3 ('natural_unlimited'): No CPS limit, no contraction, no compression passes
 */

export type DialogueCpsMode = 'strict_compact' | 'balanced_natural' | 'natural_unlimited';

export interface CpsModeConfig {
  mode: DialogueCpsMode;
  shortLabel: string;
  optionLabel: string;
  targetBadge: string;
  initialTargetCps: number | null;
  minCharFloor: number;
  useLocalContraction: boolean;
  /** Minimum CPS to trigger Global Pass 1 compression (null = disabled). */
  pass1TriggerCps: number | null;
  /** Whether Pass 1 trigger comparison is strictly greater (>) vs >=. */
  pass1StrictGreater: boolean;
  useStrictPass2: boolean;
  pass2TriggerCps: number | null;
  viewerComplianceLimit: number | null;
  viewerBadgeGreenLimit: number | null;
}

export const CPS_MODE_CONFIGS: Record<DialogueCpsMode, CpsModeConfig> = {
  strict_compact: {
    mode: 'strict_compact',
    shortLabel: 'CPS < 10',
    optionLabel: 'Level 1: Strict Compact (CPS < 10)',
    targetBadge: 'Target CPS < 10',
    initialTargetCps: 8.8,
    minCharFloor: 4,
    useLocalContraction: true,
    pass1TriggerCps: 9.4,
    pass1StrictGreater: false,
    useStrictPass2: true,
    pass2TriggerCps: 9.5,
    viewerComplianceLimit: 10.0,
    viewerBadgeGreenLimit: 10.4,
  },
  balanced_natural: {
    mode: 'balanced_natural',
    shortLabel: 'CPS ~12–14',
    optionLabel: 'Level 2: Balanced Natural (Recommended)',
    targetBadge: 'Target CPS ~12–14',
    initialTargetCps: 12.0,
    minCharFloor: 10,
    useLocalContraction: false,
    pass1TriggerCps: 14.5,
    pass1StrictGreater: true,
    useStrictPass2: false,
    pass2TriggerCps: null,
    viewerComplianceLimit: 14.0,
    viewerBadgeGreenLimit: 14.4,
  },
  natural_unlimited: {
    mode: 'natural_unlimited',
    shortLabel: 'Unlimited',
    optionLabel: 'Level 3: Natural Unlimited (No Limit)',
    targetBadge: 'No CPS Limit',
    initialTargetCps: null,
    minCharFloor: 0,
    useLocalContraction: false,
    pass1TriggerCps: null,
    pass1StrictGreater: false,
    useStrictPass2: false,
    pass2TriggerCps: null,
    viewerComplianceLimit: null,
    viewerBadgeGreenLimit: null,
  },
};

export const BALANCED_COMPRESSION_PRESERVE_RULE =
  'မူရင်း ဇာတ်ကောင်စရိုက်နဲ့ နာမ်စား (ငါ/နင်, ငါ/မင်း, ကိုယ်/မင်း, ကျွန်တော်/ကျွန်မ/ရှင်) နဲ့ ဝါကျအဆုံးသတ်/ခံစားချက် (ကွ/ဟေ့/ဘူး/တယ်/လား/ရှင့်) တွေကို မပြောင်းလဲဘဲ ခေတ်မီလူငယ်စကားပြောအတိုင်း မလိုအပ်တဲ့ စကားပိုတွေကိုပဲ ကျစ်လျစ်အောင် ချုံ့ရမည်။';

export function normalizeCpsMode(raw: unknown): DialogueCpsMode {
  if (raw === 'strict_compact' || raw === 'balanced_natural' || raw === 'natural_unlimited') {
    return raw;
  }
  return 'balanced_natural';
}

export function getCpsModeConfig(mode: DialogueCpsMode): CpsModeConfig {
  return CPS_MODE_CONFIGS[normalizeCpsMode(mode)];
}

/**
 * Returns max visible Aegisub characters allowed for a dialogue line under the chosen CPS mode,
 * or `undefined` when CPS limiting is disabled ('natural_unlimited').
 */
export function getMaxCharsForMode(durationSec: number, mode: DialogueCpsMode): number | undefined {
  const cfg = getCpsModeConfig(mode);
  if (cfg.initialTargetCps === null) {
    return undefined;
  }
  const safeDuration = durationSec > 0 ? durationSec : 1.0;
  return Math.max(cfg.minCharFloor, Math.floor(safeDuration * cfg.initialTargetCps));
}

export function shouldUseLocalContraction(mode: DialogueCpsMode): boolean {
  return getCpsModeConfig(mode).useLocalContraction;
}

export function shouldTriggerPass1(cps: number, mode: DialogueCpsMode): boolean {
  const cfg = getCpsModeConfig(mode);
  if (cfg.pass1TriggerCps === null) return false;
  return cfg.pass1StrictGreater ? cps > cfg.pass1TriggerCps : cps >= cfg.pass1TriggerCps;
}

export function shouldTriggerPass2(cps: number, mode: DialogueCpsMode): boolean {
  const cfg = getCpsModeConfig(mode);
  if (!cfg.useStrictPass2 || cfg.pass2TriggerCps === null) return false;
  return cps >= cfg.pass2TriggerCps;
}

export function isCpsCompliantForMode(cps: number, mode: DialogueCpsMode): boolean {
  const cfg = getCpsModeConfig(mode);
  if (cfg.viewerComplianceLimit === null) return true;
  return cps <= cfg.viewerComplianceLimit;
}
