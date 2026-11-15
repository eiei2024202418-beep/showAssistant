import { HarmCategory, HarmBlockThreshold } from '@google/genai';

export interface SafetyLevel {
  id: 'block_most' | 'block_some' | 'block_few' | 'block_none';
  label: string;
  threshold: HarmBlockThreshold;
}

/**
 * Safety Auto Step-Down ladder:
 * Always starts at "Block most" (BLOCK_LOW_AND_ABOVE) and automatically relaxes
 * step-by-step only if Gemini blocks a response due to safety filtering.
 */
export const SAFETY_STEP_DOWN_LEVELS: readonly SafetyLevel[] = [
  {
    id: 'block_most',
    label: 'Block most',
    threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
  },
  {
    id: 'block_some',
    label: 'Block some',
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    id: 'block_few',
    label: 'Block few',
    threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
  },
  {
    id: 'block_none',
    label: 'Block none',
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
] as const;

export interface SafetySettingEntry {
  category: HarmCategory;
  threshold: HarmBlockThreshold;
}

export function buildSafetySettings(threshold: HarmBlockThreshold): SafetySettingEntry[] {
  return [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold },
  ];
}

export const SAFETY_SETTINGS_BLOCK_MOST: SafetySettingEntry[] = buildSafetySettings(
  HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
);

export interface SafetyBlockDetection {
  blocked: boolean;
  source?: 'promptFeedback' | 'finishReason' | 'textGetter';
  reason?: string;
}

const SAFETY_FINISH_REASONS = new Set([
  'SAFETY',
  'PROHIBITED_CONTENT',
  'BLOCKLIST',
  'SPII',
  'IMAGE_SAFETY',
]);

export interface MinimalGeminiResponse {
  promptFeedback?: {
    blockReason?: string;
    blockReasonMessage?: string;
  };
  candidates?: Array<{
    finishReason?: string;
    finishMessage?: string;
  }>;
  text?: string;
}

export class SafetyBlockError extends Error {
  readonly reason: string;
  readonly finalStepIndex: number;

  constructor(message: string, reason: string, finalStepIndex: number) {
    super(message);
    this.name = 'SafetyBlockError';
    this.reason = reason;
    this.finalStepIndex = finalStepIndex;
  }
}

export function isSafetyBlockErrorMessage(errMsg: string): boolean {
  if (!errMsg) return false;
  return /\b(SAFETY|PROHIBITED_CONTENT|BLOCKLIST|IMAGE_SAFETY)\b|blocked by.*safety|safety\s*filter|promptFeedback\.blockReason|candidates.*blocked/i.test(
    errMsg
  );
}

/**
 * Checks whether a Gemini generateContent response was blocked by safety filters
 * (either at the prompt level via promptFeedback.blockReason, at the candidate level via
 * finishReason === 'SAFETY' / 'PROHIBITED_CONTENT', or via a thrown response.text getter).
 */
export function detectSafetyBlockFromResponse(response: MinimalGeminiResponse | null | undefined): SafetyBlockDetection {
  if (!response) return { blocked: false };

  const blockReason = response.promptFeedback?.blockReason;
  if (blockReason && blockReason !== 'BLOCK_REASON_UNSPECIFIED') {
    return {
      blocked: true,
      source: 'promptFeedback',
      reason: String(blockReason),
    };
  }

  const finishReason = response.candidates?.[0]?.finishReason;
  if (finishReason && SAFETY_FINISH_REASONS.has(String(finishReason))) {
    return {
      blocked: true,
      source: 'finishReason',
      reason: String(finishReason),
    };
  }

  try {
    const _text = response.text;
    void _text;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (isSafetyBlockErrorMessage(msg)) {
      return {
        blocked: true,
        source: 'textGetter',
        reason: msg,
      };
    }
  }

  return { blocked: false };
}

export interface StepDownTransition {
  fromLevel: SafetyLevel;
  toLevel: SafetyLevel;
  reason: string;
  nextStepIndex: number;
}

export interface ExecuteWithSafetyStepDownOptions<TResponse extends MinimalGeminiResponse> {
  runAttempt: (
    safetySettings: SafetySettingEntry[],
    level: SafetyLevel,
    stepIndex: number
  ) => Promise<TResponse>;
  initialStepIndex?: number;
  delayMs?: number;
  onStepDown?: (transition: StepDownTransition) => void;
}

export interface SafetyStepDownResult<TResponse> {
  response: TResponse;
  usedLevel: SafetyLevel;
  stepIndex: number;
  didStepDown: boolean;
}

/**
 * Executes a Gemini generation call starting at "Block most" (or `initialStepIndex`),
 * automatically stepping down through Block some -> Block few -> Block none whenever
 * a safety block is encountered, while immediately rethrowing non-safety errors (429/503/timeouts).
 */
export async function executeWithSafetyStepDown<TResponse extends MinimalGeminiResponse>({
  runAttempt,
  initialStepIndex = 0,
  delayMs = 250,
  onStepDown,
}: ExecuteWithSafetyStepDownOptions<TResponse>): Promise<SafetyStepDownResult<TResponse>> {
  const startIdx = Math.max(0, Math.min(initialStepIndex, SAFETY_STEP_DOWN_LEVELS.length - 1));

  for (let stepIdx = startIdx; stepIdx < SAFETY_STEP_DOWN_LEVELS.length; stepIdx++) {
    const level = SAFETY_STEP_DOWN_LEVELS[stepIdx];
    const safetySettings = buildSafetySettings(level.threshold);

    let response: TResponse;
    let blockReason: string | undefined;

    try {
      response = await runAttempt(safetySettings, level, stepIdx);
      const detection = detectSafetyBlockFromResponse(response);
      if (!detection.blocked) {
        return {
          response,
          usedLevel: level,
          stepIndex: stepIdx,
          didStepDown: stepIdx > startIdx,
        };
      }
      blockReason = detection.reason || detection.source || 'SAFETY';
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (!isSafetyBlockErrorMessage(msg)) {
        throw err;
      }
      blockReason = msg;
    }

    if (stepIdx < SAFETY_STEP_DOWN_LEVELS.length - 1) {
      const nextLevel = SAFETY_STEP_DOWN_LEVELS[stepIdx + 1];
      onStepDown?.({
        fromLevel: level,
        toLevel: nextLevel,
        reason: blockReason || 'SAFETY',
        nextStepIndex: stepIdx + 1,
      });
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
      continue;
    }

    throw new SafetyBlockError(
      `Response blocked by safety settings even after Auto Step-Down to "${level.label}" (${blockReason || 'SAFETY'})`,
      blockReason || 'SAFETY',
      stepIdx
    );
  }

  throw new SafetyBlockError('Response blocked by safety settings', 'SAFETY', SAFETY_STEP_DOWN_LEVELS.length - 1);
}
