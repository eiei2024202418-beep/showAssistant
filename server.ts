import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import {
  executeWithSafetyStepDown,
  SafetyBlockError,
} from './src/utils/safetyStepDown.js';
import { executeBatchWithAutoSplit, shouldFastSplitLargeBatch } from './src/utils/batchResilience.js';
import {
  sanitizePreTranslatedMap,
  filterRemainingTranslatableEvents,
} from './src/utils/checkpoint.js';
import {
  parseAss,
  parseSrt,
  buildAss,
  validateAss,
  AssEvent,
  calculateCps,
  countVisibleGraphemes,
  getEventDuration,
  isSignOrTitle,
  findNearbyDuplicateClusters,
  unifyNearbyDuplicateTranslations,
  validateTranslatedLine,
  detectForbiddenScripts,
  extractForbiddenTokens,
  stripStrayForbiddenChars,
  buildWatermarkEvents,
  hasWatermarkEvents,
  mapStyleName,
} from './src/utils/subtitleParser.js';
import { reconcileTags, removeTrailingPeriods, fixCommonBurmeseSpelling } from './src/utils/tagProtector.js';
import { ModelRateLimiter } from './src/utils/rateLimiter.js';
import {
  buildEpisodeSample,
  sanitizeEpisodeLockProfile,
  formatEpisodeLockInstruction,
  formatLineSpeakerHint,
  detectSpeakerPronounViolation,
  applyDeterministicSpeakerPronounFix,
  enforceLockedGlossaryPostProcess,
  type EpisodeLockProfile,
} from './src/utils/episodeGlossary.js';
import {
  prepareSignTextForTranslation,
  extractLeadingTagBlock,
  type SignTypesettingMode,
  type DenseSignStrategy,
} from './src/utils/signTypesetting.js';
import {
  type DialogueCpsMode,
  normalizeCpsMode,
  getCpsModeConfig,
  getMaxCharsForMode,
  shouldUseLocalContraction,
  shouldTriggerPass1,
  shouldTriggerPass2,
  BALANCED_COMPRESSION_PRESERVE_RULE,
} from './src/utils/cpsMode.js';

import multer from 'multer';
import { EXACT_USER_PROMPT } from './src/utils/userPrompt.js';
import {
  verifySecretConstantTime,
  extractApiTokenFromHeaders,
  parseExternalTranslatePayload,
  parseAutopilotTranslateRequest,
  sanitizeSubtitleFileName,
  ExternalJobStore,
} from './src/utils/externalApi.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// Increase payload limits for large subtitle files
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Initialize GoogleGenAI client (Server-Side only)
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || '',
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    externalApiConfigured: Boolean(process.env.EXTERNAL_API_SECRET?.trim()),
  });
});

export interface BatchItemInput {
  index: number;
  actor: string;
  speakerLock?: string;
  style: string;
  durationSec: number;
  isDialogue: boolean;
  maxChars?: number;
  duplicateGroupId?: string;
  allowHighCps?: boolean;
  lineBreakCount?: number;
  text: string;
}

/**
 * Smart local contraction helper for spoken anime Burmese
 * to bring Aegisub CPS strictly under 10 if LLM returned slightly long phrasing.
 * Uses countVisibleGraphemes (matching Aegisub) instead of raw UTF-16 .length.
 */
function contractBurmeseForCps(text: string, maxChars: number): string {
  let res = text;
  // Common spoken contractions for fast anime reading
  const contractions: [RegExp, string][] = [
    [/ဒီဘက်ကို/g, 'ဒီကို'],
    [/ဒီဘက်မှာ/g, 'ဒီမှာ'],
    [/လာပါဦး/g, 'လာ'],
    [/လာပါ/g, 'လာ'],
    [/ချက်ချင်းပဲ/g, 'ချက်ချင်း'],
    [/သွားပါလား/g, 'သွားတယ်'],
    [/သွားကြစို့/g, 'သွားစို့'],
    [/နားလည်ပါပြီ/g, 'သိပြီ'],
    [/ကျေးဇူးတင်ပါတယ်/g, 'ကျေးဇူး'],
    [/စိတ်မပူပါနဲ့/g, 'မပူနဲ့'],
    [/မဟုတ်ပါဘူး/g, 'မဟုတ်ဘူး'],
    [/အဆင်ပြေပါတယ်/g, 'အဆင်ပြေတယ်'],
    [/သုံးနေတာလား/g, 'သုံးတာလား'],
    [/လုပ်နေတာလား/g, 'လုပ်တာလား'],
    [/ပြောနေတာလား/g, 'ပြောတာလား'],
    [/နေတာလား/g, 'တာလား'],
    [/တဲ့လား/g, 'လား'],
    [/နေတယ်/g, 'တယ်'],
    [/ခဲ့တာလား/g, 'လား'],
    [/ဖြစ်ပါသည်/g, 'ဖြစ်တယ်'],
    [/ပါလား/g, 'လား'],
    [/ပါတယ်/g, 'တယ်'],
    [/ပါဘူး/g, 'ဘူး'],
    [/ပါသည်/g, 'တယ်'],
    [/ပေးပါ/g, 'ပေး'],
    [/လိုက်ပါ/g, 'လိုက်'],
    [/တော့မယ်/g, 'မယ်'],
  ];

  for (const [pattern, replacement] of contractions) {
    if (countVisibleGraphemes(res) > maxChars) {
      res = res.replace(pattern, replacement);
    }
  }

  // Trim optional polite/filler particles if still exceeding target grapheme limit
  if (countVisibleGraphemes(res) > maxChars) {
    res = res.replace(/(\s*ပါလား|\s*ပါဦး|\s*ပါ|\s*နော်|\s*ကွာ|\s*ပေါ့|\s*လေ)(?=(\s*(\{[^}]*\}|\\N|$|\?)))/g, '');
  }

  return removeTrailingPeriods(res);
}

/**
 * Extracts episode-wide character pronoun pairs and key title/term glossary
 * from a representative episode sample before Batch 1 starts.
 */
async function extractEpisodeLockProfile(
  events: AssEvent[],
  fileName: string,
  sourceLang: string,
  preferredModel = 'gemini-3.8-flash',
  rateLimiter?: ModelRateLimiter,
  onStatus?: (msg: string) => void
): Promise<EpisodeLockProfile> {
  const empty: EpisodeLockProfile = { speakerProfiles: [], pronounLocks: [], glossaryTerms: [] };
  if (events.length < 2 || !process.env.GEMINI_API_KEY) {
    return empty;
  }

  const sample = buildEpisodeSample(events, fileName, 80);
  if (sample.sampleLines.length === 0) {
    return empty;
  }

  const modelChain =
    preferredModel === 'gemini-4-hydrogen'
      ? ['gemini-4-hydrogen', 'gemini-3.1-flash-lite', 'gemini-3.8-flash']
      : preferredModel === 'gemini-3.8-flash'
      ? ['gemini-3.1-flash-lite', 'gemini-3.8-flash']
      : ['gemini-3.1-flash-lite', 'gemini-3.8-flash'];

  const sourceHint =
    sourceLang === 'ja'
      ? 'Japanese (日本語)'
      : sourceLang === 'en'
      ? 'English'
      : 'Auto-detect (Japanese or English)';

  const systemInstruction = `You are a lead Burmese anime localization director preparing an Episode-Wide Consistency Sheet before subtitle translation begins.
Inspect the episode filename, speaker list ("actors"), detected proper phrases ("detectedProperPhrases"), and sample dialogue/sign lines (${sourceHint}) and extract ALL 3 sections strictly:

1. "speakerProfiles": MANDATORY Speaker-Gender Strict Pronoun Lock for EVERY recurring actor in "actors":
   - Specify exact "gender": "male" or "female" (only use "unknown" if genuinely impossible to tell).
   - Specify ONE SINGLE "fixedSelfPronoun" (e.g., "ကျွန်တော်" OR "ကျွန်ုပ်" OR "ကိုယ်" OR "ငါ" for males; "ကျွန်မ" OR "ငါ" for females — NEVER list multiple options with "/"!).
     * CRITICAL: A single character must NEVER switch between "ကျွန်ုပ်" and "ကျွန်တော်" across batches. Pick ONE and lock it!
     * CRITICAL: Male characters NEVER use "ကျွန်မ" / "ကျွန်မတို့". Female characters NEVER use "ကျွန်တော်" / "ကျွန်တော်တို့" / "ခင်ဗျား".
   - Specify exact matching "fixedPluralPronoun" (e.g. "ကျွန်တော်တို့", "ကျွန်ုပ်တို့", "ကျွန်မတို့", "ကိုယ်တို့", "ငါတို့").

2. "pronounLocks": Consistent speaker -> listener Burmese pronoun pairs so pronouns NEVER drift between batches:
   - Examples: "Dars" -> "Romelia": "ကျွန်တော် / ခင်ဗျား" or "ကျွန်ုပ် / အရှင်မ", "Romelia" -> "Dars": "ကျွန်မ / ရှင်".
   - Infer gender, age, social rank, and closeness accurately from dialogue tone and honorifics (-san, -sama, -kun, -chan).

3. "glossaryTerms": STRICT 1-FORM Place/Country, Rank/Title, and Episode Title Lock:
   - RULE A (Places, Kingdoms, Geographic Landmarks — "place"): Keep the Proper Root Name in English/Romaji EXACTLY AS SPELLED and translate ONLY the geographic suffix into ONE locked Burmese word!
     * NEVER phonetically transliterate proper place roots into Burmese (e.g. NEVER write "ဂီလီယာချောက်ကမ်းပါး"), and NEVER leave geographic suffixes in English (NEVER write "Guillier Gorge").
     * Examples: "Guillier Gorge" -> "Guillier ချောက်ကမ်းပါး", "Kingdom of Lionel" / "Lionel Kingdom" -> "Lionel နိုင်ငံတော်", "Dacan Plains" -> "Dacan လွင်ပြင်".
   - RULE B (Ranks, Noble & Military Titles — "rank"): Translate ALL ranks consistently into pure Burmese (NEVER mix English/Burmese ranks):
     * Examples: "General" -> "ဗိုလ်ချုပ်", "Crown Prince" -> "အိမ်ရှေ့မင်းသား", "His Majesty" -> "ဘုရင်မင်းမြတ်", "Commander" -> "တပ်မှူး", "Duke" -> "မြို့စားကြီး".
   - RULE C (Titles & Key Concepts — "title" / "term"): Lock episode titles and common nouns ("Firefly Wedding" -> "ပိုးစုန်းကြူးမင်္ဂလာပွဲ", "Heaven" -> "ကောင်းကင်ဘုံ", "Devil" -> "နတ်ဆိုး"). For Japanese Kanji unit/mecha/device codenames (e.g. "無号機" / "無號機"), translate or romanize into English/Burmese (e.g. "Unit 0", "Mugouki", or "နံပါတ်မဲ့ယူနစ်") — NEVER put Kanji/Kana/CJK inside "burmeseTerm"!
   - Return ONLY Myanmar Unicode (\\u1000-\\u109F) plus Basic Latin/digits for proper Romaji names. NEVER output Thai, Lao, Ethiopic, Japanese, or Chinese/CJK characters anywhere in "burmeseTerm" or "pronouns".`;

  const prompt = `Analyze this episode sample and build the strict episode consistency lock JSON:
${JSON.stringify(sample)}`;

  for (const modelToUse of modelChain) {
    if (rateLimiter?.isDailyExhausted(modelToUse)) continue;

    try {
      if (rateLimiter) {
        await rateLimiter.acquireSlot(modelToUse, (waitMs, reason) => {
          onStatus?.(`${reason} — waiting ${Math.ceil(waitMs / 1000)}s before episode character analysis...`);
        });
      }

      const { response } = await executeWithSafetyStepDown({
        onStepDown: ({ fromLevel, toLevel }) => {
          console.warn(`[glossary] Safety step-down (${modelToUse}): ${fromLevel.label} -> ${toLevel.label}`);
        },
        runAttempt: async (safetySettings) => {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`Episode lock extraction (${modelToUse}) timed out`)), 35000)
          );
          const generatePromise = ai.models.generateContent({
            model: modelToUse,
            contents: prompt,
            config: {
              systemInstruction,
              safetySettings,
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  speakerProfiles: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        actor: { type: Type.STRING },
                        gender: { type: Type.STRING },
                        fixedSelfPronoun: { type: Type.STRING },
                        fixedPluralPronoun: { type: Type.STRING },
                        toneNote: { type: Type.STRING },
                      },
                      required: ['actor', 'gender', 'fixedSelfPronoun', 'fixedPluralPronoun'],
                    },
                  },
                  pronounLocks: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        speaker: { type: Type.STRING },
                        listener: { type: Type.STRING },
                        pronouns: { type: Type.STRING },
                        toneNote: { type: Type.STRING },
                      },
                      required: ['speaker', 'listener', 'pronouns'],
                    },
                  },
                  glossaryTerms: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        sourceTerm: { type: Type.STRING },
                        burmeseTerm: { type: Type.STRING },
                        category: { type: Type.STRING },
                      },
                      required: ['sourceTerm', 'burmeseTerm'],
                    },
                  },
                },
                required: ['speakerProfiles', 'pronounLocks', 'glossaryTerms'],
              },
            },
          });
          return Promise.race([generatePromise, timeoutPromise]);
        },
      });
      rateLimiter?.recordSuccess(modelToUse);
      const parsed = JSON.parse(response.text || '{}');
      const sanitized = sanitizeEpisodeLockProfile(parsed);
      console.log(
        `[glossary] Extracted ${sanitized.speakerProfiles?.length ?? 0} speaker profile(s), ${sanitized.pronounLocks.length} pronoun lock(s) & ${sanitized.glossaryTerms.length} glossary term(s) via ${modelToUse}`
      );
      return sanitized;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      rateLimiter?.recordError(modelToUse, errMsg);
      console.warn(`[glossary] Episode lock extraction warning (${modelToUse}):`, errMsg);
    }
  }

  return empty;
}

/**
 * Fast LLM compression pass for lines exceeding the selected CPS mode threshold
 */
async function compressLinesToTargetCps(
  linesToCompress: { index: number; currentBurmese: string; currentGraphemes?: number; durationSec: number; maxChars: number }[],
  preferredModel = 'gemini-3.8-flash',
  strictSecondPass = false,
  rateLimiter?: ModelRateLimiter,
  onStatus?: (msg: string) => void,
  cpsMode: DialogueCpsMode = 'balanced_natural',
  episodeLockInstruction = ''
): Promise<Map<number, string>> {
  const resultMap = new Map<number, string>();
  if (linesToCompress.length === 0) return resultMap;

  const modelChain =
    preferredModel === 'gemini-3.1-flash-lite'
      ? ['gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-4-hydrogen']
      : preferredModel === 'gemini-4-hydrogen'
      ? ['gemini-4-hydrogen', 'gemini-3.8-flash', 'gemini-3.1-flash-lite']
      : ['gemini-3.8-flash', 'gemini-4-hydrogen', 'gemini-3.1-flash-lite'];

  const isBalancedMode = cpsMode === 'balanced_natural';

  const systemInstruction = (isBalancedMode
    ? `You are an expert Burmese anime subtitle editor for Aegisub (Balanced Natural Pacing Mode).
The following Burmese anime dialogue lines currently exceed 14.5 CPS (Characters Per Second) in Aegisub and need gentle, natural tightening.

CRITICAL AEGISUB CHARACTER COUNTING RULE:
In Aegisub, Burmese combining vowel/tone marks (ေ, ာ, ါ, း, etc.) count as SEPARATE characters!
- Example: "သုံးနေတာလား" = 8 Aegisub chars (NOT 4!). Each Burmese syllable averages ~2 Aegisub characters.

MANDATORY BALANCED NATURAL RULE (MUST OBEY):
- ${BALANCED_COMPRESSION_PRESERVE_RULE}
- Do NOT strip polite/respectful endings or character tone markers (e.g. "သခင်ကြီးရှင့်", "ပါတယ်", "ပါဘူး", "နော်", "လား", "ရှင့်").
- Gently tighten ONLY redundant filler clauses or overly wordy phrasing so its Aegisub character count is <= "maxChars" (targeting ~12–14 CPS) while keeping complete natural grammar and character emotion.
- Output ONLY pure Burmese Unicode (\\u1000-\\u109F) plus ASCII character names when needed. NEVER output Ethiopic (ድ), Thai, Lao, Japanese, CJK, Korean, Hindi, Arabic, or Cyrillic characters.
- Do NOT include ASS {\\...} tags (tags are preserved separately).
- NEVER end with a period (.).`
    : `You are an expert anime subtitle compression editor for Aegisub.
The following Burmese anime dialogue lines currently reach or exceed CPS 10 (Characters Per Second) in Aegisub.

CRITICAL AEGISUB CHARACTER COUNTING RULE:
In Aegisub, Burmese combining vowel/tone marks (ေ, ာ, ါ, း, etc.) count as SEPARATE characters!
- Example: "သုံးနေတာလား" = 8 Aegisub chars (NOT 4!). "လား" = 3 chars. "Skill သုံးနေတာလား" = 13 chars!
- Each Burmese syllable averages ~2 Aegisub characters.

YOUR TASK:
Condense each line into super-short, natural spoken anime Burmese so its Aegisub character count (excluding spaces and tags) is STRICTLY <= "maxChars" (guaranteeing CPS <= 9 in Aegisub, NEVER 10+).
- Keep core anime emotion and meaning, drop filler/pronouns/progressive markers ("နေ", "တဲ့", "ပါ", "တွေ", "တော့").
- ${strictSecondPass ? 'STRICT OVERRIDE: Even if an English skill/attack/item name makes the line exceed maxChars (e.g. "Blackened Flameblade" = 19 chars, or "Skill သုံးထား" = 10 chars when maxChars=8!), you MUST drop/translate the English word into short Burmese (e.g. "သုံးတာလား" = 7 chars, "ဓားနက်လား" = 8 chars) so total Aegisub chars is STRICTLY <= maxChars!' : 'Keep short character first-names in Romaji when they fit under maxChars; if an English skill/item phrase pushes total length over maxChars on a short <1.5s line, translate or shorten it into brief Burmese so total length <= maxChars.'}
- Output ONLY pure Burmese Unicode (\\u1000-\\u109F) plus ASCII character names when needed. NEVER output Ethiopic (ድ), Thai, Lao, Japanese, CJK, Korean, Hindi, Arabic, or Cyrillic characters.
- Do NOT include ASS {\\...} tags (tags are preserved separately).
- NEVER end with a period (.).`) + episodeLockInstruction;

  // Send clean text without ASS tags so maxChars constraint is purely visible text
  const cleanPayload = linesToCompress.map((l) => {
    const cleanText = l.currentBurmese.replace(/\{[^}]*\}/g, '').replace(/\\[Nnh]/g, ' ').trim();
    const currentAegisubChars = l.currentGraphemes ?? countVisibleGraphemes(cleanText);
    return {
      index: l.index,
      currentDialogue: cleanText,
      currentAegisubChars,
      maxChars: l.maxChars,
      mustRemoveAtLeastChars: Math.max(1, currentAegisubChars - l.maxChars),
      durationSec: l.durationSec,
    };
  });

  const prompt = isBalancedMode
    ? `Tighten ONLY redundant wording in each Burmese dialogue line below so its Aegisub char count is <= maxChars (Target ~12–14 CPS), WITHOUT breaking grammar or changing pronouns/sentence endings:\n⚠️ ${BALANCED_COMPRESSION_PRESERVE_RULE}\n${JSON.stringify(cleanPayload)}\n\nReturn a JSON array of objects with "index" (number) and "shortened" (natural Burmese string <= maxChars).`
    : `${strictSecondPass ? 'URGENT FINAL PASS — Previous version STILL exceeded maxChars! Drop English words/particles and write ultra-short Burmese (remember: ~2 Aegisub chars per syllable, so maxChars=8 means max 3-4 short Burmese syllables!) so EVERY line has Aegisub char count STRICTLY <= maxChars:\n' : 'Shorten each Burmese dialogue line below so its Aegisub char count (remember: each Burmese syllable ≈ 2 chars!) is STRICTLY <= maxChars (CPS < 10):\n'}${JSON.stringify(cleanPayload)}

Return a JSON array of objects with "index" (number) and "shortened" (concise Burmese string <= maxChars).`;

  for (const modelToUse of modelChain) {
    if (rateLimiter?.isDailyExhausted(modelToUse)) {
      console.log(`[compress] Skipping ${modelToUse} (daily quota exhausted)`);
      continue;
    }

    try {
      if (rateLimiter) {
        await rateLimiter.acquireSlot(modelToUse, (waitMs, reason) => {
          onStatus?.(`${reason} — waiting ${Math.ceil(waitMs / 1000)}s before CPS optimization...`);
        });
      }

      const baseConfig: Record<string, unknown> = {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              index: { type: Type.INTEGER },
              shortened: { type: Type.STRING },
            },
            required: ['index', 'shortened'],
          },
        },
      };

      if (modelToUse === 'gemini-4-hydrogen' || modelToUse === 'gemini-3.8-flash') {
        baseConfig.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
      }

      const { response } = await executeWithSafetyStepDown({
        onStepDown: ({ fromLevel, toLevel, reason }) => {
          console.warn(
            `[safety] Compression (${modelToUse}) blocked at "${fromLevel.label}" (${reason}) -> Auto Step-Down to "${toLevel.label}"`
          );
          onStatus?.(
            `Safety filter ("${fromLevel.label}") triggered during CPS optimization — Auto Step-Down to "${toLevel.label}" & retrying...`
          );
        },
        runAttempt: async (safetySettings) => {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`Compression (${modelToUse}) timed out`)), 90000)
          );
          const generatePromise = ai.models.generateContent({
            model: modelToUse,
            contents: prompt,
            config: {
              ...baseConfig,
              safetySettings,
            },
          });
          return Promise.race([generatePromise, timeoutPromise]);
        },
      });
      rateLimiter?.recordSuccess(modelToUse);
      const parsed: { index: number; shortened: string }[] = JSON.parse(response.text || '[]');

      const origBurmeseByIndex = new Map(linesToCompress.map((l) => [l.index, l.currentBurmese]));
      for (const item of parsed) {
        if (typeof item.index === 'number' && typeof item.shortened === 'string') {
          const cleaned = removeTrailingPeriods(item.shortened).trim();
          const prevText = origBurmeseByIndex.get(item.index) || '';
          if (cleaned.length > 0 && validateTranslatedLine(prevText, cleaned).valid) {
            resultMap.set(item.index, cleaned);
          } else {
            console.warn(`[compress] Rejected invalid compressed output for index ${item.index}: "${cleaned}"`);
          }
        }
      }
      if (resultMap.size > 0) break;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      rateLimiter?.recordError(modelToUse, errMsg);
      console.warn(`[compress] LLM compression warning (${modelToUse}):`, errMsg);
    }
  }

  return resultMap;
}

interface InvalidLineRepairInput {
  index: number;
  actor?: string;
  speakerLock?: string;
  originalText: string;
  badOutput: string;
  reason: string;
  forbiddenScripts?: string[];
  isDialogue: boolean;
  maxChars?: number;
}

/**
 * Surgical last-resort sanitizer ONLY for lines that already contain valid Burmese dialogue
 * plus stray foreign tokens (e.g. CJK/Kanji `無號機`, Hiragana/Katakana, Thai `ค`, Lao `ະ`, Ethiopic `ድ`)
 * after multi-line + solo LLM repair retries fail.
 * Never strips lines that lack Burmese because completely untranslated source lines must not be deleted.
 */
function stripStrayNonBurmeseTokensIfBurmesePresent(text: string): string {
  const hasBurmese = /[\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF]/.test(text);
  if (!hasBurmese) return text;

  return stripStrayForbiddenChars(text);
}

/**
 * Targeted per-line repair pass for any lines that failed script/translation/speaker-pronoun validation
 * (e.g., untranslated Japanese blocks, Ethiopic `ድ` / Thai `ค` / Lao `ະ` script glitches,
 * male actor using `ကျွန်မ`, female actor using `ခင်ဗျား`/`ကျွန်တော်`, or accidentally blanked lines).
 */
async function repairInvalidLines(
  invalidLines: InvalidLineRepairInput[],
  sourceLang: string,
  preferredModel = 'gemini-3.8-flash',
  rateLimiter?: ModelRateLimiter,
  onStatus?: (msg: string) => void,
  episodeLockInstruction = '',
  episodeLock?: EpisodeLockProfile
): Promise<Map<number, string>> {
  const repairedMap = new Map<number, string>();
  if (invalidLines.length === 0) return repairedMap;

  const sourceHint =
    sourceLang === 'ja'
      ? 'Japanese (日本語)'
      : sourceLang === 'en'
      ? 'English'
      : 'Auto-detect (Japanese or English)';

  const repairPlans: { model: string; thinking?: ThinkingLevel; timeoutMs: number }[] =
    preferredModel === 'gemini-3.1-flash-lite'
      ? [
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 75000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 75000 },
        ]
      : preferredModel === 'gemini-4-hydrogen'
      ? [
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 75000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 75000 },
        ]
      : [
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 75000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 75000 },
        ];

  let remaining = [...invalidLines];

  for (let round = 0; round < repairPlans.length && remaining.length > 0; round++) {
    const plan = repairPlans[round];
    if (rateLimiter?.isDailyExhausted(plan.model)) {
      continue;
    }

    const promptItems = remaining.map((item) => {
      const forbiddenTokens = extractForbiddenTokens(item.badOutput);
      return forbiddenTokens.length > 0 ? { ...item, forbiddenTokens } : item;
    });

    const prompt = `# Source Language: ${sourceHint}
# URGENT QUALITY & PRONOUN REPAIR PASS:
The following subtitle lines FAILED automated Burmese quality or Speaker-Gender Pronoun validation in the previous pass (untranslated Japanese/English, leftover Kanji/Hanzi terms like \`無號機\`, blank output, foreign Unicode glitch such as Ethiopic \`ድ\`, Thai \`ค\`, Lao \`ະ\`, OR wrong speaker gender pronoun such as a male actor using \`ကျွန်မ\`, female actor using \`ခင်ဗျား\`/\`ကျွန်တော်\`, or switching between \`ကျွန်ုပ်\` and \`ကျွန်တော်\`).

YOUR TASK:
Translate EACH line cleanly and accurately into natural spoken Burmese following the exact system prompt & Episode Lock rules:
1. NEVER copy or leave source Japanese/Kanji/Hanzi/Kana or ANY non-allowed Unicode scripts (CJK/Kanji \`無號機\`, Hiragana, Katakana, Ethiopic \`ድ\`, Arabic, Thai, Lao, Korean, Indic, Cyrillic) in "translated" — even if the source line has ASS tags like {\\i1}, quotation marks like "部長", or Kanji unit/mecha codenames (translate/romanize \`無号機\`/\`無號機\` as \`Unit 0\`, \`Mugouki\`, or \`နံပါတ်မဲ့ယူနစ်\` — NEVER output CJK/Kanji!).
2. If "forbiddenTokens" is present on an item, you MUST replace every listed foreign token with natural Burmese or English/Romaji so ZERO forbidden characters remain!
3. Output ONLY pure Myanmar Unicode (\\u1000-\\u109F) plus Basic Latin ASCII/digits & standard punctuation ONLY (use \`…\` for interrupted pauses instead of a single \`. \`; fix any \`{\\i0}{\\i1}\` into \`{\\i1}...{\\i0}\`; ALWAYS spell \`ကတိ\` instead of \`ဂတိ\`).
4. OBEY "speakerLock" STRICTLY: Male actors NEVER use \`ကျွန်မ/ကျွန်မတို့\`; Female actors NEVER use \`ခင်ဗျား/ကျွန်တော်/ကျွန်တော်တို့\`; NEVER switch self-pronoun between \`ကျွန်ုပ်\` and \`ကျွန်တော်\`.
5. Translate ALL common English nouns & ranks into Burmese (General -> ဗိုလ်ချုပ်, Crown Prince -> အိမ်ရှေ့မင်းသား, His Majesty -> ဘုရင်မင်းမြတ်, Heaven -> ကောင်းကင်ဘုံ, Boss -> သူဌေး, Devil -> နတ်ဆိုး) and keep proper place roots in English + Burmese suffix (e.g. Guillier ချောက်ကမ်းပါး, Lionel နိုင်ငံတော်, Dacan လွင်ပြင်).
6. Preserve all ASS tags {\\...} from "originalText". If "isDialogue" is true, output a SINGLE line without \\N line breaks; if "isDialogue" is false (Signs/RPG UI/Text/Title/Lyrics), translate ALL descriptive labels/stats/disclaimers into concise Burmese (only standard short game abbreviations like HP/MP/LV or proper names stay English) and preserve \\N line breaks intact. Never output timecodes or trailing periods (.).
7. If "isDialogue" is true and "maxChars" is provided, keep Aegisub visible char count <= maxChars. If "maxChars" is omitted, translate naturally without artificial character shortening.
8. Only return "" if "originalText" is purely a parenthetical SFX/groan note like （ため息）.

# LINES TO REPAIR:
${JSON.stringify(promptItems)}

Return a JSON array of objects with "index" (integer) and "translated" (clean Burmese string).`;

    try {
      if (rateLimiter) {
        await rateLimiter.acquireSlot(plan.model, (waitMs, reason) => {
          onStatus?.(`${reason} — waiting ${Math.ceil(waitMs / 1000)}s before quality repair...`);
        });
      }

      const baseConfig: Record<string, unknown> = {
        systemInstruction: `${EXACT_USER_PROMPT}${episodeLockInstruction}`,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              index: { type: Type.INTEGER },
              translated: { type: Type.STRING },
            },
            required: ['index', 'translated'],
          },
        },
      };
      if (plan.thinking) {
        baseConfig.thinkingConfig = { thinkingLevel: plan.thinking };
      }

      console.log(`[repair] Round ${round + 1}/${repairPlans.length}: repairing ${remaining.length} invalid line(s) via ${plan.model} (thinking=${plan.thinking || 'default'})...`);
      const { response } = await executeWithSafetyStepDown({
        onStepDown: ({ fromLevel, toLevel, reason }) => {
          console.warn(
            `[safety] Repair (${plan.model}) blocked at "${fromLevel.label}" (${reason}) -> Auto Step-Down to "${toLevel.label}"`
          );
          onStatus?.(
            `Safety filter ("${fromLevel.label}") triggered during quality repair — Auto Step-Down to "${toLevel.label}" & retrying...`
          );
        },
        runAttempt: async (safetySettings) => {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`Repair (${plan.model}) timed out`)), plan.timeoutMs)
          );
          const generatePromise = ai.models.generateContent({
            model: plan.model,
            contents: prompt,
            config: {
              ...baseConfig,
              safetySettings,
            },
          });
          return Promise.race([generatePromise, timeoutPromise]);
        },
      });
      rateLimiter?.recordSuccess(plan.model);
      const parsed: { index: number; translated: string }[] = JSON.parse(response.text || '[]');
      const byIndex = new Map<number, string>();
      for (const item of parsed) {
        if (typeof item.index === 'number' && typeof item.translated === 'string') {
          byIndex.set(item.index, item.translated);
        }
      }

      const nextRemaining: InvalidLineRepairInput[] = [];
      for (const reqItem of remaining) {
        const candidate = byIndex.get(reqItem.index);
        if (candidate !== undefined) {
          const check = validateTranslatedLine(reqItem.originalText, candidate);
          const spCheck = detectSpeakerPronounViolation(reqItem.actor || '', candidate, episodeLock);
          if (check.valid && spCheck.valid) {
            repairedMap.set(reqItem.index, candidate);
            continue;
          }
          nextRemaining.push({
            ...reqItem,
            badOutput: candidate,
            reason: !check.valid ? (check.reason || reqItem.reason) : (spCheck.reason || reqItem.reason),
            forbiddenScripts: check.forbiddenScripts,
          });
        } else {
          nextRemaining.push(reqItem);
        }
      }
      remaining = nextRemaining;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      rateLimiter?.recordError(plan.model, errMsg);
      console.warn(`[repair] Round ${round + 1} (${plan.model}) failed:`, errMsg);
    }
  }

  // Focused single-line solo repair for any stubborn lines that survived multi-line repair rounds
  if (remaining.length > 0 && remaining.length <= 15) {
    const soloRemaining: InvalidLineRepairInput[] = [];
    for (const item of remaining) {
      const forbiddenTokens = extractForbiddenTokens(item.badOutput);
      const soloModel = !rateLimiter?.isDailyExhausted('gemini-3.8-flash')
        ? 'gemini-3.8-flash'
        : !rateLimiter?.isDailyExhausted('gemini-4-hydrogen')
        ? 'gemini-4-hydrogen'
        : 'gemini-3.1-flash-lite';
      if (rateLimiter?.isDailyExhausted(soloModel)) {
        soloRemaining.push(item);
        continue;
      }

      const soloPrompt = `# SINGLE-LINE STRICT CLEANUP REPAIR (${sourceHint})
Translate THIS ONE subtitle line into natural spoken Burmese with ZERO foreign Unicode characters.
${forbiddenTokens.length > 0 ? `CRITICAL ERROR IN PREVIOUS ATTEMPT: You left foreign tokens ${JSON.stringify(forbiddenTokens)} inside "${item.badOutput}". Replace/translate/romanize EVERY character of ${JSON.stringify(forbiddenTokens)} into Burmese or English/Romaji (e.g. 無号機/無號機 -> Unit 0 or နံပါတ်မဲ့ယူနစ်). DO NOT output any Kanji, Hiragana, Katakana, Hanzi, Korean, Thai, Lao, Ethiopic, or non-Burmese/non-ASCII script!` : `Previous invalid output: "${item.badOutput}" (${item.reason}).`}

Input line:
${JSON.stringify({
  index: item.index,
  actor: item.actor || '',
  speakerLock: item.speakerLock,
  isDialogue: item.isDialogue,
  maxChars: item.maxChars,
  originalText: item.originalText,
})}

Return a JSON object with keys "index" (integer) and "translated" (clean Burmese string only).`;

      try {
        if (rateLimiter) {
          await rateLimiter.acquireSlot(soloModel);
        }
        const soloConfig: Record<string, unknown> = {
          systemInstruction: `${EXACT_USER_PROMPT}${episodeLockInstruction}`,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              index: { type: Type.INTEGER },
              translated: { type: Type.STRING },
            },
            required: ['index', 'translated'],
          },
        };
        if (soloModel === 'gemini-3.8-flash' || soloModel === 'gemini-4-hydrogen') {
          soloConfig.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
        }
        const { response } = await executeWithSafetyStepDown({
          runAttempt: async (safetySettings) => {
            const timeoutPromise = new Promise<never>((_, reject) =>
              setTimeout(() => reject(new Error(`Solo repair (${soloModel}) timed out`)), 45000)
            );
            const generatePromise = ai.models.generateContent({
              model: soloModel,
              contents: soloPrompt,
              config: { ...soloConfig, safetySettings },
            });
            return Promise.race([generatePromise, timeoutPromise]);
          },
        });
        rateLimiter?.recordSuccess(soloModel);
        const parsedSolo: { index?: number; translated?: string } = JSON.parse(response.text || '{}');
        const candidate = typeof parsedSolo.translated === 'string' ? parsedSolo.translated : undefined;
        if (
          candidate !== undefined &&
          validateTranslatedLine(item.originalText, candidate).valid &&
          detectSpeakerPronounViolation(item.actor || '', candidate, episodeLock).valid
        ) {
          repairedMap.set(item.index, candidate);
          continue;
        }
        soloRemaining.push(
          candidate !== undefined ? { ...item, badOutput: candidate } : item
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        rateLimiter?.recordError(soloModel, errMsg);
        soloRemaining.push(item);
      }
    }
    remaining = soloRemaining;
  }

  // Last-resort surgical scrub ONLY if a line already has valid Burmese and just a stray non-Burmese/Kanji token or pronoun violation
  for (const leftover of remaining) {
    let scrubbed = stripStrayNonBurmeseTokensIfBurmesePresent(leftover.badOutput);
    scrubbed = applyDeterministicSpeakerPronounFix(leftover.actor || '', scrubbed, episodeLock);
    if (
      scrubbed !== leftover.badOutput &&
      validateTranslatedLine(leftover.originalText, scrubbed).valid &&
      detectSpeakerPronounViolation(leftover.actor || '', scrubbed, episodeLock).valid
    ) {
      console.warn(`[repair] Scrubbed residual glitch/pronoun on line ${leftover.index}: "${leftover.badOutput}" -> "${scrubbed}"`);
      repairedMap.set(leftover.index, scrubbed);
    }
  }

  return repairedMap;
}

/**
 * Helper to translate a batch of lines with retries and model fallback
 * Uses the exact Burmese anime translation prompt with CPS (< 10) controls.
 */
async function translateBatchWithRetry(
  batch: BatchItemInput[],
  sourceLang: string,
  preferredModel = 'gemini-3.8-flash',
  rateLimiter?: ModelRateLimiter,
  onStatus?: (msg: string) => void,
  cpsMode: DialogueCpsMode = 'balanced_natural',
  episodeLockInstruction = '',
  episodeLock?: EpisodeLockProfile
): Promise<{ index: number; translated: string }[]> {
  const systemInstruction = `${EXACT_USER_PROMPT}${episodeLockInstruction}`;

  const sourceHint =
    sourceLang === 'ja'
      ? 'Japanese (日本語)'
      : sourceLang === 'en'
      ? 'English'
      : 'Auto-detect (Japanese or English)';

  const dialogueCpsRule =
    cpsMode === 'strict_compact'
      ? `၁။ ဇာတ်ကောင်ပြောသော Dialogue များ (isDialogue: true) [Strict Compact — CPS < 10]:
   - ကြည့်ရှုသူ စာဖတ်ချိန်မီစေရန် Aegisub CPS ကို 10 အောက် (CPS < 10) ဖြစ်အောင် အတိုဆုံးနှင့် အထိမိဆုံး ပြန်ပေးရမည်။
   - ⚠️ အရေးကြီး: Aegisub သည် မြန်မာစာတွင် သရ/အသတ်/ဝစ္စပေါက် (ေ, ာ, ါ, း) များကိုပါ စာလုံးတစ်လုံးစီ ရေတွက်သောကြောင့် မြန်မာတစ်သံလျှင် ~2 chars ခန့် ရှိသည် (ဥပမာ: "သုံးနေတာလား" = 8 chars)။ ထို့ကြောင့် "maxChars" အောက် ဝင်စေရန် အတိုဆုံး စကားပြောကိုသာ ရွေးပါ။
   - duration တိုသော (<1.5s) စာကြောင်းများတွင် အင်္ဂလိပ် Skill/Item နာမည်ရှည်ကြီးများက maxChars ကို ကျော်နေပါက အတိုချုံး/မြန်မာလို အတိုပြန်၍ maxChars အောက် ဝင်အောင် ရေးပါ။
   - ဇာတ်ကောင်ပြောသော Spoken Dialogue (isDialogue: true) များတွင် \\N (line break) များ မထည့်ဘဲ တစ်ကြောင်းတည်း (single line) အဖြစ်သာ ပြန်ပေးပါ။`
      : cpsMode === 'balanced_natural'
      ? `၁။ ဇာတ်ကောင်ပြောသော Dialogue များ (isDialogue: true) [Balanced Natural — Target CPS ~12–14]:
   - မြန်မာစကားပြော လေသံ၊ ဇာတ်ကောင်စရိုက်/ယဉ်ကျေးမှု (ဥပမာ: "သခင်ကြီးရှင့်", "ကိုယ်/မင်း", "ရှင်/ကျွန်မ", "ငါ/နင်") နှင့် ဝါကျအဆုံးသတ် (ဘူး/တယ်/လား/ရှင့်) များကို မပျက်စေဘဲ သဘာဝကျကျ ကျစ်လျစ်အောင် ပြန်ပေးပါ (Target ~12–14 CPS, "maxChars" ဝန်းကျင် ထိန်းပါ)။
   - ⚠️ အတိုချုံးရန်အတွက် ဇာတ်ကောင်၏ ယဉ်ကျေးသော လေသံ/အဆုံးသတ် စကားလုံးများကို မဖြုတ်ပစ်ရ — ဝါကျအပြည့်အစုံ ဖတ်ရလွယ်ကူပြီး သဘာဝကျပါစေ။
   - ဇာတ်ကောင်ပြောသော Spoken Dialogue (isDialogue: true) များတွင် \\N (line break) များ မထည့်ဘဲ တစ်ကြောင်းတည်း (single line) အဖြစ်သာ ပြန်ပေးပါ။`
      : `၁။ ဇာတ်ကောင်ပြောသော Dialogue များ (isDialogue: true) [Natural Unlimited — No CPS Limit]:
   - CPS / စာလုံးအရေအတွက် ကန့်သတ်ချက် လုံးဝ မရှိပါ။ မူရင်းအဓိပ္ပာယ်၊ ခံစားချက်နှင့် ဇာတ်ကောင်စရိုက်အတိုင်း အသဘာဝအကျဆုံး၊ အပြည့်စုံဆုံး စကားပြောမြန်မာလို ပြန်ပေးပါ။
   - ဇာတ်ကောင်ပြောသော Spoken Dialogue (isDialogue: true) များတွင် \\N (line break) များ မထည့်ဘဲ တစ်ကြောင်းတည်း (single line) အဖြစ်သာ ပြန်ပေးပါ။`;

  const prompt = `# Source Language: ${sourceHint}
# ⏱️ CPS (Characters Per Second) & JP Subtitle Rules:
${dialogueCpsRule}
၂။ On-screen Signs / RPG UI / Text / Title / Lyrics များ (isDialogue: false):
   - Actor သို့မဟုတ် Style သည် SIGN, TEXT, TITLE, LYRICS, NOTE ဖြစ်ပါက CPS ကန့်သတ်ချက် မလိုအပ်ပါ။
   - RPG Status Box / Stats / Skill / Equipment / Disclaimer များတွင် ပါဝင်သော အင်္ဂလိပ် သို့မဟုတ် ဂျပန် စာသား/ခေါင်းစဉ်/အချက်အလက်အားလုံးကို ဘာသာမပြန်ဘဲ မချန်ထားရ (HP, MP, LV, S-rank ကဲ့သို့ Game အတိုကောက်နှင့် ဇာတ်ကောင်/နာမည်ရင်းများမှလွဲ၍ ကျန်ဖော်ပြချက်အားလုံးကို တိုတိုကျစ်ကျစ် မြန်မာလို အပြည့်အစုံ ပြန်ပေးပါ)။
   - Box ထဲတွင် သပ်ရပ်စွာ ဝင်ဆံ့စေရန် တိုတိုကျစ်ကျစ် ရေးပြီး မူရင်း \\N line break အရေအတွက်နှင့် နေရာ (lineBreakCount) ကို အတိအကျ ဆက်လက် ထိန်းသိမ်းပေးပါ။
၃။ Jp စာတန်းထိုးတွေအတွက်သီးသန့် - () စာသားတွေ နဲ့ ညည်းသံလေသံတွေပါလာရင် ဖျက်ထုတ်ပါ Text ပဲ ဖျက်ပါ လှိုင်းမဖျက်ပါနဲ့:
   - စာကြောင်းတစ်ခုလုံးသည် (...) သို့မဟုတ် ညည်းသံ/အသက်ရှူသံ/လေသံသက်သက်သာဖြစ်ပါက "translated" ကို "" (empty string) အဖြစ် ပြန်ပေးပါ (index ကို လုံးဝ မချန်ထားရ၊ လိုင်းမဖျက်ရ)။
၄။ ပူးနေသော သို့မဟုတ် ၁ ကြောင်းခြား စာသားတူလိုင်းများ (duplicateGroupId ပါဝင်သော လိုင်းများ):
   - မူရင်းစာသားတူနေ၍ "duplicateGroupId" တူညီနေသော လိုင်းအားလုံးကို မြန်မာဘာသာပြန် စာသား တစ်ထပ်တည်း (EXACT SAME Burmese dialogue) ထွက်အောင် ပြန်ပေးရမည်။ CPS များနေလည်း ပြဿနာမရှိပါ (allowHighCps: true) — CPS ချရန်အတွက် ဘာသာပြန် မတူအောင် မပြောင်းရ။
၅။ ASS Tags & Ellipsis Discipline:
   - Timecode များ လုံးဝ မထည့်ရ။ ASS Tag {\\...} များကို မပျက်မစီး ထိန်းသိမ်းပါ (isDialogue: true တွင် \\N ဖျက်၍ တစ်ကြောင်းတည်း ရေးပါ၊ Signs/Title/Lyrics တွင် \\N ကို ဆက်လက် ထိန်းသိမ်းပါ)။
   - Italic tag များကို {\\i1}စာသား{\\i0} ဟု မှန်ကန်စွာ အဖွင့်အပိတ် ထားပါ ({{\\i0}{\\i1}} ဟု ပြောင်းပြန် မရေးရ)။
   - စကားပြော ရပ်တန့်/ဆိုင်းငံ့သော နေရာများ (ဥပမာ "ဒါပေမဲ့…") တွင် single period (.) မသုံးဘဲ Unicode ellipsis (…) သာ သုံးပါ။ ဝါကျအဆုံးတွင် ပုဒ်မ (.) လုံးဝ မထည့်ရ။
၆။ ဘာသာစကား သန့်ရှင်းမှု & အင်္ဂလိပ်စကားလုံး အပြည့်အစုံ ဘာသာပြန်ခြင်း (STRICT LANGUAGE PURITY):
   - မူရင်း ဂျပန်/အင်္ဂလိပ်စာသားများကို ဘာသာမပြန်ဘဲ လုံးဝ မချန်ထားရ ({\\i1} ကဲ့သို့ ASS Tag များ သို့မဟုတ် "..." မျက်တောင်အဖွင့်အပိတ်များ ပါနေသော်လည်း စာသားအားလုံးကို မြန်မာလို အပြည့်အစုံ ပြန်ပေးရမည်)။
   - ဇာတ်ကောင်နာမည် (Character Name) / နေရာနာမည်ရင်း သို့မဟုတ် HP/MP/LV ကဲ့သို့ Game stat tag များမှလွဲ၍ Heaven, Boss, Devil, Master, Princess, Monster, Guild, Wedding, Promise စသည့် သာမန်အင်္ဂလိပ် စကားလုံးများကို အင်္ဂလိပ်လို မချန်ဘဲ မြန်မာလို အပြည့်အစုံ (ဥပမာ - ကောင်းကင်ဘုံ, သူဌေး/ခေါင်းဆောင်, နတ်ဆိုး/မိစ္ဆာ) ဘာသာပြန်ပေးရမည်။
   - မြန်မာစာထဲတွင် အီသီယိုးပီးယား (ဥပမာ ድ), ထိုင်း (ค), လာအို (ະ), အာရပ်, ကိုရီးယား, ဟိန္ဒီ, ရုရှား, သို့မဟုတ် ဂျပန်/တရုတ် Kanji/Hanzi/Kana စာလုံးများ (ဥပမာ \`無號機\`, \`無号機\`) လုံးဝ မရောထည့်ရ။ Kanji ဖြင့် ရေးထားသော Unit/Mecha/ကုဒ်နာမည်များကို "Unit 0", "Mugouki" သို့မဟုတ် "နံပါတ်မဲ့ယူနစ်" ဟု အင်္ဂလိပ်/မြန်မာလို ပြောင်းရေးပါ။ Allowed scripts (မြန်မာ Unicode \\u1000-\\u109F + Basic Latin ASCII + digits + standard punctuation) ကိုသာ ထုတ်ပေးရမည်။
   - စာလုံးပေါင်း မှန်ကန်စေရန် "ကတိ" ကို "ဂတိ" ဟု မှားမရေးရ (တကယ် ကို တစ်ကယ် ဟု မရေးရ)။
၇။ SPEAKER-GENDER STRICT PRONOUN LOCK & PLACE/RANK CONSISTENCY:
   - Line တစ်ခုချင်းစီရှိ "actor" နှင့် "speakerLock" ကို အတိအကျ စစ်ဆေးပါ — ယောက်ျားလေး Actor (MALE) သည် "ကျွန်မ / ကျွန်မတို့" လုံးဝ မသုံးရ၊ မိန်းကလေး Actor (FEMALE) သည် "ခင်ဗျား / ကျွန်တော် / ကျွန်တော်တို့" လုံးဝ မသုံးရ။ Character တစ်ဦးတည်းတွင် "ကျွန်ုပ်" နှင့် "ကျွန်တော်" ရောမသုံးဘဲ Lock ချထားသော Self-Pronoun တစ်မျိုးတည်းသာ သုံးပါ။
   - နေရာ/နိုင်ငံနာမည် အရင်းကို Romaji/English အတိုင်း ထားပြီး နောက်ဆက်တွဲကိုသာ မြန်မာလို တစ်မျိုးတည်း ပြန်ပါ (ဥပမာ- Guillier ချောက်ကမ်းပါး, Lionel နိုင်ငံတော်, Dacan လွင်ပြင် — ဂီလီယာ ဟု အသံဖလှယ်ခြင်း သို့မဟုတ် Guillier Gorge ဟု အင်္ဂလိပ်ချန်ခြင်း မလုပ်ရ)။
   - ရာထူးနာမည်များ (General -> ဗိုလ်ချုပ်, Crown Prince -> အိမ်ရှေ့မင်းသား, His Majesty -> ဘုရင်မင်းမြတ်) ကို အင်္ဂလိပ်/မြန်မာ မရောဘဲ မြန်မာလို တစ်ညီတည်း ပြန်ပေးပါ။

# DIALOGUE BATCH TO TRANSLATE (JSON format):
${JSON.stringify(batch)}

Return a JSON array of objects with "index" (number matching input) and "translated" (translated dialogue text):`;

  // Attempt strategy: try the user's chosen model first, then immediately rotate across
  // alternate Gemini models and lighter thinking tiers to survive 503 demand spikes and timeouts.
  const attemptPlans: { model: string; thinking?: ThinkingLevel; timeoutMs: number }[] =
    preferredModel === 'gemini-3.1-flash-lite'
      ? [
          { model: 'gemini-3.1-flash-lite', timeoutMs: 60000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 65000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 65000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 55000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.LOW, timeoutMs: 55000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.LOW, timeoutMs: 55000 },
        ]
      : preferredModel === 'gemini-4-hydrogen'
      ? [
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 70000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 75000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 60000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 60000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 65000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 55000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.LOW, timeoutMs: 55000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.LOW, timeoutMs: 55000 },
        ]
      : [
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 75000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 70000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 60000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 65000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 60000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 55000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.LOW, timeoutMs: 55000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.LOW, timeoutMs: 55000 },
        ];

  let lastError: unknown;
  const failedDistinctModels = new Set<string>();
  const totalAttempts = attemptPlans.length;

  for (let attempt = 1; attempt <= totalAttempts; attempt++) {
    const plan = attemptPlans[attempt - 1];
    const modelToUse = plan.model;

    // Never re-call a model whose daily Free Tier quota is already exhausted
    if (rateLimiter?.isDailyExhausted(modelToUse)) {
      console.log(`[translate] Skipping attempt ${attempt}/${totalAttempts} (${modelToUse} daily quota exhausted)`);
      continue;
    }

    // Skip if model has a long >15s cooldown while a healthy alternate model exists later
    const modelCooldownMs = rateLimiter?.getRemainingCooldownMs(modelToUse) ?? 0;
    const hasHealthyOtherModelLater = attemptPlans
      .slice(attempt)
      .some(
        (p) =>
          p.model !== modelToUse &&
          !rateLimiter?.isDailyExhausted(p.model) &&
          (rateLimiter?.getRemainingCooldownMs(p.model) ?? 0) === 0
      );

    if (modelCooldownMs > 15000 && hasHealthyOtherModelLater) {
      console.log(
        `[translate] Skipping attempt ${attempt}/${totalAttempts} (${modelToUse} cooldown=${Math.ceil(modelCooldownMs / 1000)}s) in favor of healthy alternate model`
      );
      continue;
    }

    try {
      if (rateLimiter) {
        await rateLimiter.acquireSlot(modelToUse, (waitMs, reason) => {
          onStatus?.(`${reason} — waiting ${Math.ceil(waitMs / 1000)}s before resuming seamlessly...`);
        });
      }

      const baseConfig: Record<string, unknown> = {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              index: { type: Type.INTEGER, description: 'The index of the line from the input batch' },
              translated: { type: Type.STRING, description: 'The translated Burmese subtitle line' },
            },
            required: ['index', 'translated'],
          },
        },
      };

      if (plan.thinking) {
        baseConfig.thinkingConfig = { thinkingLevel: plan.thinking };
      }

      console.log(`[translate] Calling ${modelToUse} (thinking=${plan.thinking || 'default'}) for ${batch.length} lines (attempt ${attempt}/${totalAttempts})...`);
      const t0 = Date.now();

      const { response, usedLevel, didStepDown } = await executeWithSafetyStepDown({
        onStepDown: ({ fromLevel, toLevel, reason }) => {
          console.warn(
            `[safety] ${modelToUse} blocked at "${fromLevel.label}" (${reason}) for ${batch.length} lines -> Auto Step-Down to "${toLevel.label}"`
          );
          onStatus?.(
            `Safety filter ("${fromLevel.label}") triggered — Auto Step-Down to "${toLevel.label}" & retrying...`
          );
        },
        runAttempt: async (safetySettings) => {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`Model ${modelToUse} timed out after ${plan.timeoutMs / 1000}s`)), plan.timeoutMs)
          );

          const generatePromise = ai.models.generateContent({
            model: modelToUse,
            contents: prompt,
            config: {
              ...baseConfig,
              safetySettings,
            },
          });

          return Promise.race([generatePromise, timeoutPromise]);
        },
      });
      rateLimiter?.recordSuccess(modelToUse);
      console.log(
        `[translate] ${modelToUse} responded in ${Date.now() - t0}ms (safety="${usedLevel.label}"${didStepDown ? ', stepped-down' : ''})`
      );

      const responseText = response.text || '[]';
      const parsed: { index: number; translated: string }[] = JSON.parse(responseText);

      // Verify that all items are present
      if (Array.isArray(parsed) && parsed.length > 0) {
        const map = new Map<number, string>();
        for (const item of parsed) {
          if (typeof item.index === 'number' && typeof item.translated === 'string') {
            map.set(item.index, item.translated);
          }
        }

        const completedList: { index: number; translated: string }[] = [];
        const invalidLines: InvalidLineRepairInput[] = [];

        for (const item of batch) {
          const trans = map.get(item.index);
          if (trans !== undefined) {
            const check = validateTranslatedLine(item.text, trans);
            const spCheck = detectSpeakerPronounViolation(item.actor, trans, episodeLock);
            if (!check.valid || !spCheck.valid) {
              invalidLines.push({
                index: item.index,
                actor: item.actor,
                speakerLock: item.speakerLock,
                originalText: item.text,
                badOutput: trans,
                reason: !check.valid ? (check.reason || 'invalid_output') : (spCheck.reason || 'pronoun_violation'),
                forbiddenScripts: check.forbiddenScripts,
                isDialogue: item.isDialogue,
                maxChars: item.maxChars,
              });
            }
            completedList.push({ index: item.index, translated: trans });
          } else {
            throw new Error(`Model omitted index ${item.index} in batch response`);
          }
        }

        if (invalidLines.length > 0) {
          console.warn(
            `[translate] Detected ${invalidLines.length}/${batch.length} invalid line(s) (${invalidLines
              .map((l) => `#${l.index}:${l.reason}${l.forbiddenScripts ? `[${l.forbiddenScripts.join(',')}]` : ''}`)
              .join(', ')}). Running targeted repair...`
          );

          const repairedMap = await repairInvalidLines(
            invalidLines,
            sourceLang,
            preferredModel,
            rateLimiter,
            onStatus,
            episodeLockInstruction,
            episodeLock
          );
          for (const entry of completedList) {
            const fixed = repairedMap.get(entry.index);
            if (fixed !== undefined) {
              entry.translated = fixed;
            }
          }

          // Check if any foreign Unicode glitch or many lines STILL failed validation after repair -> retry batch on non-final attempt
          let stillInvalidCount = 0;
          let stillForbiddenCount = 0;
          for (let idx = 0; idx < completedList.length; idx++) {
            const recheck = validateTranslatedLine(batch[idx].text, completedList[idx].translated);
            if (!recheck.valid) {
              stillInvalidCount++;
              if (recheck.reason === 'forbidden_script') {
                stillForbiddenCount++;
              }
            }
          }

          if (
            (stillForbiddenCount > 0 || stillInvalidCount > Math.max(1, Math.floor(batch.length / 3))) &&
            attempt < totalAttempts
          ) {
            throw new Error(
              `${stillInvalidCount} line(s) (${stillForbiddenCount} forbidden-script) remained invalid after targeted repair`
            );
          }
        }

        return completedList;
      }
      throw new Error('Received empty or invalid array from translation model');
    } catch (err) {
      lastError = err;
      failedDistinctModels.add(modelToUse);
      const errMsg = err instanceof Error ? err.message : String(err);
      const classified = rateLimiter?.recordError(modelToUse, errMsg);
      console.warn(`[translate] Attempt ${attempt}/${totalAttempts} (${modelToUse}, thinking=${plan.thinking || 'default'}) failed: ${errMsg}`);

      // Fast-split large batches (>= 10 lines) as soon as >= 2 distinct models fail due to 503/timeout/overload
      if (shouldFastSplitLargeBatch(batch.length, failedDistinctModels.size, err, 5)) {
        console.warn(
          `[translate] Fast-splitting large batch (${batch.length} lines) after transient failure across ${failedDistinctModels.size} models`
        );
        throw err;
      }

      if (attempt < totalAttempts) {
        const nextPlan = attemptPlans[attempt];
        const switchingHealthyModel =
          nextPlan &&
          nextPlan.model !== modelToUse &&
          !rateLimiter?.isDailyExhausted(nextPlan.model) &&
          (rateLimiter?.getRemainingCooldownMs(nextPlan.model) ?? 0) === 0;

        const hasHealthyFallbackLater = attemptPlans
          .slice(attempt)
          .some(
            (p) =>
              p.model !== modelToUse &&
              !rateLimiter?.isDailyExhausted(p.model) &&
              (rateLimiter?.getRemainingCooldownMs(p.model) ?? 0) === 0
          );

        if (switchingHealthyModel || err instanceof SafetyBlockError) {
          await new Promise((resolve) => setTimeout(resolve, err instanceof SafetyBlockError ? 500 : 1000));
        } else {
          const backoffMs = classified?.isRateLimit
            ? Math.max(classified.retryDelayMs, 10000)
            : classified?.isTransientUnavailable
            ? Math.min(20000, 4000 * attempt)
            : 1500 * attempt;
          if (backoffMs > 15000 && hasHealthyFallbackLater) {
            // Skip long stall on throttled model so next loop immediately switches to healthy alternate
            await new Promise((resolve) => setTimeout(resolve, 500));
          } else {
            if (backoffMs >= 4000) {
              const label = classified?.isTransientUnavailable ? 'High demand (503)' : 'API busy/rate-limited';
              onStatus?.(`${label} — waiting ${Math.ceil(backoffMs / 1000)}s before retrying automatically...`);
            }
            await new Promise((resolve) => setTimeout(resolve, backoffMs));
          }
        }
      }
    }
  }

  if (!lastError) {
    const dailyMsg = rateLimiter?.getLastDailyQuotaError();
    throw new Error(
      dailyMsg || 'All configured Gemini models have exceeded their Free Tier RequestsPerDay daily quota.'
    );
  }
  throw lastError;
}

interface PipelineParams {
  fileContent: string;
  fileName: string;
  sourceLanguage: string;
  modelChoice: string;
  signMode: SignTypesettingMode;
  denseSignStrategy: DenseSignStrategy;
  cpsMode: DialogueCpsMode;
  preTranslatedLines?: unknown;
  preExtractedEpisodeLock?: unknown;
}

interface PipelineCallbacks {
  isAborted?: () => boolean;
  onEvent?: (data: Record<string, unknown>) => void;
  onProgress?: (percent: number, statusMessage: string) => void;
}

class PipelineValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PipelineValidationError';
  }
}

async function runSubtitleTranslationPipeline(
  params: PipelineParams,
  callbacks: PipelineCallbacks = {}
): Promise<{
  assContent: string;
  finalPreview: unknown[];
  outputFileName: string;
  cpsMode: DialogueCpsMode;
  signMode: SignTypesettingMode;
  eventCount: number;
  translatableCount: number;
}> {
  const {
    fileContent,
    fileName,
    sourceLanguage,
    modelChoice,
    signMode: safeSignMode,
    denseSignStrategy: safeDenseSignStrategy,
    cpsMode: safeCpsMode,
    preTranslatedLines,
    preExtractedEpisodeLock,
  } = params;

  const cpsConfig = getCpsModeConfig(safeCpsMode);
  const isAborted = () => Boolean(callbacks.isAborted?.());
  const emit = (data: Record<string, unknown>) => {
    callbacks.onEvent?.(data);
    const pct = typeof data.percent === 'number' ? data.percent : undefined;
    const msg =
      typeof data.statusMessage === 'string'
        ? data.statusMessage
        : typeof data.processedCount === 'number' && typeof data.totalCount === 'number'
        ? `Translated ${data.processedCount}/${data.totalCount} lines...`
        : undefined;
    if (pct !== undefined && msg) {
      callbacks.onProgress?.(pct, msg);
    }
  };

  // Detect format: SRT or ASS
  const isSrt =
    fileName.toLowerCase().endsWith('.srt') ||
    (!fileContent.includes('[Script Info]') && fileContent.includes('-->'));

  const parsedData = isSrt ? parseSrt(fileContent) : parseAss(fileContent);

  if (parsedData.events.length === 0) {
    throw new PipelineValidationError(
      'No dialogue events found in the uploaded subtitle file. Please verify file format.'
    );
  }

  const translatableEvents = parsedData.events.filter((e) => e.isTranslatable);
  const validTranslatableIndices = new Set(translatableEvents.map((e) => e.index));
  const translatedMap = sanitizePreTranslatedMap(preTranslatedLines, validTranslatableIndices);
  const { alreadyTranslatedCount, remainingEvents } = filterRemainingTranslatableEvents(
    translatableEvents,
    translatedMap
  );

  emit({
    type: 'init',
    totalEvents: parsedData.events.length,
    translatableCount: translatableEvents.length,
    resumedCount: alreadyTranslatedCount,
    format: parsedData.format,
    fileName,
  });

  if (translatableEvents.length === 0) {
    const watermarkEvents = hasWatermarkEvents(parsedData.events)
      ? []
      : buildWatermarkEvents(fileName, parsedData.events);
    const finalAss = buildAss(parsedData, undefined, true, watermarkEvents, {
      mode: safeSignMode,
      denseSignStrategy: safeDenseSignStrategy,
    });
    const outputFileName = fileName.replace(/\.(ass|srt)$/i, '') + '.translated.ass';
    return {
      assContent: finalAss,
      finalPreview: [],
      outputFileName,
      cpsMode: safeCpsMode,
      signMode: safeSignMode,
      eventCount: parsedData.events.length + watermarkEvents.length,
      translatableCount: 0,
    };
  }

  // Detect nearby duplicate lines (consecutive like 1,2,3 or 1-line gap like 1 & 3)
  const nearbyDuplicateClusters = findNearbyDuplicateClusters(parsedData.events, 2);

  const rateLimiter = new ModelRateLimiter();
  let episodeLock: EpisodeLockProfile = sanitizeEpisodeLockProfile(preExtractedEpisodeLock);
  const hasPreExtractedLock =
    (episodeLock.speakerProfiles?.length ?? 0) > 0 ||
    episodeLock.pronounLocks.length > 0 ||
    episodeLock.glossaryTerms.length > 0;

  let completedBatchesCount = 0;
  let completedLinesCount = alreadyTranslatedCount;
  const startPercent = Math.max(
    5,
    Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))
  );

  if (remainingEvents.length > 0 && !hasPreExtractedLock && translatableEvents.length >= 2) {
    emit({
      type: 'status',
      batchIndex: 0,
      totalBatches: Math.ceil(remainingEvents.length / 20),
      statusMessage: 'Analyzing episode speakers (gender & pronoun lock), places & ranks glossary...',
      percent: startPercent,
    });
    episodeLock = await extractEpisodeLockProfile(
      translatableEvents,
      fileName,
      sourceLanguage,
      modelChoice,
      rateLimiter,
      (msg) => {
        emit({
          type: 'status',
          batchIndex: 0,
          totalBatches: Math.ceil(remainingEvents.length / 20),
          statusMessage: msg,
          percent: startPercent,
        });
      }
    );
    if (
      (episodeLock.speakerProfiles?.length ?? 0) > 0 ||
      episodeLock.pronounLocks.length > 0 ||
      episodeLock.glossaryTerms.length > 0
    ) {
      emit({
        type: 'episode_lock',
        episodeLock,
      });
    }
  }

  const episodeLockInstruction = formatEpisodeLockInstruction(episodeLock);

  const BATCH_SIZE = 20;
  const batches: BatchItemInput[][] = [];
  let cursor = 0;
  while (cursor < remainingEvents.length) {
    let end = Math.min(cursor + BATCH_SIZE, remainingEvents.length);
    let extra = 0;
    while (end < remainingEvents.length && extra < 6) {
      const prevCluster = nearbyDuplicateClusters.get(remainingEvents[end - 1].index);
      const nextCluster = nearbyDuplicateClusters.get(remainingEvents[end].index);
      if (prevCluster && nextCluster && prevCluster.clusterId === nextCluster.clusterId) {
        end++;
        extra++;
      } else {
        break;
      }
    }

    const slice = remainingEvents.slice(cursor, end);
    batches.push(
      slice.map((e) => {
        const duration = getEventDuration(e.start, e.end);
        const isSign = isSignOrTitle(e.name, e.style, e.text);
        const dupCluster = nearbyDuplicateClusters.get(e.index);
        const maxChars = !isSign && !dupCluster ? getMaxCharsForMode(duration, safeCpsMode) : undefined;
        const textForLlm =
          isSign && safeSignMode !== 'preserve_original'
            ? prepareSignTextForTranslation(e.text)
            : e.text;
        const lbCount = isSign ? (textForLlm.match(/\\N/g) || []).length : 0;
        const speakerLock = !isSign ? formatLineSpeakerHint(e.name || '', episodeLock) : undefined;
        return {
          index: e.index,
          actor: e.name || '',
          ...(speakerLock ? { speakerLock } : {}),
          style: e.style,
          durationSec: duration,
          isDialogue: !isSign,
          maxChars,
          duplicateGroupId: dupCluster?.clusterId,
          allowHighCps: dupCluster ? true : undefined,
          lineBreakCount: lbCount > 0 ? lbCount : undefined,
          text: textForLlm,
        };
      })
    );
    cursor = end;
  }

  const getReconcileSource = (origText: string, isSign: boolean): string => {
    if (!isSign || safeSignMode === 'preserve_original') {
      return origText;
    }
    const cleanBody = prepareSignTextForTranslation(origText);
    if (safeSignMode === 'clean_top') {
      return cleanBody;
    }
    return `${extractLeadingTagBlock(origText)}${cleanBody}`;
  };

  if (batches.length > 0) {
    emit({
      type: 'status',
      batchIndex: 0,
      totalBatches: batches.length,
      statusMessage:
        alreadyTranslatedCount > 0
          ? `Resuming from checkpoint: ${alreadyTranslatedCount}/${translatableEvents.length} lines restored. Translating remaining ${remainingEvents.length} lines (${batches.length} micro-batches)...`
          : `Starting focused line-by-line translation for ${translatableEvents.length} events (${batches.length} micro-batches, ${cpsConfig.targetBadge})...`,
      percent: startPercent,
    });
  } else {
    emit({
      type: 'status',
      batchIndex: 0,
      totalBatches: 0,
      statusMessage: `Resuming from checkpoint: all ${alreadyTranslatedCount}/${translatableEvents.length} lines restored. Running final verification & ASS build...`,
      percent: 87,
    });
  }

  const CONCURRENCY = Math.min(2, batches.length);
  let nextBatchIdx = 0;

  const worker = async () => {
    while (nextBatchIdx < batches.length && !isAborted()) {
      const bIndex = nextBatchIdx++;
      const currentBatch = batches[bIndex];

      emit({
        type: 'status',
        batchIndex: bIndex + 1,
        totalBatches: batches.length,
        statusMessage: `Translating micro-batch ${bIndex + 1}/${batches.length} (${completedLinesCount}/${translatableEvents.length} lines done)...`,
        percent: Math.max(5, Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))),
      });

      const translatedBatch = await executeBatchWithAutoSplit({
        batch: currentBatch,
        minSplitSize: 5,
        onSplit: (origLen, firstLen, secondLen) => {
          console.warn(`[translate] Auto-splitting timed-out/failed batch (${origLen} -> ${firstLen} + ${secondLen} lines)`);
          emit({
            type: 'status',
            batchIndex: bIndex + 1,
            totalBatches: batches.length,
            statusMessage: `Dense batch (${origLen} lines) timed out — auto-splitting into ${firstLen} + ${secondLen} lines & retrying seamlessly...`,
            percent: Math.max(5, Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))),
          });
        },
        translateFn: (subBatch) =>
          translateBatchWithRetry(
            subBatch,
            sourceLanguage,
            modelChoice,
            rateLimiter,
            (msg) => {
              emit({
                type: 'status',
                batchIndex: bIndex + 1,
                totalBatches: batches.length,
                statusMessage: msg,
                percent: Math.max(5, Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))),
              });
            },
            safeCpsMode,
            episodeLockInstruction,
            episodeLock
          ),
      });

      const batchResults = new Map<number, string>();
      for (const item of translatedBatch) {
        const origEvent = parsedData.events[item.index];
        const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
        const reconcileSource = getReconcileSource(origEvent.text, isSign);
        let safeTranslated = reconcileTags(reconcileSource, item.translated, { preserveLineBreaks: isSign });
        if (!isSign) {
          safeTranslated = applyDeterministicSpeakerPronounFix(origEvent.name || '', safeTranslated, episodeLock);
        }
        safeTranslated = fixCommonBurmeseSpelling(enforceLockedGlossaryPostProcess(safeTranslated, episodeLock));
        const duration = getEventDuration(origEvent.start, origEvent.end);
        const isNearbyDup = nearbyDuplicateClusters.has(item.index);

        if (!isSign && !isNearbyDup && shouldUseLocalContraction(safeCpsMode)) {
          const maxAllowed = getMaxCharsForMode(duration, safeCpsMode);
          if (maxAllowed !== undefined) {
            safeTranslated = contractBurmeseForCps(safeTranslated, maxAllowed);
          }
        }
        batchResults.set(item.index, safeTranslated);
      }

      unifyNearbyDuplicateTranslations(parsedData.events, batchResults, nearbyDuplicateClusters);

      const previewList: {
        index: number;
        original: string;
        translated: string;
        time: string;
        start: string;
        end: string;
        durationSec: number;
        cps: number;
        isSignOrTitle: boolean;
        actor: string;
        style: string;
      }[] = [];

      for (const item of translatedBatch) {
        const origEvent = parsedData.events[item.index];
        const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
        const reconcileSource = getReconcileSource(origEvent.text, isSign);
        const finalText =
          batchResults.get(item.index) ??
          reconcileTags(reconcileSource, item.translated, { preserveLineBreaks: isSign });
        translatedMap.set(item.index, finalText);

        const duration = getEventDuration(origEvent.start, origEvent.end);
        const finalCps = calculateCps(finalText, duration);

        previewList.push({
          index: item.index,
          original: origEvent.text,
          translated: finalText,
          time: `${origEvent.start} - ${origEvent.end}`,
          start: origEvent.start,
          end: origEvent.end,
          durationSec: duration,
          cps: finalCps,
          isSignOrTitle: isSign,
          actor: origEvent.name || '',
          style: mapStyleName(origEvent.style, origEvent.name),
        });
      }

      completedBatchesCount++;
      completedLinesCount += currentBatch.length;

      emit({
        type: 'batch_complete',
        batchIndex: bIndex + 1,
        totalBatches: batches.length,
        completedBatches: completedBatchesCount,
        processedCount: completedLinesCount,
        totalCount: translatableEvents.length,
        percent: Math.max(5, Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))),
        preview: previewList,
      });
    }
  };

  if (CONCURRENCY > 0) {
    await Promise.all(new Array(CONCURRENCY).fill(null).map(() => worker()));
  }

  if (isAborted()) {
    throw new Error('Client disconnected before translation completed.');
  }

  const finalInvalidLines: InvalidLineRepairInput[] = [];
  for (const origEvent of translatableEvents) {
    const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
    const currentText = translatedMap.get(origEvent.index);
    const check =
      currentText === undefined
        ? { valid: false, reason: 'missing_translation' }
        : validateTranslatedLine(origEvent.text, currentText);
    const spCheck =
      !isSign && currentText !== undefined
        ? detectSpeakerPronounViolation(origEvent.name || '', currentText, episodeLock)
        : { valid: true };

    if (!check.valid || !spCheck.valid) {
      const duration = getEventDuration(origEvent.start, origEvent.end);
      const isNearbyDup = nearbyDuplicateClusters.has(origEvent.index);
      finalInvalidLines.push({
        index: origEvent.index,
        actor: origEvent.name || '',
        speakerLock: !isSign ? formatLineSpeakerHint(origEvent.name || '', episodeLock) : undefined,
        originalText:
          isSign && safeSignMode !== 'preserve_original'
            ? prepareSignTextForTranslation(origEvent.text)
            : origEvent.text,
        badOutput: currentText ?? '',
        reason: !check.valid ? (check.reason || 'invalid_output') : (spCheck.reason || 'pronoun_violation'),
        forbiddenScripts: check.forbiddenScripts,
        isDialogue: !isSign && !isNearbyDup,
        maxChars: !isSign && !isNearbyDup ? getMaxCharsForMode(duration, safeCpsMode) : undefined,
      });
    }
  }

  if (finalInvalidLines.length > 0) {
    console.warn(`[pipeline] Final quality gate caught ${finalInvalidLines.length} invalid/pronoun line(s). Running dedicated repair pass...`);
    emit({
      type: 'status',
      statusMessage: `Repairing ${finalInvalidLines.length} line(s) for language purity & speaker pronoun lock...`,
      percent: 88,
    });

    for (let i = 0; i < finalInvalidLines.length; i += 10) {
      const chunk = finalInvalidLines.slice(i, i + 10);
      const repairedChunk = await repairInvalidLines(
        chunk,
        sourceLanguage,
        modelChoice,
        rateLimiter,
        (msg) => {
          emit({ type: 'status', statusMessage: msg, percent: 89 });
        },
        episodeLockInstruction,
        episodeLock
      );
      for (const reqItem of chunk) {
        const fixed = repairedChunk.get(reqItem.index);
        if (fixed !== undefined) {
          const origEvent = parsedData.events[reqItem.index];
          const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
          const reconcileSource = getReconcileSource(origEvent.text, isSign);
          let safeFixed = reconcileTags(reconcileSource, fixed, { preserveLineBreaks: isSign });
          if (!isSign) {
            safeFixed = applyDeterministicSpeakerPronounFix(origEvent.name || '', safeFixed, episodeLock);
          }
          safeFixed = fixCommonBurmeseSpelling(enforceLockedGlossaryPostProcess(safeFixed, episodeLock));
          if (reqItem.isDialogue && reqItem.maxChars && shouldUseLocalContraction(safeCpsMode)) {
            safeFixed = contractBurmeseForCps(safeFixed, reqItem.maxChars);
          }
          if (validateTranslatedLine(origEvent.text, safeFixed).valid) {
            translatedMap.set(reqItem.index, safeFixed);
          }
        }
      }
    }
  }

  const globalLinesNeedingCompression: {
    index: number;
    currentBurmese: string;
    currentGraphemes: number;
    durationSec: number;
    maxChars: number;
  }[] = [];

  for (const origEvent of translatableEvents) {
    const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
    const isNearbyDup = nearbyDuplicateClusters.has(origEvent.index);
    if (isSign || isNearbyDup) continue;

    const currentText = translatedMap.get(origEvent.index) ?? '';
    const duration = getEventDuration(origEvent.start, origEvent.end);
    const cps = calculateCps(currentText, duration);
    if (shouldTriggerPass1(cps, safeCpsMode)) {
      const modeMaxChars = getMaxCharsForMode(duration, safeCpsMode);
      if (modeMaxChars !== undefined) {
        globalLinesNeedingCompression.push({
          index: origEvent.index,
          currentBurmese: currentText,
          currentGraphemes: countVisibleGraphemes(currentText),
          durationSec: duration,
          maxChars: modeMaxChars,
        });
      }
    }
  }

  if (globalLinesNeedingCompression.length > 0) {
    console.log(`[CPS] Global pass (${safeCpsMode}): ${globalLinesNeedingCompression.length} dialogue line(s) flagged. Compressing in chunks of 20...`);
    emit({
      type: 'status',
      statusMessage:
        safeCpsMode === 'balanced_natural'
          ? `Gently tightening ${globalLinesNeedingCompression.length} long dialogue line(s) (> 14.5 CPS) while keeping tone & endings...`
          : `Optimizing ${globalLinesNeedingCompression.length} dialogue line(s) to strictly stay under 10 CPS...`,
      percent: 91,
    });

    const stillTooLong: typeof globalLinesNeedingCompression = [];
    for (let i = 0; i < globalLinesNeedingCompression.length; i += 20) {
      const chunk = globalLinesNeedingCompression.slice(i, i + 20);
      const compressedMap = await compressLinesToTargetCps(
        chunk,
        modelChoice,
        false,
        rateLimiter,
        (msg) => {
          emit({ type: 'status', statusMessage: msg, percent: 92 });
        },
        safeCpsMode,
        episodeLockInstruction
      );

      for (const lineToFix of chunk) {
        const origEvent = parsedData.events[lineToFix.index];
        const candidate = compressedMap.get(lineToFix.index);
        let compressed = candidate ? candidate : lineToFix.currentBurmese;
        compressed = reconcileTags(origEvent.text, compressed, { preserveLineBreaks: false });
        compressed = applyDeterministicSpeakerPronounFix(origEvent.name || '', compressed, episodeLock);
        compressed = fixCommonBurmeseSpelling(enforceLockedGlossaryPostProcess(compressed, episodeLock));
        if (shouldUseLocalContraction(safeCpsMode)) {
          compressed = contractBurmeseForCps(compressed, lineToFix.maxChars);
        }

        if (
          validateTranslatedLine(origEvent.text, compressed).valid &&
          calculateCps(compressed, lineToFix.durationSec) <= calculateCps(lineToFix.currentBurmese, lineToFix.durationSec)
        ) {
          translatedMap.set(lineToFix.index, compressed);
        }

        const updatedText = translatedMap.get(lineToFix.index) || lineToFix.currentBurmese;
        if (shouldTriggerPass2(calculateCps(updatedText, lineToFix.durationSec), safeCpsMode)) {
          stillTooLong.push({
            index: lineToFix.index,
            currentBurmese: updatedText,
            currentGraphemes: countVisibleGraphemes(updatedText),
            durationSec: lineToFix.durationSec,
            maxChars: lineToFix.maxChars,
          });
        }
      }
    }

    if (stillTooLong.length > 0 && cpsConfig.useStrictPass2) {
      console.log(`[CPS] Global strict pass 2: ${stillTooLong.length} line(s) still >= 9.5 CPS...`);
      for (let i = 0; i < stillTooLong.length; i += 20) {
        const chunk = stillTooLong.slice(i, i + 20);
        const strictMap = await compressLinesToTargetCps(
          chunk,
          modelChoice,
          true,
          rateLimiter,
          (msg) => {
            emit({ type: 'status', statusMessage: msg, percent: 94 });
          },
          safeCpsMode,
          episodeLockInstruction
        );

        for (const lineToFix of chunk) {
          const origEvent = parsedData.events[lineToFix.index];
          const candidate = strictMap.get(lineToFix.index);
          if (candidate) {
            let finalCompressed = reconcileTags(origEvent.text, candidate, { preserveLineBreaks: false });
            finalCompressed = applyDeterministicSpeakerPronounFix(origEvent.name || '', finalCompressed, episodeLock);
            finalCompressed = fixCommonBurmeseSpelling(enforceLockedGlossaryPostProcess(finalCompressed, episodeLock));
            if (shouldUseLocalContraction(safeCpsMode)) {
              finalCompressed = contractBurmeseForCps(finalCompressed, lineToFix.maxChars);
            }
            const prevText = translatedMap.get(lineToFix.index) || lineToFix.currentBurmese;
            if (
              validateTranslatedLine(origEvent.text, finalCompressed).valid &&
              calculateCps(finalCompressed, lineToFix.durationSec) < calculateCps(prevText, lineToFix.durationSec)
            ) {
              translatedMap.set(lineToFix.index, finalCompressed);
            }
          }
        }
      }
    }
  }

  emit({
    type: 'status',
    statusMessage: 'Rebuilding ASS subtitle file and verifying all formatting tags...',
    percent: 96,
  });

  // Final safety sweep before building .ass so ZERO foreign/CJK/Kanji tokens survive into Burmese output lines
  for (const origEvent of translatableEvents) {
    const currentText = translatedMap.get(origEvent.index);
    if (!currentText) continue;
    if (detectForbiddenScripts(currentText).hasForbidden) {
      const scrubbed = stripStrayNonBurmeseTokensIfBurmesePresent(currentText);
      if (scrubbed !== currentText && !detectForbiddenScripts(scrubbed).hasForbidden && scrubbed.trim().length > 0) {
        const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
        const reconcileSource = getReconcileSource(origEvent.text, isSign);
        const normalized = reconcileTags(reconcileSource, scrubbed, { preserveLineBreaks: isSign });
        console.warn(`[pipeline] Final foreign-script scrub on line ${origEvent.index}: "${currentText}" -> "${normalized}"`);
        translatedMap.set(origEvent.index, normalized);
      }
    }
  }

  unifyNearbyDuplicateTranslations(parsedData.events, translatedMap, nearbyDuplicateClusters);

  const updatedEvents: AssEvent[] = parsedData.events.map((event) => {
    if (translatedMap.has(event.index)) {
      return {
        ...event,
        text: translatedMap.get(event.index)!,
      };
    }
    return event;
  });

  const watermarkEvents = hasWatermarkEvents(parsedData.events)
    ? []
    : buildWatermarkEvents(fileName, updatedEvents);

  const finalAssContent = buildAss(parsedData, updatedEvents, true, watermarkEvents, {
    mode: safeSignMode,
    denseSignStrategy: safeDenseSignStrategy,
  });

  const validation = validateAss(parsedData, finalAssContent, watermarkEvents.length);
  if (!validation.valid) {
    throw new PipelineValidationError(`Validation failed: ${validation.errors.join('; ')}`);
  }

  const finalBuiltEvents = parseAss(finalAssContent).events.slice(watermarkEvents.length);
  const finalPreview = translatableEvents.map((origEvent) => {
    const builtEv = finalBuiltEvents[origEvent.index];
    const finalText = builtEv ? builtEv.text : (translatedMap.get(origEvent.index) ?? origEvent.text);
    const finalStyle = builtEv ? builtEv.style : mapStyleName(origEvent.style, origEvent.name);
    const duration = getEventDuration(origEvent.start, origEvent.end);
    const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
    return {
      index: origEvent.index,
      original: origEvent.text,
      translated: builtEv?.type === 'Comment' ? `[Comment] ${finalText}` : finalText,
      time: `${origEvent.start} - ${origEvent.end}`,
      start: origEvent.start,
      end: origEvent.end,
      durationSec: duration,
      cps: calculateCps(finalText, duration),
      isSignOrTitle: isSign,
      actor: origEvent.name || '',
      style: finalStyle,
    };
  });

  const cleanBase = fileName.replace(/\.(ass|srt)$/i, '');
  const outputFileName = `${cleanBase}.translated.ass`;

  return {
    assContent: finalAssContent,
    finalPreview,
    outputFileName,
    cpsMode: safeCpsMode,
    signMode: safeSignMode,
    eventCount: updatedEvents.length + watermarkEvents.length,
    translatableCount: translatableEvents.length,
  };
}

function formatFriendlyTranslationError(err: unknown): string {
  const errorMsg = err instanceof Error ? err.message : String(err);
  const isDailyQuota =
    /RequestsPerDay|PerDay|daily/i.test(errorMsg) ||
    (/429|RESOURCE_EXHAUSTED/i.test(errorMsg) && /retry in \d+h/i.test(errorMsg));
  return isDailyQuota
    ? `Free Tier Daily API Quota Exceeded (Google's daily request cap on your API key was reached). Please wait for the daily quota reset or select a Billing-enabled Paid API Key in Settings > Secrets. Details: ${errorMsg}`
    : `Translation process encountered an error: ${errorMsg}`;
}

/**
 * Server-Sent Events (SSE) endpoint for translating subtitles with live progress
 */
app.post('/api/translate/stream', async (req, res) => {
  const {
    fileContent,
    fileName = 'subtitle.ass',
    sourceLanguage = 'auto',
    modelChoice = 'gemini-3.8-flash',
    signMode = 'smart_hybrid',
    denseSignStrategy = 'merge',
    cpsMode = 'balanced_natural',
    preTranslatedLines,
    preExtractedEpisodeLock,
  } = req.body;

  const safeSignMode: SignTypesettingMode =
    signMode === 'clean_top' || signMode === 'preserve_original' || signMode === 'smart_hybrid'
      ? signMode
      : 'smart_hybrid';

  const safeDenseSignStrategy: DenseSignStrategy =
    denseSignStrategy === 'comment' || denseSignStrategy === 'keep' || denseSignStrategy === 'merge'
      ? denseSignStrategy
      : 'merge';

  const safeCpsMode: DialogueCpsMode = normalizeCpsMode(cpsMode);

  if (!fileContent || typeof fileContent !== 'string') {
    res.status(400).json({ error: 'Missing or invalid fileContent' });
    return;
  }

  res.setTimeout(0);
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(': ' + ' '.repeat(2048) + '\n\n');

  const sendEvent = (data: Record<string, unknown>) => {
    if (!isClientClosed && !res.writableEnded) {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  };

  let isClientClosed = false;
  res.on('close', () => {
    if (!res.writableEnded) {
      isClientClosed = true;
    }
  });

  const heartbeatTimer = setInterval(() => {
    if (!isClientClosed && !res.writableEnded) {
      try {
        res.write(': keepalive\n\n');
      } catch {
        // ignore closed socket
      }
    }
  }, 4000);

  try {
    const result = await runSubtitleTranslationPipeline(
      {
        fileContent,
        fileName: sanitizeSubtitleFileName(fileName),
        sourceLanguage,
        modelChoice,
        signMode: safeSignMode,
        denseSignStrategy: safeDenseSignStrategy,
        cpsMode: safeCpsMode,
        preTranslatedLines,
        preExtractedEpisodeLock,
      },
      {
        isAborted: () => isClientClosed,
        onEvent: (data) => sendEvent(data),
      }
    );

    if (!isClientClosed) {
      sendEvent({
        type: 'complete',
        assContent: result.assContent,
        finalPreview: result.finalPreview,
        outputFileName: result.outputFileName,
        cpsMode: result.cpsMode,
        eventCount: result.eventCount,
        translatableCount: result.translatableCount,
        percent: 100,
        ...(result.translatableCount === 0
          ? { message: 'No translatable dialogue text found. Header and structure preserved.' }
          : {}),
      });
    }
  } catch (err: unknown) {
    console.error('[stream] Error in translate stream:', err);
    if (!isClientClosed) {
      sendEvent({
        type: 'error',
        message:
          err instanceof PipelineValidationError
            ? err.message
            : formatFriendlyTranslationError(err),
      });
    }
  } finally {
    clearInterval(heartbeatTimer);
    if (!res.writableEnded) {
      res.end();
    }
  }
});

// =============================================================================
// External Integration API (/api/v1/*) for Anime Downloader Apps / Bots / CLI
// Protected by EXTERNAL_API_SECRET (Authorization: Bearer <token> or x-api-key)
// =============================================================================

const externalJobStore = new ExternalJobStore({ maxJobs: 50, ttlMs: 60 * 60 * 1000 });
const rawSubtitleBodyParser = express.text({
  type: ['text/plain', 'text/x-ssa', 'text/x-ass'],
  limit: '50mb',
});

function requireExternalApiKey(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
): void {
  const expectedSecret = process.env.EXTERNAL_API_SECRET?.trim() || '';
  if (!expectedSecret) {
    res.status(503).json({
      ok: false,
      error:
        'External API is not configured. Set EXTERNAL_API_SECRET in server environment secrets before calling /api/v1/*.',
    });
    return;
  }

  const providedToken = extractApiTokenFromHeaders(req.headers);
  if (!verifySecretConstantTime(providedToken, expectedSecret)) {
    res.status(401).json({
      ok: false,
      error: 'Unauthorized. Provide valid Authorization: Bearer <token> or x-api-key header.',
    });
    return;
  }

  next();
}

/**
 * Direct REST endpoint: sends .ass / .srt and returns translated .ass upon completion.
 * Supports JSON ({ fileContent, fileName, signMode, cpsMode, responseFormat })
 * OR raw text subtitle body (with optional ?fileName=...&format=ass).
 */
app.post(
  '/api/v1/translate',
  requireExternalApiKey,
  rawSubtitleBodyParser,
  async (req, res) => {
    const parsedReq = parseExternalTranslatePayload(
      req.body,
      req.query as Record<string, unknown>,
      req.headers.accept || ''
    );
    if (!parsedReq.ok) {
      res.status(400).json({ ok: false, error: parsedReq.error });
      return;
    }

    const opts = parsedReq.data;
    res.setTimeout(0);
    let isClientClosed = false;
    res.on('close', () => {
      if (!res.writableEnded) {
        isClientClosed = true;
      }
    });

    try {
      const result = await runSubtitleTranslationPipeline(
        {
          fileContent: opts.fileContent,
          fileName: opts.fileName,
          sourceLanguage: opts.sourceLanguage,
          modelChoice: opts.modelChoice,
          signMode: opts.signMode,
          denseSignStrategy: opts.denseSignStrategy,
          cpsMode: opts.cpsMode,
        },
        {
          isAborted: () => isClientClosed,
        }
      );

      if (isClientClosed || res.writableEnded) return;

      if (opts.responseFormat === 'ass') {
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${result.outputFileName}"`);
        res.status(200).send(result.assContent);
        return;
      }

      res.status(200).json({
        ok: true,
        outputFileName: result.outputFileName,
        assContent: result.assContent,
        eventCount: result.eventCount,
        translatableCount: result.translatableCount,
        cpsMode: result.cpsMode,
        signMode: result.signMode,
      });
    } catch (err: unknown) {
      if (isClientClosed || res.writableEnded) return;
      if (err instanceof PipelineValidationError) {
        res.status(400).json({ ok: false, error: err.message });
        return;
      }
      console.error('[api/v1/translate] Error:', err);
      res.status(500).json({
        ok: false,
        error: formatFriendlyTranslationError(err),
      });
    }
  }
);

/**
 * Async Job endpoint: immediately returns 202 with jobId, processes in background,
 * and allows polling status + downloading the finished .translated.ass file.
 * Note on Cloud Run: enable "CPU always allocated" if polling with gaps between requests.
 */
app.post(
  '/api/v1/jobs',
  requireExternalApiKey,
  rawSubtitleBodyParser,
  (req, res) => {
    if (externalJobStore.activeCount() >= 5) {
      res.status(429).json({
        ok: false,
        error: 'Too many active translation jobs running (max 5 concurrent). Please retry shortly.',
      });
      return;
    }

    const parsedReq = parseExternalTranslatePayload(
      req.body,
      req.query as Record<string, unknown>,
      req.headers.accept || ''
    );
    if (!parsedReq.ok) {
      res.status(400).json({ ok: false, error: parsedReq.error });
      return;
    }

    const opts = parsedReq.data;
    const job = externalJobStore.createJob({
      fileName: opts.fileName,
      cpsMode: opts.cpsMode,
      signMode: opts.signMode,
    });

    // Launch background pipeline without awaiting HTTP response
    void (async () => {
      try {
        externalJobStore.updateProgress(job.id, 3, 'Starting subtitle translation pipeline...');
        const result = await runSubtitleTranslationPipeline(
          {
            fileContent: opts.fileContent,
            fileName: opts.fileName,
            sourceLanguage: opts.sourceLanguage,
            modelChoice: opts.modelChoice,
            signMode: opts.signMode,
            denseSignStrategy: opts.denseSignStrategy,
            cpsMode: opts.cpsMode,
          },
          {
            onProgress: (percent, statusMessage) => {
              externalJobStore.updateProgress(job.id, percent, statusMessage);
            },
          }
        );

        externalJobStore.completeJob(job.id, {
          assContent: result.assContent,
          outputFileName: result.outputFileName,
          eventCount: result.eventCount,
          translatableCount: result.translatableCount,
        });
      } catch (err: unknown) {
        const message =
          err instanceof PipelineValidationError
            ? err.message
            : formatFriendlyTranslationError(err);
        externalJobStore.failJob(job.id, message);
      }
    })();

    res.status(202).json({
      ok: true,
      jobId: job.id,
      status: job.status,
      outputFileName: job.outputFileName,
      statusUrl: `/api/v1/jobs/${job.id}`,
      downloadUrl: `/api/v1/jobs/${job.id}/download`,
    });
  }
);

app.get('/api/v1/jobs/:id', requireExternalApiKey, (req, res) => {
  const jobId = String(req.params.id || '').trim();
  const job = externalJobStore.getJob(jobId);
  if (!job) {
    res.status(404).json({ ok: false, error: 'Translation job not found or expired.' });
    return;
  }

  const includeContent = req.query.includeContent !== 'false';
  res.status(200).json({
    ok: true,
    id: job.id,
    status: job.status,
    percent: job.percent,
    statusMessage: job.statusMessage,
    fileName: job.fileName,
    outputFileName: job.outputFileName,
    cpsMode: job.cpsMode,
    signMode: job.signMode,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    completedAt: job.completedAt,
    eventCount: job.eventCount,
    translatableCount: job.translatableCount,
    ...(job.status === 'completed' && includeContent ? { assContent: job.assContent } : {}),
    ...(job.status === 'failed' ? { error: job.error } : {}),
  });
});

app.get('/api/v1/jobs/:id/download', requireExternalApiKey, (req, res) => {
  const jobId = String(req.params.id || '').trim();
  const job = externalJobStore.getJob(jobId);
  if (!job) {
    res.status(404).json({ ok: false, error: 'Translation job not found or expired.' });
    return;
  }
  if (job.status === 'failed') {
    res.status(500).json({ ok: false, error: job.error || 'Translation job failed.' });
    return;
  }
  if (job.status !== 'completed' || typeof job.assContent !== 'string') {
    res.status(409).json({
      ok: false,
      status: job.status,
      percent: job.percent,
      error: 'Translation job is not completed yet.',
    });
    return;
  }

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename="${job.outputFileName}"`);
  res.status(200).send(job.assContent);
});

const autopilotMulter = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024,
    files: 1,
    fields: 20,
  },
}).any();

function parseAutopilotMultipart(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
): void {
  autopilotMulter(req, res, (err: unknown) => {
    if (err) {
      const msg = err instanceof Error ? err.message : 'Invalid multipart upload.';
      res.status(400).json({ ok: false, error: msg });
      return;
    }
    next();
  });
}

let warnedUnprotectedAutopilot = false;

/**
 * Autopilot endpoint: accepts JSON { assContent, targetLanguage } OR multipart/form-data .ass file
 * (or raw text/plain), runs through Gemini 3.8 Flash + Smart Hybrid + CPS pacing pipeline,
 * and returns translated .ass content as text/plain.
 */
app.post(
  '/api/autopilot-translate-ass',
  (req, res, next) => {
    const expectedSecret = process.env.EXTERNAL_API_SECRET?.trim() || '';
    if (expectedSecret) {
      const providedToken = extractApiTokenFromHeaders(req.headers);
      if (!verifySecretConstantTime(providedToken, expectedSecret)) {
        res.status(401).json({
          ok: false,
          error: 'Unauthorized. Provide valid Authorization: Bearer <token> or x-api-key header.',
        });
        return;
      }
    } else if (!warnedUnprotectedAutopilot) {
      warnedUnprotectedAutopilot = true;
      console.warn('[autopilot] EXTERNAL_API_SECRET is not set; /api/autopilot-translate-ass is accepting requests without token auth.');
    }
    next();
  },
  rawSubtitleBodyParser,
  parseAutopilotMultipart,
  async (req, res) => {
    const uploadedFiles = Array.isArray((req as express.Request & { files?: Express.Multer.File[] }).files)
      ? (req as express.Request & { files?: Express.Multer.File[] }).files
      : undefined;
    const firstFile = uploadedFiles?.[0];

    const parsedReq = parseAutopilotTranslateRequest({
      body: req.body,
      query: req.query as Record<string, unknown>,
      uploadedFile: firstFile
        ? { buffer: firstFile.buffer, originalname: firstFile.originalname }
        : undefined,
    });

    if (!parsedReq.ok) {
      res.status(400).json({ ok: false, error: parsedReq.error });
      return;
    }

    const opts = parsedReq.data;
    res.setTimeout(0);
    let isClientClosed = false;
    res.on('close', () => {
      if (!res.writableEnded) {
        isClientClosed = true;
      }
    });

    try {
      const result = await runSubtitleTranslationPipeline(
        {
          fileContent: opts.fileContent,
          fileName: opts.fileName,
          sourceLanguage: opts.sourceLanguage,
          modelChoice: opts.modelChoice || 'gemini-3.8-flash',
          signMode: opts.signMode,
          denseSignStrategy: opts.denseSignStrategy,
          cpsMode: opts.cpsMode,
        },
        {
          isAborted: () => isClientClosed,
        }
      );

      if (isClientClosed || res.writableEnded) return;

      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('X-Output-Filename', result.outputFileName);
      res.setHeader('Content-Disposition', `inline; filename="${result.outputFileName}"`);
      res.status(200).send(result.assContent);
    } catch (err: unknown) {
      if (isClientClosed || res.writableEnded) return;
      if (err instanceof PipelineValidationError) {
        res.status(400).json({ ok: false, error: err.message });
        return;
      }
      console.error('[api/autopilot-translate-ass] Error:', err);
      res.status(500).json({
        ok: false,
        error: formatFriendlyTranslationError(err),
      });
    }
  }
);

// Serve frontend in production or through Vite in development
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });

    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Anime Subtitle Translator server running on http://0.0.0.0:${PORT}`);
    });
    return;
  }

  let viteMiddleware: express.RequestHandler | null = null;
  const viteReady = import('vite')
    .then(({ createServer: createViteServer }) =>
      createViteServer({
        server: { middlewareMode: true },
        appType: 'spa',
      })
    )
    .then((vite) => {
      viteMiddleware = vite.middlewares;
      console.log('[vite] Dev middleware ready');
      return vite.middlewares;
    })
    .catch((err) => {
      console.error('[vite] Failed to initialize Vite dev middleware:', err);
      throw err;
    });

  app.use(async (req, res, next) => {
    try {
      const mw = viteMiddleware ?? (await viteReady);
      return mw(req, res, next);
    } catch (err) {
      return next(err);
    }
  });

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Anime Subtitle Translator server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
