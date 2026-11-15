import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type, ThinkingLevel, HarmCategory, HarmBlockThreshold } from '@google/genai';

const SAFETY_SETTINGS_BLOCK_MOST = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
  },
];
import {
  parseAss,
  parseSrt,
  buildAss,
  validateAss,
  AssEvent,
  calculateCps,
  countVisibleGraphemes,
  getEventDuration,
  getMaxCharsForCps,
  isSignOrTitle,
  findNearbyDuplicateClusters,
  unifyNearbyDuplicateTranslations,
  validateTranslatedLine,
  buildWatermarkEvents,
  hasWatermarkEvents,
  mapStyleName,
} from './src/utils/subtitleParser.js';
import { reconcileTags, removeTrailingPeriods } from './src/utils/tagProtector.js';
import { ModelRateLimiter } from './src/utils/rateLimiter.js';

import { EXACT_USER_PROMPT } from './src/utils/userPrompt.js';

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
  });
});

export interface BatchItemInput {
  index: number;
  actor: string;
  style: string;
  durationSec: number;
  isDialogue: boolean;
  maxChars?: number;
  duplicateGroupId?: string;
  allowHighCps?: boolean;
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
 * Fast LLM compression pass for lines exceeding CPS 10
 */
async function compressLinesToTargetCps(
  linesToCompress: { index: number; currentBurmese: string; currentGraphemes?: number; durationSec: number; maxChars: number }[],
  preferredModel = 'gemini-3.8-flash',
  strictSecondPass = false,
  rateLimiter?: ModelRateLimiter,
  onStatus?: (msg: string) => void
): Promise<Map<number, string>> {
  const resultMap = new Map<number, string>();
  if (linesToCompress.length === 0) return resultMap;

  const modelChain =
    preferredModel === 'gemini-3.1-flash-lite'
      ? ['gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-4-hydrogen']
      : preferredModel === 'gemini-4-hydrogen'
      ? ['gemini-4-hydrogen', 'gemini-3.8-flash', 'gemini-3.1-flash-lite']
      : ['gemini-3.8-flash', 'gemini-4-hydrogen', 'gemini-3.1-flash-lite'];

  const systemInstruction = `You are an expert anime subtitle compression editor for Aegisub.
The following Burmese anime dialogue lines currently reach or exceed CPS 10 (Characters Per Second) in Aegisub.

CRITICAL AEGISUB CHARACTER COUNTING RULE:
In Aegisub, Burmese combining vowel/tone marks (ေ, ာ, ါ, း, etc.) count as SEPARATE characters!
- Example: "သုံးနေတာလား" = 8 Aegisub chars (NOT 4!). "လား" = 3 chars. "Skill သုံးနေတာလား" = 13 chars!
- Each Burmese syllable averages ~2 Aegisub characters.

YOUR TASK:
Condense each line into super-short, natural spoken anime Burmese so its Aegisub character count (excluding spaces and tags) is STRICTLY <= "maxChars" (guaranteeing CPS <= 9 in Aegisub, NEVER 10+).
- Keep core anime emotion and meaning, drop filler/pronouns/progressive markers ("နေ", "တဲ့", "ပါ", "တွေ", "တော့").
- ${strictSecondPass ? 'STRICT OVERRIDE: Even if an English skill/attack/item name makes the line exceed maxChars (e.g. "Blackened Flameblade" = 19 chars, or "Skill သုံးထား" = 10 chars when maxChars=8!), you MUST drop/translate the English word into short Burmese (e.g. "သုံးတာလား" = 7 chars, "ဓားနက်လား" = 8 chars) so total Aegisub chars is STRICTLY <= maxChars!' : 'Keep short character first-names in Romaji when they fit under maxChars; if an English skill/item phrase pushes total length over maxChars on a short <1.5s line, translate or shorten it into brief Burmese so total length <= maxChars.'}
- Output ONLY pure Burmese script (plus ASCII character names when needed). NEVER output Thai, Lao, Japanese, CJK, Korean, Hindi, or Cyrillic characters.
- Do NOT include ASS {\\...} tags (tags are preserved separately).
- NEVER end with a period (.).`;

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

  const prompt = `${strictSecondPass ? 'URGENT FINAL PASS — Previous version STILL exceeded maxChars! Drop English words/particles and write ultra-short Burmese (remember: ~2 Aegisub chars per syllable, so maxChars=8 means max 3-4 short Burmese syllables!) so EVERY line has Aegisub char count STRICTLY <= maxChars:\n' : 'Shorten each Burmese dialogue line below so its Aegisub char count (remember: each Burmese syllable ≈ 2 chars!) is STRICTLY <= maxChars (CPS < 10):\n'}${JSON.stringify(cleanPayload)}

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

      const config: Record<string, unknown> = {
        systemInstruction,
        safetySettings: SAFETY_SETTINGS_BLOCK_MOST,
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
        config.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
      }

      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Compression (${modelToUse}) timed out`)), 60000)
      );

      const generatePromise = ai.models.generateContent({
        model: modelToUse,
        contents: prompt,
        config,
      });

      const response = await Promise.race([generatePromise, timeoutPromise]);
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
  originalText: string;
  badOutput: string;
  reason: string;
  forbiddenScripts?: string[];
  isDialogue: boolean;
  maxChars?: number;
}

/**
 * Surgical last-resort sanitizer ONLY for lines that already contain valid Burmese dialogue
 * plus 1-2 stray foreign tokens (e.g. Thai `ค` or Lao `ະ` leaked mid-token) after LLM repair retries.
 * Never strips Japanese/CJK lines because untranslated Japanese lines must be re-translated, not deleted.
 */
function stripStrayNonBurmeseTokensIfBurmesePresent(text: string): string {
  const hasBurmese = /[\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF]/.test(text);
  if (!hasBurmese) return text;

  // Preserve ASS {...} tags intact, only clean visible segments
  return text.replace(/(\{[^}]*\})|([^{]+)/g, (segment, tag) => {
    if (tag) return tag;
    return segment.replace(/[\u0E00-\u0E7F\u0E80-\u0EFF\u1780-\u17FF\u0900-\u0DFF\u0400-\u04FF\u0600-\u06FF\u0590-\u05FF\uAC00-\uD7AF]/g, '');
  });
}

/**
 * Targeted per-line repair pass for any lines that failed script/translation validation
 * (e.g., untranslated Japanese blocks, Thai/Lao script leaks like `คະ`, or accidentally blanked lines).
 */
async function repairInvalidLines(
  invalidLines: InvalidLineRepairInput[],
  sourceLang: string,
  preferredModel = 'gemini-3.8-flash',
  rateLimiter?: ModelRateLimiter,
  onStatus?: (msg: string) => void
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
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 60000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
        ]
      : preferredModel === 'gemini-4-hydrogen'
      ? [
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 60000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
        ]
      : [
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 60000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 90000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
        ];

  let remaining = [...invalidLines];

  for (let round = 0; round < repairPlans.length && remaining.length > 0; round++) {
    const plan = repairPlans[round];
    if (rateLimiter?.isDailyExhausted(plan.model)) {
      continue;
    }

    const prompt = `# Source Language: ${sourceHint}
# URGENT QUALITY REPAIR PASS:
The following subtitle lines FAILED automated Burmese quality validation in the previous pass (they were either left untranslated in Japanese/English, accidentally blanked out, or contaminated with foreign characters such as Thai \`ค\`, Lao \`ະ\`, Korean, Hindi, or Japanese Kana/Kanji).

YOUR TASK:
Translate EACH line cleanly and accurately into natural spoken Burmese following the exact system prompt rules:
1. NEVER copy or leave source Japanese/Kanji/Kana or foreign scripts (Thai, Lao, Korean, Indic, Cyrillic) in "translated" — even if the source line has ASS tags like {\\i1} or quotation marks like "部長"!
2. Output ONLY pure Burmese Unicode (မြန်မာစာ) plus ASCII/Romaji character names where natural.
3. Preserve all ASS tags {\\...} and \\N line breaks from "originalText". Never output timecodes or trailing periods (.).
4. If "isDialogue" is true and "maxChars" is provided, keep Aegisub visible char count <= maxChars (CPS < 10).
5. Only return "" if "originalText" is purely a parenthetical SFX/groan note like （ため息）.

# LINES TO REPAIR:
${JSON.stringify(remaining)}

Return a JSON array of objects with "index" (integer) and "translated" (clean Burmese string).`;

    try {
      if (rateLimiter) {
        await rateLimiter.acquireSlot(plan.model, (waitMs, reason) => {
          onStatus?.(`${reason} — waiting ${Math.ceil(waitMs / 1000)}s before quality repair...`);
        });
      }

      const config: Record<string, unknown> = {
        systemInstruction: EXACT_USER_PROMPT,
        safetySettings: SAFETY_SETTINGS_BLOCK_MOST,
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
        config.thinkingConfig = { thinkingLevel: plan.thinking };
      }

      console.log(`[repair] Round ${round + 1}/${repairPlans.length}: repairing ${remaining.length} invalid line(s) via ${plan.model} (thinking=${plan.thinking || 'default'})...`);
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Repair (${plan.model}) timed out`)), plan.timeoutMs)
      );
      const generatePromise = ai.models.generateContent({
        model: plan.model,
        contents: prompt,
        config,
      });

      const response = await Promise.race([generatePromise, timeoutPromise]);
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
          if (check.valid) {
            repairedMap.set(reqItem.index, candidate);
            continue;
          }
          nextRemaining.push({
            ...reqItem,
            badOutput: candidate,
            reason: check.reason || reqItem.reason,
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

  // Last-resort surgical scrub ONLY if a line already has valid Burmese and just a stray Thai/Lao/Indic token
  for (const leftover of remaining) {
    const scrubbed = stripStrayNonBurmeseTokensIfBurmesePresent(leftover.badOutput);
    if (scrubbed !== leftover.badOutput && validateTranslatedLine(leftover.originalText, scrubbed).valid) {
      console.warn(`[repair] Scrubbed stray non-Burmese token on line ${leftover.index}: "${leftover.badOutput}" -> "${scrubbed}"`);
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
  onStatus?: (msg: string) => void
): Promise<{ index: number; translated: string }[]> {
  const systemInstruction = EXACT_USER_PROMPT;

  const sourceHint =
    sourceLang === 'ja'
      ? 'Japanese (日本語)'
      : sourceLang === 'en'
      ? 'English'
      : 'Auto-detect (Japanese or English)';

  const prompt = `# Source Language: ${sourceHint}
# ⏱️ CPS (Characters Per Second) & JP Subtitle Rules:
၁။ ဇာတ်ကောင်ပြောသော Dialogue များ (isDialogue: true):
   - ကြည့်ရှုသူ စာဖတ်ချိန်မီစေရန် Aegisub CPS ကို 10 အောက် (CPS < 10) ဖြစ်အောင် အတိုဆုံးနှင့် အထိမိဆုံး ပြန်ပေးရမည်။
   - ⚠️ အရေးကြီး: Aegisub သည် မြန်မာစာတွင် သရ/အသတ်/ဝစ္စပေါက် (ေ, ာ, ါ, း) များကိုပါ စာလုံးတစ်လုံးစီ ရေတွက်သောကြောင့် မြန်မာတစ်သံလျှင် ~2 chars ခန့် ရှိသည် (ဥပမာ: "သုံးနေတာလား" = 8 chars)။ ထို့ကြောင့် "maxChars" အောက် ဝင်စေရန် အတိုဆုံး စကားပြောကိုသာ ရွေးပါ။
   - duration တိုသော (<1.5s) စာကြောင်းများတွင် အင်္ဂလိပ် Skill/Item နာမည်ရှည်ကြီးများက maxChars ကို ကျော်နေပါက အတိုချုံး/မြန်မာလို အတိုပြန်၍ maxChars အောက် ဝင်အောင် ရေးပါ။
၂။ On-screen Signs / Text / Title များ (isDialogue: false):
   - Actor သို့မဟုတ် Style သည် SIGN, TEXT, TITLE, NOTE ဖြစ်ပါက CPS ကန့်သတ်ချက် မလိုအပ်ပါ။ မူရင်းအတိုင်း တိကျစွာ ပြန်ပါ။
၃။ Jp စာတန်းထိုးတွေအတွက်သီးသန့် - () စာသားတွေ နဲ့ ညည်းသံလေသံတွေပါလာရင် ဖျက်ထုတ်ပါ Text ပဲ ဖျက်ပါ လှိုင်းမဖျက်ပါနဲ့:
   - စာကြောင်းတစ်ခုလုံးသည် (...) သို့မဟုတ် ညည်းသံ/အသက်ရှူသံ/လေသံသက်သက်သာဖြစ်ပါက "translated" ကို "" (empty string) အဖြစ် ပြန်ပေးပါ (index ကို လုံးဝ မချန်ထားရ၊ လိုင်းမဖျက်ရ)။
၄။ ပူးနေသော သို့မဟုတ် ၁ ကြောင်းခြား စာသားတူလိုင်းများ (duplicateGroupId ပါဝင်သော လိုင်းများ):
   - မူရင်းစာသားတူနေ၍ "duplicateGroupId" တူညီနေသော လိုင်းအားလုံးကို မြန်မာဘာသာပြန် စာသား တစ်ထပ်တည်း (EXACT SAME Burmese dialogue) ထွက်အောင် ပြန်ပေးရမည်။ CPS များနေလည်း ပြဿနာမရှိပါ (allowHighCps: true) — CPS ချရန်အတွက် ဘာသာပြန် မတူအောင် မပြောင်းရ။ (duplicateGroupId မပါသည့် ဝေးကွာသော လိုင်းများတွင်မူ ပုံမှန် CPS < 10 ထိန်းပါ)။
၅။ ဘာသာပြန်ထဲတွင် Timecode များ လုံးဝ မထည့်ရ။ ASS Tag {\\...} များနှင့် \\N များကို မပျက်မစီး ထိန်းသိမ်းပါ။ အဆုံးတွင် ပုဒ်မ (.) မထည့်ရ။
၆။ ဘာသာစကား သန့်ရှင်းမှု (STRICT LANGUAGE PURITY):
   - မူရင်း ဂျပန်/အင်္ဂလိပ်စာသားများကို ဘာသာမပြန်ဘဲ လုံးဝ မချန်ထားရ ({\\i1} ကဲ့သို့ ASS Tag များ သို့မဟုတ် "..." မျက်တောင်အဖွင့်အပိတ်များ ပါနေသော်လည်း စာသားအားလုံးကို မြန်မာလို အပြည့်အစုံ ပြန်ပေးရမည်)။
   - မြန်မာစာထဲတွင် ထိုင်း (ဥပမာ ค), လာအို (ဥပမာ ະ), ကိုရီးယား, ဟိန္ဒီ, ရုရှား, သို့မဟုတ် ဂျပန်/တရုတ် စာလုံးများ လုံးဝ မရောထည့်ရ။ မြန်မာ Unicode စာလုံးများ (နှင့် လိုအပ်သော Romaji နာမည်များ) ကိုသာ ထုတ်ပေးရမည်။

# DIALOGUE BATCH TO TRANSLATE (JSON format):
${JSON.stringify(batch)}

Return a JSON array of objects with "index" (number matching input) and "translated" (translated dialogue text):`;

  // Attempt strategy: prioritize user's chosen model (gemini-3.8-flash with HIGH thinking by default)
  // and only rotate to alternate models if the primary model hits quota/overload errors.
  const attemptPlans: { model: string; thinking?: ThinkingLevel; timeoutMs: number }[] =
    preferredModel === 'gemini-3.1-flash-lite'
      ? [
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 120000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 90000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 90000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.LOW, timeoutMs: 75000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
        ]
      : preferredModel === 'gemini-4-hydrogen'
      ? [
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 120000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 120000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 90000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 90000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.LOW, timeoutMs: 75000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.LOW, timeoutMs: 75000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.LOW, timeoutMs: 75000 },
        ]
      : [
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 120000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.HIGH, timeoutMs: 120000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.HIGH, timeoutMs: 120000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.MEDIUM, timeoutMs: 90000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
          { model: 'gemini-4-hydrogen', thinking: ThinkingLevel.MEDIUM, timeoutMs: 90000 },
          { model: 'gemini-3.8-flash', thinking: ThinkingLevel.LOW, timeoutMs: 75000 },
          { model: 'gemini-3.1-flash-lite', timeoutMs: 45000 },
        ];

  let lastError: unknown;
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

      const config: Record<string, unknown> = {
        systemInstruction,
        safetySettings: SAFETY_SETTINGS_BLOCK_MOST,
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
        config.thinkingConfig = { thinkingLevel: plan.thinking };
      }

      console.log(`[translate] Calling ${modelToUse} (thinking=${plan.thinking || 'default'}) for ${batch.length} lines (attempt ${attempt}/${totalAttempts})...`);
      const t0 = Date.now();

      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Model ${modelToUse} timed out after ${plan.timeoutMs / 1000}s`)), plan.timeoutMs)
      );

      const generatePromise = ai.models.generateContent({
        model: modelToUse,
        contents: prompt,
        config,
      });

      const response = await Promise.race([generatePromise, timeoutPromise]);
      rateLimiter?.recordSuccess(modelToUse);
      console.log(`[translate] ${modelToUse} responded in ${Date.now() - t0}ms`);

      const finishReason = response.candidates?.[0]?.finishReason;
      if (finishReason === 'SAFETY') {
        throw new Error(`Response blocked by "Block most" safety settings (${modelToUse})`);
      }

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
            if (!check.valid) {
              invalidLines.push({
                index: item.index,
                originalText: item.text,
                badOutput: trans,
                reason: check.reason || 'invalid_output',
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

          const repairedMap = await repairInvalidLines(invalidLines, sourceLang, preferredModel, rateLimiter, onStatus);
          for (const entry of completedList) {
            const fixed = repairedMap.get(entry.index);
            if (fixed !== undefined) {
              entry.translated = fixed;
            }
          }

          // Check if many lines STILL failed validation after repair -> retry batch on non-final attempt
          const stillInvalidCount = completedList.filter((entry, idx) => {
            return !validateTranslatedLine(batch[idx].text, entry.translated).valid;
          }).length;

          if (stillInvalidCount > Math.max(1, Math.floor(batch.length / 3)) && attempt < totalAttempts) {
            throw new Error(`${stillInvalidCount} lines remained untranslated/invalid after targeted repair`);
          }
        }

        return completedList;
      }
      throw new Error('Received empty or invalid array from translation model');
    } catch (err) {
      lastError = err;
      const errMsg = err instanceof Error ? err.message : String(err);
      const classified = rateLimiter?.recordError(modelToUse, errMsg);
      console.warn(`[translate] Attempt ${attempt}/${totalAttempts} (${modelToUse}, thinking=${plan.thinking || 'default'}) failed: ${errMsg}`);
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

        if (switchingHealthyModel) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
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

/**
 * Server-Sent Events (SSE) endpoint for translating subtitles with live progress
 */
app.post('/api/translate/stream', async (req, res) => {
  const {
    fileContent,
    fileName = 'subtitle.ass',
    sourceLanguage = 'auto',
    modelChoice = 'gemini-3.8-flash',
  } = req.body;

  if (!fileContent || typeof fileContent !== 'string') {
    res.status(400).json({ error: 'Missing or invalid fileContent' });
    return;
  }

  // Set SSE headers with anti-buffering for Cloud Run / Nginx
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  // Flush 2KB comment padding immediately to defeat Cloud Run buffer threshold
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

  try {
    // Detect format: SRT or ASS
    const isSrt = fileName.toLowerCase().endsWith('.srt') ||
      (!fileContent.includes('[Script Info]') && fileContent.includes('-->'));

    const parsedData = isSrt ? parseSrt(fileContent) : parseAss(fileContent);

    if (parsedData.events.length === 0) {
      sendEvent({
        type: 'error',
        message: 'No dialogue events found in the uploaded subtitle file. Please verify file format.',
      });
      res.end();
      return;
    }

    const translatableEvents = parsedData.events.filter((e) => e.isTranslatable);

    sendEvent({
      type: 'init',
      totalEvents: parsedData.events.length,
      translatableCount: translatableEvents.length,
      format: parsedData.format,
      fileName,
    });

    if (translatableEvents.length === 0) {
      // Nothing to translate, return original as ASS with NaJu watermark header
      const watermarkEvents = hasWatermarkEvents(parsedData.events)
        ? []
        : buildWatermarkEvents(fileName, parsedData.events);
      const finalAss = buildAss(parsedData, undefined, true, watermarkEvents);
      const outputFileName = fileName.replace(/\.(ass|srt)$/i, '') + '.translated.ass';
      sendEvent({
        type: 'complete',
        assContent: finalAss,
        outputFileName,
        eventCount: parsedData.events.length + watermarkEvents.length,
        message: 'No translatable dialogue text found. Header and structure preserved.',
      });
      res.end();
      return;
    }

    // Detect nearby duplicate lines (consecutive like 1,2,3 or 1-line gap like 1 & 3)
    const nearbyDuplicateClusters = findNearbyDuplicateClusters(parsedData.events, 2);

    // Process in balanced batches (~28 lines) to slash Free Tier RPD/RPM quota usage by >55% (~16 calls/episode)
    // while never splitting a nearby duplicate cluster across batch edges when avoidable
    const BATCH_SIZE = 28;
    const batches: BatchItemInput[][] = [];
    let cursor = 0;
    while (cursor < translatableEvents.length) {
      let end = Math.min(cursor + BATCH_SIZE, translatableEvents.length);
      let extra = 0;
      while (end < translatableEvents.length && extra < 6) {
        const prevCluster = nearbyDuplicateClusters.get(translatableEvents[end - 1].index);
        const nextCluster = nearbyDuplicateClusters.get(translatableEvents[end].index);
        if (prevCluster && nextCluster && prevCluster.clusterId === nextCluster.clusterId) {
          end++;
          extra++;
        } else {
          break;
        }
      }

      const slice = translatableEvents.slice(cursor, end);
      batches.push(
        slice.map((e) => {
          const duration = getEventDuration(e.start, e.end);
          const isSign = isSignOrTitle(e.name, e.style, e.text);
          const dupCluster = nearbyDuplicateClusters.get(e.index);
          const enforceCps = !isSign && !dupCluster;
          return {
            index: e.index,
            actor: e.name || '',
            style: e.style,
            durationSec: duration,
            isDialogue: !isSign,
            maxChars: enforceCps ? getMaxCharsForCps(duration, 8.8) : undefined,
            duplicateGroupId: dupCluster?.clusterId,
            allowHighCps: dupCluster ? true : undefined,
            text: e.text,
          };
        })
      );
      cursor = end;
    }

    sendEvent({
      type: 'status',
      batchIndex: 0,
      totalBatches: batches.length,
      statusMessage: `Starting focused line-by-line translation for ${translatableEvents.length} events (${batches.length} micro-batches, Target CPS < 10)...`,
      percent: 5,
    });

    const translatedMap = new Map<number, string>();
    const rateLimiter = new ModelRateLimiter();
    let completedBatchesCount = 0;
    let completedLinesCount = 0;

    // Parallel Worker Pool: Concurrency of 2 prevents Free-Tier RPM/503 bursts while keeping smooth throughput
    const CONCURRENCY = Math.min(2, batches.length);
    let nextBatchIdx = 0;

    // SSE keep-alive heartbeat every 4s so Cloud Run / proxy never drops an active thinking stream
    const heartbeatTimer = setInterval(() => {
      if (!isClientClosed && !res.writableEnded) {
        try {
          res.write(': keepalive\n\n');
        } catch {
          // ignore write error on closed socket
        }
      }
    }, 4000);

    try {
      const worker = async () => {
        while (nextBatchIdx < batches.length && !isClientClosed) {
          const bIndex = nextBatchIdx++;
          const currentBatch = batches[bIndex];

          sendEvent({
            type: 'status',
            batchIndex: bIndex + 1,
            totalBatches: batches.length,
            statusMessage: `Translating micro-batch ${bIndex + 1}/${batches.length} (${completedLinesCount}/${translatableEvents.length} lines done)...`,
            percent: Math.max(5, Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))),
          });

          const translatedBatch = await translateBatchWithRetry(
            currentBatch,
            sourceLanguage,
            modelChoice,
            rateLimiter,
            (msg) => {
              sendEvent({
                type: 'status',
                batchIndex: bIndex + 1,
                totalBatches: batches.length,
                statusMessage: msg,
                percent: Math.max(5, Math.min(86, Math.round((completedLinesCount / translatableEvents.length) * 82))),
              });
            }
          );

          // Instant local post-processing: Reconcile tags & apply fast local Burmese contractions (0 extra API calls per batch!)
          const batchResults = new Map<number, string>();
          for (const item of translatedBatch) {
            const origEvent = parsedData.events[item.index];
            let safeTranslated = reconcileTags(origEvent.text, item.translated);
            const duration = getEventDuration(origEvent.start, origEvent.end);
            const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
            const isNearbyDup = nearbyDuplicateClusters.has(item.index);

            if (!isSign && !isNearbyDup) {
              const maxAllowed = getMaxCharsForCps(duration, 8.8);
              safeTranslated = contractBurmeseForCps(safeTranslated, maxAllowed);
            }
            batchResults.set(item.index, safeTranslated);
          }

          // Unify any nearby duplicate clusters present within this batch before emitting live preview
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
            const finalText = batchResults.get(item.index) ?? reconcileTags(origEvent.text, item.translated);
            translatedMap.set(item.index, finalText);

            const duration = getEventDuration(origEvent.start, origEvent.end);
            const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
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

          sendEvent({
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

      // Run parallel workers
      await Promise.all(new Array(CONCURRENCY).fill(null).map(() => worker()));
    } finally {
      clearInterval(heartbeatTimer);
    }

    console.log('[stream] All batches finished, completedLinesCount:', completedLinesCount);

    if (isClientClosed) {
      console.log('[stream] Client already closed, ending.');
      res.end();
      return;
    }

    // Final quality & purity safety net before CPS compression & ASS rebuild:
    // Catch any line still missing, untranslated, blanked, or containing leaked foreign scripts
    const finalInvalidLines: InvalidLineRepairInput[] = [];
    for (const origEvent of translatableEvents) {
      const currentText = translatedMap.get(origEvent.index);
      const check = currentText === undefined
        ? { valid: false, reason: 'missing_translation' }
        : validateTranslatedLine(origEvent.text, currentText);

      if (!check.valid) {
        const duration = getEventDuration(origEvent.start, origEvent.end);
        const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
        const isNearbyDup = nearbyDuplicateClusters.has(origEvent.index);
        finalInvalidLines.push({
          index: origEvent.index,
          originalText: origEvent.text,
          badOutput: currentText ?? '',
          reason: check.reason || 'invalid_output',
          forbiddenScripts: check.forbiddenScripts,
          isDialogue: !isSign && !isNearbyDup,
          maxChars: !isSign && !isNearbyDup ? getMaxCharsForCps(duration, 8.8) : undefined,
        });
      }
    }

    if (finalInvalidLines.length > 0) {
      console.warn(`[stream] Final quality gate caught ${finalInvalidLines.length} invalid line(s). Running dedicated repair pass...`);
      sendEvent({
        type: 'status',
        statusMessage: `Repairing ${finalInvalidLines.length} line(s) with untranslated or foreign characters...`,
        percent: 88,
      });

      for (let i = 0; i < finalInvalidLines.length; i += 10) {
        const chunk = finalInvalidLines.slice(i, i + 10);
        const repairedChunk = await repairInvalidLines(chunk, sourceLanguage, modelChoice, rateLimiter, (msg) => {
          sendEvent({ type: 'status', statusMessage: msg, percent: 89 });
        });
        for (const reqItem of chunk) {
          const fixed = repairedChunk.get(reqItem.index);
          if (fixed !== undefined) {
            const origEvent = parsedData.events[reqItem.index];
            let safeFixed = reconcileTags(origEvent.text, fixed);
            if (reqItem.isDialogue && reqItem.maxChars) {
              safeFixed = contractBurmeseForCps(safeFixed, reqItem.maxChars);
            }
            if (validateTranslatedLine(origEvent.text, safeFixed).valid) {
              translatedMap.set(reqItem.index, safeFixed);
            }
          }
        }
      }
    }

    // Batched global CPS < 10 compression pass (cuts API calls from ~80 down to 1-3 total calls for the whole episode!)
    const globalLinesNeedingCompression: { index: number; currentBurmese: string; currentGraphemes: number; durationSec: number; maxChars: number }[] = [];
    for (const origEvent of translatableEvents) {
      const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
      const isNearbyDup = nearbyDuplicateClusters.has(origEvent.index);
      if (isSign || isNearbyDup) continue;

      const currentText = translatedMap.get(origEvent.index) ?? '';
      const duration = getEventDuration(origEvent.start, origEvent.end);
      const cps = calculateCps(currentText, duration);
      if (cps >= 9.4) {
        globalLinesNeedingCompression.push({
          index: origEvent.index,
          currentBurmese: currentText,
          currentGraphemes: countVisibleGraphemes(currentText),
          durationSec: duration,
          maxChars: getMaxCharsForCps(duration, 8.8),
        });
      }
    }

    if (globalLinesNeedingCompression.length > 0) {
      console.log(`[CPS] Global pass: ${globalLinesNeedingCompression.length} dialogue line(s) >= 9.4 CPS. Compressing in chunks of 20...`);
      sendEvent({
        type: 'status',
        statusMessage: `Optimizing ${globalLinesNeedingCompression.length} dialogue line(s) to strictly stay under 10 CPS...`,
        percent: 91,
      });

      const stillTooLong: typeof globalLinesNeedingCompression = [];
      for (let i = 0; i < globalLinesNeedingCompression.length; i += 20) {
        const chunk = globalLinesNeedingCompression.slice(i, i + 20);
        const compressedMap = await compressLinesToTargetCps(chunk, modelChoice, false, rateLimiter, (msg) => {
          sendEvent({ type: 'status', statusMessage: msg, percent: 92 });
        });

        for (const lineToFix of chunk) {
          const origEvent = parsedData.events[lineToFix.index];
          let compressed = compressedMap.get(lineToFix.index) || lineToFix.currentBurmese;
          compressed = reconcileTags(origEvent.text, compressed);
          compressed = contractBurmeseForCps(compressed, lineToFix.maxChars);
          translatedMap.set(lineToFix.index, compressed);

          if (calculateCps(compressed, lineToFix.durationSec) >= 9.5) {
            stillTooLong.push({
              index: lineToFix.index,
              currentBurmese: compressed,
              currentGraphemes: countVisibleGraphemes(compressed),
              durationSec: lineToFix.durationSec,
              maxChars: lineToFix.maxChars,
            });
          }
        }
      }

      if (stillTooLong.length > 0) {
        console.log(`[CPS] Global strict pass 2: ${stillTooLong.length} line(s) still >= 9.5 CPS...`);
        for (let i = 0; i < stillTooLong.length; i += 20) {
          const chunk = stillTooLong.slice(i, i + 20);
          const strictMap = await compressLinesToTargetCps(chunk, modelChoice, true, rateLimiter, (msg) => {
            sendEvent({ type: 'status', statusMessage: msg, percent: 94 });
          });

          for (const lineToFix of chunk) {
            const origEvent = parsedData.events[lineToFix.index];
            const candidate = strictMap.get(lineToFix.index);
            if (candidate) {
              let finalCompressed = reconcileTags(origEvent.text, candidate);
              finalCompressed = contractBurmeseForCps(finalCompressed, lineToFix.maxChars);
              const prevText = translatedMap.get(lineToFix.index) || lineToFix.currentBurmese;
              if (calculateCps(finalCompressed, lineToFix.durationSec) < calculateCps(prevText, lineToFix.durationSec)) {
                translatedMap.set(lineToFix.index, finalCompressed);
              }
            }
          }
        }
      }
    }

    sendEvent({
      type: 'status',
      statusMessage: 'Rebuilding ASS subtitle file and verifying all formatting tags...',
      percent: 96,
    });

    console.log('[stream] Unifying nearby duplicate lines and reconstructing events...');
    // Final deterministic unification across all batches for nearby duplicate clusters
    unifyNearbyDuplicateTranslations(parsedData.events, translatedMap, nearbyDuplicateClusters);

    // Reconstruct updated events preserving original order and all metadata
    const updatedEvents: AssEvent[] = parsedData.events.map((event) => {
      if (translatedMap.has(event.index)) {
        return {
          ...event,
          text: translatedMap.get(event.index)!,
        };
      }
      return event;
    });

    console.log('[stream] Building final ASS content...');
    const watermarkEvents = hasWatermarkEvents(parsedData.events)
      ? []
      : buildWatermarkEvents(fileName, updatedEvents);

    // Build the final ASS content
    const finalAssContent = buildAss(parsedData, updatedEvents, true, watermarkEvents);

    console.log('[stream] Validating final ASS content...');
    // Validate the resulting ASS file
    const validation = validateAss(parsedData, finalAssContent, watermarkEvents.length);
    if (!validation.valid) {
      console.log('[stream] Validation failed:', validation.errors);
      sendEvent({
        type: 'error',
        message: `Validation failed: ${validation.errors.join('; ')}`,
      });
      res.end();
      return;
    }

    // Build synchronized finalPreview for UI SubtitleViewer (reflects global CPS compression & repairs)
    const finalPreview = translatableEvents.map((origEvent) => {
      const finalText = translatedMap.get(origEvent.index) ?? origEvent.text;
      const duration = getEventDuration(origEvent.start, origEvent.end);
      const isSign = isSignOrTitle(origEvent.name, origEvent.style, origEvent.text);
      return {
        index: origEvent.index,
        original: origEvent.text,
        translated: finalText,
        time: `${origEvent.start} - ${origEvent.end}`,
        start: origEvent.start,
        end: origEvent.end,
        durationSec: duration,
        cps: calculateCps(finalText, duration),
        isSignOrTitle: isSign,
        actor: origEvent.name || '',
        style: mapStyleName(origEvent.style, origEvent.name),
      };
    });

    // Output filename must end with .translated.ass
    const cleanBase = fileName.replace(/\.(ass|srt)$/i, '');
    const outputFileName = `${cleanBase}.translated.ass`;

    console.log('[stream] Sending complete event, filename:', outputFileName);
    sendEvent({
      type: 'complete',
      assContent: finalAssContent,
      finalPreview,
      outputFileName,
      eventCount: updatedEvents.length + watermarkEvents.length,
      translatableCount: translatableEvents.length,
      percent: 100,
    });

    console.log('[stream] Ending response stream.');
    res.end();
  } catch (err: unknown) {
    console.error('[stream] Error in translate stream:', err);
    const errorMsg = err instanceof Error ? err.message : String(err);
    const isDailyQuota = /RequestsPerDay|PerDay|daily/i.test(errorMsg) || (/429|RESOURCE_EXHAUSTED/i.test(errorMsg) && /retry in \d+h/i.test(errorMsg));
    const friendlyMessage = isDailyQuota
      ? `Free Tier Daily API Quota Exceeded (Google's daily request cap on your API key was reached). Please wait for the daily quota reset or select a Billing-enabled Paid API Key in Settings > Secrets. Details: ${errorMsg}`
      : `Translation process encountered an error: ${errorMsg}`;
    sendEvent({
      type: 'error',
      message: friendlyMessage,
    });
    res.end();
  }
});

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
