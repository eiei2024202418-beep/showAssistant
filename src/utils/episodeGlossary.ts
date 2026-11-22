/**
 * Episode-Wide Character Pronoun & Glossary Lock utilities.
 * - Extracts per-actor gender + fixed self/plural pronouns (Speaker-Gender Strict Pronoun Lock)
 * - Extracts speaker->listener pronoun pairs
 * - Extracts consistent Place/Name (<English Root> + <Burmese Suffix>) and Rank/Title locks
 * - Provides deterministic validation and fallback enforcement so pronouns/glossary never drift across batches.
 */

import { detectForbiddenScripts, type AssEvent } from './subtitleParser';

export type SpeakerGender = 'male' | 'female' | 'unknown';

export interface SpeakerPronounProfile {
  actor: string;
  gender: SpeakerGender;
  fixedSelfPronoun: string;
  fixedPluralPronoun: string;
  toneNote?: string;
}

export interface CharacterPronounLock {
  speaker: string;
  listener: string;
  pronouns: string;
  toneNote?: string;
}

export interface GlossaryTermLock {
  sourceTerm: string;
  burmeseTerm: string;
  category?: 'title' | 'character' | 'place' | 'rank' | 'term';
}

export interface EpisodeLockProfile {
  speakerProfiles?: SpeakerPronounProfile[];
  pronounLocks: CharacterPronounLock[];
  glossaryTerms: GlossaryTermLock[];
}

export interface EpisodeSamplePayload {
  fileName: string;
  actors: string[];
  detectedProperPhrases: string[];
  sampleLines: { index: number; actor: string; style: string; text: string }[];
}

const GEOGRAPHIC_OR_RANK_PHRASE_REGEX =
  /\b(?:(?:The\s+)?Kingdom\s+of\s+[A-Z][a-zA-Z'-]+|(?:The\s+)?Empire\s+of\s+[A-Z][a-zA-Z'-]+|Crown\s+Prince(?:\s+[A-Z][a-zA-Z'-]+)?|His\s+Majesty|Your\s+Majesty|Her\s+Majesty|Your\s+Highness|His\s+Highness|General\s+[A-Z][a-zA-Z'-]+|Commander\s+[A-Z][a-zA-Z'-]+|[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?\s+(?:Gorge|Kingdom|Empire|Plains|Plateau|Valley|Forest|City|Town|Village|Castle|Fortress|Mountain|Mountains|Duchy|Territory|Border|Archduke))\b/g;

const STANDALONE_RANK_REGEX =
  /\b(?:General|Crown Prince|His Majesty|Your Majesty|Her Majesty|Your Highness|Commander|Archduke)\b/g;

/**
 * Collects a compact, representative sample of lines across the full episode:
 * - Guarantees at least 1 dialogue line per unique named actor (so every male/female speaker gets profiled)
 * - Fills remaining slots across opening, middle, and ending
 * - Extracts recurring multi-word place/country/rank phrases from ALL events
 */
export function buildEpisodeSample(
  events: AssEvent[],
  fileName: string,
  maxLines = 80
): EpisodeSamplePayload {
  const cleanEvents = events.filter((ev) => {
    const body = (ev.text || '').replace(/\{[^}]*\}/g, '').trim();
    return body.length > 0;
  });

  const actorSet = new Set<string>();
  const firstEventIdxByActor = new Map<string, number>();
  const phraseCounts = new Map<string, number>();

  for (let i = 0; i < cleanEvents.length; i++) {
    const ev = cleanEvents[i];
    const actor = (ev.name || '').trim();
    if (actor && !/^(sign|text|title|op|ed|lyrics|note)$/i.test(actor)) {
      actorSet.add(actor);
      const actorKey = actor.toLowerCase();
      if (!firstEventIdxByActor.has(actorKey)) {
        firstEventIdxByActor.set(actorKey, i);
      }
    }

    const plainText = (ev.text || '').replace(/\{[^}]*\}/g, '').replace(/\\N/g, ' ').trim();
    for (const m of plainText.matchAll(GEOGRAPHIC_OR_RANK_PHRASE_REGEX)) {
      const phrase = m[0].trim();
      phraseCounts.set(phrase, (phraseCounts.get(phrase) ?? 0) + 1);
    }
    for (const m of plainText.matchAll(STANDALONE_RANK_REGEX)) {
      const phrase = m[0].trim();
      phraseCounts.set(phrase, (phraseCounts.get(phrase) ?? 0) + 1);
    }
  }

  let sampled: AssEvent[] = [];
  if (cleanEvents.length <= maxLines) {
    sampled = cleanEvents;
  } else {
    const pickedIndices = new Set<number>();

    // Priority 1: Include at least 1 representative line for every named actor
    for (const idx of firstEventIdxByActor.values()) {
      if (pickedIndices.size < maxLines) {
        pickedIndices.add(idx);
      }
    }

    // Priority 2: Include opening lines
    const headCount = Math.min(30, Math.floor(maxLines * 0.4));
    for (let i = 0; i < headCount && pickedIndices.size < maxLines; i++) {
      pickedIndices.add(i);
    }

    // Priority 3: Distribute remaining slots evenly through the rest of the episode
    const remainingSlots = maxLines - pickedIndices.size;
    if (remainingSlots > 0) {
      const step = cleanEvents.length / remainingSlots;
      for (let k = 0; k < remainingSlots && pickedIndices.size < maxLines; k++) {
        let idx = Math.min(cleanEvents.length - 1, Math.floor(k * step));
        while (pickedIndices.has(idx) && idx + 1 < cleanEvents.length) {
          idx++;
        }
        pickedIndices.add(idx);
      }
    }

    sampled = [...pickedIndices]
      .sort((a, b) => a - b)
      .map((i) => cleanEvents[i]);
  }

  const detectedProperPhrases = [...phraseCounts.entries()]
    .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
    .slice(0, 30)
    .map(([phrase]) => phrase);

  return {
    fileName: fileName || 'episode.ass',
    actors: [...actorSet].slice(0, 45),
    detectedProperPhrases,
    sampleLines: sampled.map((ev) => ({
      index: ev.index,
      actor: (ev.name || '').trim(),
      style: ev.style,
      text: (ev.text || '').replace(/\{[^}]*\}/g, '').replace(/\\N/g, ' ').trim(),
    })),
  };
}

function normalizeGender(val: unknown): SpeakerGender {
  if (typeof val !== 'string') return 'unknown';
  const low = val.trim().toLowerCase();
  if (low === 'male' || low === 'm') return 'male';
  if (low === 'female' || low === 'f') return 'female';
  return 'unknown';
}

/**
 * Ensures self pronoun obeys gender lock and uses a single unambiguous pronoun variant
 * (e.g. "ကျွန်တော် OR ကျွန်ုပ်", never "ကျွန်တော် / ကျွန်ုပ်", and male never gets "ကျွန်မ").
 */
function normalizeFixedSelfPronoun(rawSelf: string, gender: SpeakerGender): string {
  const firstOption = rawSelf.split(/[/,|]/)[0].trim();

  if (gender === 'male') {
    if (!firstOption || /ကျွန်မ|ကျမ/.test(firstOption)) {
      return 'ကျွန်တော်';
    }
    if (/ကျွန်ုပ်/.test(firstOption)) return 'ကျွန်ုပ်';
    if (/ကျွန်တော်/.test(firstOption)) return 'ကျွန်တော်';
    if (/ကိုယ်/.test(firstOption)) return 'ကိုယ်';
    if (/ငါ/.test(firstOption)) return 'ငါ';
    return firstOption;
  }

  if (gender === 'female') {
    if (!firstOption || /ကျွန်တော်/.test(firstOption)) {
      return 'ကျွန်မ';
    }
    if (/ကျွန်မ|ကျမ/.test(firstOption)) return 'ကျွန်မ';
    if (/ကျွန်ုပ်/.test(firstOption)) return 'ကျွန်ုပ်';
    if (/ငါ/.test(firstOption)) return 'ငါ';
    return firstOption;
  }

  return firstOption;
}

function derivePluralPronoun(rawPlural: string, fixedSelf: string, gender: SpeakerGender): string {
  const firstPlural = rawPlural.split(/[/,|]/)[0].trim();

  if (fixedSelf === 'ကျွန်တော်') return 'ကျွန်တော်တို့';
  if (fixedSelf === 'ကျွန်ုပ်') return 'ကျွန်ုပ်တို့';
  if (fixedSelf === 'ကျွန်မ') return 'ကျွန်မတို့';
  if (fixedSelf === 'ကိုယ်') return gender === 'female' ? 'ကျွန်မတို့' : 'ကိုယ်တို့';
  if (fixedSelf === 'ငါ') return 'ငါတို့';

  if (firstPlural) {
    if (gender === 'male' && /ကျွန်မ|ကျမ/.test(firstPlural)) return 'ကျွန်တော်တို့';
    if (gender === 'female' && /ကျွန်တော်/.test(firstPlural)) return 'ကျွန်မတို့';
    return firstPlural.endsWith('တို့') ? firstPlural : `${firstPlural}တို့`;
  }

  return `${fixedSelf}တို့`;
}

/**
 * Sanitizes and deduplicates an extracted EpisodeLockProfile returned by the LLM.
 */
export function sanitizeEpisodeLockProfile(raw: unknown): EpisodeLockProfile {
  const empty: EpisodeLockProfile = { speakerProfiles: [], pronounLocks: [], glossaryTerms: [] };
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return empty;
  }

  const obj = raw as Record<string, unknown>;

  const speakerProfiles: SpeakerPronounProfile[] = [];
  const seenActors = new Set<string>();
  if (Array.isArray(obj.speakerProfiles)) {
    for (const item of obj.speakerProfiles) {
      if (!item || typeof item !== 'object') continue;
      const rec = item as Record<string, unknown>;
      const actor = typeof rec.actor === 'string' ? rec.actor.trim() : '';
      if (!actor) continue;
      const key = actor.toLowerCase();
      if (seenActors.has(key)) continue;
      seenActors.add(key);

      const gender = normalizeGender(rec.gender);
      const rawSelf = typeof rec.fixedSelfPronoun === 'string' ? rec.fixedSelfPronoun.trim() : '';
      const rawPlural = typeof rec.fixedPluralPronoun === 'string' ? rec.fixedPluralPronoun.trim() : '';
      let fixedSelfPronoun = normalizeFixedSelfPronoun(rawSelf, gender);
      if (!fixedSelfPronoun) continue;
      if (detectForbiddenScripts(fixedSelfPronoun).hasForbidden) {
        if (gender === 'male') fixedSelfPronoun = 'ကျွန်တော်';
        else if (gender === 'female') fixedSelfPronoun = 'ကျွန်မ';
        else continue;
      }

      let fixedPluralPronoun = derivePluralPronoun(rawPlural, fixedSelfPronoun, gender);
      if (detectForbiddenScripts(fixedPluralPronoun).hasForbidden) {
        fixedPluralPronoun = `${fixedSelfPronoun}တို့`;
      }
      const toneNote = typeof rec.toneNote === 'string' ? rec.toneNote.trim() : undefined;

      speakerProfiles.push({
        actor,
        gender,
        fixedSelfPronoun,
        fixedPluralPronoun,
        ...(toneNote ? { toneNote } : {}),
      });
    }
  }

  const pronounLocks: CharacterPronounLock[] = [];
  const seenPairs = new Set<string>();

  if (Array.isArray(obj.pronounLocks)) {
    for (const item of obj.pronounLocks) {
      if (!item || typeof item !== 'object') continue;
      const rec = item as Record<string, unknown>;
      const speaker = typeof rec.speaker === 'string' ? rec.speaker.trim() : '';
      const listener = typeof rec.listener === 'string' ? rec.listener.trim() : 'General';
      const pronouns = typeof rec.pronouns === 'string' ? rec.pronouns.trim() : '';
      const toneNote = typeof rec.toneNote === 'string' ? rec.toneNote.trim() : undefined;
      if (!speaker || !pronouns || detectForbiddenScripts(pronouns).hasForbidden) continue;
      const key = `${speaker.toLowerCase()}->${listener.toLowerCase()}`;
      if (seenPairs.has(key)) continue;
      seenPairs.add(key);
      pronounLocks.push({
        speaker,
        listener: listener || 'General',
        pronouns,
        ...(toneNote ? { toneNote } : {}),
      });
    }
  }

  const glossaryTerms: GlossaryTermLock[] = [];
  const seenTerms = new Set<string>();

  if (Array.isArray(obj.glossaryTerms)) {
    for (const item of obj.glossaryTerms) {
      if (!item || typeof item !== 'object') continue;
      const rec = item as Record<string, unknown>;
      const sourceTerm = typeof rec.sourceTerm === 'string' ? rec.sourceTerm.trim() : '';
      const burmeseTerm = typeof rec.burmeseTerm === 'string' ? rec.burmeseTerm.trim() : '';
      const cat = rec.category;
      const category =
        cat === 'title' || cat === 'character' || cat === 'place' || cat === 'rank' || cat === 'term'
          ? cat
          : undefined;
      // Never lock a glossary translation that still contains Kanji/Kana/CJK/Thai/Lao/foreign scripts
      if (!sourceTerm || !burmeseTerm || detectForbiddenScripts(burmeseTerm).hasForbidden) continue;
      const key = sourceTerm.toLowerCase();
      if (seenTerms.has(key)) continue;
      seenTerms.add(key);
      glossaryTerms.push({
        sourceTerm,
        burmeseTerm,
        ...(category ? { category } : {}),
      });
    }
  }

  return {
    speakerProfiles: speakerProfiles.slice(0, 45),
    pronounLocks: pronounLocks.slice(0, 25),
    glossaryTerms: glossaryTerms.slice(0, 40),
  };
}

/**
 * Looks up a speaker's locked gender & pronoun profile by ASS Actor name.
 */
export function findSpeakerProfile(
  actor: string,
  profile: EpisodeLockProfile | null | undefined
): SpeakerPronounProfile | undefined {
  if (!actor || !profile?.speakerProfiles?.length) return undefined;
  const trimmed = actor.trim().toLowerCase();
  if (!trimmed) return undefined;

  const exact = profile.speakerProfiles.find((s) => s.actor.trim().toLowerCase() === trimmed);
  if (exact) return exact;

  // Match base character name when ASS actor includes variant suffix like "Romelia (Flashback)"
  const baseActor = trimmed.replace(/\s*\([^)]*\)\s*$/, '').trim();
  if (baseActor && baseActor !== trimmed) {
    return profile.speakerProfiles.find((s) => s.actor.trim().toLowerCase() === baseActor);
  }
  return undefined;
}

/**
 * Generates an inline per-line speaker lock hint injected directly into each batch item
 * so the model sees the speaker's exact locked gender & self/plural pronouns next to the text.
 */
export function formatLineSpeakerHint(
  actor: string,
  profile: EpisodeLockProfile | null | undefined
): string | undefined {
  const sp = findSpeakerProfile(actor, profile);
  if (!sp) return undefined;

  const genderLabel = sp.gender === 'male' ? 'MALE' : sp.gender === 'female' ? 'FEMALE' : 'SPEAKER';
  const forbiddenList =
    sp.gender === 'male'
      ? sp.fixedSelfPronoun === 'ကျွန်တော်'
        ? 'ကျွန်မ, ကျွန်မတို့, ကျွန်ုပ်, ကျွန်ုပ်တို့, ရှင့်'
        : sp.fixedSelfPronoun === 'ကျွန်ုပ်'
        ? 'ကျွန်မ, ကျွန်မတို့, ကျွန်တော်, ကျွန်တော်တို့, ရှင့်'
        : 'ကျွန်မ, ကျွန်မတို့, ရှင့်'
      : sp.gender === 'female'
      ? sp.fixedSelfPronoun === 'ကျွန်မ'
        ? 'ကျွန်တော်, ကျွန်တော်တို့, ကျွန်ုပ်, ကျွန်ုပ်တို့, ခင်ဗျား, ခင်ဗျာ'
        : 'ကျွန်တော်, ကျွန်တော်တို့, ခင်ဗျား, ခင်ဗျာ'
      : '';

  return `${genderLabel} | Self: ${sp.fixedSelfPronoun} / ${sp.fixedPluralPronoun}${
    forbiddenList ? ` | FORBIDDEN: ${forbiddenList}` : ''
  }`;
}

/**
 * Strips ASS override tags and quoted speech snippets so quoted lines don't false-trigger gender checks.
 */
function getUnquotedVisibleSegments(text: string): string {
  return text
    .replace(/\{[^}]*\}/g, '')
    .replace(/\\[Nnh]/g, ' ')
    .replace(/"[^"]*"/g, ' ')
    .replace(/“[^”]*”/g, ' ')
    .replace(/「[^」]*」/g, ' ')
    .replace(/『[^』]*』/g, ' ');
}

export interface SpeakerViolationResult {
  valid: boolean;
  reason?: 'male_used_female_pronoun' | 'female_used_male_pronoun' | 'self_pronoun_drift';
  detail?: string;
}

/**
 * Deterministic check catching cross-gender pronoun leaks or single-character self-pronoun drift:
 * - Male actors MUST NEVER use female `ကျွန်မ` / `ကျွန်မတို့`
 * - Female actors MUST NEVER use `ခင်ဗျား` / `ကျွန်တော်` / `ကျွန်တော်တို့`
 * - Characters locked to `ကျွန်တော်` MUST NOT drift to `ကျွန်ုပ်` (and vice versa)
 */
export function detectSpeakerPronounViolation(
  actor: string,
  translatedText: string,
  profile: EpisodeLockProfile | null | undefined
): SpeakerViolationResult {
  const sp = findSpeakerProfile(actor, profile);
  if (!sp || !translatedText) return { valid: true };

  const inspectable = getUnquotedVisibleSegments(translatedText);
  if (!inspectable.trim()) return { valid: true };

  if (sp.gender === 'male') {
    if (/(?:ကျွန်မတို့|ကျွန်မ|ကျမတို့|ကျမ)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'male_used_female_pronoun',
        detail: `Male actor "${sp.actor}" used female self-pronoun (must use "${sp.fixedSelfPronoun}" / "${sp.fixedPluralPronoun}")`,
      };
    }
    if (sp.fixedSelfPronoun === 'ကျွန်တော်' && /(?:ကျွန်ုပ်တို့|ကျွန်ုပ်)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'self_pronoun_drift',
        detail: `Actor "${sp.actor}" drifted to "ကျွန်ုပ်" (locked to "ကျွန်တော်")`,
      };
    }
    if (sp.fixedSelfPronoun === 'ကျွန်ုပ်' && /(?:ကျွန်တော်တို့|ကျွန်တော်)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'self_pronoun_drift',
        detail: `Actor "${sp.actor}" drifted to "ကျွန်တော်" (locked to "ကျွန်ုပ်")`,
      };
    }
  } else if (sp.gender === 'female') {
    if (/(?:ကျွန်တော်တို့|ကျွန်တော်|ခင်ဗျားတို့|ခင်ဗျား|ခင်ဗျာ)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'female_used_male_pronoun',
        detail: `Female actor "${sp.actor}" used male-register pronoun "ကျွန်တော်"/"ခင်ဗျား"`,
      };
    }
    if (sp.fixedSelfPronoun === 'ကျွန်မ' && /(?:ကျွန်ုပ်တို့|ကျွန်ုပ်)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'self_pronoun_drift',
        detail: `Female actor "${sp.actor}" drifted to "ကျွန်ုပ်" (locked to "ကျွန်မ")`,
      };
    }
  } else {
    // Unknown gender: still enforce fixed self-pronoun lock between ကျွန်တော် <-> ကျွန်ုပ် <-> ကျွန်မ
    if (sp.fixedSelfPronoun === 'ကျွန်တော်' && /(?:ကျွန်ုပ်တို့|ကျွန်ုပ်)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'self_pronoun_drift',
        detail: `Actor "${sp.actor}" drifted to "ကျွန်ုပ်" (locked to "ကျွန်တော်")`,
      };
    }
    if (sp.fixedSelfPronoun === 'ကျွန်ုပ်' && /(?:ကျွန်တော်တို့|ကျွန်တော်)/.test(inspectable)) {
      return {
        valid: false,
        reason: 'self_pronoun_drift',
        detail: `Actor "${sp.actor}" drifted to "ကျွန်တော်" (locked to "ကျွန်ုပ်")`,
      };
    }
  }

  return { valid: true };
}

/**
 * Applies a segment-safe replacement on unquoted, non-tag text while preserving `{...}` ASS tags and quoted snippets.
 */
function transformUnquotedVisibleSegments(text: string, fn: (seg: string) => string): string {
  const tokenRegex = /(\{[^}]*\}|"[^"]*"|“[^”]*”|「[^」]*」|『[^』]*』)/g;
  return text
    .split(tokenRegex)
    .map((part, idx) => (idx % 2 === 1 ? part : fn(part)))
    .join('');
}

/**
 * Deterministic fallback repair for any residual speaker-gender / self-pronoun violations
 * after targeted LLM repair rounds complete.
 */
export function applyDeterministicSpeakerPronounFix(
  actor: string,
  translatedText: string,
  profile: EpisodeLockProfile | null | undefined
): string {
  const sp = findSpeakerProfile(actor, profile);
  if (!sp || !translatedText) return translatedText;

  return transformUnquotedVisibleSegments(translatedText, (seg) => {
    let s = seg;
    if (sp.gender === 'male') {
      const targetSelf =
        sp.fixedSelfPronoun && !/ကျွန်မ|ကျမ/.test(sp.fixedSelfPronoun)
          ? sp.fixedSelfPronoun
          : 'ကျွန်တော်';
      const targetPlural =
        sp.fixedPluralPronoun && !/ကျွန်မ|ကျမ/.test(sp.fixedPluralPronoun)
          ? sp.fixedPluralPronoun
          : `${targetSelf}တို့`;

      s = s.replace(/(?:ကျွန်မတို့|ကျမတို့)/g, targetPlural);
      s = s.replace(/(?:ကျွန်မ|ကျမ)/g, targetSelf);

      if (targetSelf === 'ကျွန်တော်') {
        s = s.replace(/ကျွန်ုပ်တို့/g, 'ကျွန်တော်တို့').replace(/ကျွန်ုပ်/g, 'ကျွန်တော်');
      } else if (targetSelf === 'ကျွန်ုပ်') {
        s = s.replace(/ကျွန်တော်တို့/g, 'ကျွန်ုပ်တို့').replace(/ကျွန်တော်/g, 'ကျွန်ုပ်');
      }
    } else if (sp.gender === 'female') {
      const targetSelf =
        sp.fixedSelfPronoun && !/ကျွန်တော်/.test(sp.fixedSelfPronoun)
          ? sp.fixedSelfPronoun
          : 'ကျွန်မ';
      const targetPlural =
        sp.fixedPluralPronoun && !/ကျွန်တော်/.test(sp.fixedPluralPronoun)
          ? sp.fixedPluralPronoun
          : `${targetSelf}တို့`;

      s = s.replace(/ကျွန်တော်တို့/g, targetPlural);
      s = s.replace(/ကျွန်တော်/g, targetSelf);
      s = s.replace(/ခင်ဗျားတို့/g, 'ရှင်တို့');
      s = s.replace(/(?:ခင်ဗျား|ခင်ဗျာ)/g, 'ရှင်');

      if (targetSelf === 'ကျွန်မ') {
        s = s.replace(/ကျွန်ုပ်တို့/g, 'ကျွန်မတို့').replace(/ကျွန်ုပ်/g, 'ကျွန်မ');
      }
    } else {
      if (sp.fixedSelfPronoun === 'ကျွန်တော်') {
        s = s.replace(/ကျွန်ုပ်တို့/g, 'ကျွန်တော်တို့').replace(/ကျွန်ုပ်/g, 'ကျွန်တော်');
      } else if (sp.fixedSelfPronoun === 'ကျွန်ုပ်') {
        s = s.replace(/ကျွန်တော်တို့/g, 'ကျွန်ုပ်တို့').replace(/ကျွန်တော်/g, 'ကျွန်ုပ်');
      }
    }
    return s;
  });
}

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Replaces any source multi-word place/rank terms accidentally left verbatim in English
 * inside Burmese dialogue with the locked Burmese glossary term.
 */
export function enforceLockedGlossaryPostProcess(
  translatedText: string,
  profile: EpisodeLockProfile | null | undefined
): string {
  if (!translatedText || !profile?.glossaryTerms?.length) return translatedText;
  if (!/[\u1000-\u109F]/.test(translatedText)) return translatedText;

  const eligible = [...profile.glossaryTerms]
    .filter((g) => {
      const src = g.sourceTerm.trim();
      const dst = g.burmeseTerm.trim();
      if (!src || !dst || src.toLowerCase() === dst.toLowerCase()) return false;
      if (detectForbiddenScripts(dst).hasForbidden) return false;
      const isMultiWord = /\s/.test(src);
      const isLockedRankOrPlaceWord =
        (g.category === 'rank' || g.category === 'place') && src.length >= 6 && /^[A-Z]/.test(src);
      return isMultiWord || isLockedRankOrPlaceWord;
    })
    .sort((a, b) => b.sourceTerm.length - a.sourceTerm.length);

  if (eligible.length === 0) return translatedText;

  return translatedText.replace(/(\{[^}]*\})|([^{]+)/g, (segment, tag) => {
    if (tag) return tag;
    let out = segment;
    for (const entry of eligible) {
      const escaped = escapeRegExp(entry.sourceTerm.trim());
      const flags = /\s/.test(entry.sourceTerm) ? 'gi' : 'g';
      const rx = new RegExp(`(?<![A-Za-z0-9])${escaped}(?![A-Za-z0-9])`, flags);
      out = out.replace(rx, entry.burmeseTerm);
    }
    return out;
  });
}

/**
 * Formats an EpisodeLockProfile into a concise system-prompt instruction block
 * injected across EVERY batch so pronouns and glossary terms stay consistent episode-wide.
 */
export function formatEpisodeLockInstruction(profile: EpisodeLockProfile | null | undefined): string {
  if (!profile) return '';
  const hasSpeakers = (profile.speakerProfiles?.length ?? 0) > 0;
  const hasPronouns = profile.pronounLocks.length > 0;
  const hasGlossary = profile.glossaryTerms.length > 0;
  if (!hasSpeakers && !hasPronouns && !hasGlossary) return '';

  const sections: string[] = [
    '\n\n# 🔒 EPISODE-WIDE SPEAKER GENDER, PRONOUN & GLOSSARY LOCK (MANDATORY FOR EVERY BATCH)',
    'Batch တိုင်းတွင် "actor" နာမည်ကို သေချာစစ်ဆေးပြီး အောက်ပါ Gender/Pronoun Lock နှင့် Glossary Lock များကို 100% EXACTLY လိုက်နာရမည်:',
  ];

  if (hasSpeakers && profile.speakerProfiles) {
    sections.push('\n1. Speaker-Gender Strict Pronoun Lock (Actor -> Gender | Fixed Self / Plural | FORBIDDEN):');
    sections.push(
      '⚠️ ယောက်ျားလေး Actor သည် "ကျွန်မ / ကျွန်မတို့ / ရှင့်" ကို လုံးဝ မသုံးရ။ မိန်းကလေး Actor သည် "ခင်ဗျား / ကျွန်တော် / ကျွန်တော်တို့" ကို လုံးဝ မသုံးရ။ Actor တစ်ဦးအတွက် သတ်မှတ်ထားသော Self-Pronoun (ဥပမာ- ကျွန်တော် OR ကျွန်ုပ် OR ကျွန်မ) တစ်မျိုးတည်းကိုသာ အစအဆုံး သုံးရမည်:'
    );
    for (const sp of profile.speakerProfiles) {
      const genderTag = sp.gender.toUpperCase();
      const note = sp.toneNote ? ` [${sp.toneNote}]` : '';
      const forbidden =
        sp.gender === 'male'
          ? 'NEVER use ကျွန်မ/ကျွန်မတို့'
          : sp.gender === 'female'
          ? 'NEVER use ခင်ဗျား/ကျွန်တော်/ကျွန်တော်တို့'
          : 'Keep fixed self-pronoun';
      sections.push(
        `- ${sp.actor} (${genderTag}): ONLY Self="${sp.fixedSelfPronoun}", Plural="${sp.fixedPluralPronoun}" (${forbidden})${note}`
      );
    }
  }

  if (hasPronouns) {
    sections.push('\n2. Character Listener Pronoun Pairs (Speaker -> Listener: Locked Pronouns):');
    for (const p of profile.pronounLocks) {
      const note = p.toneNote ? ` [${p.toneNote}]` : '';
      sections.push(`- ${p.speaker} -> ${p.listener}: ${p.pronouns}${note}`);
    }
  }

  if (hasGlossary) {
    sections.push('\n3. Locked Episode Places, Ranks & Titles (STRICT 1-FORM RULE — NEVER MIX OR TRANSLITERATE PROPER ROOTS):');
    sections.push(
      '⚠️ နေရာ/နိုင်ငံနာမည် အရင်းကို Romaji/English အတိုင်း ထားပြီး နောက်ဆက်တွဲကိုသာ မြန်မာလို တစ်မျိုးတည်း ရေးပါ (ဥပမာ- Guillier ချောက်ကမ်းပါး, Lionel နိုင်ငံတော်, Dacan လွင်ပြင်)။ ရာထူးနာမည်များ (General -> ဗိုလ်ချုပ်, Crown Prince -> အိမ်ရှေ့မင်းသား, His Majesty -> ဘုရင်မင်းမြတ်) ကိုလည်း မြန်မာလို တစ်ညီတည်း ပြန်ပါ:'
    );
    for (const g of profile.glossaryTerms) {
      const catTag = g.category ? ` (${g.category})` : '';
      sections.push(`- "${g.sourceTerm}" -> "${g.burmeseTerm}"${catTag}`);
    }
  }

  return sections.join('\n');
}
