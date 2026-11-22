/**
 * Utility for preserving ASS override tags and line-break tokens.
 *
 * Rules:
 * - Never translate or modify ASS override tags inside curly braces:
 *   e.g. {\i1}, {\b1}, {\an8}, {\pos(100,200)}, {\fs30}, {\fnArial}, {\c&HFFFFFF&}, {\k10}, {\fad(200,200)}
 * - Preserve the literal ASS line-break token \N (and \n, \h if present).
 * - Keep tags intact even if LLM adjusts word order.
 */

export interface ExtractedTagData {
  originalText: string;
  tags: string[];
  hasLineBreak: boolean;
  cleanText: string;
}

/**
 * Extracts ASS tags (anything in {...}) and checks for line breaks (\N).
 */
export function extractTags(text: string): ExtractedTagData {
  const tags: string[] = [];
  const tagRegex = /\{[^}]*\}/g;
  let match: RegExpExecArray | null;

  while ((match = tagRegex.exec(text)) !== null) {
    tags.push(match[0]);
  }

  const hasLineBreak = text.includes('\\N') || text.includes('\\n');
  const cleanText = text.replace(tagRegex, '').replace(/\\[Nnh]/g, ' ').trim();

  return {
    originalText: text,
    tags,
    hasLineBreak,
    cleanText,
  };
}

/**
 * Normalizes inverted or misplaced ASS italic tag pairs:
 * - `{\i0}{\i1}Text` -> `{\i1}Text{\i0}` (or strips redundant leading `{\i0}` if `{\i0}` already follows `Text`)
 * - `{\i0}Text{\i1}` -> `{\i1}Text{\i0}`
 * - Collapses duplicate adjacent `{\i0}{\i0}` or `{\i1}{\i1}` tags without breaking legitimate mid-line switches.
 */
export function normalizeItalicTagPairs(text: string): string {
  if (!text) return text;

  let res = text;

  // Collapse identical adjacent italic tags (e.g. {\i0}{\i0} -> {\i0})
  res = res.replace(/\{\\i0\}(?:\s*\{\\i0\})+/g, '{\\i0}');
  res = res.replace(/\{\\i1\}(?:\s*\{\\i1\})+/g, '{\\i1}');

  // Handle inverted {\i0}{\i1} at the start (optionally preceded by other leading override tags)
  const invertedStartMatch = res.match(/^((?:\{[^}]*\})*?)\{\\i0\}\s*\{\\i1\}([\s\S]*)$/);
  if (invertedStartMatch) {
    const [, beforeTags, rest] = invertedStartMatch;
    if (/\{\\i0\}/.test(rest)) {
      res = `${beforeTags}{\\i1}${rest}`;
    } else {
      const trailingMatch = rest.match(/^(.*?)((?:\{[^}]*\})+)$/);
      if (trailingMatch) {
        res = `${beforeTags}{\\i1}${trailingMatch[1]}{\\i0}${trailingMatch[2]}`;
      } else {
        res = `${beforeTags}{\\i1}${rest}{\\i0}`;
      }
    }
  }

  // Handle combined {\i0\i1} inside a single tag block at start
  const combinedStartMatch = res.match(/^((?:\{[^}]*\})*?)\{([^}]*)\\i0\\i1([^}]*)\}([\s\S]*)$/);
  if (combinedStartMatch) {
    const [, beforeTags, innerPre, innerPost, rest] = combinedStartMatch;
    const rebuiltTag = `{${innerPre}\\i1${innerPost}}`;
    if (/\{\\i0\}/.test(rest)) {
      res = `${beforeTags}${rebuiltTag}${rest}`;
    } else {
      res = `${beforeTags}${rebuiltTag}${rest}{\\i0}`;
    }
  }

  // Handle swapped {\i0}Text{\i1} (closing tag at start, opening tag at end)
  const swappedEndsMatch = res.match(/^((?:\{[^}]*\})*?)\{\\i0\}([\s\S]*?)\{\\i1\}((?:\{[^}]*\})*)$/);
  if (swappedEndsMatch && !/\{\\i[01]\}/.test(swappedEndsMatch[2])) {
    const [, beforeTags, body, afterTags] = swappedEndsMatch;
    res = `${beforeTags}{\\i1}${body}{\\i0}${afterTags}`;
  }

  return res;
}

const INTERRUPTED_BURMESE_PAUSE_END_REGEX =
  /(?:ဒါပေမဲ့|ဆိုပေမဲ့|ဖြစ်ပေမဲ့|ဒါကြောင့်|မို့လို့|ဆိုတော့|တကယ်လို့|အဲဒါ|အင်း|အဲ|ဟို|ခဏ|ဘာလို့|မဟုတ်မှ|နောက်ပြီး|ပြီးတော့|ဒါဆို|အခုတော့|တကယ်တော့)\.(?=\s*(?:\\N|\\n|$|\{))/;

/**
 * Converts a stray single '.' at interrupted Burmese speech pauses (e.g. "ဒါပေမဲ့.") into Unicode ellipsis "…":
 * 1. Any mid-sentence Burmese char + single '.' followed by further text/clause ("ဒါပေမဲ့. ငါ") -> "ဒါပေမဲ့… ငါ"
 * 2. Any trailing Burmese char + single '.' when originalText had ellipsis/dash ("...", "…", "—", "--") -> "…"
 * 3. Any trailing Burmese pause conjunction + single '.' ("ဒါပေမဲ့.") -> "ဒါပေမဲ့…"
 * Leaves ASCII names/sentences ("Naruto.", "Hello world.") and decimals ("1.5") untouched for removeTrailingPeriods.
 */
export function fixInterruptedEllipsis(translatedText: string, originalText = ''): string {
  if (!translatedText) return translatedText;

  let res = translatedText;

  // Convert accidental double-dot '..' attached to Burmese script into '…'
  res = res.replace(/([\u1000-\u109F])(?<!\.)\.\.(?!\.)/g, '$1…');

  // 1. Mid-sentence interrupted single dot after Burmese char followed by continuing speech
  res = res.replace(
    /([\u1000-\u109F])(?<!\.)\.(?!\.)(?=(?:\s*(?:\\N|\\n)?\s*[^\s\d.{}]))/g,
    '$1…'
  );

  // 2. Trailing single dot when original line indicates pause/ellipsis/cut-off
  const origHasPauseMarker = /(?:\.{2,}|…|—|--|―|・)/.test(originalText);
  if (origHasPauseMarker) {
    res = res.replace(/([\u1000-\u109F])(?<!\.)\.(?!\.)(?=\s*(?:\\N|\\n|$|\{))/g, '$1…');
  } else if (INTERRUPTED_BURMESE_PAUSE_END_REGEX.test(res)) {
    // 3. Common Burmese interrupted pause words ending with single '.' even without explicit source marker
    res = res.replace(
      /(ဒါပေမဲ့|ဆိုပေမဲ့|ဖြစ်ပေမဲ့|ဒါကြောင့်|မို့လို့|ဆိုတော့|တကယ်လို့|အဲဒါ|အင်း|အဲ|ဟို|ခဏ|ဘာလို့|မဟုတ်မှ|နောက်ပြီး|ပြီးတော့|ဒါဆို|အခုတော့|တကယ်တော့)\.(?=\s*(?:\\N|\\n|$|\{))/g,
      '$1…'
    );
  }

  return res;
}

/**
 * Strips trailing English periods '.' from sentence ends, names, or before line breaks and tags,
 * as required by the anime translation rule: "စာသားနဲ့ နာမည်တွေရဲ့ အဆုံးမှာ ပုဒ်မ (.) လုံးဝ မထည့်ရ။"
 */
export function removeTrailingPeriods(text: string): string {
  // 1. Remove period before ASS tags or line breaks, or at the end
  let res = text.replace(/\.(?=\s*(?:\\N|\\n|$|\{))/g, '');
  // 2. Also remove period after names (e.g. "Naruto." -> "Naruto")
  res = res.replace(/([A-Za-z\u1000-\u109F])\.(?=\s)/g, '$1');
  return res.trim();
}

/**
 * Removes ASS line-break tokens (\N, \n) and literal newlines from spoken dialogue while
 * keeping all ASS override tags ({...}) untouched and joining visible clauses cleanly with a single space.
 */
export function stripLineBreaksForDialogue(text: string): string {
  if (!text) return text;

  const tokens = text.split(/(\{[^}]*\})/g);
  const tokenHasVisible = tokens.map((tok, idx) => {
    if (idx % 2 === 1) return false;
    return tok.replace(/\\[Nn]|\r\n|\r|\n/g, '').trim().length > 0;
  });

  const hasVisibleAfter: boolean[] = new Array(tokens.length).fill(false);
  let seenVisibleAfter = false;
  for (let i = tokens.length - 1; i >= 0; i--) {
    hasVisibleAfter[i] = seenVisibleAfter;
    if (tokenHasVisible[i]) seenVisibleAfter = true;
  }

  let seenVisibleBefore = false;
  let out = '';

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (i % 2 === 1) {
      out += tok;
      continue;
    }

    const replaced = tok.replace(/(?:[ \t]*(?:\\[Nn]|\r\n|\r|\n)[ \t]*)+/g, (match, offset: number, whole: string) => {
      const beforeInTok = whole.slice(0, offset).replace(/\\[Nn]|\r\n|\r|\n/g, '').trim().length > 0;
      const afterInTok = whole.slice(offset + match.length).replace(/\\[Nn]|\r\n|\r|\n/g, '').trim().length > 0;

      const hasBefore = seenVisibleBefore || beforeInTok;
      const hasAfter = afterInTok || hasVisibleAfter[i];

      return hasBefore && hasAfter ? ' ' : '';
    });

    out += replaced;
    if (tokenHasVisible[i]) {
      seenVisibleBefore = true;
    }
  }

  return out;
}

/**
 * Fixes unambiguous minor Burmese subtitle spelling errors (e.g. "ဂတိ" -> "ကတိ",
 * "တစ်ကယ်" -> "တကယ်", "ကျွှန်တော်/မ/ုပ်" -> "ကျွန်တော်/မ/ုပ်") while keeping
 * ASS {...} override tags and valid Pali compounds ("အဂတိ", "သုဂတိ", "ဒုဂ္ဂတိ") untouched.
 */
export function fixCommonBurmeseSpelling(text: string): string {
  if (!text || !/[\u1000-\u109F]/.test(text)) return text;

  return text.replace(/(\{[^}]*\})|([^{]+)/g, (segment, tag) => {
    if (tag) return tag;
    return segment
      .replace(/(?<![အသုဂ္])ဂတိ/g, 'ကတိ')
      .replace(/ကျွှန်တော်/g, 'ကျွန်တော်')
      .replace(/ကျွှန်မ/g, 'ကျွန်မ')
      .replace(/ကျွှန်ုပ်/g, 'ကျွန်ုပ်')
      .replace(/တစ်ကယ်/g, 'တကယ်')
      .replace(/ပရိတ်သတ်/g, 'ပရိသတ်');
  });
}

export interface ReconcileTagsOptions {
  /** Keep \N line breaks (true for Signs/Text/Title/Lyrics; false for Spoken Dialogue). Defaults to true. */
  preserveLineBreaks?: boolean;
}

/**
 * Reconciles translated text with original tags and optional line breaks.
 * Ensures:
 * 1. Any tags {..} that were at the beginning stay at the beginning.
 * 2. Any tags {..} that were at the end stay at the end.
 * 3. Any intermediate tags that got lost are preserved.
 * 4. Literal \N line breaks are preserved ONLY when `preserveLineBreaks` is true (Signs/Titles/Lyrics),
 *    and removed cleanly into a single line when `preserveLineBreaks` is false (Spoken Dialogue).
 * 5. Trailing periods on dialogue/names are removed per anime subtitle rule.
 */
export function reconcileTags(
  originalText: string,
  translatedText: string,
  options?: ReconcileTagsOptions
): string {
  const preserveLineBreaks = options?.preserveLineBreaks ?? true;

  if (!originalText) {
    const raw = translatedText || '';
    const lineNormalized = preserveLineBreaks ? raw : stripLineBreaksForDialogue(raw);
    return normalizeItalicTagPairs(fixCommonBurmeseSpelling(lineNormalized));
  }

  // If the model intentionally emptied the dialogue text (e.g., removing JP () SFX or groans),
  // keep the ASS line intact with empty dialogue text (plus any leading override tags if needed),
  // NEVER revert back to the untranslated Japanese text.
  if (translatedText.trim() === '') {
    const prefixMatch = originalText.match(/^(\{[^}]*\})+/);
    return prefixMatch ? prefixMatch[0] : '';
  }

  // Normalize any literal newlines the LLM might have returned into ASS \N (or strip later for dialogue)
  let result = translatedText
    .replace(/\r\n/g, '\\N')
    .replace(/\n/g, '\\N')
    .replace(/\r/g, '\\N');

  // Convert interrupted Burmese single-dot pauses ("ဒါပေမဲ့.") to "…" BEFORE stripping trailing periods
  result = fixInterruptedEllipsis(result, originalText);

  // Strip remaining trailing periods per prompt rule
  result = removeTrailingPeriods(result);

  // Extract all tags from original
  const tagRegex = /\{[^}]*\}/g;
  const originalTags: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = tagRegex.exec(originalText)) !== null) {
    originalTags.push(m[0]);
  }

  // Check prefix tags in original (tags occurring before any non-tag characters)
  const prefixMatch = originalText.match(/^(\{[^}]*\})+/);
  const originalPrefix = prefixMatch ? prefixMatch[0] : '';

  // Check suffix tags in original (tags occurring at the very end after non-tag characters)
  const suffixMatch = originalText.match(/(\{[^}]*\})+$/);
  const originalSuffix = suffixMatch ? suffixMatch[0] : '';

  // If translated text doesn't start with the original prefix tags, ensure they are prepended
  if (originalPrefix) {
    const transPrefixMatch = result.match(/^(\{[^}]*\})+/);
    const transPrefix = transPrefixMatch ? transPrefixMatch[0] : '';
    if (transPrefix !== originalPrefix) {
      // Remove any partial/misplaced duplicate of prefix
      if (result.startsWith(transPrefix) && transPrefix.length > 0) {
        result = result.slice(transPrefix.length);
      }
      result = originalPrefix + result;
    }
  }

  // If translated text doesn't end with original suffix tags, ensure they are appended
  if (originalSuffix && originalSuffix !== originalPrefix) {
    const transSuffixMatch = result.match(/(\{[^}]*\})+$/);
    const transSuffix = transSuffixMatch ? transSuffixMatch[0] : '';
    if (transSuffix !== originalSuffix) {
      if (result.endsWith(transSuffix) && transSuffix.length > 0) {
        result = result.slice(0, result.length - transSuffix.length);
      }
      result = result + originalSuffix;
    }
  }

  // Ensure any other original tags are not missing completely
  for (const tag of originalTags) {
    if (!result.includes(tag)) {
      // If tag is completely missing from translated text, insert it gracefully
      // For karaoke tags or style tags, place near the start after prefix
      if (originalPrefix && result.startsWith(originalPrefix)) {
        result = originalPrefix + tag + result.slice(originalPrefix.length);
      } else {
        result = tag + result;
      }
    }
  }

  if (!preserveLineBreaks) {
    return fixCommonBurmeseSpelling(
      normalizeItalicTagPairs(
        removeTrailingPeriods(fixInterruptedEllipsis(stripLineBreaksForDialogue(result), originalText))
      )
    );
  }

  // Check \N line breaks for Sign/Title/Lyrics
  const origLineBreakCount = (originalText.match(/\\N/g) || []).length;
  const currentLineBreakCount = (result.match(/\\N/g) || []).length;

  // If original had \N but result has none, and result has a space around the middle, convert or restore
  if (origLineBreakCount > 0 && currentLineBreakCount === 0) {
    // If there's multiple sentences or commas in result, or split near middle
    const parts = result.split(/(?<=[။၊])\s*/);
    if (parts.length > 1) {
      result = parts.join('\\N');
    } else {
      // Fallback: put \N in the middle space
      const mid = Math.floor(result.length / 2);
      const spaceIdx = result.indexOf(' ', mid);
      if (spaceIdx !== -1) {
        result = result.slice(0, spaceIdx) + '\\N' + result.slice(spaceIdx + 1);
      } else {
        // Just append \N to preserve token presence if un-splittable
        result = result + '\\N';
      }
    }
  }

  return fixCommonBurmeseSpelling(normalizeItalicTagPairs(result));
}
