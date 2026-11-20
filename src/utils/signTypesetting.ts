/**
 * Advanced ASS Sign & Typesetting post-processing engine.
 * Supports three modes:
 * 1. `smart_hybrid`: Auto-fits Burmese text inside {\p1} vector boxes (measuring box width,
 *    auto-wrapping \N, applying \fscx85-90, and expanding box width +15% if needed),
 *    enforces PlayResY-proportional minimum readable Burmese font size with bold removal & thin borders,
 *    and upgrades \bord0 overlapping disclaimers to BorderStyle=3 (`Sign_Box`) opaque background boxes.
 * 2. `clean_top`: Strips complex typesetting override tags, comments out {\p1} vector masks,
 *    renders signs uniformly in `Sign_Basic`, and handles 4+ simultaneous signs via `|` auto-merge
 *    into 2-3 tidy lines or `Comment:` archival.
 * 3. `preserve_original`: Keeps original Crunchyroll/Erai-raws typesetting tags scaled to target PlayRes.
 */

import type { AssEvent } from './subtitleParser';

export type SignTypesettingMode = 'smart_hybrid' | 'clean_top' | 'preserve_original';
export type DenseSignStrategy = 'merge' | 'comment' | 'keep';

export interface SignTypesettingOptions {
  mode: SignTypesettingMode;
  denseSignStrategy?: DenseSignStrategy;
  playResX?: number;
  playResY?: number;
  /** Optional map of lowercased source Style name -> ASS Alignment (1..9) */
  styleAlignments?: Record<string, number>;
}

export interface DrawingBoxBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
  screenLeft: number;
  screenRight: number;
  screenTop: number;
  screenBottom: number;
}

const myGraphemeSegmenter = new Intl.Segmenter('my', { granularity: 'grapheme' });

function roundAss(value: number, decimals = 2): string {
  if (!Number.isFinite(value)) return '0';
  const rounded = Number(value.toFixed(decimals));
  return Object.is(rounded, -0) ? '0' : String(rounded);
}

function parseTimeSec(timeStr: string): number {
  const parts = (timeStr || '').trim().split(/[:.]/);
  if (parts.length < 4) return 0;
  return (
    (parseInt(parts[0], 10) || 0) * 3600 +
    (parseInt(parts[1], 10) || 0) * 60 +
    (parseInt(parts[2], 10) || 0) +
    (parseInt(parts[3], 10) || 0) / 100
  );
}

/**
 * PlayResY-proportional minimum readable Burmese font size (~3.2% of PlayResY):
 * - 360p -> 11.5 (inside user-requested 3.0%–3.3% / \fs11-12 range)
 * - 720p -> 23.0
 * - 1080p -> 34.6
 */
export function getMinReadableBurmeseFontSize(playResY = 360): number {
  const safeResY = playResY > 0 ? playResY : 360;
  return Math.max(9, Math.round(safeResY * 0.032 * 10) / 10);
}

/**
 * Threshold below which a Burmese sign font is considered "small" (~5.6% of PlayResY: ~20 at 360p, ~60 at 1080p)
 * and benefits from disabling Bold (\b0) and using a thin proportional border.
 */
export function getSmallSignFontThreshold(playResY = 360): number {
  const safeResY = playResY > 0 ? playResY : 360;
  return Math.round(safeResY * 0.056 * 10) / 10;
}

/**
 * Computes crisp thin outline width proportional to font size (~4.0% of font size, e.g. ~0.46-0.50 at fs11.5 on 360p).
 */
export function getProportionalThinBorder(fontSize: number, playResY = 360): number {
  const minBord = Math.max(0.4, (playResY / 360) * 0.45);
  const maxBord = Math.max(0.8, (playResY / 360) * 0.85);
  const raw = fontSize * 0.04;
  return Math.round(Math.min(maxBord, Math.max(minBord, raw)) * 100) / 100;
}

/**
 * Strips noisy inline typesetting override tags from a Sign line before sending it to the LLM
 * so word-order shifts between Japanese/English and Burmese never corrupt mid-word tags or lose \N breaks.
 */
export function prepareSignTextForTranslation(text: string): string {
  if (!text) return '';
  return text
    .replace(/\{[^}]*\}/g, '')
    .replace(/\\h/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/[ \t]*\\N[ \t]*/g, '\\N')
    .trim();
}

/**
 * Extracts only the clean leading ASS override block (dropping fragmented mid-word color/size overrides
 * that break Burmese word order) while preserving \N structure for Smart Hybrid signs.
 */
export function extractLeadingTagBlock(text: string): string {
  const match = (text || '').match(/^(\{[^}]*\})+/);
  if (!match) return '';
  // Merge multiple adjacent {...}{...} leading blocks into a single clean {...} block
  const inner = match[0].replace(/\}\{/g, '');
  return inner;
}

const DRAWING_TAG_REGEX = /\{[^}]*\\p[1-9]\d*(?![a-zA-Z])/;

/**
 * Parses an ASS {\p1} vector drawing string (e.g. `{\pos(180,100)\an7\p1}m 0 0 l 130 0 l 130 195 l 0 195`)
 * to extract its bounding box and approximate screen coordinates.
 */
export function parseDrawingBoxBounds(
  text: string,
  playResX = 640,
  playResY = 360
): DrawingBoxBounds | null {
  if (!text || !DRAWING_TAG_REGEX.test(text)) return null;

  // Strip tags to get vector drawing coordinates
  const drawingBody = text.replace(/\{[^}]*\}/g, ' ').trim();
  const tokens = drawingBody.match(/[a-zA-Z]|-?\d+(?:\.\d+)?/g);
  if (!tokens) return null;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  let expectX = true;
  let coordCount = 0;

  for (const tok of tokens) {
    if (/^[a-zA-Z]$/.test(tok)) {
      expectX = true;
      continue;
    }
    const val = parseFloat(tok);
    if (!Number.isFinite(val)) continue;
    if (expectX) {
      if (val < minX) minX = val;
      if (val > maxX) maxX = val;
      coordCount++;
    } else {
      if (val < minY) minY = val;
      if (val > maxY) maxY = val;
    }
    expectX = !expectX;
  }

  if (coordCount < 2 || !Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null;
  }

  const width = Math.max(0, maxX - minX);
  const height = Math.max(0, maxY - minY);
  if (width < 10 || height < 8) return null;

  // Determine screen placement via \pos(x,y) and \an1..9 (default \an7 for drawing if pos exists, else \an8/2)
  const posMatch = text.match(/\\pos\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)/);
  const anMatch = text.match(/\\an([1-9])/);
  const an = anMatch ? parseInt(anMatch[1], 10) : 7;

  const hasPos = Boolean(posMatch);
  const px = posMatch ? parseFloat(posMatch[1]) : minX > 5 || minY > 5 ? 0 : playResX / 2;
  const py = posMatch ? parseFloat(posMatch[2]) : minX > 5 || minY > 5 ? 0 : playResY / 2;

  const hAlign = (an - 1) % 3; // 0 = left (1,4,7), 1 = center (2,5,8), 2 = right (3,6,9)
  const vAlign = Math.floor((an - 1) / 3); // 0 = bottom (1,2,3), 1 = middle (4,5,6), 2 = top (7,8,9)

  const screenLeft = !hasPos && (minX > 5 || minY > 5)
    ? minX
    : hAlign === 0
    ? px + minX
    : hAlign === 1
    ? px - width / 2
    : px - width;
  const screenTop = !hasPos && (minX > 5 || minY > 5)
    ? minY
    : vAlign === 2
    ? py + minY
    : vAlign === 1
    ? py - height / 2
    : py - height;

  return {
    minX,
    minY,
    maxX,
    maxY,
    width,
    height,
    screenLeft,
    screenRight: screenLeft + width,
    screenTop,
    screenBottom: screenTop + height,
  };
}

/**
 * Expands the horizontal X and/or vertical Y dimensions of a {\p1} vector drawing box
 * around its anchor origin (top-left for m 0 0 boxes, or center for symmetric shapes)
 * so wider or taller wrapped Burmese text stays cleanly inside the box.
 */
export function expandDrawingBoxDimensions(
  text: string,
  widthFactor = 1.15,
  heightFactor = 1
): string {
  if (
    !text ||
    !DRAWING_TAG_REGEX.test(text) ||
    (Math.abs(widthFactor - 1) < 1e-4 && Math.abs(heightFactor - 1) < 1e-4)
  ) {
    return text;
  }
  const bounds = parseDrawingBoxBounds(text);
  if (!bounds) return text;

  const originX = bounds.minX < 0 && bounds.maxX > 0 ? (bounds.minX + bounds.maxX) / 2 : bounds.minX;
  const originY = bounds.minY < 0 && bounds.maxY > 0 ? (bounds.minY + bounds.maxY) / 2 : bounds.minY;
  let inDrawing = false;

  return text.replace(/(\{[^}]*\})|([^{]+)/g, (_seg, tagBlock: string | undefined, plain: string | undefined) => {
    if (tagBlock !== undefined) {
      const pMatches = [...tagBlock.matchAll(/\\p(\d+)/g)];
      if (pMatches.length > 0) {
        inDrawing = parseInt(pMatches[pMatches.length - 1][1], 10) > 0;
      }
      return tagBlock;
    }
    if (!inDrawing || !plain) return plain ?? '';

    let expectX = true;
    return plain.replace(/([a-zA-Z])|(-?\d+(?:\.\d+)?)/g, (_t, cmd: string | undefined, numStr: string | undefined) => {
      if (cmd) {
        expectX = true;
        return cmd;
      }
      const val = parseFloat(numStr!);
      const nextVal = expectX
        ? originX + (val - originX) * widthFactor
        : originY + (val - originY) * heightFactor;
      expectX = !expectX;
      return roundAss(nextVal, 2);
    });
  });
}

export function expandDrawingBoxWidth(text: string, factor = 1.15): string {
  return expandDrawingBoxDimensions(text, factor, 1);
}

const ZERO_WIDTH_BURMESE_MARK_REGEX =
  /^[\u102D-\u1030\u1032\u1036\u1037\u1039\u103A\u103D\u103E\u1082\u1085\u1086\u108D]$/;
const WIDE_BURMESE_CONSONANT_REGEX = /^[\u1000\u100F\u1010\u1011\u1019\u101C\u101E\u1025\u1026\u103F\u104C-\u104F]$/;
const BURMESE_CONSONANT_OR_IND_VOWEL_REGEX = /^[\u1000-\u102A\u103F\u104E\uAA60-\uAA7F\uA9E0-\uA9FF]$/;

/**
 * Estimates rendered horizontal pixel width of a single visible subtitle line (without tags or \N)
 * in `Myanmar Text` at the given ASS `fontSize` and horizontal scale `fscx` (%).
 *
 * Strictly treats Burmese zero-width combining marks
 * ([\u102D-\u1030\u1032\u1036\u1037\u1039\u103A\u103D\u103E]) and stacked virama subjoined consonants
 * as Width = 0 so multi-mark syllables like "ကြောင်း" are measured at real glyph width
 * without causing premature wrapping.
 */
export function estimateBurmeseLineWidthPx(line: string, fontSize: number, fscx = 100): number {
  const clean = (line || '').replace(/\{[^}]*\}/g, '').replace(/\\h/g, ' ');
  if (!clean) return 0;

  let widthUnits = 0;
  let afterVirama = false;

  for (const ch of clean) {
    if (ch === '\u1039') {
      // Virama stacks the next consonant vertically underneath (zero horizontal advance)
      afterVirama = true;
      continue;
    }

    if (afterVirama) {
      afterVirama = false;
      if (BURMESE_CONSONANT_OR_IND_VOWEL_REGEX.test(ch)) {
        continue; // Stacked subjoined consonant -> width 0
      }
    }

    if (ZERO_WIDTH_BURMESE_MARK_REGEX.test(ch)) {
      continue; // Width = 0
    }

    if (/\s/.test(ch)) {
      widthUnits += 0.24;
    } else if (ch === '\u103C') {
      // Ya-yit (ြ) wraps around left edge of base consonant
      widthUnits += 0.22;
    } else if (ch === '\u103B') {
      // Ya-pin (ျ) slight right extension
      widthUnits += 0.16;
    } else if (ch === '\u1031') {
      // Tha-way-htoe (ေ) left spacing vowel
      widthUnits += 0.26;
    } else if (ch === '\u102B' || ch === '\u102C') {
      // Yei-cha / Mauk-cha (ါ, ာ)
      widthUnits += 0.25;
    } else if (ch === '\u1038') {
      // Wisarga (း)
      widthUnits += 0.16;
    } else if (WIDE_BURMESE_CONSONANT_REGEX.test(ch)) {
      widthUnits += 0.45;
    } else if (BURMESE_CONSONANT_OR_IND_VOWEL_REGEX.test(ch)) {
      widthUnits += 0.41;
    } else if (/^[\u1040-\u1049]$/.test(ch)) {
      widthUnits += 0.42;
    } else if (ch === '\u104A' || ch === '\u104B') {
      widthUnits += 0.22;
    } else if (/^[A-Z0-9]$/.test(ch)) {
      widthUnits += 0.5;
    } else if (/^[a-z]$/.test(ch)) {
      widthUnits += 0.4;
    } else if (/^[.,:;!|'"()\[\]\-\/]$/.test(ch)) {
      widthUnits += 0.24;
    } else {
      widthUnits += 0.42;
    }
  }

  return widthUnits * fontSize * (fscx / 100);
}

/**
 * Wraps visible Burmese text into lines separated by `\N` so no line exceeds `maxWidthPx`.
 * Preserves existing intentional `\N` breaks and splits overlong lines cleanly at spaces,
 * punctuation, or Burmese grapheme clusters.
 */
export function wrapBurmeseTextToWidth(
  visibleText: string,
  maxWidthPx: number,
  fontSize: number,
  fscx = 100
): string {
  if (!visibleText || maxWidthPx <= 0 || fontSize <= 0) return visibleText;

  const existingLines = visibleText.split(/\\N/g);
  const wrappedAll: string[] = [];

  for (const rawLine of existingLines) {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      wrappedAll.push('');
      continue;
    }

    if (estimateBurmeseLineWidthPx(trimmed, fontSize, fscx) <= maxWidthPx) {
      wrappedAll.push(trimmed);
      continue;
    }

    // Split by spaces first; if a single space-free Burmese phrase exceeds maxWidthPx, break by graphemes
    const words = trimmed.split(/\s+/);
    let currentLine = '';

    const pushGraphemeBroken = (longWord: string) => {
      const graphemes = [...myGraphemeSegmenter.segment(longWord)].map((s) => s.segment);
      let chunk = '';
      for (const g of graphemes) {
        const candidate = chunk + g;
        if (chunk && estimateBurmeseLineWidthPx(candidate, fontSize, fscx) > maxWidthPx) {
          wrappedAll.push(chunk);
          chunk = g;
        } else {
          chunk = candidate;
        }
      }
      if (chunk) {
        currentLine = chunk;
      }
    };

    for (const word of words) {
      if (!currentLine) {
        if (estimateBurmeseLineWidthPx(word, fontSize, fscx) > maxWidthPx) {
          pushGraphemeBroken(word);
        } else {
          currentLine = word;
        }
        continue;
      }

      const candidate = `${currentLine} ${word}`;
      if (estimateBurmeseLineWidthPx(candidate, fontSize, fscx) <= maxWidthPx) {
        currentLine = candidate;
      } else {
        wrappedAll.push(currentLine);
        currentLine = '';
        if (estimateBurmeseLineWidthPx(word, fontSize, fscx) > maxWidthPx) {
          pushGraphemeBroken(word);
        } else {
          currentLine = word;
        }
      }
    }

    if (currentLine) {
      wrappedAll.push(currentLine);
    }
  }

  return wrappedAll.join('\\N');
}

/**
 * Computes maximum safe horizontal width available for a `\pos(x,y)` sign without shooting off-screen.
 */
function getAvailableScreenWidthForPos(
  posX: number,
  an: number,
  playResX: number,
  marginPx: number
): number {
  const hAlign = (an - 1) % 3; // 0 = left, 1 = center, 2 = right
  if (hAlign === 0) {
    return Math.max(80, playResX - posX - marginPx);
  }
  if (hAlign === 2) {
    return Math.max(80, posX - marginPx);
  }
  // Centered: symmetric space around posX
  const halfSpace = Math.min(posX - marginPx, playResX - posX - marginPx);
  return Math.max(80, halfSpace * 2);
}

/**
 * Checks whether a sign event is a borderless or faint-border disclaimer/section notice
 * overlapping Japanese background text that should be upgraded to BorderStyle=3 (`Sign_Box`) opaque dark box.
 */
function shouldUpgradeToOpaqueBox(
  event: AssEvent,
  hasMatchedDrawingBox: boolean,
  fullEventText: string,
  visibleText: string,
  playResY: number
): boolean {
  if (hasMatchedDrawingBox) return false;

  const actorLower = (event.name || '').toLowerCase();
  const styleLower = (event.style || '').toLowerCase();
  const isExplicitSectionOrDisclaimer =
    /disclaimer|warning|notice|caution|section/.test(actorLower) ||
    /disclaimer|warning|notice|caution|section/.test(styleLower);
  const isTextHint =
    isExplicitSectionOrDisclaimer ||
    /text|note/.test(actorLower) ||
    /text|note/.test(styleLower);

  const bordMatches = [...fullEventText.matchAll(/\\(?:x|y)?bord(?![a-zA-Z])(-?\d+(?:\.\d+)?)/g)];
  const maxThinThreshold = Math.max(0.55, (playResY / 360) * 0.55);
  const hasThinOrZeroBorder =
    bordMatches.some((m) => parseFloat(m[1]) <= maxThinThreshold) ||
    /\\3a&HFF&/i.test(fullEventText);
  const hasNoBorderTag = bordMatches.length === 0;

  // Explicit multi-section disclaimers/notices (e.g. "text (top section)", "text (3rd section)", "text (4th section)")
  // upgrade to Sign_Box even when no inline \bord0 was emitted or border was thin/transparent.
  if (isExplicitSectionOrDisclaimer && (hasThinOrZeroBorder || hasNoBorderTag)) {
    return true;
  }

  if (!hasThinOrZeroBorder) return false;

  const visibleLen = visibleText.replace(/\\N/g, '').trim().length;
  return isTextHint || visibleLen >= 18 || visibleText.includes('\\N');
}

/**
 * Updates or inserts a specific override tag inside a leading `{...}` tag block.
 */
function upsertTag(tagBlock: string, pattern: RegExp, replacement: string): string {
  const base = tagBlock && tagBlock.startsWith('{') && tagBlock.endsWith('}') ? tagBlock : '{}';
  const inner = base.slice(1, -1);
  if (pattern.test(inner)) {
    return `{${inner.replace(pattern, replacement)}}`;
  }
  return `{${inner}${replacement}}`;
}

function removeTags(tagBlock: string, patterns: RegExp[]): string {
  if (!tagBlock || !tagBlock.startsWith('{') || !tagBlock.endsWith('}')) return tagBlock;
  let inner = tagBlock.slice(1, -1);
  for (const p of patterns) {
    inner = inner.replace(p, '');
  }
  return inner ? `{${inner}}` : '';
}

/**
 * Strips all complex typesetting tags for Clean Top Mode (`clean_top`) so signs render uniformly via `Sign_Basic`.
 */
export function stripTypesettingTagsForCleanTop(text: string): string {
  if (!text) return '';
  return text
    .replace(/\{[^}]*\}/g, '')
    .replace(/\\h/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/[ \t]*\\N[ \t]*/g, '\\N')
    .replace(/^(?:\\N)+|(?:\\N)+$/g, '')
    .trim();
}

/**
 * Groups simultaneous sign lines (4+ lines sharing Start/End within `toleranceSec`) and formats them
 * into 2-3 balanced lines joined by ` | ` (for `denseSignStrategy === 'merge'`).
 */
export function formatMergedDenseSigns(texts: string[], maxLines = 3): string {
  const cleaned = texts
    .map((t) => stripTypesettingTagsForCleanTop(t).replace(/\\N/g, ' ').replace(/\s+/g, ' ').trim())
    .filter((t) => t.length > 0);

  if (cleaned.length === 0) return '';
  if (cleaned.length <= maxLines) {
    return cleaned.join('\\N');
  }

  const targetLines = Math.min(maxLines, Math.max(2, Math.ceil(cleaned.length / 4)));
  const itemsPerLine = Math.ceil(cleaned.length / targetLines);
  const rows: string[] = [];

  for (let i = 0; i < cleaned.length; i += itemsPerLine) {
    rows.push(cleaned.slice(i, i + itemsPerLine).join(' | '));
  }

  return rows.join('\\N');
}

function isPureDrawingEvent(text: string): boolean {
  if (!DRAWING_TAG_REGEX.test(text)) return false;
  const stripped = text.replace(/\{[^}]*\}/g, '').trim();
  return stripped.length > 0 && /^[0-9\smlbspc.-]+$/i.test(stripped);
}

/**
 * Applies the selected Sign & Typesetting Mode across the full list of source events
 * (already scaled to target `playResX x playResY`, e.g. 640x360).
 *
 * Always preserves exact event count and timestamps so `validateAss` remains 100% valid.
 */
export function applySignTypesettingMode(
  events: AssEvent[],
  isSignFlags: boolean[],
  options: SignTypesettingOptions
): AssEvent[] {
  const mode = options.mode || 'smart_hybrid';
  const denseSignStrategy = options.denseSignStrategy || 'merge';
  const playResX = options.playResX ?? 640;
  const playResY = options.playResY ?? 360;

  if (mode === 'preserve_original') {
    return events;
  }

  const result: AssEvent[] = events.map((e) => ({ ...e }));

  // =========================================================================
  // MODE 1: CLEAN TOP SIGNS (`clean_top`)
  // =========================================================================
  if (mode === 'clean_top') {
    // Step 1: Comment out pure {\p1} vector mask boxes & strip complex tags on sign lines
    const activeSignIndices: number[] = [];

    for (let i = 0; i < result.length; i++) {
      const ev = result[i];
      if (ev.type !== 'Dialogue') continue;

      if (isPureDrawingEvent(ev.text)) {
        result[i] = { ...ev, type: 'Comment' };
        continue;
      }

      if (!isSignFlags[i]) continue;

      const cleanedText = stripTypesettingTagsForCleanTop(ev.text);
      result[i] = {
        ...ev,
        style: 'Sign_Basic',
        text: cleanedText,
      };
      if (cleanedText.length > 0) {
        activeSignIndices.push(i);
      }
    }

    // Step 2: Detect simultaneous sign clusters (>= 4 signs overlapping same Start/End window)
    if (denseSignStrategy !== 'keep' && activeSignIndices.length >= 4) {
      const visited = new Set<number>();
      const TIME_TOLERANCE_SEC = 0.35;

      for (const idx of activeSignIndices) {
        if (visited.has(idx)) continue;
        const anchor = result[idx];
        const aStart = parseTimeSec(anchor.start);
        const aEnd = parseTimeSec(anchor.end);

        const cluster: number[] = [idx];
        visited.add(idx);

        for (const otherIdx of activeSignIndices) {
          if (visited.has(otherIdx)) continue;
          const other = result[otherIdx];
          const oStart = parseTimeSec(other.start);
          const oEnd = parseTimeSec(other.end);

          const sameWindow =
            Math.abs(oStart - aStart) <= TIME_TOLERANCE_SEC &&
            Math.abs(oEnd - aEnd) <= TIME_TOLERANCE_SEC;
          const heavyOverlap =
            Math.min(aEnd, oEnd) - Math.max(aStart, oStart) >=
            Math.min(aEnd - aStart, oEnd - oStart) * 0.75;

          if (sameWindow || heavyOverlap) {
            cluster.push(otherIdx);
            visited.add(otherIdx);
          }
        }

        if (cluster.length >= 4) {
          if (denseSignStrategy === 'comment') {
            for (const memberIdx of cluster) {
              result[memberIdx] = { ...result[memberIdx], type: 'Comment' };
            }
          } else if (denseSignStrategy === 'merge') {
            const mergedText = formatMergedDenseSigns(
              cluster.map((m) => result[m].text),
              3
            );
            const firstIdx = cluster[0];
            result[firstIdx] = {
              ...result[firstIdx],
              style: 'Sign_Basic',
              text: mergedText,
            };
            for (let k = 1; k < cluster.length; k++) {
              const memberIdx = cluster[k];
              result[memberIdx] = { ...result[memberIdx], type: 'Comment' };
            }
          }
        }
      }
    }

    return result;
  }

  // =========================================================================
  // MODE 2: SMART HYBRID (`smart_hybrid`)
  // =========================================================================
  const minFontSize = getMinReadableBurmeseFontSize(playResY);
  const smallFontThreshold = getSmallSignFontThreshold(playResY);
  const screenMargin = Math.max(10, Math.round(playResX * 0.02));
  const styleAlignments = options.styleAlignments || {};

  // Step 1: Index all {\p1} vector drawing boxes with their bounds and timing
  interface BoxRecord {
    eventIndex: number;
    startSec: number;
    endSec: number;
    bounds: DrawingBoxBounds;
    widthFactor: number;
    heightFactor: number;
  }

  const boxes: BoxRecord[] = [];
  for (let i = 0; i < result.length; i++) {
    const ev = result[i];
    if (ev.type !== 'Dialogue' || !isPureDrawingEvent(ev.text)) continue;
    const bounds = parseDrawingBoxBounds(ev.text, playResX, playResY);
    if (bounds) {
      boxes.push({
        eventIndex: i,
        startSec: parseTimeSec(ev.start),
        endSec: parseTimeSec(ev.end),
        bounds,
        widthFactor: 1,
        heightFactor: 1,
      });
    }
  }

  // Helper to match a sign event to a simultaneous {\p1} drawing box
  const findMatchingBox = (ev: AssEvent, posX: number | null, posY: number | null): BoxRecord | null => {
    const sSec = parseTimeSec(ev.start);
    const eSec = parseTimeSec(ev.end);

    let bestBox: BoxRecord | null = null;
    let bestScore = -1;

    for (const box of boxes) {
      const overlap = Math.min(eSec, box.endSec) - Math.max(sSec, box.startSec);
      if (overlap <= 0.05) continue;

      if (posX !== null && posY !== null) {
        const padX = Math.max(24, box.bounds.width * 0.25);
        const padY = Math.max(24, box.bounds.height * 0.25);
        const inside =
          posX >= box.bounds.screenLeft - padX &&
          posX <= box.bounds.screenRight + padX &&
          posY >= box.bounds.screenTop - padY &&
          posY <= box.bounds.screenBottom + padY;
        if (inside) {
          const score = 100 + overlap;
          if (score > bestScore) {
            bestScore = score;
            bestBox = box;
          }
        }
      } else {
        // No explicit \pos: match by exact same timestamp window
        const exactTime = Math.abs(sSec - box.startSec) <= 0.15 && Math.abs(eSec - box.endSec) <= 0.15;
        if (exactTime && overlap > bestScore) {
          bestScore = overlap;
          bestBox = box;
        }
      }
    }

    return bestBox;
  };

  // Pre-pass: Gather spatial coordinates of all simultaneous positioned sign lines
  // so vertically stacked lines (e.g. RPG status screens) obey a strict Y-spacing cap (fs <= y_gap * 0.9).
  interface PositionedSignMeta {
    eventIndex: number;
    startSec: number;
    endSec: number;
    posX: number;
    posY: number;
    an: number;
    leftX: number;
    rightX: number;
    matchedBox: BoxRecord | null;
    leadingBlock: string;
    visibleBody: string;
    originalFs: number;
    hasInlineFs: boolean;
  }

  const resolveAlignment = (ev: AssEvent, leadingBlock: string, posX: number | null): number => {
    const anMatch = leadingBlock.match(/\\an([1-9])/);
    if (anMatch) return parseInt(anMatch[1], 10);
    const styleKey = (ev.style || '').trim().toLowerCase();
    if (styleKey && styleAlignments[styleKey]) return styleAlignments[styleKey];
    // Infer sensible alignment near left/right edge when source style alignment wasn't explicit
    if (posX !== null && posX < playResX * 0.16) return 7;
    if (posX !== null && posX > playResX * 0.84) return 9;
    return 8;
  };

  const posSignByIndex = new Map<number, PositionedSignMeta>();
  const posSignList: PositionedSignMeta[] = [];

  for (let i = 0; i < result.length; i++) {
    const ev = result[i];
    if (ev.type !== 'Dialogue' || !isSignFlags[i] || isPureDrawingEvent(ev.text)) continue;

    const leadingBlock = extractLeadingTagBlock(ev.text);
    const visibleBody = prepareSignTextForTranslation(ev.text);
    if (!visibleBody) continue;

    const posMatch = leadingBlock.match(/\\pos\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)/);
    if (!posMatch) continue;

    const posX = parseFloat(posMatch[1]);
    const posY = parseFloat(posMatch[2]);
    const an = resolveAlignment(ev, leadingBlock, posX);
    const fsMatch = leadingBlock.match(/\\fs(?![a-zA-Z])(\d+(?:\.\d+)?)/);
    const originalFs = fsMatch ? parseFloat(fsMatch[1]) : Math.round(playResY * 0.083);
    const estFs = Math.max(minFontSize, originalFs);
    const estWidth = Math.max(
      ...visibleBody.split('\\N').map((l) => estimateBurmeseLineWidthPx(l, estFs, 100))
    );
    const hAlign = (an - 1) % 3;
    const leftX = hAlign === 0 ? posX : hAlign === 1 ? posX - estWidth / 2 : posX - estWidth;
    const rightX = leftX + estWidth;

    const meta: PositionedSignMeta = {
      eventIndex: i,
      startSec: parseTimeSec(ev.start),
      endSec: parseTimeSec(ev.end),
      posX,
      posY,
      an,
      leftX,
      rightX,
      matchedBox: findMatchingBox(ev, posX, posY),
      leadingBlock,
      visibleBody,
      originalFs,
      hasInlineFs: Boolean(fsMatch),
    };
    posSignByIndex.set(i, meta);
    posSignList.push(meta);
  }

  const getMinVerticalGapForLine = (cur: PositionedSignMeta): number | null => {
    let minGap = Number.POSITIVE_INFINITY;
    const minRowThreshold = Math.max(4, playResY * 0.012); // Ignore < 1.2% PlayResY (shadow/duplicate layer on same row)
    const hPad = Math.max(12, playResX * 0.025);

    for (const other of posSignList) {
      if (other.eventIndex === cur.eventIndex) continue;
      const timeOverlap = Math.min(cur.endSec, other.endSec) - Math.max(cur.startSec, other.startSec);
      if (timeOverlap <= 0.05) continue;

      const yGap = Math.abs(other.posY - cur.posY);
      if (yGap < minRowThreshold) continue;

      const overlapsHorizontally =
        Math.min(cur.rightX + hPad, other.rightX + hPad) >
        Math.max(cur.leftX - hPad, other.leftX - hPad);
      if (overlapsHorizontally && yGap < minGap) {
        minGap = yGap;
      }
    }

    return Number.isFinite(minGap) ? minGap : null;
  };

  // Pre-track disclaimer clusters so every section sharing a timestamp window upgrades uniformly
  const disclaimerCandidateIndices = new Set<number>();
  for (let i = 0; i < result.length; i++) {
    const ev = result[i];
    if (ev.type !== 'Dialogue' || !isSignFlags[i] || isPureDrawingEvent(ev.text)) continue;
    const posMeta = posSignByIndex.get(i);
    const matchedBox = posMeta ? posMeta.matchedBox : findMatchingBox(ev, null, null);
    const visibleBody = posMeta ? posMeta.visibleBody : prepareSignTextForTranslation(ev.text);
    if (!visibleBody) continue;

    if (shouldUpgradeToOpaqueBox(ev, Boolean(matchedBox), ev.text, visibleBody, playResY)) {
      disclaimerCandidateIndices.add(i);
    }
  }

  // Propagate Sign_Box to simultaneous sibling section/text lines within the same disclaimer card window
  for (let i = 0; i < result.length; i++) {
    if (disclaimerCandidateIndices.has(i)) continue;
    const ev = result[i];
    if (ev.type !== 'Dialogue' || !isSignFlags[i] || isPureDrawingEvent(ev.text)) continue;
    const actorStyleLower = `${ev.name || ''} ${ev.style || ''}`.toLowerCase();
    if (!/section|disclaimer|warning|notice|caution|\btext\b/.test(actorStyleLower)) continue;

    const posMeta = posSignByIndex.get(i);
    const matchedBox = posMeta ? posMeta.matchedBox : findMatchingBox(ev, null, null);
    if (matchedBox) continue;

    const sSec = parseTimeSec(ev.start);
    const eSec = parseTimeSec(ev.end);
    for (const dIdx of disclaimerCandidateIndices) {
      const dEv = result[dIdx];
      const dStart = parseTimeSec(dEv.start);
      const dEnd = parseTimeSec(dEv.end);
      if (Math.abs(sSec - dStart) <= 0.5 && Math.abs(eSec - dEnd) <= 0.5) {
        disclaimerCandidateIndices.add(i);
        break;
      }
    }
  }

  // Step 2: Optimize each sign event
  for (let i = 0; i < result.length; i++) {
    const ev = result[i];
    if (ev.type !== 'Dialogue' || !isSignFlags[i] || isPureDrawingEvent(ev.text)) continue;

    const posMeta = posSignByIndex.get(i);
    const leadingBlock = posMeta ? posMeta.leadingBlock : extractLeadingTagBlock(ev.text);
    const visibleBody = posMeta ? posMeta.visibleBody : prepareSignTextForTranslation(ev.text);
    if (!visibleBody) continue;

    const posX = posMeta ? posMeta.posX : null;
    const an = posMeta ? posMeta.an : resolveAlignment(ev, leadingBlock, null);
    const matchedBox = posMeta ? posMeta.matchedBox : findMatchingBox(ev, null, null);
    const upgradeToOpaqueBox = disclaimerCandidateIndices.has(i);

    // Determine effective font size:
    // 1) Enforce PlayResY-proportional minimum (~3.2% of PlayResY, e.g. 11.5 at 360p)
    // 2) Apply strict Y-spacing cap (fs <= y_gap * 0.9) for simultaneous stacked \pos(x,y) lines
    const fsMatch = leadingBlock.match(/\\fs(?![a-zA-Z])(\d+(?:\.\d+)?)/);
    const originalFs = posMeta
      ? posMeta.originalFs
      : fsMatch
      ? parseFloat(fsMatch[1])
      : Math.round(playResY * 0.083);

    let effectiveFs = Math.max(minFontSize, originalFs);
    const minYGap = posMeta && !matchedBox && !upgradeToOpaqueBox ? getMinVerticalGapForLine(posMeta) : null;
    const absMinFloor = Math.max(7, Math.round(playResY * 0.021 * 10) / 10);
    if (minYGap !== null) {
      const yCapFs = Math.max(absMinFloor, Math.round(minYGap * 0.9 * 10) / 10);
      if (effectiveFs > yCapFs) {
        effectiveFs = yCapFs;
      }
    }

    let updatedTag = removeTags(leadingBlock, [/\\q\d+/g]);
    let targetStyle = ev.style;

    // Step 3: Auto-Fit Burmese text to matched {\p1} box (width + height) OR screen width from \pos(x,y)
    let fittedText = visibleBody;
    let finalFs = effectiveFs;

    if (matchedBox) {
      const boxPaddingX = Math.max(6, matchedBox.bounds.width * 0.05);
      const boxPaddingY = Math.max(5, Math.round(playResY * 0.015));
      const LINE_HEIGHT_RATIO = 1.35;
      const posY = posMeta ? posMeta.posY : null;
      const vAlign = Math.floor((an - 1) / 3); // 2 = top (7,8,9), 1 = middle (4,5,6), 0 = bottom (1,2,3)
      const hAlign = (an - 1) % 3; // 0 = left (1,4,7), 1 = center (2,5,8), 2 = right (3,6,9)

      // Compute exact available horizontal width inside box (respecting text posX indent when inside box)
      let baseInnerWidth = Math.max(48, matchedBox.bounds.width - boxPaddingX);
      if (posX !== null && posX > matchedBox.bounds.screenLeft && posX < matchedBox.bounds.screenRight) {
        if (hAlign === 0) {
          baseInnerWidth = Math.max(48, matchedBox.bounds.screenRight - posX - boxPaddingX / 2);
        } else if (hAlign === 2) {
          baseInnerWidth = Math.max(48, posX - matchedBox.bounds.screenLeft - boxPaddingX / 2);
        }
      }

      // Compute exact available vertical height remaining from text_y to box edge:
      // available_h = (box_y + box_h) - text_y - padding (for top-anchored \an7/8/9 text)
      let availableHeight = Math.max(12, matchedBox.bounds.height - boxPaddingY);
      if (posY !== null) {
        if (vAlign === 2 && posY >= matchedBox.bounds.screenTop - 4 && posY < matchedBox.bounds.screenBottom) {
          availableHeight = Math.max(12, matchedBox.bounds.screenBottom - posY - boxPaddingY);
        } else if (vAlign === 0 && posY > matchedBox.bounds.screenTop && posY <= matchedBox.bounds.screenBottom + 4) {
          availableHeight = Math.max(12, posY - matchedBox.bounds.screenTop - boxPaddingY);
        } else if (vAlign === 1 && posY > matchedBox.bounds.screenTop && posY < matchedBox.bounds.screenBottom) {
          const halfH = Math.min(posY - matchedBox.bounds.screenTop, matchedBox.bounds.screenBottom - posY);
          availableHeight = Math.max(12, halfH * 2 - boxPaddingY);
        }
      }

      const fitsAvailableHeight = (textCandidate: string, fsVal: number): boolean => {
        const lineCount = textCandidate.split('\\N').length;
        return lineCount * (fsVal * LINE_HEIGHT_RATIO) <= availableHeight;
      };

      const maxBoxLinesAtFs = (fsVal: number) =>
        Math.max(1, Math.floor(availableHeight / (fsVal * LINE_HEIGHT_RATIO)));

      // Pass 1: Try wrapping at 100% horizontal scale
      let wrapped = wrapBurmeseTextToWidth(visibleBody, baseInnerWidth, finalFs, 100);
      let activeScaleX = 100;

      if (wrapped.split('\\N').length > maxBoxLinesAtFs(finalFs)) {
        // Pass 2: Try condensing horizontal scale (\fscx88)
        const wrapped88 = wrapBurmeseTextToWidth(visibleBody, baseInnerWidth, finalFs, 88);
        if (wrapped88.split('\\N').length <= maxBoxLinesAtFs(finalFs)) {
          wrapped = wrapped88;
          activeScaleX = 88;
          updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx88');
        } else {
          // Pass 3: Expand box width by +15% and try \fscx85
          const expandedWidth = baseInnerWidth + matchedBox.bounds.width * 0.15;
          wrapped = wrapBurmeseTextToWidth(visibleBody, expandedWidth, finalFs, 85);
          activeScaleX = 85;
          updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx85');
          matchedBox.widthFactor = Math.max(matchedBox.widthFactor, 1.15);
        }
      } else {
        // Check if a long unbroken word required slight horizontal condensing or +15% width expansion
        const maxSingleLinePx = Math.max(
          ...wrapped.split('\\N').map((l) => estimateBurmeseLineWidthPx(l, finalFs, 100))
        );
        if (maxSingleLinePx > baseInnerWidth * 1.03) {
          if (maxSingleLinePx * 0.86 <= baseInnerWidth) {
            activeScaleX = 86;
            updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx86');
          } else {
            activeScaleX = 85;
            updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx85');
            matchedBox.widthFactor = Math.max(matchedBox.widthFactor, 1.15);
          }
        }
      }

      // Pass 4 (Burmese 1.35x Line-Height Safety Fit):
      // (1) If line_count * (fs * 1.35) > available_h, incrementally reduce \fs down to safety floor (~9px @ 360p)
      // (2) If STILL overflowing even at safety floor, expand {\p1} box bottom height so 100% of text stays inside
      if (!fitsAvailableHeight(wrapped, finalFs)) {
        const safetyMinFs = Math.max(8, Math.round(playResY * 0.025 * 10) / 10); // 9.0 at 360p, 27.0 at 1080p
        const stepFs = Math.max(0.4, Math.round((playResY / 360) * 0.5 * 10) / 10);
        const currentWrapWidth =
          matchedBox.widthFactor > 1.001
            ? baseInnerWidth + matchedBox.bounds.width * (matchedBox.widthFactor - 1)
            : baseInnerWidth;

        let bestFs = finalFs;
        let bestWrapped = wrapped;
        let solvedByShrink = false;

        for (let candFs = finalFs - stepFs; candFs >= safetyMinFs - 0.01; candFs -= stepFs) {
          const roundedCand = Math.max(safetyMinFs, Math.round(candFs * 10) / 10);
          const candWrapped = wrapBurmeseTextToWidth(visibleBody, currentWrapWidth, roundedCand, activeScaleX);
          bestFs = roundedCand;
          bestWrapped = candWrapped;
          if (fitsAvailableHeight(candWrapped, roundedCand)) {
            solvedByShrink = true;
            break;
          }
        }

        finalFs = bestFs;
        wrapped = bestWrapped;

        if (!solvedByShrink) {
          const finalLineCount = wrapped.split('\\N').length;
          const neededTextHeight = finalLineCount * (finalFs * LINE_HEIGHT_RATIO);
          const extraHeightNeeded = Math.max(0, neededTextHeight - availableHeight);
          if (extraHeightNeeded > 0.5) {
            const neededTotalHeight = matchedBox.bounds.height + extraHeightNeeded;
            const neededScaleY = Math.min(3.0, Math.round((neededTotalHeight / matchedBox.bounds.height) * 100) / 100);
            matchedBox.heightFactor = Math.max(matchedBox.heightFactor, neededScaleY);
          }
        }
      }

      fittedText = wrapped;
    } else if (posX !== null) {
      // Positioned sign without a {\p1} drawing box:
      // Prevent text from shooting off the screen edge, while respecting tight stacked Y-spacing rows
      const maxSafeWidth = getAvailableScreenWidthForPos(posX, an, playResX, screenMargin);
      const rawMaxLinePx = Math.max(
        ...visibleBody.split('\\N').map((l) => estimateBurmeseLineWidthPx(l, finalFs, 100))
      );

      if (rawMaxLinePx > maxSafeWidth) {
        const tightVerticalStack = minYGap !== null && minYGap < finalFs * 2.2;
        if (rawMaxLinePx * 0.9 <= maxSafeWidth) {
          updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx90');
        } else if (tightVerticalStack && rawMaxLinePx * 0.85 <= maxSafeWidth) {
          // Condense horizontally so tight vertically-stacked status lines don't wrap into the next row
          updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx85');
        } else {
          updatedTag = upsertTag(updatedTag, /\\fscx\d+(?:\.\d+)?/, '\\fscx90');
          fittedText = wrapBurmeseTextToWidth(visibleBody, maxSafeWidth, finalFs, 90);
        }
      }
    }

    const isSmallFont = finalFs <= smallFontThreshold;

    if (upgradeToOpaqueBox || targetStyle === 'Sign_Box') {
      // Upgrade Japanese-overlapping disclaimers to BorderStyle=3 (`Sign_Box`) opaque dark box:
      // NEVER inject thin \bord0.51 override so Style's box padding (Outline=3) covers Japanese text cleanly.
      targetStyle = 'Sign_Box';
      updatedTag = removeTags(updatedTag, [
        /\\(?:x|y)?bord(?![a-zA-Z])-?\d+(?:\.\d+)?/g,
        /\\(?:x|y)?shad(?![a-zA-Z])-?\d+(?:\.\d+)?/g,
        /\\3c&H[0-9A-Fa-f]+&/g,
        /\\3a&H[0-9A-Fa-f]+&/g,
        /\\b\d+/g,
      ]);
      updatedTag = upsertTag(updatedTag, /\\fs(?![a-zA-Z])\d+(?:\.\d+)?/, `\\fs${roundAss(finalFs, 1)}`);
      updatedTag = upsertTag(updatedTag, /\\b\d+/, '\\b0');
    } else {
      if (fsMatch || finalFs !== originalFs) {
        updatedTag = upsertTag(updatedTag, /\\fs(?![a-zA-Z])\d+(?:\.\d+)?/, `\\fs${roundAss(finalFs, 1)}`);
      }
      // Small sign Burmese clarity: disable Bold (\b0) and set thin proportional outline (\bord)
      if (isSmallFont) {
        updatedTag = upsertTag(updatedTag, /\\b\d+/, '\\b0');
        const thinBord = getProportionalThinBorder(finalFs, playResY);
        updatedTag = upsertTag(updatedTag, /\\bord(?![a-zA-Z])-?\d+(?:\.\d+)?/, `\\bord${thinBord}`);
      }
    }

    result[i] = {
      ...ev,
      style: targetStyle,
      text: `${updatedTag}${fittedText}`,
    };
  }

  // Step 4: Expand any {\p1} vector drawing boxes horizontally (+15%) and/or vertically for clean Burmese fit
  for (const box of boxes) {
    if (box.widthFactor > 1.001 || box.heightFactor > 1.001) {
      result[box.eventIndex] = {
        ...result[box.eventIndex],
        text: expandDrawingBoxDimensions(
          result[box.eventIndex].text,
          box.widthFactor,
          box.heightFactor
        ),
      };
    }
  }

  return result;
}
