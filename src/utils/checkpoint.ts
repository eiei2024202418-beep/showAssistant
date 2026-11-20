/**
 * Auto-Save Checkpoint & Resume utilities for Anime Subtitle Translator.
 * Pure, storage-injectable logic shared between browser UI (localStorage) and server resume handler.
 */

import type { SignTypesettingMode, DenseSignStrategy } from './signTypesetting';

export interface CheckpointPreviewItem {
  index: number;
  time: string;
  start?: string;
  end?: string;
  durationSec?: number;
  cps?: number;
  isSignOrTitle?: boolean;
  actor?: string;
  style: string;
  original: string;
  translated: string;
}

export interface TranslationCheckpoint {
  version: 1;
  signature: string;
  fileName: string;
  fileSize: number;
  fileContent: string;
  sourceLanguage: 'auto' | 'ja' | 'en';
  modelChoice: 'gemini-3.8-flash' | 'gemini-4-hydrogen' | 'gemini-3.1-flash-lite';
  signMode?: SignTypesettingMode;
  denseSignStrategy?: DenseSignStrategy;
  totalEvents: number;
  translatableCount: number;
  /** Map of AssEvent.index (as string key) -> translated ASS text */
  translatedLines: Record<string, string>;
  /** Preview rows collected so far for live display when resuming */
  previewItems: CheckpointPreviewItem[];
  updatedAt: number;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  key(index: number): string | null;
  readonly length: number;
}

export const CHECKPOINT_KEY_PREFIX = 'anime_sub_checkpoint_v1:';
const MAX_STORED_CHECKPOINTS = 3;

function getDefaultStorage(): StorageLike | null {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      return window.localStorage;
    }
  } catch {
    // Storage disabled or restricted
  }
  return null;
}

/**
 * Deterministic fast 64-bit-like fingerprint (two 32-bit FNV-1a hashes) for a subtitle file.
 * Ensures checkpoints strictly match the exact file name and content.
 */
export function computeFileSignature(fileName: string, fileContent: string): string {
  const normalizedName = (fileName || 'subtitle.ass').trim().toLowerCase();
  const normalizedContent = (fileContent || '').replace(/\r\n/g, '\n');

  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;

  for (let i = 0; i < normalizedContent.length; i++) {
    const code = normalizedContent.charCodeAt(i);
    h1 ^= code;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= code + (i & 0xff);
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  }

  let nameHash = 0x811c9dc5;
  for (let i = 0; i < normalizedName.length; i++) {
    nameHash ^= normalizedName.charCodeAt(i);
    nameHash = Math.imul(nameHash, 0x01000193) >>> 0;
  }

  return `${nameHash.toString(36)}_${normalizedContent.length.toString(36)}_${h1.toString(36)}_${h2.toString(36)}`;
}

/**
 * Server-side sanitizer for client-supplied preTranslatedLines.
 * Accepts only valid non-negative integer indices that exist in `validTranslatableIndices`
 * and have string values.
 */
export function sanitizePreTranslatedMap(
  raw: unknown,
  validTranslatableIndices: Set<number>
): Map<number, string> {
  const result = new Map<number, string>();
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return result;
  }

  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof value !== 'string') continue;
    const idx = Number(key);
    if (!Number.isInteger(idx) || idx < 0) continue;
    if (!validTranslatableIndices.has(idx)) continue;
    result.set(idx, value);
  }

  return result;
}

/**
 * Splits translatable events into already-translated count vs remaining events that still
 * need to be sent to Gemini API.
 */
export function filterRemainingTranslatableEvents<T extends { index: number }>(
  translatableEvents: T[],
  preTranslatedMap: Map<number, string>
): { alreadyTranslatedCount: number; remainingEvents: T[] } {
  const remainingEvents: T[] = [];
  let alreadyTranslatedCount = 0;

  for (const ev of translatableEvents) {
    if (preTranslatedMap.has(ev.index)) {
      alreadyTranslatedCount++;
    } else {
      remainingEvents.push(ev);
    }
  }

  return { alreadyTranslatedCount, remainingEvents };
}

/**
 * Merges incoming batch preview items into an existing sorted list without duplicates.
 */
export function mergeCheckpointPreviewItems(
  existing: CheckpointPreviewItem[],
  incoming: CheckpointPreviewItem[]
): CheckpointPreviewItem[] {
  const byIndex = new Map<number, CheckpointPreviewItem>();
  for (const item of existing) {
    byIndex.set(item.index, item);
  }
  for (const item of incoming) {
    byIndex.set(item.index, item);
  }
  return [...byIndex.values()].sort((a, b) => a.index - b.index);
}

/**
 * Extracts a clean index -> translated string record from preview items.
 */
export function buildPreTranslatedRecord(items: CheckpointPreviewItem[]): Record<string, string> {
  const map: Record<string, string> = {};
  for (const item of items) {
    if (typeof item.index === 'number' && typeof item.translated === 'string') {
      map[String(item.index)] = item.translated;
    }
  }
  return map;
}

export function getCheckpointTranslatedCount(cp: TranslationCheckpoint | null | undefined): number {
  if (!cp || !cp.translatedLines) return 0;
  return Object.keys(cp.translatedLines).length;
}

export function isCheckpointResumable(cp: TranslationCheckpoint | null | undefined): boolean {
  if (!cp) return false;
  if (!cp.fileContent || !cp.fileName) return false;
  return getCheckpointTranslatedCount(cp) > 0;
}

function listAllCheckpoints(storage: StorageLike): TranslationCheckpoint[] {
  const items: TranslationCheckpoint[] = [];
  for (let i = 0; i < storage.length; i++) {
    const k = storage.key(i);
    if (!k || !k.startsWith(CHECKPOINT_KEY_PREFIX)) continue;
    try {
      const raw = storage.getItem(k);
      if (!raw) continue;
      const parsed = JSON.parse(raw) as TranslationCheckpoint;
      if (parsed && parsed.version === 1 && parsed.signature && parsed.fileName && parsed.translatedLines) {
        items.push(parsed);
      }
    } catch {
      // Ignore corrupt entries
    }
  }
  return items.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

function evictOldCheckpoints(storage: StorageLike, keepSignature: string, maxToKeep: number): void {
  const all = listAllCheckpoints(storage).filter((c) => c.signature !== keepSignature);
  const toRemove = all.slice(Math.max(0, maxToKeep - 1));
  for (const cp of toRemove) {
    try {
      storage.removeItem(CHECKPOINT_KEY_PREFIX + cp.signature);
    } catch {
      // ignore
    }
  }
}

/**
 * Persists a translation checkpoint to storage with automatic oldest-entry eviction
 * if storage quota is tight.
 */
export function saveCheckpoint(
  checkpoint: TranslationCheckpoint,
  storage: StorageLike | null = getDefaultStorage()
): boolean {
  if (!storage) return false;
  const key = CHECKPOINT_KEY_PREFIX + checkpoint.signature;
  const payload = JSON.stringify(checkpoint);

  try {
    evictOldCheckpoints(storage, checkpoint.signature, MAX_STORED_CHECKPOINTS);
    storage.setItem(key, payload);
    return true;
  } catch {
    // QuotaExceededError: evict all other checkpoints and retry once
    try {
      evictOldCheckpoints(storage, checkpoint.signature, 1);
      storage.setItem(key, payload);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Loads a saved checkpoint matching a specific file signature.
 */
export function loadCheckpointBySignature(
  signature: string,
  storage: StorageLike | null = getDefaultStorage()
): TranslationCheckpoint | null {
  if (!storage || !signature) return null;
  try {
    const raw = storage.getItem(CHECKPOINT_KEY_PREFIX + signature);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as TranslationCheckpoint;
    if (parsed && parsed.version === 1 && parsed.signature === signature && isCheckpointResumable(parsed)) {
      return parsed;
    }
  } catch {
    // Ignore parse error
  }
  return null;
}

/**
 * Returns the most recently updated resumable checkpoint in storage (if any).
 */
export function findLatestCheckpoint(
  storage: StorageLike | null = getDefaultStorage()
): TranslationCheckpoint | null {
  if (!storage) return null;
  const all = listAllCheckpoints(storage).filter(isCheckpointResumable);
  return all.length > 0 ? all[0] : null;
}

/**
 * Removes a saved checkpoint by signature.
 */
export function clearCheckpoint(
  signature: string,
  storage: StorageLike | null = getDefaultStorage()
): void {
  if (!storage || !signature) return;
  try {
    storage.removeItem(CHECKPOINT_KEY_PREFIX + signature);
  } catch {
    // ignore
  }
}
