import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeFileSignature,
  sanitizePreTranslatedMap,
  filterRemainingTranslatableEvents,
  mergeCheckpointPreviewItems,
  buildPreTranslatedRecord,
  saveCheckpoint,
  loadCheckpointBySignature,
  findLatestCheckpoint,
  clearCheckpoint,
  isCheckpointResumable,
  getCheckpointTranslatedCount,
  type StorageLike,
  type TranslationCheckpoint,
} from './checkpoint';

class MemoryStorage implements StorageLike {
  private map = new Map<string, string>();
  private maxBytes: number;

  constructor(maxBytes = Number.POSITIVE_INFINITY) {
    this.maxBytes = maxBytes;
  }

  get length(): number {
    return this.map.size;
  }

  key(index: number): string | null {
    return [...this.map.keys()][index] ?? null;
  }

  getItem(key: string): string | null {
    return this.map.has(key) ? this.map.get(key)! : null;
  }

  setItem(key: string, value: string): void {
    let total = value.length;
    for (const [k, v] of this.map.entries()) {
      if (k !== key) total += v.length;
    }
    if (total > this.maxBytes) {
      throw new Error('QuotaExceededError');
    }
    this.map.set(key, value);
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }
}

test('computeFileSignature is deterministic, normalizes CRLF, and differentiates distinct files', () => {
  const sig1 = computeFileSignature('Episode 01.ass', 'line1\r\nline2');
  const sig2 = computeFileSignature('episode 01.ass', 'line1\nline2');
  const sig3 = computeFileSignature('Episode 02.ass', 'line1\nline2');
  const sig4 = computeFileSignature('Episode 01.ass', 'line1\nline2 modified');

  assert.equal(sig1, sig2);
  assert.notEqual(sig1, sig3);
  assert.notEqual(sig1, sig4);
});

test('sanitizePreTranslatedMap keeps only valid translatable indices and ignores invalid entries', () => {
  const valid = new Set([0, 2, 5]);
  const sanitized = sanitizePreTranslatedMap(
    {
      '0': 'မင်္ဂလာပါ',
      '2': '',
      '3': 'not translatable index',
      '-1': 'negative',
      abc: 'non-numeric',
      '5': 12345,
    },
    valid
  );

  assert.equal(sanitized.size, 2);
  assert.equal(sanitized.get(0), 'မင်္ဂလာပါ');
  assert.equal(sanitized.get(2), '');
  assert.equal(sanitized.has(3), false);
  assert.equal(sanitized.has(5), false);
});

test('filterRemainingTranslatableEvents skips already translated lines and preserves order', () => {
  const events = [{ index: 0 }, { index: 1 }, { index: 3 }, { index: 4 }];
  const preMap = new Map<number, string>([
    [0, 'လိုင်း ၁'],
    [3, 'လိုင်း ၃'],
  ]);

  const { alreadyTranslatedCount, remainingEvents } = filterRemainingTranslatableEvents(events, preMap);
  assert.equal(alreadyTranslatedCount, 2);
  assert.deepEqual(
    remainingEvents.map((e) => e.index),
    [1, 4]
  );
});

test('mergeCheckpointPreviewItems deduplicates by index and sorts ascending', () => {
  const merged = mergeCheckpointPreviewItems(
    [
      { index: 2, time: '0:02 - 0:04', style: 'Myanmar1', original: 'B', translated: 'old B' },
      { index: 0, time: '0:00 - 0:02', style: 'Myanmar1', original: 'A', translated: 'A1' },
    ],
    [
      { index: 2, time: '0:02 - 0:04', style: 'Myanmar1', original: 'B', translated: 'new B' },
      { index: 1, time: '0:01 - 0:02', style: 'Myanmar1', original: 'M', translated: 'M1' },
    ]
  );

  assert.deepEqual(
    merged.map((m) => [m.index, m.translated]),
    [
      [0, 'A1'],
      [1, 'M1'],
      [2, 'new B'],
    ]
  );
  assert.deepEqual(buildPreTranslatedRecord(merged), { '0': 'A1', '1': 'M1', '2': 'new B' });
});

test('saveCheckpoint, loadCheckpointBySignature, findLatestCheckpoint, quota eviction, and clearCheckpoint work cleanly', () => {
  const storage = new MemoryStorage();
  const makeCp = (name: string, updatedAt: number, linesCount = 2): TranslationCheckpoint => {
    const content = `Content for ${name}`;
    const signature = computeFileSignature(name, content);
    const translatedLines: Record<string, string> = {};
    for (let i = 0; i < linesCount; i++) {
      translatedLines[String(i)] = `Line ${i}`;
    }
    return {
      version: 1,
      signature,
      fileName: name,
      fileSize: content.length,
      fileContent: content,
      sourceLanguage: 'ja',
      modelChoice: 'gemini-3.8-flash',
      totalEvents: 10,
      translatableCount: 8,
      translatedLines,
      previewItems: [],
      updatedAt,
    };
  };

  const cp1 = makeCp('ep1.ass', 1000);
  const cp2 = makeCp('ep2.ass', 2000);
  const cp3 = makeCp('ep3.ass', 3000);
  const cp4 = makeCp('ep4.ass', 4000);

  assert.equal(saveCheckpoint(cp1, storage), true);
  assert.equal(saveCheckpoint(cp2, storage), true);
  assert.equal(saveCheckpoint(cp3, storage), true);
  assert.equal(saveCheckpoint(cp4, storage), true);

  // Oldest checkpoint (ep1) should be evicted when saving 4th (max 3)
  assert.equal(loadCheckpointBySignature(cp1.signature, storage), null);
  assert.equal(loadCheckpointBySignature(cp4.signature, storage)?.fileName, 'ep4.ass');
  assert.equal(findLatestCheckpoint(storage)?.fileName, 'ep4.ass');
  assert.equal(isCheckpointResumable(cp4), true);
  assert.equal(getCheckpointTranslatedCount(cp4), 2);

  // Clearing ep4 makes ep3 the latest checkpoint
  clearCheckpoint(cp4.signature, storage);
  assert.equal(loadCheckpointBySignature(cp4.signature, storage), null);
  assert.equal(findLatestCheckpoint(storage)?.fileName, 'ep3.ass');

  // Empty translatedLines is not resumable
  const emptyCp = makeCp('empty.ass', 5000, 0);
  assert.equal(isCheckpointResumable(emptyCp), false);
});

test('saveCheckpoint retries after evicting older entries when storage quota is tight', () => {
  const cp1: TranslationCheckpoint = {
    version: 1,
    signature: computeFileSignature('old.ass', 'A'.repeat(200)),
    fileName: 'old.ass',
    fileSize: 200,
    fileContent: 'A'.repeat(200),
    sourceLanguage: 'auto',
    modelChoice: 'gemini-3.8-flash',
    totalEvents: 5,
    translatableCount: 5,
    translatedLines: { '0': 'ဟိုင်း' },
    previewItems: [],
    updatedAt: 100,
  };
  const cp2: TranslationCheckpoint = {
    ...cp1,
    signature: computeFileSignature('new.ass', 'B'.repeat(200)),
    fileName: 'new.ass',
    fileContent: 'B'.repeat(200),
    updatedAt: 200,
  };

  // Budget big enough for 1 checkpoint, too small for 2
  const tightStorage = new MemoryStorage(JSON.stringify(cp1).length + 50);
  assert.equal(saveCheckpoint(cp1, tightStorage), true);
  assert.equal(saveCheckpoint(cp2, tightStorage), true);
  assert.equal(loadCheckpointBySignature(cp1.signature, tightStorage), null);
  assert.equal(loadCheckpointBySignature(cp2.signature, tightStorage)?.fileName, 'new.ass');
});
