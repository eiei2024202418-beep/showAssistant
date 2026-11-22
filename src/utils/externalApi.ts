import crypto from 'node:crypto';
import type { SignTypesettingMode, DenseSignStrategy } from './signTypesetting';
import { type DialogueCpsMode, normalizeCpsMode } from './cpsMode';

export interface ExternalTranslateOptions {
  fileContent: string;
  fileName: string;
  sourceLanguage: string;
  modelChoice: string;
  signMode: SignTypesettingMode;
  denseSignStrategy: DenseSignStrategy;
  cpsMode: DialogueCpsMode;
  responseFormat: 'json' | 'ass';
}

export type ExternalJobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export interface ExternalTranslationJob {
  id: string;
  status: ExternalJobStatus;
  fileName: string;
  outputFileName: string;
  percent: number;
  statusMessage: string;
  createdAt: number;
  updatedAt: number;
  completedAt?: number;
  assContent?: string;
  eventCount?: number;
  translatableCount?: number;
  cpsMode: DialogueCpsMode;
  signMode: SignTypesettingMode;
  error?: string;
}

/**
 * Constant-time token verification using SHA-256 digests so comparison timing
 * never leaks secret length or prefix characters.
 */
export function verifySecretConstantTime(provided: string, expected: string): boolean {
  if (!provided || !expected) return false;
  const a = crypto.createHash('sha256').update(provided, 'utf8').digest();
  const b = crypto.createHash('sha256').update(expected, 'utf8').digest();
  return crypto.timingSafeEqual(a, b);
}

/**
 * Extracts caller token from `Authorization: Bearer <token>` or `x-api-key: <token>`.
 */
export function extractApiTokenFromHeaders(headers: {
  authorization?: string | string[];
  'x-api-key'?: string | string[];
}): string {
  const authHeader = Array.isArray(headers.authorization)
    ? headers.authorization[0]
    : headers.authorization;
  if (typeof authHeader === 'string') {
    const match = authHeader.match(/^\s*Bearer\s+(.+?)\s*$/i);
    if (match?.[1]) return match[1].trim();
  }

  const apiKeyHeader = Array.isArray(headers['x-api-key'])
    ? headers['x-api-key'][0]
    : headers['x-api-key'];
  if (typeof apiKeyHeader === 'string' && apiKeyHeader.trim()) {
    return apiKeyHeader.trim();
  }

  return '';
}

/**
 * Sanitizes incoming fileName to prevent path traversal or header injection
 * in Content-Disposition and output filenames.
 */
export function sanitizeSubtitleFileName(rawName: unknown, fallback = 'subtitle.ass'): string {
  if (typeof rawName !== 'string') return fallback;
  const stripped = rawName
    .replace(/[\r\n\0"<>|:*?]/g, '')
    .split(/[\\/]/)
    .pop()
    ?.trim();
  if (!stripped || stripped === '.' || stripped === '..') return fallback;
  if (!/\.(ass|srt)$/i.test(stripped)) {
    return `${stripped}.ass`;
  }
  return stripped.slice(0, 180);
}

export function buildTranslatedOutputFileName(fileName: string): string {
  const safeName = sanitizeSubtitleFileName(fileName);
  const cleanBase = safeName.replace(/\.(ass|srt)$/i, '') || 'subtitle';
  return `${cleanBase}.translated.ass`;
}

export function isSupportedBurmeseTargetLanguage(targetLanguage: unknown): boolean {
  if (targetLanguage === undefined || targetLanguage === null) return true;
  if (typeof targetLanguage !== 'string') return false;
  const normalized = targetLanguage.trim().toLowerCase();
  if (!normalized) return true;
  if (
    normalized === 'my' ||
    normalized === 'mm' ||
    normalized === 'my-mm' ||
    normalized === 'auto'
  ) {
    return true;
  }
  return (
    normalized.includes('burm') ||
    normalized.includes('myan') ||
    normalized.includes('မြန်မာ')
  );
}

export function parseExternalTranslatePayload(
  body: unknown,
  query: Record<string, unknown> = {},
  acceptHeader = ''
): { ok: true; data: ExternalTranslateOptions } | { ok: false; error: string } {
  let fileContent = '';
  let rawFileName: unknown = query.fileName ?? 'subtitle.ass';
  let sourceLanguage = typeof query.sourceLanguage === 'string' ? query.sourceLanguage : 'auto';
  let modelChoice = typeof query.modelChoice === 'string' ? query.modelChoice : 'gemini-3.8-flash';
  let signModeRaw: unknown = query.signMode ?? 'smart_hybrid';
  let denseSignRaw: unknown = query.denseSignStrategy ?? 'merge';
  let cpsModeRaw: unknown = query.cpsMode ?? 'balanced_natural';
  let formatRaw: unknown = query.format ?? query.responseFormat;
  let targetLanguageRaw: unknown = query.targetLanguage;

  if (typeof body === 'string') {
    fileContent = body;
  } else if (body && typeof body === 'object' && !Array.isArray(body)) {
    const record = body as Record<string, unknown>;
    if (typeof record.assContent === 'string') {
      fileContent = record.assContent;
    } else if (typeof record.fileContent === 'string') {
      fileContent = record.fileContent;
    } else if (typeof record.subtitle === 'string') {
      fileContent = record.subtitle;
    } else if (typeof record.content === 'string') {
      fileContent = record.content;
    }
    if (record.fileName !== undefined) rawFileName = record.fileName;
    if (record.targetLanguage !== undefined) targetLanguageRaw = record.targetLanguage;
    if (typeof record.sourceLanguage === 'string' && record.sourceLanguage.trim()) {
      sourceLanguage = record.sourceLanguage.trim().slice(0, 40);
    }
    if (typeof record.modelChoice === 'string' && record.modelChoice.trim()) {
      modelChoice = record.modelChoice.trim().slice(0, 80);
    }
    if (record.signMode !== undefined) signModeRaw = record.signMode;
    if (record.denseSignStrategy !== undefined) denseSignRaw = record.denseSignStrategy;
    if (record.cpsMode !== undefined) cpsModeRaw = record.cpsMode;
    if (record.responseFormat !== undefined) formatRaw = record.responseFormat;
    else if (record.format !== undefined) formatRaw = record.format;
  } else {
    return {
      ok: false,
      error: 'Request body must be JSON with "assContent"/"fileContent" or plain text subtitle content.',
    };
  }

  if (!isSupportedBurmeseTargetLanguage(targetLanguageRaw)) {
    return {
      ok: false,
      error: `Unsupported targetLanguage "${String(targetLanguageRaw)}". This pipeline translates into Burmese ("my", "Burmese", "Myanmar").`,
    };
  }

  if (!fileContent || !fileContent.trim()) {
    return {
      ok: false,
      error: 'Missing or empty subtitle content ("assContent" / "fileContent").',
    };
  }

  const safeSignMode: SignTypesettingMode =
    signModeRaw === 'clean_top' || signModeRaw === 'preserve_original' || signModeRaw === 'smart_hybrid'
      ? signModeRaw
      : 'smart_hybrid';

  const safeDenseSignStrategy: DenseSignStrategy =
    denseSignRaw === 'comment' || denseSignRaw === 'keep' || denseSignRaw === 'merge'
      ? denseSignRaw
      : 'merge';

  const safeCpsMode = normalizeCpsMode(cpsModeRaw);
  const fileName = sanitizeSubtitleFileName(rawFileName);

  const wantsAss =
    formatRaw === 'ass' ||
    formatRaw === 'raw' ||
    (formatRaw === undefined && typeof body === 'string') ||
    (formatRaw === undefined && /text\/(plain|x-ssa|x-ass)/i.test(acceptHeader));

  return {
    ok: true,
    data: {
      fileContent,
      fileName,
      sourceLanguage,
      modelChoice,
      signMode: safeSignMode,
      denseSignStrategy: safeDenseSignStrategy,
      cpsMode: safeCpsMode,
      responseFormat: wantsAss ? 'ass' : 'json',
    },
  };
}

export function parseAutopilotTranslateRequest(params: {
  body: unknown;
  query?: Record<string, unknown>;
  uploadedFile?: { buffer: Buffer; originalname?: string };
}): { ok: true; data: ExternalTranslateOptions } | { ok: false; error: string } {
  const { body, query = {}, uploadedFile } = params;

  if (uploadedFile && Buffer.isBuffer(uploadedFile.buffer)) {
    const fileText = uploadedFile.buffer.toString('utf8');
    const bodyRecord =
      body && typeof body === 'object' && !Array.isArray(body)
        ? (body as Record<string, unknown>)
        : {};
    const mergedBody: Record<string, unknown> = {
      ...bodyRecord,
      assContent: fileText,
      fileName: bodyRecord.fileName ?? query.fileName ?? uploadedFile.originalname ?? 'subtitle.ass',
      responseFormat: 'ass',
    };
    return parseExternalTranslatePayload(mergedBody, query, 'text/plain');
  }

  const parsed = parseExternalTranslatePayload(body, query, 'text/plain');
  if (!parsed.ok) return parsed;

  return {
    ok: true,
    data: {
      ...parsed.data,
      responseFormat: 'ass',
    },
  };
}

/**
 * Bounded in-memory async job store with TTL eviction so long-lived Cloud Run
 * instances cannot leak memory across many translation jobs.
 */
export class ExternalJobStore {
  private readonly jobs = new Map<string, ExternalTranslationJob>();
  private readonly maxJobs: number;
  private readonly ttlMs: number;

  constructor(options: { maxJobs?: number; ttlMs?: number } = {}) {
    this.maxJobs = options.maxJobs ?? 50;
    this.ttlMs = options.ttlMs ?? 60 * 60 * 1000; // 1 hour
  }

  prune(now = Date.now()): void {
    for (const [id, job] of this.jobs.entries()) {
      if (job.status !== 'processing' && job.status !== 'queued' && now - job.updatedAt > this.ttlMs) {
        this.jobs.delete(id);
      }
    }
    while (this.jobs.size > this.maxJobs) {
      let oldestEvictableId: string | undefined;
      for (const [id, job] of this.jobs.entries()) {
        if (job.status === 'completed' || job.status === 'failed') {
          oldestEvictableId = id;
          break;
        }
      }
      if (!oldestEvictableId) break;
      this.jobs.delete(oldestEvictableId);
    }
  }

  createJob(params: {
    fileName: string;
    cpsMode: DialogueCpsMode;
    signMode: SignTypesettingMode;
  }, now = Date.now()): ExternalTranslationJob {
    this.prune(now);
    const id = `job_${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`;
    const safeFileName = sanitizeSubtitleFileName(params.fileName);
    const job: ExternalTranslationJob = {
      id,
      status: 'queued',
      fileName: safeFileName,
      outputFileName: buildTranslatedOutputFileName(safeFileName),
      percent: 0,
      statusMessage: 'Queued for translation...',
      createdAt: now,
      updatedAt: now,
      cpsMode: params.cpsMode,
      signMode: params.signMode,
    };
    this.jobs.set(id, job);
    this.prune(now);
    return { ...job };
  }

  getJob(id: string, now = Date.now()): ExternalTranslationJob | undefined {
    this.prune(now);
    const job = this.jobs.get(id);
    return job ? { ...job } : undefined;
  }

  updateProgress(id: string, percent: number, statusMessage: string, now = Date.now()): void {
    const job = this.jobs.get(id);
    if (!job) return;
    job.status = 'processing';
    job.percent = Math.max(0, Math.min(99, Math.round(percent)));
    job.statusMessage = statusMessage;
    job.updatedAt = now;
  }

  completeJob(
    id: string,
    result: {
      assContent: string;
      outputFileName: string;
      eventCount: number;
      translatableCount: number;
    },
    now = Date.now()
  ): void {
    const job = this.jobs.get(id);
    if (!job) return;
    job.status = 'completed';
    job.percent = 100;
    job.statusMessage = 'Translation completed.';
    job.assContent = result.assContent;
    job.outputFileName = result.outputFileName;
    job.eventCount = result.eventCount;
    job.translatableCount = result.translatableCount;
    job.updatedAt = now;
    job.completedAt = now;
  }

  failJob(id: string, error: string, now = Date.now()): void {
    const job = this.jobs.get(id);
    if (!job) return;
    job.status = 'failed';
    job.statusMessage = 'Translation failed.';
    job.error = error;
    job.updatedAt = now;
    job.completedAt = now;
  }

  activeCount(): number {
    let count = 0;
    for (const job of this.jobs.values()) {
      if (job.status === 'queued' || job.status === 'processing') count++;
    }
    return count;
  }
}
