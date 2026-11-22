import test from 'node:test';
import assert from 'node:assert/strict';
import {
  verifySecretConstantTime,
  extractApiTokenFromHeaders,
  sanitizeSubtitleFileName,
  buildTranslatedOutputFileName,
  isSupportedBurmeseTargetLanguage,
  parseExternalTranslatePayload,
  parseAutopilotTranslateRequest,
  ExternalJobStore,
} from './externalApi';

test('verifySecretConstantTime & extractApiTokenFromHeaders validate Bearer and x-api-key', () => {
  assert.equal(verifySecretConstantTime('my-secret-123', 'my-secret-123'), true);
  assert.equal(verifySecretConstantTime('wrong-secret', 'my-secret-123'), false);
  assert.equal(verifySecretConstantTime('', 'my-secret-123'), false);
  assert.equal(verifySecretConstantTime('my-secret-123', ''), false);

  assert.equal(
    extractApiTokenFromHeaders({ authorization: 'Bearer tok_abc_123' }),
    'tok_abc_123'
  );
  assert.equal(
    extractApiTokenFromHeaders({ 'x-api-key': 'tok_xyz_999' }),
    'tok_xyz_999'
  );
  assert.equal(
    extractApiTokenFromHeaders({ authorization: 'Basic dXNlcjpwYXNz' }),
    ''
  );
});

test('sanitizeSubtitleFileName prevents path traversal and normalizes extension', () => {
  assert.equal(sanitizeSubtitleFileName('../../etc/passwd'), 'passwd.ass');
  assert.equal(sanitizeSubtitleFileName('C:\\Anime\\Ep01.ass'), 'Ep01.ass');
  assert.equal(sanitizeSubtitleFileName('Ep02\r\nEvil.srt'), 'Ep02Evil.srt');
  assert.equal(buildTranslatedOutputFileName('Episode_05.srt'), 'Episode_05.translated.ass');
});

test('parseExternalTranslatePayload supports JSON and raw ASS text payloads', () => {
  const jsonRes = parseExternalTranslatePayload({
    fileContent: '[Script Info]\n[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello!',
    fileName: 'Test Ep 01.ass',
    signMode: 'smart_hybrid',
    cpsMode: 'strict_compact',
  });
  assert.equal(jsonRes.ok, true);
  if (jsonRes.ok) {
    assert.equal(jsonRes.data.fileName, 'Test Ep 01.ass');
    assert.equal(jsonRes.data.signMode, 'smart_hybrid');
    assert.equal(jsonRes.data.cpsMode, 'strict_compact');
    assert.equal(jsonRes.data.responseFormat, 'json');
  }

  const rawRes = parseExternalTranslatePayload(
    '1\n00:00:01,000 --> 00:00:03,000\nHello!',
    { fileName: 'ep02.srt' }
  );
  assert.equal(rawRes.ok, true);
  if (rawRes.ok) {
    assert.equal(rawRes.data.fileName, 'ep02.srt');
    assert.equal(rawRes.data.responseFormat, 'ass');
    assert.equal(rawRes.data.cpsMode, 'balanced_natural');
  }

  const emptyRes = parseExternalTranslatePayload({ fileContent: '   ' });
  assert.equal(emptyRes.ok, false);
});

test('parseAutopilotTranslateRequest handles { assContent, targetLanguage } and multipart uploaded .ass files', () => {
  assert.equal(isSupportedBurmeseTargetLanguage('Burmese'), true);
  assert.equal(isSupportedBurmeseTargetLanguage('my'), true);
  assert.equal(isSupportedBurmeseTargetLanguage('Myanmar (မြန်မာ)'), true);
  assert.equal(isSupportedBurmeseTargetLanguage('French'), false);

  const jsonReq = parseAutopilotTranslateRequest({
    body: {
      assContent: '[Script Info]\n[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hi!',
      targetLanguage: 'Burmese',
    },
  });
  assert.equal(jsonReq.ok, true);
  if (jsonReq.ok) {
    assert.equal(jsonReq.data.modelChoice, 'gemini-3.8-flash');
    assert.equal(jsonReq.data.signMode, 'smart_hybrid');
    assert.equal(jsonReq.data.cpsMode, 'balanced_natural');
    assert.equal(jsonReq.data.responseFormat, 'ass');
  }

  const multipartReq = parseAutopilotTranslateRequest({
    body: { targetLanguage: 'my', cpsMode: 'strict_compact' },
    uploadedFile: {
      buffer: Buffer.from('[Script Info]\n[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Run!', 'utf8'),
      originalname: '../shows/ReZero_01.ass',
    },
  });
  assert.equal(multipartReq.ok, true);
  if (multipartReq.ok) {
    assert.equal(multipartReq.data.fileName, 'ReZero_01.ass');
    assert.equal(multipartReq.data.cpsMode, 'strict_compact');
    assert.equal(multipartReq.data.responseFormat, 'ass');
  }

  const badLang = parseAutopilotTranslateRequest({
    body: { assContent: '[Script Info]', targetLanguage: 'Spanish' },
  });
  assert.equal(badLang.ok, false);
});

test('ExternalJobStore tracks lifecycle and evicts expired completed jobs', () => {
  const store = new ExternalJobStore({ maxJobs: 2, ttlMs: 1000 });
  const j1 = store.createJob({ fileName: 'ep1.ass', cpsMode: 'balanced_natural', signMode: 'smart_hybrid' }, 100);
  assert.equal(j1.status, 'queued');
  assert.equal(store.activeCount(), 1);

  store.updateProgress(j1.id, 45, 'Translating batch 2/4', 200);
  assert.equal(store.getJob(j1.id, 250)?.status, 'processing');
  assert.equal(store.getJob(j1.id, 250)?.percent, 45);

  store.completeJob(
    j1.id,
    { assContent: '[Script Info]', outputFileName: 'ep1.translated.ass', eventCount: 10, translatableCount: 8 },
    300
  );
  assert.equal(store.getJob(j1.id, 350)?.status, 'completed');
  assert.equal(store.activeCount(), 0);

  const j2 = store.createJob({ fileName: 'ep2.ass', cpsMode: 'balanced_natural', signMode: 'smart_hybrid' }, 400);
  const j3 = store.createJob({ fileName: 'ep3.ass', cpsMode: 'balanced_natural', signMode: 'smart_hybrid' }, 500);
  // maxJobs=2 evicts completed j1 while keeping active j2 & j3
  assert.equal(store.getJob(j1.id, 550), undefined);
  assert.ok(store.getJob(j2.id, 550));
  assert.ok(store.getJob(j3.id, 550));
});
