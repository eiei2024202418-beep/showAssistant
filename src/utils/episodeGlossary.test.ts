import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildEpisodeSample,
  sanitizeEpisodeLockProfile,
  formatEpisodeLockInstruction,
  formatLineSpeakerHint,
  detectSpeakerPronounViolation,
  applyDeterministicSpeakerPronounFix,
  enforceLockedGlossaryPostProcess,
  type EpisodeLockProfile,
} from './episodeGlossary';
import type { AssEvent } from './subtitleParser';

function makeEvent(index: number, name: string, text: string): AssEvent {
  return {
    type: 'Dialogue',
    layer: '0',
    start: '0:00:01.00',
    end: '0:00:03.00',
    style: 'Default',
    name,
    marginL: '0',
    marginR: '0',
    marginV: '0',
    effect: '',
    text,
    originalText: text,
    index,
    isTranslatable: true,
  };
}

test('buildEpisodeSample collects unique actors, detected proper place/rank phrases, and spans full episode', () => {
  const events: AssEvent[] = [];
  for (let i = 0; i < 120; i++) {
    const actor = i === 75 ? 'Seryure' : i % 2 === 0 ? 'Shinpei' : 'Satoko';
    const text =
      i === 42
        ? 'We must defend Guillier Gorge for Crown Prince and the Kingdom of Lionel!'
        : `{\\i1}Line ${i} dialogue`;
    events.push(makeEvent(i, actor, text));
  }

  const sample = buildEpisodeSample(events, 'Firefly Wedding - 01.ass', 50);
  assert.equal(sample.fileName, 'Firefly Wedding - 01.ass');
  assert.ok(sample.actors.includes('Shinpei'));
  assert.ok(sample.actors.includes('Satoko'));
  assert.ok(sample.actors.includes('Seryure'));
  // Every named actor gets at least one sample line even if rare (Seryure at index 75)
  assert.ok(sample.sampleLines.some((l) => l.actor === 'Seryure'));
  // Full-episode scan captures Place & Rank phrases
  assert.ok(sample.detectedProperPhrases.includes('Guillier Gorge'));
  assert.ok(sample.detectedProperPhrases.includes('Crown Prince'));
  assert.ok(sample.detectedProperPhrases.includes('Kingdom of Lionel'));
});

test('sanitizeEpisodeLockProfile enforces single fixed self-pronoun and gender consistency', () => {
  const cleaned = sanitizeEpisodeLockProfile({
    speakerProfiles: [
      { actor: 'Dars', gender: 'male', fixedSelfPronoun: 'ကျွန်မ / ကျွန်တော်', fixedPluralPronoun: 'ကျွန်မတို့' }, // female leak for male corrected!
      { actor: 'Romelia', gender: 'female', fixedSelfPronoun: 'ကျွန်မ', fixedPluralPronoun: 'ကျွန်မတို့' },
      { actor: 'Hyde', gender: 'male', fixedSelfPronoun: 'ကျွန်ုပ် / ကျွန်တော်', fixedPluralPronoun: '' },
    ],
    pronounLocks: [
      { speaker: ' Shinpei ', listener: 'Satoko', pronouns: 'ကိုယ်/မင်း', toneNote: 'playful' },
      { speaker: 'Shinpei', listener: 'satoko', pronouns: 'ငါ/နင်' },
      { speaker: 'Satoko', listener: 'Shinpei', pronouns: 'ကျွန်မ/ရှင်' },
    ],
    glossaryTerms: [
      { sourceTerm: 'Guillier Gorge', burmeseTerm: 'Guillier ချောက်ကမ်းပါး', category: 'place' },
      { sourceTerm: 'guillier gorge', burmeseTerm: 'duplicate' },
      { sourceTerm: 'Crown Prince', burmeseTerm: 'အိမ်ရှေ့မင်းသား', category: 'rank' },
      { sourceTerm: 'General', burmeseTerm: 'ဗိုလ်ချုပ်', category: 'rank' },
    ],
  });

  assert.equal(cleaned.speakerProfiles?.length, 3);
  assert.equal(cleaned.speakerProfiles![0].fixedSelfPronoun, 'ကျွန်တော်');
  assert.equal(cleaned.speakerProfiles![0].fixedPluralPronoun, 'ကျွန်တော်တို့');
  assert.equal(cleaned.speakerProfiles![1].fixedSelfPronoun, 'ကျွန်မ');
  assert.equal(cleaned.speakerProfiles![2].fixedSelfPronoun, 'ကျွန်ုပ်');
  assert.equal(cleaned.speakerProfiles![2].fixedPluralPronoun, 'ကျွန်ုပ်တို့');

  assert.equal(cleaned.glossaryTerms.length, 3);
  assert.equal(cleaned.glossaryTerms[0].burmeseTerm, 'Guillier ချောက်ကမ်းပါး');
});

test('detectSpeakerPronounViolation & applyDeterministicSpeakerPronounFix prevent male ကျွန်မ, female ခင်ဗျား, and ကျွန်ုပ်/ကျွန်တော် drift', () => {
  const profile: EpisodeLockProfile = {
    speakerProfiles: [
      { actor: 'Romelia', gender: 'female', fixedSelfPronoun: 'ကျွန်မ', fixedPluralPronoun: 'ကျွန်မတို့' },
      { actor: 'Dars', gender: 'male', fixedSelfPronoun: 'ကျွန်တော်', fixedPluralPronoun: 'ကျွန်တော်တို့' },
      { actor: 'Hyde', gender: 'male', fixedSelfPronoun: 'ကျွန်ုပ်', fixedPluralPronoun: 'ကျွန်ုပ်တို့' },
    ],
    pronounLocks: [],
    glossaryTerms: [
      { sourceTerm: 'Guillier Gorge', burmeseTerm: 'Guillier ချောက်ကမ်းပါး', category: 'place' },
      { sourceTerm: 'Crown Prince', burmeseTerm: 'အိမ်ရှေ့မင်းသား', category: 'rank' },
      { sourceTerm: 'General', burmeseTerm: 'ဗိုလ်ချုပ်', category: 'rank' },
    ],
  };

  // Male Dars accidentally used female ကျွန်မတို့ -> caught & fixed to ကျွန်တော်တို့
  const maleLeak = detectSpeakerPronounViolation('Dars', 'ကျွန်မတို့ ဒီနေရာကို ကာကွယ်ရမယ်', profile);
  assert.equal(maleLeak.valid, false);
  assert.equal(maleLeak.reason, 'male_used_female_pronoun');
  assert.equal(
    applyDeterministicSpeakerPronounFix('Dars', 'ကျွန်မတို့ ဒီနေရာကို ကာကွယ်ရမယ်', profile),
    'ကျွန်တော်တို့ ဒီနေရာကို ကာကွယ်ရမယ်'
  );

  // Female Romelia accidentally used ခင်ဗျား or ကျွန်တော် -> caught & fixed to ရှင် / ကျွန်မ
  const femaleLeak = detectSpeakerPronounViolation('Romelia', 'ခင်ဗျား ဘာလုပ်နေတာလဲ', profile);
  assert.equal(femaleLeak.valid, false);
  assert.equal(femaleLeak.reason, 'female_used_male_pronoun');
  assert.equal(
    applyDeterministicSpeakerPronounFix('Romelia', 'ခင်ဗျား ဘာလုပ်နေတာလဲ', profile),
    'ရှင် ဘာလုပ်နေတာလဲ'
  );

  // Male Hyde locked to ကျွန်ုပ် drifted to ကျွန်တော် -> caught & fixed to ကျွန်ုပ်
  const drift = detectSpeakerPronounViolation('Hyde', '{\\i1}ကျွန်တော် သွားမယ်{\\i0}', profile);
  assert.equal(drift.valid, false);
  assert.equal(drift.reason, 'self_pronoun_drift');
  assert.equal(
    applyDeterministicSpeakerPronounFix('Hyde', '{\\i1}ကျွန်တော် သွားမယ်{\\i0}', profile),
    '{\\i1}ကျွန်ုပ် သွားမယ်{\\i0}'
  );

  // Quoted speech should NOT trigger false violation
  assert.equal(
    detectSpeakerPronounViolation('Dars', 'သူက "ကျွန်မ သွားမယ်" လို့ ပြောတယ်', profile).valid,
    true
  );

  // Per-line inline speaker hint & lock prompt block
  const hint = formatLineSpeakerHint('Dars', profile);
  assert.ok(hint?.includes('MALE'));
  assert.ok(hint?.includes('ကျွန်တော်'));
  assert.ok(hint?.includes('FORBIDDEN: ကျွန်မ'));
  assert.ok(hint?.includes('ငါ/မင်း/နင်'));

  const instruction = formatEpisodeLockInstruction(profile);
  assert.ok(instruction.includes('Speaker-Gender & Character Vibe Lock'));
  assert.ok(instruction.includes('Dars (MALE): Primary Self="ကျွန်တော်"'));
  assert.ok(instruction.includes('"Guillier Gorge" -> "Guillier ချောက်ကမ်းပါး"'));

  // Male actor using casual "ငါ" in rough/peer scene when primary is "ကျွန်တော်" MUST be allowed (not forced polite!)
  assert.equal(
    detectSpeakerPronounViolation('Dars', 'ငါ ဒီကောင်ကို သတ်ပစ်မယ်', profile).valid,
    true
  );

  // Glossary post-process catches leftover English place/rank words inside Burmese text
  assert.equal(
    enforceLockedGlossaryPostProcess('Crown Prince က General နဲ့ Guillier Gorge ကို သွားတယ်', profile),
    'အိမ်ရှေ့မင်းသား က ဗိုလ်ချုပ် နဲ့ Guillier ချောက်ကမ်းပါး ကို သွားတယ်'
  );
});

test('sanitizeEpisodeLockProfile & enforceLockedGlossaryPostProcess reject glossary/pronoun entries with Kanji/CJK/foreign scripts', () => {
  const poisoned = sanitizeEpisodeLockProfile({
    speakerProfiles: [
      { actor: 'Hero', gender: 'male', fixedSelfPronoun: 'ကျွန်တော်', fixedPluralPronoun: 'ကျွန်တော်တို့' },
    ],
    pronounLocks: [
      { speaker: 'A', listener: 'B', pronouns: 'ငါ / 俺' }, // Japanese Kanji must be rejected
      { speaker: 'C', listener: 'D', pronouns: 'ကိုယ် / မင်း' },
    ],
    glossaryTerms: [
      { sourceTerm: '無号機', burmeseTerm: '無號機', category: 'term' }, // Kanji leak must be rejected
      { sourceTerm: 'Zero Unit', burmeseTerm: 'Unit 0', category: 'term' }, // Valid ASCII allowed
      { sourceTerm: 'Devil', burmeseTerm: 'နတ်ဆိုး', category: 'term' },
    ],
  });

  assert.equal(poisoned.pronounLocks.length, 1);
  assert.equal(poisoned.pronounLocks[0].speaker, 'C');
  assert.equal(poisoned.glossaryTerms.length, 2);
  assert.ok(!poisoned.glossaryTerms.some((g) => g.burmeseTerm === '無號機'));

  // Even if a pre-sanitized profile object somehow contains Kanji in burmeseTerm, enforceLockedGlossaryPostProcess skips it
  const unsafeManualProfile: EpisodeLockProfile = {
    pronounLocks: [],
    glossaryTerms: [
      { sourceTerm: 'Unit Zero', burmeseTerm: '無號機', category: 'term' },
      { sourceTerm: 'Crown Prince', burmeseTerm: 'အိမ်ရှေ့မင်းသား', category: 'rank' },
    ],
  };
  assert.equal(
    enforceLockedGlossaryPostProcess('Unit Zero နဲ့ Crown Prince ရောက်လာပြီ', unsafeManualProfile),
    'Unit Zero နဲ့ အိမ်ရှေ့မင်းသား ရောက်လာပြီ'
  );
});
