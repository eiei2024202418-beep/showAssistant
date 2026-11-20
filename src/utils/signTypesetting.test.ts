import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getMinReadableBurmeseFontSize,
  getSmallSignFontThreshold,
  getProportionalThinBorder,
  prepareSignTextForTranslation,
  extractLeadingTagBlock,
  parseDrawingBoxBounds,
  expandDrawingBoxWidth,
  estimateBurmeseLineWidthPx,
  wrapBurmeseTextToWidth,
  stripTypesettingTagsForCleanTop,
  formatMergedDenseSigns,
  applySignTypesettingMode,
} from './signTypesetting';
import {
  parseAss,
  buildAss,
  validateAss,
  isSignOrTitle,
  isPureDrawing,
  type AssEvent,
} from './subtitleParser';

function makeEvent(partial: Partial<AssEvent>): AssEvent {
  const text = partial.text ?? '';
  return {
    index: partial.index ?? 0,
    type: partial.type ?? 'Dialogue',
    layer: partial.layer ?? '0',
    start: partial.start ?? '0:00:01.00',
    end: partial.end ?? '0:00:05.00',
    style: partial.style ?? 'Sign_Basic',
    name: partial.name ?? 'SIGN',
    marginL: '0',
    marginR: '0',
    marginV: '0',
    effect: '',
    text,
    originalText: text,
    isTranslatable: partial.isTranslatable ?? true,
  };
}

test('PlayResY-proportional font size and thin border scale cleanly across 360p, 720p, 1080p', () => {
  assert.equal(getMinReadableBurmeseFontSize(360), 11.5);
  assert.equal(getMinReadableBurmeseFontSize(720), 23);
  assert.equal(getMinReadableBurmeseFontSize(1080), 34.6);

  assert.equal(getSmallSignFontThreshold(360), 20.2);
  assert.equal(getSmallSignFontThreshold(1080), 60.5);

  const bord360 = getProportionalThinBorder(11.5, 360);
  assert.ok(bord360 >= 0.45 && bord360 <= 0.55, `expected ~0.46 at 360p, got ${bord360}`);

  const bord1080 = getProportionalThinBorder(34.6, 1080);
  assert.ok(bord1080 >= 1.35 && bord1080 <= 1.6, `expected ~1.38 at 1080p, got ${bord1080}`);
});

test('prepareSignTextForTranslation cleans inline mid-word tags while preserving \\N line breaks', () => {
  const raw = '{\\pos(120,80)\\fs11\\c&HFFFFFF&}S{\\c&H00FFFF&}-rank {\\b1}Guild\\N{\\fs9}Special Quest';
  assert.equal(prepareSignTextForTranslation(raw), 'S-rank Guild\\NSpecial Quest');
  assert.equal(extractLeadingTagBlock(raw), '{\\pos(120,80)\\fs11\\c&HFFFFFF&}');
});

test('parseDrawingBoxBounds parses {\p1} vector box and expandDrawingBoxWidth expands width by +15%', () => {
  const boxText = '{\\an7\\pos(180,100)\\bord0\\c&HFFFFFF&\\p1}m 0 0 l 120 0 l 120 180 l 0 180';
  assert.equal(isPureDrawing(boxText), true);

  const bounds = parseDrawingBoxBounds(boxText, 640, 360);
  assert.ok(bounds);
  assert.equal(bounds.width, 120);
  assert.equal(bounds.height, 180);
  assert.equal(bounds.screenLeft, 180);
  assert.equal(bounds.screenRight, 300);
  assert.equal(bounds.screenTop, 100);
  assert.equal(bounds.screenBottom, 280);

  const expanded = expandDrawingBoxWidth(boxText, 1.15);
  const expandedBounds = parseDrawingBoxBounds(expanded, 640, 360);
  assert.ok(expandedBounds);
  assert.equal(expandedBounds.width, 138); // 120 * 1.15 = 138
  assert.equal(expandedBounds.height, 180); // height stays unchanged
});

test('Smart Hybrid: auto-fits Burmese text inside {\p1} box (expands width +15% and height when \\N overflows), \\fs11.5, \\b0 & thin \\bord', () => {
  const boxEvent = makeEvent({
    index: 0,
    style: 'Sign_Basic',
    name: 'SIGN',
    start: '0:01:00.00',
    end: '0:01:06.00',
    text: '{\\an7\\pos(180,100)\\bord0\\c&HFFFFFF&\\p1}m 0 0 l 110 0 l 110 34 l 0 34',
    isTranslatable: false,
  });

  const signTextEvent = makeEvent({
    index: 1,
    style: 'Sign_Basic',
    name: 'SIGN',
    start: '0:01:00.00',
    end: '0:01:06.00',
    text: '{\\an7\\pos(188,106)\\fs8\\b1\\bord1.5}အထူးစွမ်းရည် ဓားရှည်တိုက်ခိုက်မှု အဆင့်မြှင့်တင်ထားသည်\\Nရန်သူအားလုံးကို တစ်ချက်တည်းဖြင့် ချေမှုန်းနိုင်စွမ်း ရှိပါသည်',
  });

  const out = applySignTypesettingMode([boxEvent, signTextEvent], [true, true], {
    mode: 'smart_hybrid',
    playResX: 640,
    playResY: 360,
  });

  // Box should expand by +15% width (110 -> 126.5) AND expand height > 34 so wrapped lines never spill out the bottom
  const expandedBounds = parseDrawingBoxBounds(out[0].text, 640, 360);
  assert.ok(expandedBounds);
  assert.equal(expandedBounds.width, 126.5);
  assert.ok(expandedBounds.height > 34, `expected box height > 34 to prevent bottom spill, got ${expandedBounds.height}`);

  // Sign text should have PlayResY min font (11.5) or incrementally shrunk safety floor (down to 9.0), \b0, thin \bord, \fscx85, and \N auto-wrap
  assert.match(out[1].text, /\\fs(?:11\.5|11|10\.5|10|9\.5|9)(?![\d.])/);
  assert.match(out[1].text, /\\b0/);
  assert.match(out[1].text, /\\bord0\.45/);
  assert.match(out[1].text, /\\fscx85/);
  assert.ok(out[1].text.includes('\\N'), `expected auto-wrapped \\N in ${out[1].text}`);
});

test('Zero-width Burmese combining marks do not inflate line width or cause premature 35-40% early wraps', () => {
  const plain = 'တက'; // 2 wide consonants (2 * 0.45 * 10 = 9.0px)
  const withMarks = 'တိုက်'; // တ + ို (\u102D\u102F = 0) + က + ် (\u103A = 0) -> must equal exact same width 9.0px!
  assert.equal(
    Math.round(estimateBurmeseLineWidthPx(withMarks, 10, 100) * 100),
    Math.round(estimateBurmeseLineWidthPx(plain, 10, 100) * 100)
  );

  // Also stacked virama consonant (စ + ် + စ) shouldn't add full second consonant horizontal width
  const stacked = 'ပစ္စည်း';
  const widthAt12 = estimateBurmeseLineWidthPx(stacked, 12, 100);
  assert.ok(widthAt12 < 26, `expected compact width for 7-codepoint stacked word "${stacked}" at fs12, got ${widthAt12}`);
});

test('Smart Hybrid {\p1} box uses available_h = (box_y + box_h) - text_y - padding and 1.35x Burmese line-height safety fit', () => {
  // Box starts at y=100, height=135 (bottom=235). Text starts near bottom-half at y=195 -> available_h = 235 - 195 - 5 = 35px!
  const boxEvent = makeEvent({
    index: 0,
    style: 'Sign_Basic',
    name: 'SIGN',
    start: '0:02:00.00',
    end: '0:02:06.00',
    text: '{\\an7\\pos(180,100)\\bord0\\c&HFFFFFF&\\p1}m 0 0 l 250 0 l 250 135 l 0 135',
    isTranslatable: false,
  });

  // Two lines at fs11.5 * 1.35 = 31.05 <= 35 (fits), whereas 3 lines at fs11.5 * 1.35 = 46.58 > 35!
  // Shrinking to fs9.0 -> 3 * 9 * 1.35 = 36.45 > 35 (or fits in 2 lines when re-wrapped? Let's test 4 explicit \N lines starting at y=195!)
  const midBoxText = makeEvent({
    index: 1,
    style: 'Sign_Basic',
    name: 'SIGN',
    start: '0:02:00.00',
    end: '0:02:06.00',
    text: '{\\an7\\pos(190,195)\\fs11.5\\b1}စာကြောင်း ၁\\Nစာကြောင်း ၂\\Nစာကြောင်း ၃\\Nစာကြောင်း ၄',
  });

  const out = applySignTypesettingMode([boxEvent, midBoxText], [true, true], {
    mode: 'smart_hybrid',
    playResX: 640,
    playResY: 360,
  });

  // Even though 4 lines (~48-62px) is MUCH smaller than full box_h (135px), because text starts at y=195
  // available_h is only 35px! Safety fit must first try shrinking \fs down to 9.0 AND expand box bottom height
  // so (100 + newBoxHeight) covers y=195 + 4 * 9.0 * 1.35 = 243.6 -> newBoxHeight >= 144!
  assert.match(out[1].text, /\\fs9(?![\d.])/);
  const boundsAfter = parseDrawingBoxBounds(out[0].text, 640, 360);
  assert.ok(boundsAfter);
  assert.ok(boundsAfter.height >= 144, `expected box height >= 144 so y=195 4-line text stays inside, got ${boundsAfter.height}`);
});

test('Smart Hybrid: applies Y-spacing cap (fs <= y_gap * 0.9) on stacked RPG status lines without cross-column interference', () => {
  const stackedLines: AssEvent[] = [
    makeEvent({ index: 0, start: '0:05:00.00', end: '0:05:05.00', text: '{\\an7\\pos(40,100)\\fs14\\b1}တိုက်ခိုက်အား ၅၅' }),
    makeEvent({ index: 1, start: '0:05:00.00', end: '0:05:05.00', text: '{\\an7\\pos(40,111)\\fs14\\b1}ကာကွယ်အား ၁၁၂' }),
    makeEvent({ index: 2, start: '0:05:00.00', end: '0:05:05.00', text: '{\\an7\\pos(40,122)\\fs14\\b1}မှော်စွမ်းအား ၆၅' }),
    // Distant column on the right with same narrow Y-gap shouldn't shrink unless its own column is tight; this one has big Y gap 40
    makeEvent({ index: 3, start: '0:05:00.00', end: '0:05:05.00', text: '{\\an9\\pos(600,105)\\fs14\\b1}အဆင့် ၆၄' }),
  ];

  const out = applySignTypesettingMode(stackedLines, [true, true, true, true], {
    mode: 'smart_hybrid',
    playResX: 640,
    playResY: 360,
  });

  // Left stacked rows have y_gap = 11 -> max fs = 11 * 0.9 = 9.9!
  assert.match(out[0].text, /\\fs9\.9/);
  assert.match(out[1].text, /\\fs9\.9/);
  assert.match(out[2].text, /\\fs9\.9/);
  // Must NOT wrap into 2 lines since left-aligned \pos(40,y) has plenty of horizontal room
  assert.ok(!out[0].text.includes('\\N'));
  // Right column row is horizontally separate so it keeps its original fs14
  assert.match(out[3].text, /\\fs14/);
});

test('Smart Hybrid: upgrades ALL disclaimer sections (including 3rd/4th section without inline \\bord0) to Sign_Box with zero \\bord override', () => {
  const disclaimerLines: AssEvent[] = [
    makeEvent({
      index: 0,
      style: 'Sign_Basic',
      name: 'text (top section)',
      start: '0:00:00.00',
      end: '0:00:05.00',
      text: '{\\an8\\c&HCECECE&\\bord0\\shad0\\fs11\\q2\\pos(316,120)}ကုန်သွယ်မှုလွန်ကဲရင် နေ့စဉ်ဘဝကို ထိခိုက်နိုင်ပါတယ်\\Nသတိထားပါ',
    }),
    makeEvent({
      index: 1,
      style: 'Sign_Basic',
      name: 'text (3rd section)',
      start: '0:00:00.00',
      end: '0:00:05.00',
      // Simulates 3rd/4th section that inherited borderless style or scaled thin \bord0.33 without explicit \bord0
      text: '{\\an8\\c&HCECECE&\\bord0.33\\shad0\\fs11\\q2\\pos(316,220)}တရားဝင်မှတ်ပုံတင်မထားသော လုပ်ငန်းများဖြင့် မဆောင်ရွက်ပါနှင့်',
    }),
    makeEvent({
      index: 2,
      style: 'Sign_Basic',
      name: 'text (4th section)',
      start: '0:00:00.00',
      end: '0:00:05.00',
      // No \bord tag at all inline
      text: '{\\an8\\c&HCECECE&\\fs11\\pos(316,290)}မိမိကိုယ်ပိုင် ဆုံးဖြတ်ချက်ဖြင့်သာ ဆောင်ရွက်ပါ',
    }),
  ];

  const out = applySignTypesettingMode(disclaimerLines, [true, true, true], {
    mode: 'smart_hybrid',
    playResX: 640,
    playResY: 360,
  });

  for (const ev of out) {
    assert.equal(ev.style, 'Sign_Box');
    assert.doesNotMatch(ev.text, /\\bord/); // Must NEVER inject \bord0.46/.51 so Style's bord=3 padding applies!
    assert.doesNotMatch(ev.text, /\\q2/);
    assert.match(ev.text, /\\fs11\.5/);
    assert.match(ev.text, /\\b0/);
  }
});

test('Smart Hybrid: wraps unboxed \\pos(x,y) sign text so it never shoots off the screen edge', () => {
  const longPosSign = makeEvent({
    index: 0,
    style: 'Sign_Basic',
    name: 'SIGN',
    text: '{\\an7\\pos(420,160)\\fs16\\b1}ရန်သူအားလုံးကို တစ်ချက်တည်းဖြင့် အနိုင်ယူနိုင်သော ရှေးဟောင်းဒဏ္ဍာရီလာ ဓားမီးလျှံ',
  });

  const out = applySignTypesettingMode([longPosSign], [true], {
    mode: 'smart_hybrid',
    playResX: 640,
    playResY: 360,
  });

  assert.ok(out[0].text.includes('\\N'), 'long left-aligned \\pos(420,160) sign must auto-wrap before 640px edge');
  assert.match(out[0].text, /\\fscx90/);
});

test('Clean Top Mode: comments out {\p1} boxes, strips tags, sets Sign_Basic, and auto-merges 4+ simultaneous signs with |', () => {
  const events: AssEvent[] = [
    makeEvent({
      index: 0,
      text: '{\\an7\\pos(100,100)\\p1}m 0 0 l 150 0 l 150 100 l 0 100',
      isTranslatable: false,
    }),
    makeEvent({ index: 1, style: 'CustomTitle', text: '{\\pos(40,100)\\fs8\\c&HFFFF00&}တိုက်ခိုက်အား ၅၅' }),
    makeEvent({ index: 2, style: 'CustomTitle', text: '{\\pos(40,120)\\fs8\\c&HFFFF00&}ကာကွယ်အား ၁၁၂' }),
    makeEvent({ index: 3, style: 'CustomTitle', text: '{\\pos(40,140)\\fs8\\c&HFFFF00&}မှော်စွမ်းအား ၆၅' }),
    makeEvent({ index: 4, style: 'CustomTitle', text: '{\\pos(40,160)\\fs8\\c&HFFFF00&}အမြန်နှုန်း ၆၅' }),
    makeEvent({ index: 5, style: 'CustomTitle', text: '{\\pos(40,180)\\fs8\\c&HFFFF00&}အဆင့် ၆၄' }),
  ];

  const merged = applySignTypesettingMode(events, [true, true, true, true, true, true], {
    mode: 'clean_top',
    denseSignStrategy: 'merge',
    playResX: 640,
    playResY: 360,
  });

  assert.equal(merged[0].type, 'Comment'); // {\p1} mask commented out
  assert.equal(merged[1].type, 'Dialogue');
  assert.equal(merged[1].style, 'Sign_Basic');
  assert.doesNotMatch(merged[1].text, /\{[^}]*\}/); // tags stripped
  assert.ok(merged[1].text.includes(' | '), `expected pipe-merged text, got: ${merged[1].text}`);
  assert.ok(merged[1].text.split('\\N').length <= 3, 'merged dense signs should stay within 2-3 lines');
  // Remaining cluster members converted to Comment: so timestamps & event count stay 100% intact
  assert.equal(merged[2].type, 'Comment');
  assert.equal(merged[3].type, 'Comment');
  assert.equal(merged[4].type, 'Comment');
  assert.equal(merged[5].type, 'Comment');

  // Also test denseSignStrategy === 'comment'
  const commented = applySignTypesettingMode(events, [true, true, true, true, true, true], {
    mode: 'clean_top',
    denseSignStrategy: 'comment',
    playResX: 640,
    playResY: 360,
  });
  for (const ev of commented) {
    assert.equal(ev.type, 'Comment');
  }
});

test('buildAss + validateAss integration succeeds across smart_hybrid, clean_top, and preserve_original with Sign_Box in styles', () => {
  const sampleAss = `[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Sign_01,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Sign_01,text (top section),0,0,0,,{\\an8\\c&HCECECE&\\bord0\\shad0\\fs33\\q2\\pos(948,579)}အလွန်အကျွံ အရောင်းအဝယ်ပြုလုပ်ခြင်းသည် နေ့စဉ်ဘဝကို ထိခိုက်စေနိုင်သည်\\Nသေချာစဉ်းစားပါ
Dialogue: 0,0:00:06.00,0:00:10.00,Sign_01,SIGN,0,0,0,,{\\an7\\pos(300,300)\\p1}m 0 0 l 360 0 l 360 210 l 0 210
Dialogue: 0,0:00:06.00,0:00:10.00,Sign_01,SIGN,0,0,0,,{\\an7\\pos(320,320)\\fs27\\b1\\bord3}သံမဏိနက်ချပ်ဝတ်တန်ဆာ အထူးကာကွယ်မှုစွမ်းရည်
Dialogue: 0,0:00:11.00,0:00:14.00,Default,Hero,0,0,0,,မင်္ဂလာပါ\\Nသွားကြစို့
`;

  const parsed = parseAss(sampleAss);
  assert.equal(isSignOrTitle(parsed.events[0].name, parsed.events[0].style, parsed.events[0].text), true);

  // 1. Smart Hybrid
  const hybridOut = buildAss(parsed, parsed.events, true, [], { mode: 'smart_hybrid' });
  assert.ok(hybridOut.includes('Style: Sign_Box,Myanmar Text,14,&H00FFFFFF,&H000000FF,&H40000000,&H40000000,0,0,0,0,100,100,0,0,3,3,0,8,14,14,14,1'));
  assert.ok(hybridOut.includes('PlayResX: 640'));
  assert.ok(hybridOut.includes(',Sign_Box,text (top section),'));
  // Spoken dialogue still strips \N into single line
  assert.ok(hybridOut.includes('မင်္ဂလာပါ သွားကြစို့'));
  const hybridVal = validateAss(parsed, hybridOut, 0);
  assert.equal(hybridVal.valid, true, `hybrid validation errors: ${hybridVal.errors.join('; ')}`);

  // 2. Clean Top
  const cleanTopOut = buildAss(parsed, parsed.events, true, [], {
    mode: 'clean_top',
    denseSignStrategy: 'merge',
  });
  assert.ok(cleanTopOut.includes('Comment: 0,0:00:06.00,0:00:10.00,Sign_Basic,SIGN,'));
  const cleanVal = validateAss(parsed, cleanTopOut, 0);
  assert.equal(cleanVal.valid, true, `clean_top validation errors: ${cleanVal.errors.join('; ')}`);

  // 3. Preserve Original
  const origOut = buildAss(parsed, parsed.events, true, [], { mode: 'preserve_original' });
  assert.ok(origOut.includes('\\q2'));
  const origVal = validateAss(parsed, origOut, 0);
  assert.equal(origVal.valid, true, `preserve_original validation errors: ${origVal.errors.join('; ')}`);
});

test('stripTypesettingTagsForCleanTop and formatMergedDenseSigns handle edge cases cleanly', () => {
  assert.equal(stripTypesettingTagsForCleanTop('{\\pos(10,20)\\fs12}Hello\\hWorld\\N'), 'Hello World');
  const merged = formatMergedDenseSigns(['A', 'B', 'C', 'D', 'E', 'F'], 3);
  assert.equal(merged, 'A | B | C\\ND | E | F');
  assert.equal(wrapBurmeseTextToWidth('စမ်းသပ်စာသား', 200, 16, 100), 'စမ်းသပ်စာသား');
});
