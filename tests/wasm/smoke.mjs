// WebAssembly smoke test.
//
// The load-bearing assertion is that the browser build produces byte-identical
// JSON to the native CLI on real corpora. A demo that quietly disagreed with
// the library would be worse than no demo, and the two builds differ in ways
// that could bite: a 32-bit target, a different compiler, and merkmal's
// Unicode fallback.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const repo = dirname(dirname(here));
const cli = process.env.REGULAE_CLI ?? join(repo, 'build/c/regulae');

const createRegulae = (await import(join(repo, 'web/regulae.js'))).default;

let cancelAfter = 0;
let progressCalls = [];
const Module = await createRegulae({
  onProgress: (stage, completed, total) => {
    progressCalls.push({ stage, completed, total });
    return cancelAfter > 0 && progressCalls.length >= cancelAfter;
  },
});

const train = Module.cwrap('regulae_train_json', 'number', ['string', 'string', 'string']);
const segment = Module.cwrap('regulae_segment_json', 'number', ['string']);
const version = Module.cwrap('regulae_version', 'string', []);

function call(fn, ...args) {
  const ptr = fn(...args);
  const text = Module.UTF8ToString(ptr);
  Module.ccall('regulae_free', null, ['number'], [ptr]);
  return text;
}

const checks = [];
function check(name, fn) {
  try {
    fn();
    checks.push([name, null]);
  } catch (error) {
    checks.push([name, error.message.split('\n')[0]]);
  }
}

check('version matches the library', () => {
  assert.match(version(), /^\d+\.\d+\.\d+$/);
});

// The whole point of the exercise.
for (const name of [
  'three_lect_basic.tsv',
  'conditioned_multilect.tsv',
  'long_range.tsv',
  'real_contaminated.tsv',
  'real_tone_chinese.tsv',
  'real_ppn_hawaiian.tsv',
  'real_latin_spanish.tsv',
]) {
  check(`byte-identical to native: ${name}`, () => {
    const path = join(repo, 'testdata/parity', name);
    cancelAfter = 0;
    const fromWasm = call(train, readFileSync(path, 'utf8'), 'tsv', null);
    const fromNative = execFileSync(cli, ['train', '--json', path], {
      encoding: 'utf8',
      maxBuffer: 1 << 28,
    }).trim();
    assert.equal(fromWasm, fromNative);
  });
}

// The wide loader runs merkmal's segmenter, which is where a Unicode
// difference between the two builds would show up.
check('byte-identical to native: wide format with non-ASCII', () => {
  const path = join(repo, 'experiments/latin_spanish/cognates.tsv');
  cancelAfter = 0;
  const fromWasm = call(train, readFileSync(path, 'utf8'), 'wide', null);
  const fromNative = execFileSync(cli, ['train', '--json', '--format', 'wide', path], {
    encoding: 'utf8',
    maxBuffer: 1 << 28,
  }).trim();
  assert.equal(fromWasm, fromNative);
});

check('segmentation keeps multi-codepoint graphemes whole', () => {
  const parsed = JSON.parse(call(segment, 'pʰater'));
  assert.deepEqual(parsed.segments, ['pʰ', 'a', 't', 'e', 'r']);
  // A combining diacritic attaches to its base, and both normalisation forms
  // agree. This is the case merkmal's fallback path could plausibly get wrong.
  assert.deepEqual(JSON.parse(call(segment, 'ẽ')).segments,
                   JSON.parse(call(segment, 'ẽ')).segments);
});

check('repeated calls are deterministic', () => {
  const path = join(repo, 'testdata/parity/conditioned_multilect.tsv');
  const text = readFileSync(path, 'utf8');
  cancelAfter = 0;
  assert.equal(call(train, text, 'tsv', null), call(train, text, 'tsv', null));
});

check('progress is reported and monotonic', () => {
  cancelAfter = 0;
  progressCalls = [];
  call(train, readFileSync(join(repo, 'testdata/parity/three_lect_basic.tsv'), 'utf8'), 'tsv', null);
  assert.ok(progressCalls.length > 0);
  const { total } = progressCalls[0];
  let previous = 0;
  for (const p of progressCalls) {
    assert.equal(p.total, total);
    assert.ok(p.completed >= previous);
    assert.ok(p.completed <= p.total);
    previous = p.completed;
  }
  assert.equal(progressCalls[progressCalls.length - 1].completed, total);
});

check('cancellation stops the run', () => {
  cancelAfter = 3;
  progressCalls = [];
  const parsed = JSON.parse(
    call(train, readFileSync(join(repo, 'testdata/parity/real_ppn_hawaiian.tsv'), 'utf8'), 'tsv', null));
  assert.equal(parsed.ok, false);
  assert.equal(parsed.status, 'cancelled');
  assert.ok(progressCalls.length < 8, 'stopped early rather than running to completion');
  cancelAfter = 0;
});

check('errors come back as payloads, not exceptions', () => {
  const cases = [
    ['gloss\ta\tb\nx\tpQQa\tfa\ny\tpi\tpi', 'wide', null, 'unknown grapheme'],
    ['', 'wide', null, 'invalid argument'],
    ['gloss\ta\tb\nx\tpa\tfa', 'nope', null, 'unsupported option'],
    ['gloss\ta\nx\tpa', 'wide', null, 'parse error'],
    ['gloss\ta\tb\nx\tpa\tfa', 'wide', '{"max_chunk_sizze":2}', 'unsupported option'],
    ['gloss\ta\tb\nx\tpa\tfa', 'wide', 'not json', 'parse error'],
  ];
  for (const [corpus, format, options, expected] of cases) {
    const parsed = JSON.parse(call(train, corpus, format, options));
    assert.equal(parsed.ok, false);
    assert.equal(parsed.status, expected, `for format=${format} options=${options}`);
  }
});

check('options are honoured', () => {
  // gaps_and_length is the corpus that actually promotes chunks; on one where
  // chunking never fires, max_chunk_size correctly changes nothing.
  const text = readFileSync(join(repo, 'testdata/parity/gaps_and_length.tsv'), 'utf8');
  const defaults = call(train, text, 'tsv', null);
  const narrowed = call(train, text, 'tsv', '{"max_chunk_size":1}');
  assert.notEqual(defaults, narrowed, 'max_chunk_size should change the model');
});

let failed = 0;
for (const [name, error] of checks) {
  if (error) {
    failed++;
    console.log(`FAIL  ${name}\n      ${error}`);
  } else {
    console.log(`ok    ${name}`);
  }
}
console.log(`\n${checks.length - failed}/${checks.length} checks passed`);
process.exit(failed === 0 ? 0 : 1);
