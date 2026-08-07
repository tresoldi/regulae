// Exercises web/worker.js under a shim of the worker globals.
//
// The worker cannot be run as a real Web Worker here, but its message protocol
// is the contract the page depends on, and that is testable: importScripts,
// self.postMessage and self.onmessage are all it touches.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const repo = dirname(dirname(here));
const require = createRequire(import.meta.url);

const messages = [];
const context = {
  console,
  Date,
  Infinity,
  Object,
  String,
  setTimeout,
  clearTimeout,
  fetch,
  TextDecoder,
  performance,
  importScripts(name) {
    context.createRegulae = require(join(repo, 'web', name));
  },
  postMessage(message) {
    messages.push(message);
  },
};
context.self = context;
context.globalThis = context;
vm.createContext(context);
vm.runInContext(readFileSync(join(repo, 'web/worker.js'), 'utf8'), context, { filename: 'worker.js' });

const ready = await new Promise((resolve, reject) => {
  const started = Date.now();
  const poll = () => {
    const message = messages.find((m) => m.type === 'ready' || m.type === 'fatal');
    if (message) return resolve(message);
    if (Date.now() - started > 20000) return reject(new Error('worker never became ready'));
    setTimeout(poll, 20);
  };
  poll();
});

const checks = [];
const check = (name, fn) => {
  try { fn(); checks.push([name, null]); }
  catch (error) { checks.push([name, error.message.split('\n')[0]]); }
};

check('reports ready with a version', () => {
  assert.equal(ready.type, 'ready');
  assert.match(ready.version, /^\d+\.\d+\.\d+$/);
});

const run = (data) => {
  messages.length = 0;
  context.onmessage({ data });
  return messages;
};

check('a train request yields progress then a result', () => {
  const corpus = readFileSync(join(repo, 'testdata/parity/three_lect_basic.tsv'), 'utf8');
  const out = run({ type: 'train', corpus, format: 'tsv' });
  const progress = out.filter((m) => m.type === 'progress');
  const result = out.find((m) => m.type === 'result');
  assert.ok(progress.length > 0, 'expected progress messages');
  assert.ok(result, 'expected a result message');
  assert.equal(progress[progress.length - 1].completed, progress[0].total);
  assert.ok(typeof result.elapsedMs === 'number');
  const model = JSON.parse(result.json);
  assert.equal(model.ok, true);
  assert.deepEqual(model.lects, ['alpha', 'beta', 'gamma']);
});

check('a bad corpus comes back as a result, not a fatal', () => {
  const out = run({ type: 'train', corpus: 'gloss\ta\tb\nx\tpQQ\tfa\ny\tpi\tpi', format: 'wide' });
  assert.ok(!out.some((m) => m.type === 'fatal'), 'corpus problems are not engine failures');
  const model = JSON.parse(out.find((m) => m.type === 'result').json);
  assert.equal(model.ok, false);
  assert.equal(model.status, 'unknown grapheme');
  assert.match(model.detail, /grapheme/);
});

check('the deadline cancels a run', () => {
  const corpus = readFileSync(join(repo, 'testdata/parity/real_ppn_hawaiian.tsv'), 'utf8');
  const out = run({ type: 'train', corpus, format: 'tsv', timeoutMs: 1 });
  const model = JSON.parse(out.find((m) => m.type === 'result').json);
  assert.equal(model.ok, false);
  assert.equal(model.status, 'cancelled');
});

check('the deadline does not leak into the next run', () => {
  const corpus = readFileSync(join(repo, 'testdata/parity/three_lect_basic.tsv'), 'utf8');
  const model = JSON.parse(run({ type: 'train', corpus, format: 'tsv' })
    .find((m) => m.type === 'result').json);
  assert.equal(model.ok, true);
});

check('an unknown request is reported', () => {
  const out = run({ type: 'nonsense' });
  assert.equal(out[0].type, 'fatal');
});

let failed = 0;
for (const [name, error] of checks) {
  if (error) { failed++; console.log(`FAIL  ${name}\n      ${error}`); }
  else { console.log(`ok    ${name}`); }
}
console.log(`\n${checks.length - failed}/${checks.length} checks passed`);
process.exit(failed === 0 ? 0 : 1);
