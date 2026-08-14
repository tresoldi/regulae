// Exercises web/app.js against a real model, under a minimal DOM shim.
//
// There is no browser here, but the thing worth checking does not need one:
// that the page's rendering agrees with the shape the library actually emits,
// and that selecting a class filters the alignments to exactly the ones the
// model says realise it. A mismatch between app.js and the payload is the most
// likely way this page breaks, and the least visible.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const repo = dirname(dirname(here));
const cli = process.env.REGULAE_CLI ?? join(repo, 'build/c/regulae');

/* ---- a DOM, to the extent app.js uses one ------------------------------ */

class ClassList {
  constructor(element) { this.element = element; this.set = new Set(); }
  add(...names) { names.forEach((n) => this.set.add(n)); }
  remove(...names) { names.forEach((n) => this.set.delete(n)); }
  contains(name) { return this.set.has(name); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : force;
    if (on) this.set.add(name); else this.set.delete(name);
    return on;
  }
}

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.classList = new ClassList(this);
    this.listeners = {};
    this.textContent = '';
    this.hidden = false;
    this.value = '';
    this.disabled = false;
  }
  set className(value) { this.classList.set = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classList.set].join(' '); }
  appendChild(child) { this.children.push(child); return child; }
  append(...nodes) { nodes.forEach((n) => this.appendChild(n)); }
  set innerHTML(value) { if (value === '') this.children = []; this._html = value; }
  get innerHTML() { return this._html ?? ''; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  click() { (this.listeners.click || []).forEach((h) => h({ stopPropagation() {} })); }
  scrollIntoView() {}
  descendants() { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  querySelectorAll(selector) { return this.descendants().filter(matcher(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
}

/* Supports exactly the selector forms app.js uses. */
function matcher(selector) {
  const not = selector.match(/:not\(\.([\w-]+)\)/);
  const base = selector.replace(/:not\([^)]*\)/, '');
  const attr = base.match(/\[data-([\w-]+)(?:="([^"]*)")?\]/);
  const cls = base.match(/\.([\w-]+)/);
  const tag = base.match(/^([a-z]+)/);
  const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
  return (element) => {
    if (tag && element.tagName !== tag[1].toUpperCase()) return false;
    if (cls && !element.classList.contains(cls[1])) return false;
    if (attr) {
      const value = element.dataset[camel(attr[1])];
      if (value === undefined) return false;
      if (attr[2] !== undefined && String(value) !== attr[2]) return false;
    }
    if (not && element.classList.contains(not[1])) return false;
    return true;
  };
}

const byId = new Map();
for (const id of [
  'editor', 'run', 'cancel', 'progress', 'progress-fill', 'progress-label', 'error',
  'results', 'summary', 'classes', 'alignments', 'alignments-hint', 'examples',
  'example-note', 'version', 'timing', 'guide', 'guide-open', 'guide-close',
  'guide-title', 'guide-body', 'guide-steps', 'guide-load', 'file',
  'download-json', 'download-summary',
]) {
  byId.set(id, new Element(id === 'classes' ? 'table' : 'div'));
}
// The classes table needs a tbody for app.js to fill.
const tbody = new Element('tbody');
byId.get('classes').appendChild(tbody);

const posted = [];
class FakeWorker {
  constructor() { FakeWorker.instances.push(this); this.terminated = false; }
  postMessage(message) { posted.push(message); }
  terminate() { this.terminated = true; }
}
FakeWorker.instances = [];

const context = {
  console, JSON, Object, Number, String, Array, Set, Math, Boolean, URL,
  document: {
    getElementById: (id) => byId.get(id) ?? new Element('div'),
    createElement: (tag) => new Element(tag),
    addEventListener() {},
  },
  Worker: FakeWorker,
  Blob: class {},
};
context.window = context;
context.globalThis = context;
context.URL = { createObjectURL: () => 'blob:', revokeObjectURL() {} };

vm.createContext(context);
for (const file of ['corpora.js', 'guide-content.js']) {
  vm.runInContext(readFileSync(join(repo, 'web', file), 'utf8'), context, { filename: file });
}
vm.runInContext(readFileSync(join(repo, 'web/app.js'), 'utf8'), context, { filename: 'app.js' });

// The generated files declare their data with const, which lives in the
// context's lexical scope rather than as a property of it.
const CORPORA = vm.runInContext('CORPORA', context);
const CORPUS_LIST = vm.runInContext('CORPUS_LIST', context);
const GUIDE_STEPS = vm.runInContext('GUIDE_STEPS', context);

/* ---- checks ------------------------------------------------------------ */

const checks = [];
const check = (name, fn) => {
  try { fn(); checks.push([name, null]); }
  catch (error) { checks.push([name, error.message.split('\n')[0]]); }
};

check('the editor starts on the ladder\'s first corpus', () => {
  assert.ok(byId.get('editor').value.length > 0, 'editor is empty on load');
  assert.equal(byId.get('editor').value, CORPORA[CORPUS_LIST[0].path]);
});

check('the example list offers every corpus, grouped', () => {
  const options = byId.get('examples').descendants().filter((e) => e.tagName === 'OPTION');
  // One placeholder plus every corpus.
  assert.equal(options.length, CORPUS_LIST.length + 1);
  const unreadable = CORPUS_LIST.filter((e) => !e.readable);
  assert.ok(unreadable.length > 0, 'expected some corpora to be unreadable');
  for (const entry of unreadable) {
    const option = options.find((o) => o.value === entry.path);
    assert.ok(option.textContent.includes(entry.blockedBy),
      `${entry.path} does not say what blocks it`);
  }
});

check('the guide renders and its steps are wired', () => {
  assert.equal(byId.get('guide-steps').children.length, GUIDE_STEPS.length);
  assert.equal(byId.get('guide-title').textContent, GUIDE_STEPS[0].title);
});

// The heart of it: render a real model and check the linkage.
const model = JSON.parse(execFileSync(
  cli, ['train', '--json', join(repo, 'testdata/corpora/real_latin_spanish.tsv')],
  { encoding: 'utf8', maxBuffer: 1 << 28 }));

const worker = FakeWorker.instances[0];
worker.onmessage({ data: { type: 'ready', version: '0.1.0' } });
worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1234 } });

check('a result renders classes and alignments', () => {
  const rows = byId.get('classes').querySelectorAll('tr[data-class-id]');
  assert.equal(rows.length,
    model.classes.unconditioned.length + model.classes.conditioned.length);
  assert.equal(byId.get('alignments').querySelectorAll('.alignment').length,
    model.alignments.length);
  assert.ok(byId.get('results').classList.contains('active'));
});

check('every conditioned class shows its environment', () => {
  const rows = byId.get('classes').querySelectorAll('tr[data-class-id]');
  for (const entry of model.classes.conditioned) {
    const row = rows.find((r) => r.dataset.classId === String(entry.id));
    const env = row.querySelectorAll('.env')[0];
    assert.ok(env, `conditioned class ${entry.id} has no environment shown`);
    assert.ok(env.textContent.length > 0 && env.textContent !== 'conditioned',
      `conditioned class ${entry.id} rendered an empty environment`);
  }
});

check('selecting a class shows exactly the alignments the model says realise it', () => {
  const conditioned = model.classes.conditioned[0];
  const expected = model.alignments
    .filter((a) => a.links.some((l) => (l.classes || []).includes(conditioned.id)))
    .length;
  assert.ok(expected > 0, 'the model claims no alignments for this class');

  const row = byId.get('classes')
    .querySelectorAll('tr[data-class-id]')
    .find((r) => r.dataset.classId === String(conditioned.id));
  row.click();

  const shown = byId.get('alignments').querySelectorAll('.alignment:not(.hidden)').length;
  assert.equal(shown, expected);
  const lit = byId.get('alignments').querySelectorAll('.col')
    .filter((c) => c.classList.contains('lit')).length;
  assert.ok(lit > 0, 'no columns were highlighted');
  assert.ok(byId.get('alignments-hint').textContent.includes(String(expected)));
});

check('a conditioned class lights fewer columns than its unconditioned twin', () => {
  const key = (c) => c.segments.map((s) => `${s.lect}:${s.grapheme}`).join('|');
  const plain = new Map(model.classes.unconditioned.map((c) => [key(c), c.id]));
  const pair = model.classes.conditioned.find((c) => plain.has(key(c)));
  assert.ok(pair, 'expected a conditioned class sharing segments with an unconditioned one');

  const litFor = (id) => {
    const row = byId.get('classes').querySelectorAll('tr[data-class-id]')
      .find((r) => r.dataset.classId === String(id));
    row.click();
    const count = byId.get('alignments').querySelectorAll('.col')
      .filter((c) => c.classList.contains('lit')).length;
    row.click();
    return count;
  };
  assert.ok(litFor(pair.id) <= litFor(plain.get(key(pair))));
});

check('selecting again clears the selection', () => {
  const row = byId.get('classes').querySelectorAll('tr[data-class-id]')[0];
  row.click();
  row.click();
  assert.equal(
    byId.get('alignments').querySelectorAll('.alignment:not(.hidden)').length,
    model.alignments.length);
});

check('a failed run reports the status rather than rendering', () => {
  worker.onmessage({ data: { type: 'result', elapsedMs: 1,
    json: JSON.stringify({ ok: false, status: 'unknown grapheme', detail: 'grapheme "+" is unknown' }) } });
  assert.ok(byId.get('error').children.length > 0, 'nothing was reported');
  const text = byId.get('error').children[0].children.map((c) => c.textContent).join(' ');
  assert.ok(text.includes('unknown grapheme'));
  assert.ok(text.includes('+'));
});

check('cancelling terminates the worker and starts another', () => {
  const before = FakeWorker.instances.length;
  byId.get('cancel').click();
  assert.ok(worker.terminated, 'the running worker was not terminated');
  assert.equal(FakeWorker.instances.length, before + 1, 'no replacement worker was started');
});

let failed = 0;
for (const [name, error] of checks) {
  if (error) { failed++; console.log(`FAIL  ${name}\n      ${error}`); }
  else { console.log(`ok    ${name}`); }
}
console.log(`\n${checks.length - failed}/${checks.length} checks passed`);
process.exit(failed === 0 ? 0 : 1);
