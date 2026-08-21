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
    this._own = '';
    this.hidden = false;
    this.value = '';
    this.disabled = false;
  }
  set className(value) { this.classList.set = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classList.set].join(' '); }
  /* textContent aggregates child text the way the real DOM does, so a cell built
     from mixed nodes (a caret button plus the correspondence text) reads back as
     its full text. Setting it replaces the children, as the DOM does. */
  set textContent(value) {
    this._own = value === undefined || value === null ? '' : String(value);
    this.children = [];
  }
  get textContent() {
    return this._own + this.children.map((c) => c.textContent).join('');
  }
  appendChild(child) {
    if (typeof child === 'string') {
      const text = new Element('#text');
      text._own = child;
      this.children.push(text);
      return text;
    }
    this.children.push(child);
    return child;
  }
  append(...nodes) { nodes.forEach((n) => this.appendChild(n)); }
  set innerHTML(value) { if (value === '') this.children = []; this._html = value; }
  get innerHTML() { return this._html ?? ''; }
  setAttribute(name, value) { (this.attributes ||= {})[name] = String(value); }
  getAttribute(name) { return (this.attributes ||= {})[name] ?? null; }
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
  'example-note', 'example-stats', 'version', 'timing', 'guide', 'guide-open',
  'guide-close', 'guide-title', 'guide-body', 'guide-steps', 'guide-load', 'file',
  'download-json', 'download-summary', 'residue', 'residue-hint',
  'events', 'events-hint', 'crossdim', 'crossdim-panel', 'crossdim-hint',
  'baseline-panel', 'baseline-body', 'scorer', 'baseline', 'bootstrap',
  'classes-hint', 'segment-input', 'segment-preview',
  'format', 'class-filter', 'class-chips', 'class-count', 'copy-summary',
  'baseline-check', 'baseline-unavailable', 'predictive', 'predictive-panel',
  'predictive-body', 'predictive-table', 'provenance', 'provenance-body',
  'copy-csv', 'copy-markdown',
  'lect-pair', 'lect-pair-wrap', 'drift-panel', 'drift-list',
  'chunks-panel', 'chunks', 'chunks-hint',
]) {
  byId.set(id, new Element(
    ['classes', 'residue', 'events', 'crossdim', 'predictive-table', 'chunks'].includes(id) ? 'table' : 'div'));
}
// The tables need a tbody for app.js to fill.
const tbody = new Element('tbody');
byId.get('classes').appendChild(tbody);
byId.get('residue').appendChild(new Element('tbody'));
byId.get('events').appendChild(new Element('tbody'));
byId.get('crossdim').appendChild(new Element('tbody'));
byId.get('predictive-table').appendChild(new Element('tbody'));
byId.get('chunks').appendChild(new Element('tbody'));

// The classes table has sortable headers app.js wires and reads; the shim
// carries them so a sort click has something to act on.
for (const key of ['default', 'count', 'sets', 'rate']) {
  const th = new Element('th');
  th.dataset.sort = key;
  byId.get('classes').appendChild(th);
}
// The filter chips, so a chip click reaches the same handler the page wires.
for (const chip of ['all', 'conditioned', 'unconditioned', 'recurring', 'changes', 'weak', 'stands']) {
  const button = new Element('button');
  button.dataset.chip = chip;
  // The page defaults: the kind filter is 'all' and the retention rows hide.
  if (chip === 'all' || chip === 'changes') button.classList.add('on');
  byId.get('class-chips').appendChild(button);
}

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
  fetch: () => Promise.resolve({ ok: false, text: () => Promise.resolve('') }),
};
context.window = context;
context.globalThis = context;
context.URL = { createObjectURL: () => 'blob:', revokeObjectURL() {} };

vm.createContext(context);
for (const file of ['reading.js', 'corpora.js', 'guide-content.js']) {
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
  // None need be blocked; whatever is must say what blocks it.
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

/* Re-render the latin/spanish model, so a view test starts from a known table
   rather than whichever model the previous check left rendered. */
const rerender = () => worker.onmessage(
  { data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1 } });

const classRowCount = () => byId.get('classes').querySelectorAll('tr[data-class-id]').length;

/* The page defaults hide retentions and weak conditioned rows. Checks that
   read the whole table lift both first, through the same chips a visitor
   would use. */
const chipButton = (name) =>
  byId.get('class-chips').children.find((c) => c.dataset.chip === name);
const showEverything = () => {
  rerender();
  if (chipButton('changes').classList.contains('on')) chipButton('changes').click();
  if (!chipButton('weak').classList.contains('on')) chipButton('weak').click();
};

/* The decision list is a finding: each rule was committed against what the
   earlier ones left unexplained. The JSON sorts every table by id, so a
   consumer that iterates it as it arrives publishes the list scrambled -- which
   is what this page did until decisionOrder went in, rendering Verner as
   0, 4, 1, 3 where the CLI rendered 0, 1, 3, 4. The shim was here and the
   linkage was checked; the order simply was not. */
check('conditioned classes render in the order they were decided', () => {
  const conditioned = model.classes.conditioned;
  assert.ok(conditioned.length > 1, 'need at least two conditioned classes to order');
  const byRow = new Map(conditioned.map((c) => [String(c.id), c.decision_index]));
  const rendered = byId.get('classes').querySelectorAll('tr[data-class-id]')
    .map((row) => byRow.get(row.dataset.classId))
    .filter((index) => index !== undefined);
  assert.deepEqual(rendered, [...rendered].sort((a, b) => a - b),
    `conditioned rows rendered out of decision order: ${rendered.join(', ')}`);
});

/* Two in five committed rules are X ~ X, and most are the retention side of a
   real split with the change in the contrast class. "p ~ p before a vowel" is a
   null statement to read, so the row has to say which half is the event. */
check('a retention row names the change it is the complement of', () => {
  showEverything();
  const conditioned = model.classes.conditioned;
  const identity = conditioned.find((entry) => {
    if (new Set(entry.segments.map((s) => s.grapheme)).size !== 1) return false;
    const contrast = [...conditioned, ...model.classes.unconditioned]
      .find((c) => c.id === entry.contrast_class_id);
    return contrast && new Set(contrast.segments.map((s) => s.grapheme)).size > 1;
  });
  assert.ok(identity, 'no retention row with a non-identity contrast in this corpus');
  const row = byId.get('classes')
    .querySelectorAll(`tr[data-class-id="${identity.id}"]`)[0];
  const env = row.querySelectorAll('.env')[0];
  assert.match(env.textContent, /unchanged here; the change is/);
});

/* Cognate sets ranked by how badly they align, with the two split statistics
   read together above them. The residue is the part a comparativist wants
   first, and it is not an error term. */
/* The events pane. Latin/Spanish, the corpus this file already trains, has
   nothing to group -- so this asserts the quiet answer, which is the one a
   grouper that fires on anything would get wrong. The loud answer is asserted
   in C against the natural-class fixture. */
check('the events pane renders one row per proposed event', () => {
  const events = model.proposed_events || [];
  const rendered = byId.get('events').querySelectorAll('td.corr');
  assert.equal(rendered.length, events.length);
  for (let i = 0; i < events.length; i += 1) {
    for (const member of events[i].members) {
      assert.ok(rendered[i].textContent.includes(member.lect),
        `event row omits ${member.lect}`);
    }
  }
});

check('the residue pane ranks every scored set and reads the split', () => {
  const rows = byId.get('residue').querySelectorAll('tr[data-cognate]');
  assert.equal(rows.length, model.outliers.length);
  const scores = rows.map((row) => Number(row.querySelectorAll('.z')[0].textContent));
  assert.deepEqual(scores, [...scores].sort((a, b) => b - a),
    'residue rows are not ranked worst-first');
  assert.ok(byId.get('residue-hint').textContent.length > 0, 'no reading of the split');
});

check('selecting a residue row shows that set and nothing else', () => {
  const rows = byId.get('residue').querySelectorAll('tr[data-cognate]');
  const target = rows[0].dataset.cognate;
  rows[0].click();
  const shown = byId.get('alignments').querySelectorAll('.alignment:not(.hidden)');
  assert.ok(shown.length > 0, 'selecting a set hid everything');
  for (const block of shown) {
    assert.equal(block.dataset.cognate, target);
  }
  rows[0].click();
});

check('a result renders classes and alignments', () => {
  showEverything();
  const rows = byId.get('classes').querySelectorAll('tr[data-class-id]');
  assert.equal(rows.length,
    model.classes.unconditioned.length + model.classes.conditioned.length);
  assert.equal(byId.get('alignments').querySelectorAll('.alignment').length,
    model.alignments.length);
  assert.ok(byId.get('results').classList.contains('active'));
});

check('every conditioned class shows its environment', () => {
  showEverything();
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
  showEverything();
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
  showEverything();
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

/* A second model through the same page, because the corpus above has nothing
   to group and a pane is only tested by content. natural_class is four
   conditioned classes that are one change, and the fixture exists so both
   halves -- the grouping and the naming -- have a corpus that shows them. */
check('the events pane names a grouped class', () => {
  const grouped = JSON.parse(execFileSync(
    cli, ['train', '--json', join(repo, 'testdata/soundlaws/natural_class.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  assert.ok(grouped.proposed_events.length > 0, 'fixture stopped producing events');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(grouped), elapsedMs: 1 } });

  const rendered = byId.get('events').querySelectorAll('td.corr');
  assert.equal(rendered.length, grouped.proposed_events.length);
  const voicing = rendered.find((cell) => cell.textContent.includes('v,z'));
  assert.ok(voicing, `no event row for the voicing class: ${
    rendered.map((c) => c.textContent).join(' | ')}`);
  assert.match(voicing.textContent, /\[[^\]]* & [^\]]*\]/,
    'the grapheme set is not named by its features');
  assert.match(voicing.textContent, /Δ:.*voiced/,
    'the event row should show the shared displacement');

  /* Clicking an event row selects its member classes and filters alignments to
     those realising them; clicking a member class highlights the event back.
     The member classes in the natural_class fixture are conditioned with
     standing=unmeasured, so enable the weak chip first. */
  if (!chipButton('weak').classList.contains('on')) chipButton('weak').click();

  const eventRows = byId.get('events').querySelectorAll('tr[data-event-index]');
  assert.ok(eventRows.length > 0, 'no event rows have data-event-index');
  const firstEvent = grouped.proposed_events[0];
  eventRows[0].click();
  const selectedClasses = byId.get('classes').querySelectorAll('tr.selected');
  assert.equal(selectedClasses.length, firstEvent.class_ids.length,
    'selecting an event did not highlight its member classes');
  const visibleAlignments = byId.get('alignments').querySelectorAll('.alignment:not(.hidden)');
  assert.ok(visibleAlignments.length > 0, 'selecting an event hid all alignments');
  assert.ok(visibleAlignments.length < grouped.alignments.length,
    'selecting an event did not filter alignments');

  eventRows[0].click();
  const afterDeselect = byId.get('classes').querySelectorAll('tr.selected');
  assert.equal(afterDeselect.length, 0, 'clicking the event again did not clear selection');

  const classRow = byId.get('classes').querySelectorAll('tr[data-class-id]')[0];
  const classId = Number(classRow.dataset.classId);
  classRow.click();
  const highlightedEvents = byId.get('events').querySelectorAll('tr.selected');
  const owningEvents = grouped.proposed_events.filter((e) => e.class_ids.includes(classId));
  assert.equal(highlightedEvents.length, owningEvents.length,
    'selecting a class did not highlight the events containing it');
  classRow.click();
});

/* A change on one segment still reaches the events pane.
   umlaut_synthetic states its change on a single conditioned class and has no
   second row anywhere to group with, which used to leave the pane blank --
   read by anyone opening a textbook umlaut corpus as "no change was found". */
check('a change stated on one class still reaches the events pane', () => {
  const single = JSON.parse(execFileSync(
    cli, ['train', '--json', '--format', 'wide',
      join(repo, 'experiments/umlaut_synthetic/cognates.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  assert.equal(single.proposed_events.length, 1, 'fixture stopped stating one change');
  assert.equal(single.proposed_events[0].class_ids.length, 1,
    'the event grouped, so this no longer tests the single-class case');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(single), elapsedMs: 1 } });

  const rendered = byId.get('events').querySelectorAll('td.corr');
  assert.equal(rendered.length, 1, 'the single-class event did not render');
  assert.match(rendered[0].textContent, /æ/,
    `the event row does not name the fronting: ${rendered[0].textContent}`);
});

/* An event that conditions on something has to show it, or the row reads as an
   unconditioned correspondence and the half that makes it a finding is gone.
   conditioned_confound is `k ~ tʃ` after a sonorant, and is built so the corpus
   cannot tell that conditioner from "before a front vowel" -- so the row has to
   carry the tie as well. */
check('an event shows the environment its members condition on', () => {
  const confound = JSON.parse(execFileSync(
    cli, ['train', '--json', join(repo, 'testdata/soundlaws/conditioned_confound.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  assert.equal(confound.proposed_events.length, 1, 'fixture stopped stating one change');
  assert.ok(confound.proposed_events[0].environment_alternatives > 0,
    'fixture stopped being confounded');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(confound), elapsedMs: 1 } });

  const cells = byId.get('events').querySelectorAll('td.env-cell');
  assert.equal(cells.length, 1, 'the events table has no environment column');
  assert.notEqual(cells[0].textContent.replace('tied', '').trim(), '—',
    'the event row shows no environment');
  assert.match(cells[0].textContent, /sonorant/,
    `the environment is not the one the class conditions on: ${cells[0].textContent}`);
  assert.match(cells[0].textContent, /tied/,
    'the row does not mark the environment as one the corpus cannot pin down');

  /* And where there is nothing to condition on, the column says so plainly
     rather than being left to read as an environment of none. Grimm's shifts
     are unconditioned; the corpus also states one conditioned vowel class,
     which is an event in its own right and does carry an environment, so this
     checks the displacement rows rather than all of them. */
  const grimm = JSON.parse(execFileSync(
    cli, ['train', '--json', join(repo, 'testdata/soundlaws/grimm.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  const shifts = grimm.proposed_events.filter((e) => e.axis === 'displacement');
  assert.ok(shifts.length > 1, 'grimm stopped grouping its shifts');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(grimm), elapsedMs: 1 } });
  const cellsByRow = byId.get('events').querySelectorAll('td.env-cell');
  assert.equal(cellsByRow.length, grimm.proposed_events.length, 'a row is missing its cell');
  for (let i = 0; i < grimm.proposed_events.length; i += 1) {
    if (grimm.proposed_events[i].axis !== 'displacement') continue;
    assert.equal(cellsByRow[i].textContent, '—',
      'an unconditioned grouping claimed an environment');
  }
});

/* And when the pane really is empty, it says why rather than "(none)": a bare
   "(none)" reads as "the change was not found". metathesis_adjacent conditions
   nothing -- what moved there is the order of two segments, which is a chunk. */
check('an empty events pane says why it is empty', () => {
  const none = JSON.parse(execFileSync(
    cli, ['train', '--json', join(repo, 'testdata/soundlaws/metathesis_adjacent.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  assert.equal(none.proposed_events.length, 0, 'fixture started producing events');
  assert.equal(none.classes.conditioned.length, 0, 'fixture started conditioning');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(none), elapsedMs: 1 } });

  const hint = byId.get('events-hint').textContent;
  assert.ok(hint.length > 0, 'the empty pane said nothing about why it is empty');
  assert.match(hint, /chunk/, `the hint does not say where a reordering went: ${hint}`);
});

/* Cross-dimensional rules -- a segmental environment predicting a tone -- have
   a pane of their own, hidden on the corpora that produce none so it is never
   an empty promise. Latin/Spanish has none; the tone fixture has them, which is
   why it earned a place on the ladder. */
check('the cross-dimensional pane renders its rules, and hides when there are none', () => {
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1 } });
  assert.equal(model.cross_dimensional.length, 0, 'latin/spanish unexpectedly has cross-dim rules');
  assert.ok(byId.get('crossdim-panel').hidden, 'the empty cross-dim pane was not hidden');

  const tone = JSON.parse(execFileSync(
    cli, ['train', '--json', '--format', 'wide',
      join(repo, 'experiments/tone_chinese_like/cognates.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  assert.ok(tone.cross_dimensional.length > 0, 'tone fixture stopped producing cross-dim rules');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(tone), elapsedMs: 1 } });

  const rendered = byId.get('crossdim').querySelectorAll('td.corr');
  assert.equal(rendered.length, tone.cross_dimensional.length);
  assert.ok(!byId.get('crossdim-panel').hidden, 'the cross-dim pane hid despite having rows');
  for (let i = 0; i < tone.cross_dimensional.length; i += 1) {
    const row = tone.cross_dimensional[i];
    assert.ok(rendered[i].textContent.includes(row.conditioned_lect),
      `cross-dim row omits the conditioned lect ${row.conditioned_lect}`);
    assert.ok(rendered[i].textContent.includes(row.dimension),
      `cross-dim row omits the dimension ${row.dimension}`);
  }
});

/* The shuffled-baseline verdict shows only when the shuffle was run, since the
   default run does not measure it. The fields are injected here rather than
   spending twenty retrainings in a unit test; that the shuffle produces them is
   checked in C. */
check('the baseline pane appears only when the shuffle was run', () => {
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1 } });
  assert.ok(byId.get('baseline-panel').hidden, 'the baseline pane showed without a shuffle');

  const shuffled = JSON.parse(JSON.stringify(model));
  Object.assign(shuffled.fit, {
    permutation_count: 20,
    null_cost_per_segment_mean: -0.6,
    null_cost_per_segment_sd: 0.05,
    cost_per_segment_z: -24.8,
    rules_measured: 3,
    rules_above_noise: 3,
  });
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(shuffled), elapsedMs: 1 } });
  assert.ok(!byId.get('baseline-panel').hidden, 'the baseline pane hid despite a shuffle');
  assert.ok(byId.get('baseline-body').textContent.includes('shuffled baseline'),
    'the baseline pane did not read the verdict');
});

/* Every class carries an interval on the rate it applies at -- Wilson on a
   plain run, bootstrap once resampling is on. The bar is drawn from lower/upper
   and takes the accent class only when the method is bootstrap, which is the
   one visible sign the option did anything. */
check('every class shows its rate interval, marked when bootstrapped', () => {
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1 } });
  if (chipButton('changes').classList.contains('on')) chipButton('changes').click();
  if (!chipButton('weak').classList.contains('on')) chipButton('weak').click();
  const first = model.classes.unconditioned[0];
  const rows = byId.get('classes').querySelectorAll('tr[data-class-id]');
  const row = rows.find((r) => r.dataset.classId === String(first.id));
  const point = row.querySelectorAll('.point')[0];
  assert.ok(point, 'no rate point rendered');
  assert.equal(point.textContent, first.uncertainty.estimate.toFixed(2));
  const track = row.querySelectorAll('.ci')[0];
  assert.ok(track && !track.classList.contains('boot'),
    'the Wilson interval was marked as bootstrap');
  assert.ok(byId.get('classes-hint').textContent.includes('Wilson'),
    'the hint did not name the Wilson interval');

  const boot = JSON.parse(JSON.stringify(model));
  /* Bootstrap is all-or-nothing: a real run marks every class, so the test
     does too rather than flipping one and reading the hint off another. */
  for (const c of [...boot.classes.unconditioned, ...boot.classes.conditioned]) {
    c.uncertainty.method = 'bootstrap';
  }
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(boot), elapsedMs: 1 } });
  /* A fresh render resets the defaults; lift them again before reading rows. */
  if (chipButton('changes').classList.contains('on')) chipButton('changes').click();
  if (!chipButton('weak').classList.contains('on')) chipButton('weak').click();
  const brow = byId.get('classes').querySelectorAll('tr[data-class-id]')
    .find((r) => r.dataset.classId === String(first.id));
  assert.ok(brow.querySelectorAll('.ci')[0].classList.contains('boot'),
    'the bootstrap interval was not marked');
  assert.ok(byId.get('classes-hint').textContent.includes('bootstrap'),
    'the hint did not switch to bootstrap');
});

/* A real corpus trains a hundred classes; the filter is how a comparativist
   reaches the one they came for. It matches the written correspondence, so a
   segment typed into it narrows the table to the rows that mention it. */
check('the filter narrows the classes to those matching the query', () => {
  showEverything();
  const total = classRowCount();
  assert.ok(total > 5, 'need a corpus with enough classes to filter');
  const filter = byId.get('class-filter');
  filter.value = 'spanish:f';
  (filter.listeners.input || []).forEach((h) => h({ target: filter }));
  const shown = classRowCount();
  assert.ok(shown > 0 && shown < total, `filter did not narrow: ${shown}/${total}`);
  for (const row of byId.get('classes').querySelectorAll('tr[data-class-id]')) {
    assert.match(row.querySelectorAll('td.corr')[0].textContent, /f/);
  }
  assert.ok(byId.get('class-count').textContent.includes(`of ${total}`),
    'the count did not report the filtered total');
});

/* The chips are layered filters. "changes only" hides the retention rows --
   the X ~ X classes where every lect keeps the segment -- which is how a reader
   asks the table for what actually moved. It is on by default: the first table
   a visitor sees is the changes, and the retentions are one click away. */
check('the changes-only chip hides the identity rows, by default', () => {
  rerender();
  /* Weak rows stay hidden here so the count compares like with like. */
  if (!chipButton('weak').classList.contains('on')) chipButton('weak').click();
  const isIdentity = (entry) => new Set(entry.segments.map((s) => s.grapheme)).size === 1;
  const total = [...model.classes.unconditioned, ...model.classes.conditioned].length;
  const changing = [...model.classes.unconditioned, ...model.classes.conditioned]
    .filter((c) => !isIdentity(c)).length;
  const chip = chipButton('changes');
  assert.ok(chip.classList.contains('on'), 'changes-only is not the default');
  assert.equal(classRowCount(), changing,
    `default table showed ${classRowCount()}, expected the ${changing} changes`);
  chip.click();
  assert.equal(classRowCount(), total, 'toggling the chip off did not restore the retentions');
  chip.click();
  assert.equal(classRowCount(), changing, 'toggling the chip back on did not hide the retentions');
  chipButton('weak').click();
});

/* Conditioned rows that are thin or were never measured against the shuffle
   are not peers of the rows that carry the corpus; they sit behind the weak
   chip rather than in the first table a visitor reads. */
check('the weak chip reveals thin and unmeasured conditioned rows', () => {
  rerender();
  if (chipButton('changes').classList.contains('on')) chipButton('changes').click();
  const isWeak = (entry) => entry.count < 8
    || entry.standing === 'unmeasured' || entry.standing === 'within noise';
  const weak = model.classes.conditioned.filter(isWeak).length;
  const strong = model.classes.conditioned.length - weak;
  const uncond = model.classes.unconditioned.length;
  assert.ok(weak > 0, 'this corpus has no weak conditioned rows to hide');
  assert.equal(classRowCount(), uncond + strong,
    `default showed ${classRowCount()}, expected ${uncond + strong}`);
  chipButton('weak').click();
  assert.equal(classRowCount(), uncond + strong + weak,
    'the weak chip did not restore the hidden rows');
});

/* An environment the corpus cannot identify uniquely is marked on the row,
   not only in a hint, so it cannot be quoted as the environment. */
check('a confounded environment is marked tied', () => {
  showEverything();
  const tied = model.classes.conditioned.find((c) => (c.environment_alternatives || 0) > 0);
  assert.ok(tied, 'this corpus has no confounded environment to mark');
  const row = byId.get('classes').querySelectorAll('tr[data-class-id]')
    .find((r) => r.dataset.classId === String(tied.id));
  assert.ok(row, 'the confounded row is not rendered');
  assert.ok(row.textContent.includes('tied'), 'the confounded row does not say tied');
});

/* Spans of more than one segment reach the page beside the model, and a
   reordering reads as one -- metathesis is visible without --pairwise. */
check('the chunks pane renders spans and reorderings', () => {
  const chunks = JSON.stringify({ ok: true, pairs: [{
    source_lect: 'latin', target_lect: 'spanish',
    chunks: [
      { source: 'k t', target: 't͡ʃ', count: 4, reordering: false, transparency: 0.8 },
      { source: 'e r', target: 'r e', count: 3, reordering: true, transparency: 0.9 },
    ],
  }] });
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), chunks, elapsedMs: 1 } });
  assert.ok(!byId.get('chunks-panel').hidden, 'the chunks pane stayed hidden');
  const rows = byId.get('chunks').querySelectorAll('tr');
  assert.equal(rows.length, 2);
  assert.ok(rows[0].textContent.includes('k t ~ t͡ʃ'), 'the span did not render');
  assert.ok(rows[1].textContent.includes('reordering'), 'the reordering was not marked');

  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1 } });
  assert.ok(byId.get('chunks-panel').hidden, 'a run without chunks did not hide the pane');
});

/* Transcription drift is said above the model: it reads as a correspondence
   once trained, and it is not one. */
check('the drift banner appears when the corpus carries drift', () => {
  const drift = JSON.stringify({ ok: true, drift: [
    { lect: 'broad', other_lect: 'narrow', grapheme: 'tʃ', written_as: 't ʃ',
      corroborated: 13, forms: 13 },
  ] });
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), drift, elapsedMs: 1 } });
  assert.ok(!byId.get('drift-panel').hidden, 'the drift banner stayed hidden');
  assert.ok(byId.get('drift-list').textContent.includes('tʃ'), 'the drift row did not render');

  worker.onmessage({ data: { type: 'result', json: JSON.stringify(model), elapsedMs: 1 } });
  assert.ok(byId.get('drift-panel').hidden, 'a run without drift did not hide the banner');
});

/* Sorting is a view over the same rows: clicking count orders the table by it,
   descending first, without changing which rows are present. */
check('sorting by count orders the visible rows', () => {
  rerender();
  const th = byId.get('classes').querySelectorAll('th[data-sort]').find((t) => t.dataset.sort === 'count');
  th.click();
  const counts = byId.get('classes').querySelectorAll('tr[data-class-id] td.count')
    .map((c) => Number(c.textContent));
  assert.deepEqual(counts, [...counts].sort((a, b) => b - a), 'rows are not sorted by count descending');
});

/* The evidence drawer: expanding a conditioned row opens a block that reads the
   score, the confound and the supporting cognates the class rests on. This is
   the "why this rule?" data that used to be JSON-only. */
check('expanding a conditioned class opens its evidence, and it reads the score and sets', () => {
  showEverything();
  const cond = model.classes.conditioned[0];
  const row = byId.get('classes').querySelectorAll('tr[data-class-id]')
    .find((r) => r.dataset.classId === String(cond.id));
  // No drawer until the caret is clicked.
  assert.equal(byId.get('classes').querySelectorAll('tr.evidence-row').length, 0,
    'a drawer was open before any caret click');
  const caret = row.querySelectorAll('.caret')[0];
  assert.ok(caret, 'the conditioned row has no evidence caret');
  caret.click();
  const drawers = byId.get('classes').querySelectorAll('tr.evidence-row');
  assert.equal(drawers.length, 1, 'the evidence drawer did not open');
  const text = drawers[0].textContent;
  assert.match(text, /ΔBIC/, 'the drawer omits the committing score');
  for (const set of cond.supporting_cognates.slice(0, 2)) {
    assert.ok(text.includes(set), `the drawer omits supporting set ${set}`);
  }
  assert.ok(drawers[0].querySelectorAll('.ev-copy')[0], 'no copy affordance for the sets');
  caret.click();
  assert.equal(byId.get('classes').querySelectorAll('tr.evidence-row').length, 0,
    'the drawer did not close');
});

/* Standing is a per-rule verdict that only exists once the shuffled baseline
   ran; injected here (a real run of it is checked in C), the drawer reads it and
   the `stands` chip filters to the rules that clear it. */
check('the drawer reads standing and the stands chip filters to it', () => {
  const withStanding = JSON.parse(JSON.stringify(model));
  /* A shuffled-baseline run is what measures standing, so the fit carries the
     baseline fields too; without them renderBaseline would read undefined. */
  Object.assign(withStanding.fit, {
    permutation_count: 20, null_cost_per_segment_mean: -0.6, null_cost_per_segment_sd: 0.05,
    cost_per_segment_z: -24.8, rules_measured: 2, rules_above_noise: 1,
  });
  const cond = withStanding.classes.conditioned;
  assert.ok(cond.length >= 2, 'need two conditioned classes for this check');
  cond.forEach((c, i) => { c.standing = i === 0 ? 'above noise' : 'within noise'; });
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(withStanding), elapsedMs: 1 } });

  const row = byId.get('classes').querySelectorAll('tr[data-class-id]')
    .find((r) => r.dataset.classId === String(cond[0].id));
  row.querySelectorAll('.caret')[0].click();
  const drawer = byId.get('classes').querySelectorAll('tr.evidence-row')[0];
  assert.match(drawer.textContent, /STANDS/, 'the drawer does not read the standing verdict');

  const chip = byId.get('class-chips').children.find((c) => c.dataset.chip === 'stands');
  chip.click();
  const shown = byId.get('classes').querySelectorAll('tr[data-class-id]');
  assert.ok(shown.length >= 1, 'the stands filter hid everything');
  for (const r of shown) {
    const c = [...cond, ...withStanding.classes.unconditioned].find((x) => String(x.id) === r.dataset.classId);
    assert.equal(c.standing, 'above noise', `stands filter kept a non-standing rule ${r.dataset.classId}`);
  }
  chip.click();
});

/* Held-out prediction has a pane of its own, present only when cross-validation
   ran. The fields are injected rather than spending real folds in a unit test;
   that a fold run produces them is checked in C. */
check('the predictive pane appears only when cross-validation ran, and reads it', () => {
  rerender();
  assert.ok(byId.get('predictive-panel').hidden, 'the predictive pane showed without a run');

  const cv = JSON.parse(JSON.stringify(model));
  const score = (t1, tk, loss) => ({
    observation_count: 100, unseen_reflexes: 2, observation_weight: 100,
    log_loss: loss, top1_coverage: t1, top_k_coverage: tk, brier_score: 0.3,
    calibration_error: 0.1, abstention_rate: 0, accepted_top1_coverage: t1,
  });
  cv.fit.predictive = {
    status: 'not_confirmed', observation_unit: 'auto', folds: 4, log_loss_gain: -0.02,
    conditioned: score(0.79, 0.9, 0.95), unconditioned: score(0.81, 0.91, 0.86),
    identity: score(0.70, 0.8, 1.2), inventory_frequency: score(0.09, 0.2, 3.0),
    feature_distance: score(0.70, 0.8, 1.2),
  };
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(cv), elapsedMs: 1 } });
  assert.ok(!byId.get('predictive-panel').hidden, 'the predictive pane hid despite a run');
  assert.match(byId.get('predictive-body').textContent, /held out|naive baseline|generalis/i);
  const rows = byId.get('predictive-table').querySelectorAll('tr');
  assert.equal(rows.length, 5, `expected five model rows, got ${rows.length}`);
  assert.ok(byId.get('predictive-table').querySelectorAll('td')
    .some((c) => c.textContent === '79%'), 'the trained coverage is not shown');
});

/* Provenance is emitted on every run; the block reads the feature system and
   the corpus checksum a citation needs. */
check('the provenance block reads the feature system and checksum', () => {
  rerender();
  const text = byId.get('provenance-body').descendants().map((n) => n.textContent).join(' ');
  assert.ok(text.includes(model.provenance.feature_system), 'no feature system in provenance');
  assert.ok(text.includes(model.provenance.corpus_checksum), 'no corpus checksum in provenance');
});

/* The alignment gloss reassembles both forms, so the block reads as the words
   it aligns and not only as a grid. */
check('an alignment names the two forms it aligns', () => {
  rerender();
  const first = model.alignments[0];
  const src = first.links.map((l) => l.source.join('')).join('');
  const gloss = byId.get('alignments').querySelectorAll('.pair')[0];
  assert.ok(gloss.textContent.includes(src), `the source form ${src} is not shown`);
});

/* A segment loss is a correspondence to ∅, not a separate kind of fact: it sits
   in the correspondence table like any other row, written with ∅ on one side.
   The gaps_and_length fixture carries several. */
check('losses appear as ∅ correspondences in the classes table', () => {
  const gapModel = JSON.parse(execFileSync(
    cli, ['train', '--json', join(repo, 'testdata/corpora/gaps_and_length.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  const nullClasses = [...gapModel.classes.unconditioned, ...gapModel.classes.conditioned]
    .filter((c) => c.segments.some((s) => s.grapheme === '∅'));
  assert.ok(nullClasses.length > 0, 'fixture stopped producing ∅ correspondences');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(gapModel), elapsedMs: 1 } });
  const withNull = [...byId.get('classes').querySelectorAll('tr[data-class-id]')]
    .filter((r) => /∅/.test(r.textContent));
  assert.ok(withNull.length > 0, 'no ∅ correspondence rendered in the classes table');
});

/* The residue carries each set's confidence, but the column is worth showing
   only where it varies -- on an unweighted corpus every set is 1.0. */
check('the residue confidence column shows only when confidence varies', () => {
  const uniform = JSON.parse(JSON.stringify(model));
  uniform.outliers.forEach((o) => { o.confidence = 1; });
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(uniform), elapsedMs: 1 } });
  assert.ok(byId.get('residue').classList.contains('hide-conf'),
    'the confidence column showed on an unweighted corpus');

  const varied = JSON.parse(JSON.stringify(model));
  varied.outliers.forEach((o, i) => { o.confidence = i === 0 ? 0.4 : 1; });
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(varied), elapsedMs: 1 } });
  assert.ok(!byId.get('residue').classList.contains('hide-conf'),
    'the confidence column hid despite variation');
  const first = byId.get('residue').querySelectorAll('tr[data-cognate]')[0];
  assert.match(first.querySelectorAll('td.conf')[0].textContent, /%/, 'no confidence percent shown');
});

/* The paper-ready export carries the evidence columns the summary download
   drops, in a table a write-up can paste. */
check('the CSV and Markdown exports carry the evidence columns', () => {
  rerender();
  const csv = vm.runInContext('correspondenceCsv()', context);
  const header = csv.split('\n')[0];
  for (const column of ['correspondence', 'environment', 'delta_bic', 'standing', 'confidence']) {
    assert.ok(header.includes(column), `CSV header omits ${column}`);
  }
  assert.ok(csv.includes('conditioned'), 'CSV has no conditioned rows');
  assert.match(csv, /-\d+\.\d{2}/, 'CSV carries no committing score');

  const md = vm.runInContext('correspondenceMarkdown()', context);
  const lines = md.split('\n');
  assert.match(lines[0], /^\| kind \| correspondence \|/, 'Markdown has no header row');
  assert.match(lines[1], /\| --- \|/, 'Markdown has no separator row');
  assert.ok(lines[2].startsWith('|'), 'Markdown has no body rows');
});

/* On a multi-lect corpus the alignments are one block per lect pair; the
   selector narrows them to one. Hidden on the two-lect corpora the other checks
   use, so it takes a four-lect model of its own. */
check('the lect-pair selector filters alignments on a multi-lect corpus', () => {
  const multi = JSON.parse(execFileSync(
    cli, ['train', '--json', join(repo, 'testdata/corpora/real_romance_4lect.tsv')],
    { encoding: 'utf8', maxBuffer: 1 << 28 }));
  assert.ok(multi.lects.length > 2, 'fixture is not multi-lect');
  worker.onmessage({ data: { type: 'result', json: JSON.stringify(multi), elapsedMs: 1 } });
  assert.ok(!byId.get('lect-pair-wrap').hidden, 'the selector is hidden on a multi-lect corpus');

  const select = byId.get('lect-pair');
  const options = select.querySelectorAll('option');
  assert.ok(options.length > 2, 'no lect pairs offered');
  const pair = options[1].value;
  select.value = pair;
  (select.listeners.change || []).forEach((h) => h({ target: select }));

  const visible = byId.get('alignments').querySelectorAll('.alignment:not(.hidden)');
  assert.ok(visible.length > 0, 'the pair filter hid every alignment');
  for (const block of visible) {
    assert.equal(block.dataset.pair, pair, 'an alignment outside the chosen pair stayed visible');
  }

  // Two-lect corpora keep the selector hidden.
  rerender();
  assert.ok(byId.get('lect-pair-wrap').hidden, 'the selector showed on a two-lect corpus');
});

/* The result surface is a set of activatable rows, columns and headers. It has
   to be reachable and announced without a mouse. */
check('the result surface is keyboard-operable and announced', () => {
  rerender();
  const row = byId.get('classes').querySelectorAll('tr[data-class-id]')[0];
  assert.equal(row.getAttribute('tabindex'), '0', 'class rows are not focusable');
  assert.equal(row.getAttribute('role'), 'button', 'class rows carry no button role');
  assert.equal(row.getAttribute('aria-selected'), 'false', 'class row is not marked unselected');
  (row.listeners.keydown || []).forEach((h) => h({ key: 'Enter', preventDefault() {} }));
  assert.equal(row.getAttribute('aria-selected'), 'true', 'Enter did not select the row');

  const chip = byId.get('class-chips').children.find((c) => c.dataset.chip === 'recurring');
  assert.equal(chip.getAttribute('aria-pressed'), 'false', 'chip has no aria-pressed');
  chip.click();
  assert.equal(chip.getAttribute('aria-pressed'), 'true', 'chip aria-pressed did not update');
  chip.click();

  const th = byId.get('classes').querySelectorAll('th[data-sort]').find((t) => t.dataset.sort === 'count');
  th.click();
  assert.equal(th.getAttribute('aria-sort'), 'descending', 'sorted header carries no aria-sort');

  const sr = row.querySelectorAll('.sr-only')[0];
  assert.ok(sr && /CI/.test(sr.textContent), 'the interval numbers are not voiced for a screen reader');
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
