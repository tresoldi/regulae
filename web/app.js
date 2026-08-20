/*
 * Page logic.
 *
 * The one interaction worth building carefully is the link between a
 * correspondence class and the alignment columns realising it. That mapping
 * comes from the model rather than being re-derived here: matching graphemes in
 * JavaScript cannot tell a conditioned class from the unconditioned one over
 * the same segments, which is precisely the distinction the page exists to
 * show. Each link in the payload carries the class ids it realises.
 */

/* global CORPORA, CORPUS_LIST, GUIDE_STEPS,
   correspondence, decisionOrder, environment, eventCorrespondence, isIdentity,
   residueReading */

const $ = (id) => document.getElementById(id);

const editor = $("editor");
const runButton = $("run");
const cancelButton = $("cancel");
const progress = $("progress");
const results = $("results");

let worker = null;
let ready = false;
let running = false;
let model = null;
let currentFormat = "wide";
let selectedClass = null;
let selectedSet = null;
let selectedPair = null;
let baselineSupported = true;
let modelChunks = null;
let modelDrift = null;

/* The classes table is a view over one array, so filtering and sorting are
   state the view reads rather than a rebuild of the model. classRows is built
   once per run; classView says which subset to show and in what order. */
let classRows = [];
const classView = { query: "", chip: "all", recurring: false, changes: true, weak: false, stands: false, sort: "default", desc: true };
/* Which classes have their evidence drawer open. Kept as ids so the open state
   survives a filter or sort re-render rather than being tied to a DOM row. */
const expandedClasses = new Set();

/* ---- worker lifecycle -------------------------------------------------- */

function startWorker() {
  worker = new Worker("worker.js");
  worker.onmessage = (event) => {
    const message = event.data;
    if (message.type === "ready") {
      ready = true;
      runButton.disabled = false;
      $("version").textContent = "v" + message.version;
      setBaselineAvailability(message.baselineSupported !== false);
    } else if (message.type === "progress") {
      showProgress(message);
    } else if (message.type === "result") {
      finish(message);
    } else if (message.type === "segment_result") {
      showSegmentation(message);
    } else if (message.type === "fatal") {
      stopRunning();
      showError("The engine failed", message.message);
    }
  };
  worker.onerror = () => {
    stopRunning();
    showError("The engine could not be loaded",
      "Check that regulae.js and regulae.wasm are being served alongside this page.");
  };
}

/* A worker sitting inside a synchronous WebAssembly call cannot receive a
 * message, so cancelling means replacing it. Re-instantiating costs a few
 * milliseconds once the browser has the bytes. */
function cancelRun() {
  if (worker) {
    worker.terminate();
  }
  ready = false;
  runButton.disabled = true;
  stopRunning();
  showError("Cancelled", "Nothing was returned. The engine is restarting.");
  startWorker();
}

/* ---- running ----------------------------------------------------------- */

function stopRunning() {
  running = false;
  progress.classList.remove("active");
  cancelButton.hidden = true;
  runButton.hidden = false;
}

function showProgress({ stage, completed, total }) {
  progress.classList.add("active");
  $("progress-fill").style.width = Math.round((completed / total) * 100) + "%";
  $("progress-label").textContent = `${stage} — ${completed}/${total}`;
}

function showError(what, detail) {
  $("error").innerHTML = "";
  const box = document.createElement("div");
  box.className = "error";
  const title = document.createElement("div");
  title.className = "what";
  title.textContent = what;
  box.appendChild(title);
  if (detail) {
    const line = document.createElement("div");
    line.className = "detail";
    line.textContent = detail;
    box.appendChild(line);
  }
  $("error").appendChild(box);
}

function run() {
  if (!ready || running) {
    return;
  }
  const corpus = editor.value.trim();
  if (!corpus) {
    showError("Nothing to run", "Paste a corpus, or pick one from the list.");
    return;
  }
  running = true;
  results.classList.remove("active");
  $("error").innerHTML = "";
  runButton.hidden = true;
  cancelButton.hidden = false;
  showProgress({ stage: "starting", completed: 0, total: 1 });
  const options = buildOptions();
  /* The shuffle and the cross-validation both retrain many times over, so a run
     that asks for either needs a longer leash than the default one; the worker
     turns this into a deadline the progress callback checks, since a synchronous
     WebAssembly call cannot be interrupted any other way. */
  const slow = options && (options.includes("permutation_count") || options.includes("predictive_folds"));
  const timeoutMs = slow ? 180000 : 60000;
  worker.postMessage({ type: "train", corpus, format: currentFormat, options, timeoutMs });
}

/* The options the run departs from the default with. Only non-default keys are
   sent, so the payload says exactly what was changed and the provenance block
   in the download records it. */
function buildOptions() {
  const options = {};
  const scorer = $("scorer").value;
  if (scorer && scorer !== "corrected_bic") {
    options.split_scorer = scorer;
  }
  if ($("bootstrap").checked) {
    /* Bootstrap only re-tallies class counts over resampled cognate sets -- it
       does not retrain -- so a high draw count costs almost nothing and buys a
       smoother interval. */
    options.bootstrap_n = 300;
  }
  if ($("predictive").checked) {
    /* Few enough folds to keep an interactive run bearable; each one retrains,
       so this is the knob that makes cross-validation the slow option it is. */
    options.predictive_folds = 4;
  }
  if (baselineSupported && $("baseline").checked) {
    /* Enough shuffles to place the fit against the baseline without making the
       browser wait a minute per corpus; the CLI default is higher and the
       guide says so. Sent only when the engine accepts it -- an older committed
       build does not, and the box is disabled there rather than failing here. */
    options.permutation_count = 20;
  }
  return Object.keys(options).length ? JSON.stringify(options) : null;
}

/* Gate the shuffled-baseline box on whether the engine accepts the option. The
   feature is real -- the CLI runs it -- but a committed WebAssembly build that
   predates its JSON surface would reject the option and fail the run, so the
   box is disabled with a note rather than left as a trap. */
function setBaselineAvailability(supported) {
  baselineSupported = supported;
  const box = $("baseline");
  const label = $("baseline-check");
  const note = $("baseline-unavailable");
  box.disabled = !supported;
  if (label && label.classList) {
    label.classList.toggle("disabled", !supported);
  }
  if (!supported) {
    box.checked = false;
    if (note) {
      note.hidden = false;
      note.textContent = "not in this engine build; runs in the CLI with --permutations";
    }
  } else if (note) {
    note.hidden = true;
  }
}

function finish({ json, chunks, drift, elapsedMs }) {
  stopRunning();
  let payload;
  try {
    payload = JSON.parse(json);
  } catch (error) {
    showError("The result could not be read", error.message);
    return;
  }
  if (!payload.ok) {
    showError(payload.status, payload.detail);
    return;
  }
  model = payload;
  /* Chunks and drift ride beside the model, off separate worker calls, so the
     documented model JSON stays byte-stable. Either may be null on an old
     engine build; the panes then stay hidden. */
  modelChunks = parseExtras(chunks, "pairs");
  modelDrift = parseExtras(drift, "drift");
  $("timing").textContent = (elapsedMs / 1000).toFixed(2) + "s";
  render();
  results.classList.add("active");
}

/* Parses a worker extra and returns the named array, or null when the run
   did not produce it. A malformed extra is not a failed run. */
function parseExtras(text, key) {
  if (!text) return null;
  try {
    const payload = JSON.parse(text);
    return payload.ok && Array.isArray(payload[key]) ? payload[key] : null;
  } catch {
    return null;
  }
}

/* ---- rendering --------------------------------------------------------- */



/* Make a non-button element operable from the keyboard: focusable, activated by
   Enter or Space, and announced with a role. The result surface is otherwise
   mouse-only -- a comparativist working by keyboard or screen reader could
   select nothing. */
function makeActivatable(el, handler, role) {
  el.setAttribute("tabindex", "0");
  if (role) {
    el.setAttribute("role", role);
  }
  el.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handler(event);
    }
  });
}

/* A class by its id, across both tables. -1 (no contrast) returns null. */
function classById(id) {
  if (id === undefined || id < 0) {
    return null;
  }
  return model.classes.unconditioned.find((c) => c.id === id)
    || model.classes.conditioned.find((c) => c.id === id)
    || null;
}


/* The rate a class applies at, with its interval drawn as a bar on a 0-1 track.
   Present on every class in the model -- Wilson by default, bootstrap once the
   option is on -- so the column never empties; the numbers ride in the title so
   the cell stays a glance rather than a paragraph. */
function rateCell(u) {
  const cell = document.createElement("td");
  cell.className = "rate";
  if (!u) {
    return cell;
  }
  cell.title = uncertaintyLabel(u);

  /* The interval is drawn as a bar, so its numbers live only in the hover title
     -- invisible to a screen reader. This says them in text, hidden from sight
     but read aloud. */
  const spoken = document.createElement("span");
  spoken.className = "sr-only";
  spoken.textContent = uncertaintyLabel(u);
  cell.appendChild(spoken);

  const point = document.createElement("span");
  point.className = "point";
  point.setAttribute("aria-hidden", "true");
  point.textContent = u.estimate.toFixed(2);

  const track = document.createElement("span");
  track.className = u.method === "bootstrap" ? "ci boot" : "ci";
  const { left, right, tick } = ciBar(u);
  const bar = document.createElement("span");
  bar.className = "ci-bar";
  bar.style.left = left + "%";
  bar.style.right = (100 - right) + "%";
  const mark = document.createElement("span");
  mark.className = "ci-tick";
  mark.style.left = tick + "%";
  track.append(bar, mark);

  cell.append(point, track);
  return cell;
}

/* The distinct cognate sets behind a row -- the count that answers whether a
   correspondence recurs, which is the question the table is usually asked. */
function setCount(entry) {
  return (entry.supporting_cognates || []).length;
}

/* One class row's DOM. Pulled out of the render loop so the filtered/sorted
   view can rebuild the tbody from the same builder the first render used. */
function buildClassRow(entry, conditioned) {
  const row = document.createElement("tr");
  row.dataset.classId = String(entry.id);

  const corr = document.createElement("td");
  corr.className = "corr";

  /* The caret opens the evidence drawer without selecting the row -- selection
     filters the alignments, evidence is a different question -- so its click is
     stopped from reaching the row. */
  const caret = document.createElement("button");
  caret.className = "caret";
  caret.textContent = expandedClasses.has(entry.id) ? "▾" : "▸";
  caret.title = "show the evidence behind this rule";
  caret.setAttribute("aria-label", `evidence for ${correspondence(entry)}`);
  caret.setAttribute("aria-expanded", expandedClasses.has(entry.id) ? "true" : "false");
  caret.addEventListener("click", (event) => {
    event.stopPropagation();
    if (expandedClasses.has(entry.id)) {
      expandedClasses.delete(entry.id);
    } else {
      expandedClasses.add(entry.id);
    }
    applyClassView();
  });
  corr.appendChild(caret);
  corr.append(correspondence(entry));
  if (conditioned && (entry.environment_alternatives || 0) > 0) {
    /* The environment on this row is one of several that carve the split the
       same way; the cell says so, not only the hint, so the row cannot be
       quoted as *the* environment. */
    const tied = document.createElement("span");
    tied.className = "env";
    tied.textContent = "tied";
    tied.title = `${entry.environment_alternatives} other environment(s) carve this split the same way`;
    corr.appendChild(tied);
  }
  if (conditioned) {
    const env = document.createElement("span");
    env.className = "env";
    env.textContent = environment(entry) || "conditioned";
    /* The pivot's other reflex out of the environment: the contrast that
       makes it a split, and the row a visitor needs to believe it. Shown
       inline so the comparison is not a separate hunt through the table. */
    const contrast = classById(entry.contrast_class_id);
    if (contrast) {
      /* Two in five committed rules are X ~ X, and most of those are the
         retention side of a real split: the change is sitting in the
         contrast. "p ~ p before a vowel" is a null statement to read, so say
         which half of the pair is the event rather than leaving the reader
         to notice that the two graphemes are the same. */
      env.textContent += isIdentity(entry) && !isIdentity(contrast)
        ? ` · unchanged here; the change is ${correspondence(contrast)}`
        : ` · else ${correspondence(contrast)}`;
    }
    corr.appendChild(env);
  }

  const count = document.createElement("td");
  count.className = "count";
  count.textContent = entry.count % 1 === 0 ? entry.count : entry.count.toFixed(1);

  const sets = document.createElement("td");
  sets.className = "sets";
  sets.textContent = setCount(entry) || "";
  sets.title = "distinct cognate sets behind this row";

  row.append(corr, count, sets, rateCell(entry.uncertainty));
  row.addEventListener("click", () => select(entry.id));
  makeActivatable(row, () => select(entry.id), "button");
  row.setAttribute("aria-selected", entry.id === selectedClass ? "true" : "false");
  return row;
}

/* The evidence drawer beneath an expanded class: what the search committed the
   rule on, whether it stands, the confounds the corpus cannot resolve, and the
   cognate sets behind it. Everything here is on the class row already -- the
   drawer only reads it -- so it answers "why this rule, and can I trust it?"
   without a trip through the JSON download. */
function buildEvidenceRow(entry, conditioned) {
  const tr = document.createElement("tr");
  tr.className = "evidence-row";
  tr.dataset.evidenceFor = String(entry.id);
  const cell = document.createElement("td");
  cell.colSpan = 4;
  const box = document.createElement("div");
  box.className = "evidence";

  const line = (text, className) => {
    if (!text) {
      return;
    }
    const div = document.createElement("div");
    div.className = "ev-line" + (className ? " " + className : "");
    div.textContent = text;
    box.appendChild(div);
  };

  line(ruleScore(entry), "ev-score");

  if (conditioned) {
    const standing = entry.standing;
    const cls = standing === "above noise" ? "ev-stands"
      : standing === "within noise" ? "ev-within" : "ev-unmeasured";
    line(ruleStanding(entry), cls);
    line(ruleConfound(entry), "ev-confound");

    /* The elsewhere case the split was scored against, named and sized: the
       contrast that makes the environment a finding rather than a description. */
    const contrast = classById(entry.contrast_class_id);
    if (contrast) {
      const mass = entry.contrast_alternative_count;
      const size = typeof mass === "number" && mass > 0
        ? ` (${mass % 1 === 0 ? mass : mass.toFixed(1)} observations)` : "";
      line(`elsewhere: ${correspondence(contrast)}${size}`, "ev-contrast");
    }
  }

  if (typeof entry.confidence === "number") {
    line(`confidence ${Math.round(entry.confidence * 100)}%, the weight this rule rests on`,
      "ev-confidence");
  }

  const cognates = entry.supporting_cognates || [];
  if (cognates.length) {
    const sets = document.createElement("div");
    sets.className = "ev-line ev-sets";
    const label = document.createElement("span");
    label.textContent = `sets (${cognates.length}): ${cognates.join(", ")}`;
    const copy = document.createElement("button");
    copy.className = "ev-copy";
    copy.textContent = "copy";
    copy.title = "copy the supporting cognate sets";
    copy.addEventListener("click", (event) => {
      event.stopPropagation();
      copyText(cognates.join(", "), copy);
    });
    sets.append(label, copy);
    box.appendChild(sets);
  }

  cell.appendChild(box);
  tr.appendChild(cell);
  return tr;
}

function renderClasses() {
  /* Default order is the decision list for the conditioned rows -- a finding,
     not a key -- then the unconditioned ones. Filtering and sorting are applied
     over this array without touching the model. */
  classRows = [
    ...decisionOrder(model.classes.conditioned).map((c) => ({ entry: c, conditioned: true })),
    ...model.classes.unconditioned.map((c) => ({ entry: c, conditioned: false })),
  ];

  /* The interval column is Wilson until resampling is on, then bootstrap; the
     hint says which so the bar is not read as one when it is the other. */
  const method = classRows.length && classRows[0].entry.uncertainty
    && classRows[0].entry.uncertainty.method;
  $("classes-hint").textContent =
    "Rate is how often the correspondence is the one taken where the class applies; "
    + `the bar is its 95% ${method === "bootstrap" ? "bootstrap" : "Wilson"} interval. `
    + "Select a row for its alignments.";

  applyClassView();
}

/* Whether a row passes the current filter: the chips narrow by kind, recurrence
   and whether anything changed, and the query matches the written
   correspondence and its environment together, so "p ~ f" and "before a vowel"
   both find their rows. Retentions are hidden unless "changes only" is off, and
   conditioned rows that are thin or unmeasured sit behind the "weak" chip: the
   table leads with what the corpus actually carries. */
function classMatches({ entry, conditioned }) {
  if (classView.chip === "conditioned" && !conditioned) return false;
  if (classView.chip === "unconditioned" && conditioned) return false;
  if (classView.recurring && setCount(entry) < 2) return false;
  if (classView.changes && isIdentity(entry)) return false;
  if (classView.stands && entry.standing !== "above noise") return false;
  if (!classView.weak && conditioned
      && (entry.count < 8 || entry.standing === "unmeasured" || entry.standing === "within noise")) {
    return false;
  }
  const query = classView.query.trim().toLowerCase();
  if (query) {
    const hay = `${correspondence(entry)} ${conditioned ? environment(entry) : ""}`.toLowerCase();
    if (!hay.includes(query)) return false;
  }
  return true;
}

/* The value a sortable column reads. Rate sorts on the point estimate; the
   correspondence column has no key -- it is the decision order, restored by the
   "default" sort. */
function sortValue(entry, key) {
  if (key === "count") return entry.count;
  if (key === "sets") return setCount(entry);
  if (key === "rate") return entry.uncertainty ? entry.uncertainty.estimate : -1;
  return 0;
}

function applyClassView() {
  const body = $("classes").querySelector("tbody");
  body.innerHTML = "";

  if (!classRows.length) {
    body.innerHTML = '<tr><td colspan="4" class="empty">No classes were found.</td></tr>';
    $("class-count").textContent = "";
    return;
  }

  let rows = classRows.filter(classMatches);
  if (classView.sort !== "default") {
    const dir = classView.desc ? -1 : 1;
    rows = [...rows].sort((a, b) =>
      dir * (sortValue(a.entry, classView.sort) - sortValue(b.entry, classView.sort)));
  }

  for (const { entry, conditioned } of rows) {
    const row = buildClassRow(entry, conditioned);
    if (entry.id === selectedClass) {
      row.classList.add("selected");
    } else if (selectedClass !== null) {
      row.classList.add("dimmed");
    }
    body.appendChild(row);
    if (expandedClasses.has(entry.id)) {
      body.appendChild(buildEvidenceRow(entry, conditioned));
    }
  }

  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="4" class="no-match">'
      + "No class matches the filter. Clear it to see the rest.</td></tr>";
  }

  $("class-count").textContent = rows.length === classRows.length
    ? `${classRows.length} classes`
    : `${rows.length} of ${classRows.length}`;

  for (const th of $("classes").querySelectorAll("th[data-sort]")) {
    const on = th.dataset.sort === classView.sort && classView.sort !== "default";
    th.classList.toggle("sorted", on);
    th.classList.toggle("desc", on && classView.desc);
    if (th.dataset.sort !== "default") {
      th.setAttribute("aria-sort", on ? (classView.desc ? "descending" : "ascending") : "none");
    }
  }
}


/* Conditioned classes that look like one change. A proposal: every member is
   still in the table above, and clicking an event shows the members' alignments
   rather than replacing anything. */
function renderEvents() {
  const body = $("events").querySelector("tbody");
  body.innerHTML = "";
  const events = model.proposed_events || [];
  $("events-hint").textContent = events.length
    ? "Classes that differ only in their graphemes and share an environment or "
      + "a feature displacement. Whether a single pooled rule beats them isn't decided here."
    : "";
  if (!events.length) {
    body.innerHTML = '<tr><td colspan="3" class="empty">(none)</td></tr>';
    return;
  }
  for (const event of events) {
    const row = document.createElement("tr");
    const corr = document.createElement("td");
    corr.className = "corr";
    corr.textContent = eventCorrespondence(event);
    if (!event.featurally_definable) {
      const note = document.createElement("span");
      note.className = "env unnamed";
      note.textContent = "no feature names this set in this corpus";
      corr.appendChild(note);
    }
    const count = document.createElement("td");
    count.className = "count";
    count.textContent = event.count % 1 === 0 ? event.count : event.count.toFixed(1);
    const sets = document.createElement("td");
    sets.className = "sets";
    sets.textContent = event.supporting_cognates.length;
    sets.title = `${event.class_ids.length} member classes`;
    row.append(corr, count, sets);
    body.appendChild(row);
  }
}

/* Rules where a segmental feature on one lect predicts a suprasegmental value
   on another -- tonogenesis and its kin. Computed on every run and, until this
   pane, only ever counted in the summary. Written a ~ b like every other
   correspondence: the value is carried where the environment holds, not derived
   from it. */
function renderCrossDimensional() {
  const body = $("crossdim").querySelector("tbody");
  body.innerHTML = "";
  const rows = model.cross_dimensional || [];
  $("crossdim-panel").hidden = rows.length === 0;
  $("crossdim-hint").textContent = rows.length
    ? "A feature on one lect predicting a suprasegmental value on another. The "
      + "conditioned lect carries the value where the environment holds."
    : "";
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="3" class="empty">'
      + "No suprasegmental value was predicted by a segmental environment.</td></tr>";
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");

    const corr = document.createElement("td");
    corr.className = "corr";
    corr.textContent = crossDimCorrespondence(row);
    const env = document.createElement("span");
    env.className = "env";
    env.textContent = crossDimEnvironment(row);
    const confound = crossDimConfound(row);
    if (confound) {
      env.textContent += ` · ${confound}`;
      env.classList.add("unnamed");
    }
    corr.appendChild(env);

    const count = document.createElement("td");
    count.className = "count";
    count.textContent = row.count % 1 === 0 ? row.count : row.count.toFixed(1);

    /* The other reflex: how often the same lect does something else in this
       environment. A rule with no contrast is exceptionless in the corpus. */
    const contrast = document.createElement("td");
    contrast.className = "sets";
    contrast.textContent = row.contrast_count ? `else ${row.contrast_count}` : "";
    contrast.title = "observations of the contrasting value in the same environment";

    tr.append(corr, count, contrast);
    body.appendChild(tr);
  }
}

/* The corpus-level verdict, shown only when the shuffled baseline was run. Read
   as prose rather than as a z-score: reading.js turns the fit fields into the
   one sentence a visitor can act on. */
function renderBaseline() {
  const panel = $("baseline-panel");
  if (!model.fit || !model.fit.permutation_count) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("baseline-body").textContent = baselineReading(model.fit);
}

/* Segments that answer to nothing across a lect pair, read off the per-pair
   tables and written x ~ ∅. Loss and epenthesis are first-class objects of
   comparative work but were only ever visible as ∅ cells inside an alignment.
   Framed as correspondences with rates, never as "deletions": the rate is how
   often, of the times the segment is present, it aligns to a gap. */
function renderGaps() {
  const panel = $("gaps-panel");
  const body = $("gaps").querySelector("tbody");
  body.innerHTML = "";

  const multiPair = (model.lects || []).length > 2;
  const rows = [];
  for (const pair of model.pairwise || []) {
    for (const gap of pair.gaps || []) {
      rows.push({ pair, gap });
    }
  }
  panel.hidden = rows.length === 0;
  if (!rows.length) {
    $("gaps-hint").textContent = "";
    return;
  }
  rows.sort((a, b) => b.gap.count - a.gap.count);
  $("gaps-hint").textContent =
    "A segment that answers to nothing, written x ~ ∅. The rate is how often it aligns to "
    + "a gap when it's present, not a claim that anything was lost.";

  for (const { pair, gap } of rows) {
    const tr = document.createElement("tr");
    tr.dataset.pair = `${pair.source_lect}~${pair.target_lect}`;

    const corr = document.createElement("td");
    corr.className = "corr";
    corr.textContent = gapCorrespondence(pair.source_lect, pair.target_lect, gap);
    if (multiPair) {
      const tag = document.createElement("span");
      tag.className = "env";
      tag.textContent = `${pair.source_lect} ~ ${pair.target_lect}`;
      corr.appendChild(tag);
    }

    const count = document.createElement("td");
    count.className = "count";
    count.textContent = gap.count;
    count.title = `of ${gap.present_total} times the segment is present`;

    tr.append(corr, count, rateCell(gap.uncertainty));
    body.appendChild(tr);
  }
}

/* Cognate sets ranked by how badly they align under the trained model. The
   residue is not an error term: a set that will not align is either not
   cognate, or cognate through a correspondence the model has not got. */
function renderResidue() {
  const body = $("residue").querySelector("tbody");
  body.innerHTML = "";
  const rows = model.outliers || [];
  if (!rows.length) {
    $("residue-hint").textContent = "";
    body.innerHTML = '<tr><td colspan="4" class="empty">No sets were scored.</td></tr>';
    return;
  }
  $("residue-hint").textContent = residueReading(model.fit);

  /* The confidence column is worth a reader's attention only where it varies:
     on an unweighted corpus every set is 1.0 and the column is noise, so it is
     hidden unless some set was supplied at a lower confidence. */
  const showConfidence = rows.some((r) => typeof r.confidence === "number" && r.confidence < 1);
  $("residue").classList.toggle("hide-conf", !showConfidence);

  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.dataset.cognate = row.cognate_id;
    /* One standard deviation above the corpus mean. Not a verdict -- the cut
       that separated the planted sets from the good ones on the contaminated
       fixture, where the five bad ones ran 1.46 to 3.84 and the worst good one
       reached -0.06. On a clean corpus nothing is marked. */
    tr.classList.toggle("apart", row.z_score >= 1);

    const id = document.createElement("td");
    id.className = "corr";
    id.textContent = row.cognate_id;

    const z = document.createElement("td");
    z.className = "z";
    z.textContent = row.z_score.toFixed(2);

    const cost = document.createElement("td");
    cost.className = "cost";
    cost.textContent = row.cost_per_segment.toFixed(3);

    /* A low-confidence set that also aligns badly is doubly suspect; a
       high-confidence one that does is the real puzzle. */
    const conf = document.createElement("td");
    conf.className = "conf conf-col";
    conf.textContent = typeof row.confidence === "number" ? `${Math.round(row.confidence * 100)}%` : "";

    tr.append(id, z, cost, conf);
    tr.addEventListener("click", () => selectSet(row.cognate_id));
    makeActivatable(tr, () => selectSet(row.cognate_id), "button");
    body.appendChild(tr);
  }
}

/* An alignment block is shown when it passes all three filters at once: the
   selected class, the selected cognate set, and the selected lect pair. The
   three used to each hide blocks on their own and stack into a blank panel, so
   they are resolved here in one place. */
function updateAlignmentVisibility() {
  for (const block of $("alignments").querySelectorAll(".alignment")) {
    const ids = JSON.parse(block.dataset.classes);
    let hidden = false;
    if (selectedClass !== null && !ids.includes(selectedClass)) hidden = true;
    if (selectedSet !== null && block.dataset.cognate !== selectedSet) hidden = true;
    if (selectedPair !== null && block.dataset.pair !== selectedPair) hidden = true;
    block.classList.toggle("hidden", hidden);
  }
}

/* Show one cognate set's alignments and nothing else, so a row in the residue
   table resolves to the words behind it in one click. */
function selectSet(cognateId) {
  selectedSet = selectedSet === cognateId ? null : cognateId;
  selectedClass = null;

  for (const row of $("residue").querySelectorAll("tr[data-cognate]")) {
    row.classList.toggle("selected", row.dataset.cognate === selectedSet);
  }
  for (const row of $("classes").querySelectorAll("tr[data-class-id]")) {
    row.classList.remove("selected", "dimmed");
    row.setAttribute("aria-selected", "false");
  }
  updateAlignmentVisibility();
  for (const col of $("alignments").querySelectorAll(".col")) {
    col.classList.remove("lit");
  }
  const hint = $("alignments-hint");
  hint.textContent = selectedSet === null
    ? "Select a column to see which class it belongs to."
    : `Showing ${selectedSet}. Select it again to show every set.`;
}

/* The lect-pair selector, shown only past two lects. A multi-lect corpus aligns
   every pair, so its alignments panel is otherwise flooded; this narrows it to
   one pair, and the gap pane with it. */
function renderLectPair() {
  const wrap = $("lect-pair-wrap");
  const select = $("lect-pair");
  selectedPair = null;

  const pairs = [];
  const seen = new Set();
  for (const a of model.alignments || []) {
    const key = `${a.lect_a}~${a.lect_b}`;
    if (!seen.has(key)) {
      seen.add(key);
      pairs.push({ key, label: `${a.lect_a} ~ ${a.lect_b}` });
    }
  }

  const multi = (model.lects || []).length > 2 && pairs.length > 1;
  wrap.hidden = !multi;
  select.innerHTML = "";
  if (!multi) {
    return;
  }
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "all lect pairs";
  select.appendChild(all);
  for (const pair of pairs) {
    const option = document.createElement("option");
    option.value = pair.key;
    option.textContent = pair.label;
    select.appendChild(option);
  }
}

/* Narrow the alignments and the gaps to the chosen lect pair, or restore all. */
function applyPairFilter() {
  selectedPair = $("lect-pair").value || null;
  updateAlignmentVisibility();
  for (const tr of $("gaps").querySelectorAll("tr[data-pair]")) {
    tr.hidden = selectedPair !== null && tr.dataset.pair !== selectedPair;
  }
}

function renderAlignments() {
  const container = $("alignments");
  container.innerHTML = "";

  if (!model.alignments || !model.alignments.length) {
    container.innerHTML = '<div class="empty">No alignments were produced.</div>';
    return;
  }

  for (const alignment of model.alignments) {
    const block = document.createElement("div");
    block.className = "alignment";
    block.dataset.classes = JSON.stringify(
      [...new Set(alignment.links.flatMap((l) => l.classes || []))]);
    block.dataset.cognate = alignment.cognate_id;
    block.dataset.pair = `${alignment.lect_a}~${alignment.lect_b}`;

    const gloss = document.createElement("div");
    gloss.className = "gloss";
    gloss.textContent = alignment.cognate_id;
    /* The two forms reassembled from the columns, so the alignment reads as the
       words it aligns rather than as a grid a visitor has to sound out. */
    const src = alignment.links.map((l) => l.source.join("")).join("");
    const tgt = alignment.links.map((l) => l.target.join("")).join("");
    const pair = document.createElement("span");
    pair.className = "pair";
    pair.textContent = `  ${alignment.lect_a} ${src} ~ ${alignment.lect_b} ${tgt}`;
    gloss.appendChild(pair);

    const cols = document.createElement("div");
    cols.className = "cols";
    for (const link of alignment.links) {
      const col = document.createElement("div");
      col.className = "col";
      if (!link.source.length || !link.target.length) {
        col.classList.add("gap");
      }
      const ids = link.classes || [];
      col.dataset.classes = JSON.stringify(ids);

      const top = document.createElement("span");
      top.textContent = link.source.join("") || "∅";
      const bottom = document.createElement("span");
      bottom.className = "b";
      bottom.textContent = link.target.join("") || "∅";
      col.append(top, bottom);

      /* Selecting a column reveals the class it belongs to, which is the
       * reverse of selecting a class to see its columns. */
      if (ids.length) {
        const activate = (event) => {
          event.stopPropagation();
          select(ids[0], true);
        };
        col.addEventListener("click", activate);
        makeActivatable(col, activate, "button");
        col.setAttribute("aria-label",
          `${top.textContent} to ${bottom.textContent}; show its class`);
      }
      cols.appendChild(col);
    }

    block.append(gloss, cols);
    container.appendChild(block);
  }
}

function select(classId, scrollToClass) {
  selectedClass = selectedClass === classId ? null : classId;
  /* The two filters would otherwise stack and hide everything. */
  selectedSet = null;
  for (const row of $("residue").querySelectorAll("tr[data-cognate]")) {
    row.classList.remove("selected");
  }

  for (const row of $("classes").querySelectorAll("tr[data-class-id]")) {
    const id = Number(row.dataset.classId);
    row.classList.toggle("selected", id === selectedClass);
    row.classList.toggle("dimmed", selectedClass !== null && id !== selectedClass);
    row.setAttribute("aria-selected", id === selectedClass ? "true" : "false");
  }

  updateAlignmentVisibility();
  for (const col of $("alignments").querySelectorAll(".col")) {
    const ids = JSON.parse(col.dataset.classes);
    col.classList.toggle("lit", selectedClass !== null && ids.includes(selectedClass));
  }

  const shown = $("alignments").querySelectorAll(".alignment:not(.hidden)").length;
  $("alignments-hint").textContent = selectedClass === null
    ? "Select a column to see which class it belongs to."
    : `${shown} of ${model.alignments.length} alignments realise this class.`;

  if (scrollToClass && selectedClass !== null) {
    const row = $("classes").querySelector(`tr[data-class-id="${selectedClass}"]`);
    if (row) {
      row.scrollIntoView({ block: "nearest" });
    }
  }
}

function renderSummary() {
  const counts = [
    ["lects", model.lects.length],
    ["classes", model.classes.unconditioned.length],
    ["conditioned", model.classes.conditioned.length],
    ["cross-dimensional", model.cross_dimensional.length],
    ["alignments", (model.alignments || []).length],
  ];
  $("summary").innerHTML = "";
  for (const [label, value] of counts) {
    const item = document.createElement("div");
    item.innerHTML = `<b>${value}</b> <span>${label}</span>`;
    $("summary").appendChild(item);
  }
  /* The trained fit as one number, so the headline count of classes sits next
     to how tightly the corpus aligns under them. Lower is a closer fit. */
  if (model.fit && typeof model.fit.cost_per_segment === "number") {
    const fit = document.createElement("div");
    fit.innerHTML = `<b>${fmtCost(model.fit.cost_per_segment)}</b> <span>cost/segment</span>`;
    $("summary").appendChild(fit);
  }
  const lects = document.createElement("div");
  lects.innerHTML = `<span>${model.lects.join(", ")}</span>`;
  $("summary").appendChild(lects);
}

function render() {
  selectedClass = null;
  resetClassView();
  renderSummary();
  renderProvenance();
  renderDrift();
  renderBaseline();
  renderPredictive();
  renderEvents();
  renderCrossDimensional();
  renderChunks();
  renderGaps();
  renderResidue();
  renderClasses();
  renderAlignments();
  renderLectPair();
  select(null);
  selectedClass = null;
}

/* Transcription drift reads as a conditioned change once trained, so it is
   said above the model, not left for a separate tool. Hidden when the corpus
   carries none. */
function renderDrift() {
  const panel = $("drift-panel");
  if (!modelDrift || !modelDrift.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const list = $("drift-list");
  list.innerHTML = "";
  for (const row of modelDrift) {
    const li = document.createElement("li");
    li.textContent = `${row.lect} writes ${row.grapheme} where ${row.other_lect} writes "${row.written_as}"`
      + ` (${row.corroborated}/${row.forms} forms)`;
    list.appendChild(li);
  }
}

/* Spans of more than one segment, read off the pair models: kt answering
   tʃ, or two segments trading places ([reordering]) is one fact, not several
   segment correspondences. Hidden when nothing promoted. */
function renderChunks() {
  const panel = $("chunks-panel");
  const body = $("chunks").querySelector("tbody");
  body.innerHTML = "";
  let shown = 0;
  for (const pair of modelChunks || []) {
    for (const chunk of pair.chunks || []) {
      const row = document.createElement("tr");
      const pairCell = document.createElement("td");
      pairCell.textContent = `${pair.source_lect} › ${pair.target_lect}`;
      const corr = document.createElement("td");
      corr.textContent = `${chunk.source} ~ ${chunk.target}`;
      if (chunk.reordering) {
        const tag = document.createElement("span");
        tag.className = "env";
        tag.textContent = "reordering";
        tag.title = "the same segments in another order, not a set of substitutions";
        corr.append(" ", tag);
      }
      const count = document.createElement("td");
      count.className = "count";
      count.textContent = chunk.count % 1 === 0 ? chunk.count : chunk.count.toFixed(1);
      row.append(pairCell, corr, count);
      body.appendChild(row);
      shown++;
    }
  }
  panel.hidden = shown === 0;
  $("chunks-hint").textContent = shown === 0 ? ""
    : "A chunk is one span answering to one span; a reordering is the same segments in another order.";
}

/* A fresh corpus starts with a clean table: an old filter left over from the
   last run would silently hide rows the new run found. */
function resetClassView() {
  classView.query = "";
  classView.chip = "all";
  classView.recurring = false;
  classView.changes = true;
  classView.weak = false;
  classView.stands = false;
  classView.sort = "default";
  classView.desc = true;
  expandedClasses.clear();
  const filter = $("class-filter");
  if (filter) {
    filter.value = "";
  }
  for (const chip of $("class-chips").querySelectorAll("button")) {
    chip.classList.toggle("on", chip.dataset.chip === "all" || chip.dataset.chip === "changes");
    chip.setAttribute("aria-pressed",
      chip.dataset.chip === "all" || chip.dataset.chip === "changes" ? "true" : "false");
  }
}

/* Held-out prediction, shown only when cross-validation ran. The prose reads
   the verdict; the table gives the trained models and the three baselines the
   verdict rests on, so a reader can weigh the numbers rather than take them. */
function renderPredictive() {
  const panel = $("predictive-panel");
  const p = model.fit && model.fit.predictive;
  if (!p || p.status === "unmeasured") {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("predictive-body").textContent = predictiveReading(p);

  const body = $("predictive-table").querySelector("tbody");
  body.innerHTML = "";
  const rows = [
    { label: "trained · conditioned", score: p.conditioned, trained: true },
    { label: "trained · unconditioned", score: p.unconditioned, trained: true },
    { label: "baseline · identity (a ↦ a)", score: p.identity },
    { label: "baseline · inventory frequency", score: p.inventory_frequency },
    { label: "baseline · feature distance", score: p.feature_distance },
  ];
  const pct = (x) => `${Math.round(x * 100)}%`;
  for (const { label, score, trained } of rows) {
    if (!score) {
      continue;
    }
    const tr = document.createElement("tr");
    if (trained) {
      tr.className = "trained";
    }
    const name = document.createElement("td");
    name.className = "corr";
    name.textContent = label;
    const top1 = document.createElement("td");
    top1.className = "num";
    top1.textContent = pct(score.top1_coverage);
    const topk = document.createElement("td");
    topk.className = "num";
    topk.textContent = pct(score.top_k_coverage);
    const loss = document.createElement("td");
    loss.className = "num";
    loss.textContent = score.log_loss.toFixed(3);
    tr.append(name, top1, topk, loss);
    body.appendChild(tr);
  }
}

/* The provenance a citation of this run needs, read straight off the model.
   Reference rather than result, so it stays in a closed disclosure. */
function renderProvenance() {
  const dl = $("provenance-body");
  dl.innerHTML = "";
  const prov = model.provenance || {};
  const opts = prov.options || {};
  const rows = [
    ["regulae version", model.regulae_version],
    ["export kind", model.export_kind],
    ["feature system", prov.feature_system],
    ["merkmal version", prov.merkmal_version],
    ["corpus checksum", prov.corpus_checksum],
    ["split scorer", opts.bic && opts.bic.split_scorer],
    ["bootstrap draws", opts.bootstrap_n],
    ["cross-validation folds", opts.predictive_folds],
    ["shuffles", opts.permutation_count],
    ["max chunk size", opts.max_chunk_size],
  ];
  for (const [label, value] of rows) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    dl.append(dt, dd);
  }
}

/* ---- segmentation preview ---------------------------------------------- */

/* A token per request, so a slow answer that arrives after the visitor has
   typed on is dropped rather than shown against the wrong word. */
let segmentToken = 0;
let segmentTimer = null;

function requestSegmentation() {
  const word = $("segment-input").value.trim();
  const preview = $("segment-preview");
  if (!word) {
    preview.innerHTML = "";
    return;
  }
  if (!ready) {
    preview.textContent = "The engine is still loading.";
    return;
  }
  segmentToken += 1;
  worker.postMessage({ type: "segment", word, token: segmentToken });
}

/* Debounced so a request goes out when typing pauses, not on every keystroke. */
function scheduleSegmentation() {
  clearTimeout(segmentTimer);
  segmentTimer = setTimeout(requestSegmentation, 180);
}

function showSegmentation({ token, json }) {
  if (token !== segmentToken) {
    return;
  }
  const preview = $("segment-preview");
  preview.innerHTML = "";
  let payload;
  try {
    payload = JSON.parse(json);
  } catch {
    return;
  }
  if (!payload.ok) {
    /* The most useful thing a bad form can say is which grapheme stopped it,
       which is exactly the detail the loader would refuse the whole corpus
       with. Named here, it is fixed before a run is spent on it. */
    preview.classList.add("bad");
    preview.textContent = payload.detail || payload.status;
    return;
  }
  preview.classList.remove("bad");
  for (const grapheme of payload.segments) {
    const chip = document.createElement("span");
    chip.className = "grapheme";
    chip.textContent = grapheme;
    preview.appendChild(chip);
  }
}

/* ---- examples ---------------------------------------------------------- */

function loadCorpus(path, format) {
  editor.value = CORPORA[path] || "";
  setFormat(format || "wide");
  results.classList.remove("active");
  $("error").innerHTML = "";
}

/* The input format is the select's value; loading an example or a file sets it,
   and a visitor pasting their own data sets it by hand. Kept in one place so
   the select and currentFormat never disagree. */
function setFormat(format) {
  currentFormat = format || "wide";
  const select = $("format");
  if (select) {
    select.value = currentFormat;
  }
}

function buildExampleList() {
  const select = $("examples");
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.textContent = "Choose an example…";
  placeholder.value = "";
  select.appendChild(placeholder);

  let group = null;
  for (const entry of CORPUS_LIST) {
    if (!group || group.label !== entry.group) {
      group = document.createElement("optgroup");
      group.label = entry.group;
      select.appendChild(group);
    }
    const option = document.createElement("option");
    option.value = entry.path;
    option.textContent = entry.readable
      ? entry.label
      : `${entry.label} — cannot read “${entry.blockedBy}”`;
    group.appendChild(option);
  }

  select.addEventListener("change", () => {
    const entry = CORPUS_LIST.find((e) => e.path === select.value);
    if (entry) {
      showExample(entry);
    }
  });
}

/* The note says what the corpus is for; the stats line says what it publishes,
   and is generated from the trained model so it cannot contradict the run. */
function showExample(entry) {
  loadCorpus(entry.path, entry.format);
  /* An example that only makes its point against the shuffle -- the unrelated
     wordlists, the neutralisation -- pre-arms the baseline so the next Run
     delivers what the note promises. Others leave it off, since it is slow, and
     an engine build without the option leaves it off regardless. */
  $("baseline").checked = !!entry.baseline && baselineSupported;
  const note = $("example-note");
  note.classList.toggle("blocked", !entry.readable);
  note.textContent = entry.readable
    ? entry.description
    : `This corpus cannot be read yet: “${entry.blockedBy}” is not in the feature system. `
      + "It is here so the gap is visible rather than hidden. Run it to see the error.";
  $("example-stats").textContent = entry.stats || "";
}

/* ---- downloads --------------------------------------------------------- */

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function summaryText() {
  const lines = ["LECTS\t" + model.lects.join(" ")];
  const sets = (entry) => (entry.supporting_cognates || []).length;
  const rate = (entry) => {
    const u = entry.uncertainty;
    return u ? `\t${u.estimate.toFixed(3)}\t${u.lower.toFixed(3)}\t${u.upper.toFixed(3)}\t${u.method}` : "";
  };
  for (const entry of model.classes.unconditioned) {
    lines.push(`UNCOND\t${entry.id}\t${correspondence(entry)}\t${entry.count}\t${sets(entry)}${rate(entry)}`);
  }
  for (const entry of model.classes.conditioned) {
    lines.push(`COND\t${entry.id}\t${correspondence(entry)}\t${entry.count}\t${sets(entry)}\t${environment(entry)}${rate(entry)}`);
  }
  for (const row of model.cross_dimensional || []) {
    lines.push(`XDIM\t${crossDimCorrespondence(row)}\t${row.count}\t${crossDimEnvironment(row)}`);
  }
  if (model.fit && model.fit.permutation_count) {
    lines.push(`BASELINE\t${baselineReading(model.fit)}`);
  }
  if (model.fit && model.fit.predictive && model.fit.predictive.status !== "unmeasured") {
    lines.push(`PREDICTIVE\t${predictiveReading(model.fit.predictive)}`);
  }
  if (model.provenance) {
    lines.push(`PROVENANCE\t${model.provenance.feature_system}\t${model.provenance.corpus_checksum}`);
  }
  return lines.join("\n") + "\n";
}

/* The correspondence table as rows of the columns a write-up wants: the
   correspondence and its environment, the counts, the rate and its interval,
   and the evidence the summary download drops -- the committing score, the
   standing verdict, the confound and the confidence. Conditioned rows first, in
   decision order, then the unconditioned ones. */
function correspondenceTableRows() {
  const num = (x, d) => (typeof x === "number" ? x.toFixed(d) : "");
  const all = [
    ...decisionOrder(model.classes.conditioned).map((e) => ({ e, conditioned: true })),
    ...model.classes.unconditioned.map((e) => ({ e, conditioned: false })),
  ];
  return all.map(({ e, conditioned }) => {
    const u = e.uncertainty || {};
    return {
      kind: conditioned ? "conditioned" : "unconditioned",
      correspondence: correspondence(e),
      environment: conditioned ? environment(e) : "",
      count: String(e.count),
      sets: String(setCount(e)),
      rate: num(u.estimate, 3),
      ci_low: num(u.lower, 3),
      ci_high: num(u.upper, 3),
      delta_bic: num(e.delta_bic, 2),
      margin: num(e.search_margin, 2),
      standing: e.standing && e.standing !== "unmeasured" ? e.standing : "",
      env_alternatives: String(e.environment_alternatives || 0),
      confidence: num(e.confidence, 3),
    };
  });
}

const TABLE_COLUMNS = [
  "kind", "correspondence", "environment", "count", "sets", "rate", "ci_low",
  "ci_high", "delta_bic", "margin", "standing", "env_alternatives", "confidence",
];

/* CSV that survives a comma or a quote in an environment string. */
function correspondenceCsv() {
  const escape = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const lines = [TABLE_COLUMNS.join(",")];
  for (const row of correspondenceTableRows()) {
    lines.push(TABLE_COLUMNS.map((c) => escape(row[c])).join(","));
  }
  return lines.join("\n") + "\n";
}

/* A GitHub-flavoured Markdown table, pipes in a cell escaped so the columns do
   not shift. */
function correspondenceMarkdown() {
  const escape = (v) => v.replace(/\|/g, "\\|");
  const lines = [
    `| ${TABLE_COLUMNS.join(" | ")} |`,
    `| ${TABLE_COLUMNS.map(() => "---").join(" | ")} |`,
  ];
  for (const row of correspondenceTableRows()) {
    lines.push(`| ${TABLE_COLUMNS.map((c) => escape(row[c])).join(" | ")} |`);
  }
  return lines.join("\n") + "\n";
}

/* Copy text to the clipboard and flash the button that asked for it. Guarded
   because the API is absent in the test's DOM shim and on an insecure origin;
   the download button covers those. */
function copyText(text, button) {
  if (!navigator.clipboard) {
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    if (button) {
      const was = button.textContent;
      button.textContent = "copied";
      setTimeout(() => { button.textContent = was; }, 1200);
    }
  }, () => {});
}

function copySummary() {
  if (model) {
    copyText(summaryText(), $("copy-summary"));
  }
}

/* ---- guide ------------------------------------------------------------- */

let guideStep = 0;

function showGuideStep(index) {
  guideStep = index;
  const step = GUIDE_STEPS[index];
  $("guide-title").textContent = step.title;
  $("guide-body").innerHTML = step.content;
  $("guide-body").scrollTop = 0;
  const load = $("guide-load");
  load.hidden = !step.example;
  load.onclick = () => {
    loadCorpus(step.example.path, step.example.format);
    $("guide").classList.remove("open");
  };
  for (const button of $("guide-steps").children) {
    button.classList.toggle("current", Number(button.dataset.step) === index);
  }
}

function buildGuide() {
  const steps = $("guide-steps");
  GUIDE_STEPS.forEach((step, index) => {
    const button = document.createElement("button");
    button.textContent = String(index + 1);
    button.title = step.title;
    button.dataset.step = String(index);
    button.addEventListener("click", () => showGuideStep(index));
    steps.appendChild(button);
  });
  showGuideStep(0);
}

/* ---- wiring ------------------------------------------------------------ */

buildExampleList();
buildGuide();
startWorker();

runButton.addEventListener("click", run);
cancelButton.addEventListener("click", cancelRun);
$("segment-input").addEventListener("input", scheduleSegmentation);
$("guide-open").addEventListener("click", () => $("guide").classList.add("open"));
$("guide-close").addEventListener("click", () => $("guide").classList.remove("open"));
$("download-json").addEventListener("click", () => download("regulae-model.json", JSON.stringify(model, null, 2)));
$("download-summary").addEventListener("click", () => download("regulae-summary.tsv", summaryText()));
$("copy-summary").addEventListener("click", copySummary);
$("copy-csv").addEventListener("click", () => {
  if (model) {
    copyText(correspondenceCsv(), $("copy-csv"));
  }
});
$("copy-markdown").addEventListener("click", () => {
  if (model) {
    copyText(correspondenceMarkdown(), $("copy-markdown"));
  }
});

$("format").addEventListener("change", (event) => setFormat(event.target.value));
$("lect-pair").addEventListener("change", applyPairFilter);

wireClassTools();

/* The filter, chips and sortable headers over the classes table. Attached once;
   they read and write classView, which applyClassView renders from. */
function wireClassTools() {
  $("class-filter").addEventListener("input", (event) => {
    classView.query = event.target.value;
    applyClassView();
  });
  for (const chip of $("class-chips").querySelectorAll("button")) {
    chip.addEventListener("click", () => handleChip(chip.dataset.chip));
    chip.setAttribute("aria-pressed", chip.classList.contains("on") ? "true" : "false");
  }
  for (const th of $("classes").querySelectorAll("th[data-sort]")) {
    th.addEventListener("click", () => handleSort(th.dataset.sort));
    if (th.dataset.sort !== "default") {
      makeActivatable(th, () => handleSort(th.dataset.sort), null);
      th.setAttribute("aria-sort", "none");
    }
  }
}

/* The kind chips (all/conditioned/unconditioned) are one choice; recurring and
   changes are independent toggles laid over it. */
function handleChip(name) {
  if (name === "recurring") {
    classView.recurring = !classView.recurring;
  } else if (name === "changes") {
    classView.changes = !classView.changes;
  } else if (name === "weak") {
    classView.weak = !classView.weak;
  } else if (name === "stands") {
    classView.stands = !classView.stands;
  } else {
    classView.chip = name;
  }
  for (const chip of $("class-chips").querySelectorAll("button")) {
    const key = chip.dataset.chip;
    const on = key === "recurring" ? classView.recurring
      : key === "changes" ? classView.changes
        : key === "weak" ? classView.weak
          : key === "stands" ? classView.stands
            : classView.chip === key;
    chip.classList.toggle("on", on);
    chip.setAttribute("aria-pressed", on ? "true" : "false");
  }
  applyClassView();
}

/* A first click sorts a column descending, a second ascending, a third returns
   to the decision order the correspondence column carries. */
function handleSort(key) {
  if (key === "default") {
    classView.sort = "default";
  } else if (classView.sort === key && classView.desc) {
    classView.desc = false;
  } else if (classView.sort === key) {
    classView.sort = "default";
    classView.desc = true;
  } else {
    classView.sort = key;
    classView.desc = true;
  }
  applyClassView();
}

$("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  file.text().then((text) => {
    editor.value = text;
    /* A .csv is Arca Verborum's long format; a .tsv can be either shape, so the
       extension only seeds the guess and the format select is left to correct. */
    setFormat(file.name.endsWith(".csv") ? "arcaverborum" : "wide");
    $("example-note").textContent =
      `Loaded ${file.name}, read as ${currentFormat} format. Change "read as" if that's wrong.`;
    $("example-note").classList.remove("blocked");
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $("guide").classList.remove("open");
  }
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    run();
  }
});

/* Start on the ladder's first rung rather than an empty editor. */
if (CORPUS_LIST.length) {
  const first = CORPUS_LIST[0];
  $("examples").value = first.path;
  showExample(first);
}
