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

/* global CORPORA, CORPUS_LIST, GUIDE_STEPS */

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

/* ---- worker lifecycle -------------------------------------------------- */

function startWorker() {
  worker = new Worker("worker.js");
  worker.onmessage = (event) => {
    const message = event.data;
    if (message.type === "ready") {
      ready = true;
      runButton.disabled = false;
      $("version").textContent = "v" + message.version;
    } else if (message.type === "progress") {
      showProgress(message);
    } else if (message.type === "result") {
      finish(message);
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
  worker.postMessage({ type: "train", corpus, format: currentFormat });
}

function finish({ json, elapsedMs }) {
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
  $("timing").textContent = (elapsedMs / 1000).toFixed(2) + "s";
  render();
  results.classList.add("active");
}

/* ---- rendering --------------------------------------------------------- */

function correspondence(entry) {
  return entry.segments.map((s) => `${s.lect}:${s.grapheme}`).join("  ~  ");
}

/* Renders an environment the way the guide reads it: a filter on where the
 * correspondence applies, not a rewrite rule. */
function environment(entry) {
  const parts = [];
  for (const segment of entry.segments) {
    const context = segment.context;
    if (!context || Object.keys(context).length === 0) {
      continue;
    }
    const bits = [];
    for (const [slot, value] of Object.entries(context)) {
      if (typeof value === "string") {
        bits.push(`${slot}: ${value}`);
      } else if (Array.isArray(value)) {
        bits.push(`${slot}: ` + value
          .map((c) => (c.offset === undefined
            ? `${c.feature}:${c.value}`
            : `${c.feature}:${c.value}@${c.offset}`))
          .join(", "));
      }
    }
    if (bits.length) {
      parts.push(`${segment.lect} — ${bits.join("; ")}`);
    }
  }
  return parts.join(" · ");
}

function renderClasses() {
  const body = $("classes").querySelector("tbody");
  body.innerHTML = "";

  const all = [
    ...model.classes.conditioned.map((c) => ({ entry: c, conditioned: true })),
    ...model.classes.unconditioned.map((c) => ({ entry: c, conditioned: false })),
  ];
  if (!all.length) {
    body.innerHTML = '<tr><td colspan="2" class="empty">No classes were found.</td></tr>';
    return;
  }

  for (const { entry, conditioned } of all) {
    const row = document.createElement("tr");
    row.dataset.classId = String(entry.id);

    const corr = document.createElement("td");
    corr.className = "corr";
    corr.textContent = correspondence(entry);
    if (conditioned) {
      const env = document.createElement("span");
      env.className = "env";
      env.textContent = environment(entry) || "conditioned";
      corr.appendChild(env);
    }

    const count = document.createElement("td");
    count.className = "count";
    count.textContent = entry.count % 1 === 0 ? entry.count : entry.count.toFixed(1);

    /* count is aligned positions, so one word with a doubled segment reads 2.
       Whether a correspondence recurs is the question a visitor is actually
       asking of this table, and only this column answers it. */
    const sets = document.createElement("td");
    sets.className = "sets";
    sets.textContent = (entry.supporting_cognates || []).length || "";
    sets.title = "distinct cognate sets behind this row";

    row.append(corr, count, sets);
    row.addEventListener("click", () => select(entry.id));
    body.appendChild(row);
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

    const gloss = document.createElement("div");
    gloss.className = "gloss";
    gloss.textContent = alignment.cognate_id;
    const pair = document.createElement("span");
    pair.className = "pair";
    pair.textContent = `  ${alignment.lect_a} ~ ${alignment.lect_b}`;
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
        col.addEventListener("click", (event) => {
          event.stopPropagation();
          select(ids[0], true);
        });
      }
      cols.appendChild(col);
    }

    block.append(gloss, cols);
    container.appendChild(block);
  }
}

function select(classId, scrollToClass) {
  selectedClass = selectedClass === classId ? null : classId;

  for (const row of $("classes").querySelectorAll("tr[data-class-id]")) {
    const id = Number(row.dataset.classId);
    row.classList.toggle("selected", id === selectedClass);
    row.classList.toggle("dimmed", selectedClass !== null && id !== selectedClass);
  }

  for (const block of $("alignments").querySelectorAll(".alignment")) {
    const ids = JSON.parse(block.dataset.classes);
    block.classList.toggle("hidden", selectedClass !== null && !ids.includes(selectedClass));
  }
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
  const lects = document.createElement("div");
  lects.innerHTML = `<span>${model.lects.join(", ")}</span>`;
  $("summary").appendChild(lects);
}

function render() {
  selectedClass = null;
  renderSummary();
  renderClasses();
  renderAlignments();
  select(null);
  selectedClass = null;
}

/* ---- examples ---------------------------------------------------------- */

function loadCorpus(path, format) {
  editor.value = CORPORA[path] || "";
  currentFormat = format || "wide";
  results.classList.remove("active");
  $("error").innerHTML = "";
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
    if (!entry) {
      return;
    }
    loadCorpus(entry.path, entry.format);
    const note = $("example-note");
    note.classList.toggle("blocked", !entry.readable);
    note.textContent = entry.readable
      ? entry.description
      : `This corpus cannot be read yet: “${entry.blockedBy}” is not in the feature system. `
        + "It is here so the gap is visible rather than hidden. Run it to see the error.";
  });
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
  for (const entry of model.classes.unconditioned) {
    lines.push(`UNCOND\t${entry.id}\t${correspondence(entry)}\t${entry.count}\t${sets(entry)}`);
  }
  for (const entry of model.classes.conditioned) {
    lines.push(`COND\t${entry.id}\t${correspondence(entry)}\t${entry.count}\t${sets(entry)}\t${environment(entry)}`);
  }
  return lines.join("\n") + "\n";
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
$("guide-open").addEventListener("click", () => $("guide").classList.add("open"));
$("guide-close").addEventListener("click", () => $("guide").classList.remove("open"));
$("download-json").addEventListener("click", () => download("regulae-model.json", JSON.stringify(model, null, 2)));
$("download-summary").addEventListener("click", () => download("regulae-summary.tsv", summaryText()));

$("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  file.text().then((text) => {
    editor.value = text;
    currentFormat = file.name.endsWith(".csv") ? "arcaverborum" : "wide";
    $("example-note").textContent =
      `Loaded ${file.name}, read as ${currentFormat} format.`;
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
  loadCorpus(first.path, first.format);
  $("example-note").textContent = first.description;
}
