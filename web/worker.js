/*
 * Runs regulae off the main thread.
 *
 * Training is a single synchronous call into WebAssembly, so while it runs this
 * worker cannot receive messages. Cancelling therefore means terminating the
 * worker from the page and starting a fresh one; re-instantiating the module
 * costs a few milliseconds once the bytes are cached. The library's own
 * cancellation is reachable here only through the deadline below, which the
 * progress callback can check without needing a message.
 */

/* global createRegulae */

importScripts("regulae.js");

let engine = null;
let deadline = Infinity;

function post(type, payload) {
  self.postMessage(Object.assign({ type }, payload));
}

/* Calls a zero-argument JSON export and hands back the string, or null when
 * the build does not export it. The pointer is always returned to the engine. */
function callJson(instance, name) {
  let pointer = 0;
  try {
    pointer = instance.ccall(name, "number", [], []);
    return instance.UTF8ToString(pointer);
  } catch {
    return null;
  } finally {
    if (pointer !== 0) {
      instance.ccall("regulae_free", null, ["number"], [pointer]);
    }
  }
}

/* Whether this engine accepts the shuffled baseline over the JSON options
 * surface. The option parser runs before the corpus is read and rejects a key
 * it does not know, so a one-line corpus is enough to ask the question; a build
 * that predates JSON support for `permutation_count` answers "unsupported
 * option" and the page disables the box rather than failing the visitor's run.
 * The CLI reaches the same feature through a flag, so this is a surface gap in
 * one build, not a missing capability. */
function baselineSupported(instance) {
  let pointer = 0;
  try {
    pointer = instance.ccall(
      "regulae_train_json", "number", ["string", "string", "string"],
      ["g\ta\tb\nx\tpa\tfa\n", "wide", '{"permutation_count":1}']);
    const payload = JSON.parse(instance.UTF8ToString(pointer));
    return !(payload.ok === false && payload.status === "unsupported option");
  } catch {
    return false;
  } finally {
    if (pointer !== 0) {
      instance.ccall("regulae_free", null, ["number"], [pointer]);
    }
  }
}

createRegulae({
  onProgress(stage, completed, total) {
    post("progress", { stage, completed, total });
    /* Returning a truthy value aborts the run; the library unwinds and reports
     * it as a cancelled status rather than a partial model. */
    return Date.now() > deadline;
  },
})
  .then((instance) => {
    engine = instance;
    post("ready", {
      version: instance.ccall("regulae_version", "string", [], []),
      baselineSupported: baselineSupported(instance),
    });
  })
  .catch((error) => {
    post("fatal", { message: "could not start the engine: " + (error && error.message) });
  });

self.onmessage = (event) => {
  const { type } = event.data;

  if (type !== "train" && type !== "segment") {
    post("fatal", { message: "unknown request: " + type });
    return;
  }
  if (engine === null) {
    post("fatal", { message: "the engine is not ready yet" });
    return;
  }

  /* Segmenting one form so the page can show how it will be read. The token
     lets the page ignore a stale answer when the visitor has typed on. This is
     the same synchronous call training is, but over a single word, so it
     returns before the next keystroke matters. */
  if (type === "segment") {
    const { word, token } = event.data;
    let pointer = 0;
    try {
      pointer = engine.ccall("regulae_segment_json", "number", ["string"], [word]);
      post("segment_result", { token, json: engine.UTF8ToString(pointer) });
    } catch (error) {
      post("segment_result", {
        token,
        json: JSON.stringify({ ok: false, status: "the word could not be segmented",
          detail: (error && error.message) || String(error) }),
      });
    } finally {
      if (pointer !== 0) {
        engine.ccall("regulae_free", null, ["number"], [pointer]);
      }
    }
    return;
  }

  const { corpus, format, options, timeoutMs } = event.data;
  deadline = timeoutMs > 0 ? Date.now() + timeoutMs : Infinity;

  let pointer = 0;
  try {
    const started = Date.now();
    pointer = engine.ccall(
      "regulae_train_json",
      "number",
      ["string", "string", "string"],
      [corpus, format || "wide", options || null],
    );
    const text = engine.UTF8ToString(pointer);
    /* The display-only extras ride beside the model rather than inside it:
     * they come from the run the module keeps live after a successful train,
     * and the documented model export stays byte-stable. An old build without
     * the exports just sends neither. */
    let chunks = null;
    let drift = null;
    try {
        const payload = JSON.parse(text);
        if (payload.ok !== false) {
            chunks = callJson(engine, "regulae_chunks_json");
            drift = callJson(engine, "regulae_drift_json");
        }
    } catch {
        /* chunks and drift are extras; the result stands without them. */
    }
    post("result", { json: text, chunks, drift, elapsedMs: Date.now() - started });
  } catch (error) {
    /* An exception here is the engine itself failing, not the corpus being
     * wrong: those come back as an ok:false payload. */
    post("fatal", { message: (error && error.message) || String(error) });
  } finally {
    if (pointer !== 0) {
      engine.ccall("regulae_free", null, ["number"], [pointer]);
    }
    deadline = Infinity;
  }
};
