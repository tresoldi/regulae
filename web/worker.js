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
    post("ready", { version: instance.ccall("regulae_version", "string", [], []) });
  })
  .catch((error) => {
    post("fatal", { message: "could not start the engine: " + (error && error.message) });
  });

self.onmessage = (event) => {
  const { type } = event.data;

  if (type !== "train") {
    post("fatal", { message: "unknown request: " + type });
    return;
  }
  if (engine === null) {
    post("fatal", { message: "the engine is not ready yet" });
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
    post("result", { json: text, elapsedMs: Date.now() - started });
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
