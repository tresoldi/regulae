/*
 * How the page reads a model, separated from how it builds DOM.
 *
 * The WebAssembly build exports only the JSON entry points, so format.c is
 * dead-stripped and this page renders the model itself. Every presentation
 * decision therefore exists twice, once in src/format.c and once here, and the
 * two have drifted twice: conditioned rows were ordered on a summary of the
 * environment rather than on the environment, and this page iterated
 * classes.conditioned in the order the JSON hands it over -- class-id order --
 * while the CLI sorted by decision_index, so the same model was published as
 * two different decision lists.
 *
 * Both were reading decisions with no DOM in them, and both were untested,
 * because a function that builds elements needs a browser to check and nobody
 * built one. These do not. tests/web/reading_test.mjs runs them against real
 * `train --json` output, which is the check that would have caught both.
 *
 * Loaded as a plain script before app.js, so everything here is a global. Keep
 * it that way: a module would need index.html to change how it loads scripts,
 * and the reason to move a function here is that it can be checked without a
 * browser, not that it can be imported.
 */

/* Conditioned classes in the order they were committed.
 *
 * They are a decision list: each rule was committed against what the earlier
 * ones left unexplained, so a later one refines what an earlier one did not
 * settle, and the order is a finding rather than a presentation choice. Every
 * published table is sorted by key, which is why each row carries
 * decision_index -- restoring the order is the consumer's job and this is
 * where the page does it. */
function decisionOrder(conditioned) { // eslint-disable-line no-unused-vars
  return [...conditioned].sort(
    (a, b) => (a.decision_index - b.decision_index) || (a.id - b.id),
  );
}

/* Whether every lect in the class shows the same grapheme: a retention rather
 * than a change. Two in five committed rules are one of these, and most are
 * the retention side of a real split with the change in the contrast class. */
function isIdentity(entry) { // eslint-disable-line no-unused-vars
  return new Set(entry.segments.map((s) => s.grapheme)).size === 1;
}

function correspondence(entry) { // eslint-disable-line no-unused-vars
  return entry.segments.map((s) => `${s.lect}:${s.grapheme}`).join("  ~  ");
}

/* An environment the way the guide reads it: a filter on where the
 * correspondence applies, not a rewrite rule. */
function environment(entry) { // eslint-disable-line no-unused-vars
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

/* What the two split statistics say together, which is the only way either is
 * readable. `cost_split_fraction` says the shape -- a small fraction is a tail,
 * about half is two groups -- and `cost_split_separation` says how far apart
 * they sit.
 *
 * Calibrated on the fixtures, and the thresholds are theirs rather than
 * anything derived: grimm 2.3, chance 2.5 and Latin/Spanish 3.1 have nothing
 * standing apart; the two contaminated corpora read 7.2 and 8.2 at fractions
 * of 0.11 and 0.16; stratum and diffusion read 23.0 and 31.5 at exactly half.
 *
 * It is not a borrowing test, and saying so is not a hedge: `stratum` and
 * `diffusion`, where nothing is borrowed, both score higher than `contact`,
 * where half the wordlist is. */
function residueReading(fit) { // eslint-disable-line no-unused-vars
  const separation = fit.cost_split_separation;
  const fraction = fit.cost_split_fraction;
  if (!(separation >= 4)) {
    return "The sets fit the correspondences evenly — no group of them stands apart. "
      + "Rows are still ranked, and the worst is still the one to read first.";
  }
  if (fraction <= 0.25) {
    return `A tail of about ${Math.round(fraction * 100)}% of sets aligns much worse than `
      + "the rest. That is the shape a handful of mistaken cognate judgements makes — start "
      + "at the top and check whether those sets are cognate at all.";
  }
  return `About ${Math.round(fraction * 100)}% of the corpus aligns differently from the rest: `
    + "two populations rather than a tail. That is a shape and not a cause — a borrowed "
    + "stratum, a change spread over the lexicon and unrelated contact all make it, and this "
    + "number cannot tell them apart.";
}

/* An event's correspondence, with each lect's grapheme set named by the
   features that pick it out when any do. A set no feature names is still an
   event -- see rg_proposed_event_row on why that is ordinary. */
function eventCorrespondence(event) { // eslint-disable-line no-unused-vars
  return event.members
    .map((m) => {
      const set = `${m.lect}:{${m.graphemes.join(",")}}`;
      return m.class_features.length ? `${set}=[${m.class_features.join(" & ")}]` : set;
    })
    .join("  ~  ");
}
