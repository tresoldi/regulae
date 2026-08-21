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

/* Whether some lect answers to nothing: a correspondence to ∅, the loss (or
 * epenthesis) that the "losses" chip narrows to. Not a separate kind of row --
 * it sits in the correspondence table like any other. */
function hasNullSegment(entry) { // eslint-disable-line no-unused-vars
  return entry.segments.some((s) => s.grapheme === "∅");
}

/* The suprasegmentals a segment carries, bracketed as the CLI renders them --
 * `a[⁵⁵]` -- and empty when it carries none. A tone correspondence is part of
 * the outcome, so it reads on the row. */
function suprasegmentals(segment) { // eslint-disable-line no-unused-vars
  const bits = [];
  if (segment.tone) bits.push(segment.tone);
  if (segment.length) bits.push(`len:${segment.length}`);
  if (segment.stress) bits.push(`str:${segment.stress}`);
  return bits.length ? `[${bits.join(",")}]` : "";
}

function correspondence(entry) { // eslint-disable-line no-unused-vars
  return entry.segments
    .map((s) => `${s.lect}:${s.grapheme}${suprasegmentals(s)}`)
    .join("  ~  ");
}

/* The constraints in one context object, as a readable list. A conditioned
 * class carries one of these per segment; a cross-dimensional rule carries a
 * single one for its environment. The two rendered the same slot two ways until
 * this was shared, so it lives here once. */
function describeContext(context) { // eslint-disable-line no-unused-vars
  const bits = [];
  for (const [slot, value] of Object.entries(context || {})) {
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
  return bits.join("; ");
}

/* An environment the way the guide reads it: a filter on where the
 * correspondence applies, not a rewrite rule. */
function environment(entry) { // eslint-disable-line no-unused-vars
  const parts = [];
  for (const segment of entry.segments) {
    const bits = describeContext(segment.context);
    if (bits) {
      parts.push(`${segment.lect} — ${bits}`);
    }
  }
  return parts.join(" · ");
}

/* The environment an event's members all state, read like a class row's.
 *
 * Empty means two different things and the axis says which: on an event
 * grouped by its outcome the members differ in environment -- that is what
 * groups them -- and an empty cell there would read as "unconditioned", which
 * is the opposite. On a displacement grouping there was no environment to
 * begin with. */
function eventEnvironment(event) { // eslint-disable-line no-unused-vars
  const parts = [];
  for (const member of event.members) {
    const bits = describeContext(member.context);
    if (bits) {
      parts.push(`${member.lect} — ${bits}`);
    }
  }
  if (parts.length) {
    return parts.join(" · ");
  }
  if (event.axis === "outcome") {
    return `one change, ${event.class_ids.length} environments — see members`;
  }
  return "";
}

/* The environments the corpus cannot tell a committed one from, written the
 * way the committed one is. A count alone is a warning; these are something a
 * reader can go and check against the wordlist.
 *
 * `inverted` marks a rival that holds exactly where the committed environment
 * does not -- the same split seen from the other side -- so it has to read as
 * a negation or it states the complement of what was found. */
function environmentRivals(row) { // eslint-disable-line no-unused-vars
  const name = (r) => {
    const where = r.slot ? r.slot.replace(/_/g, " ") : "self";
    const value = r.value === undefined ? "+" : r.value;
    const lect = r.lect ? `${r.lect} — ` : "";
    return `${lect}${r.inverted ? "not " : ""}${where} [${r.feature}:${value}]`;
  };
  const rivals = row.environment_rivals || [];
  const confounds = rivals.filter((r) => r.same_partition).map(name);
  const near = rivals.filter((r) => !r.same_partition)
    .map((r) => `${name(r)} at ${r.search_margin.toFixed(2)}`);
  const lines = [];
  /* Two findings, and they must not read alike. A confound is an environment no
     evidence collected this way could separate from the committed one; a near
     tie is one the corpus did separate and preferred against, and its margin
     says by how much. A reader who takes the second for the first stops looking
     for evidence that exists. */
  if (confounds.length) {
    lines.push({ kind: "confound", text: `or equally: ${confounds.join(", ")}` });
  }
  if (near.length) {
    lines.push({ kind: "near", text: `also fits: ${near.join(", ")}` });
  }
  return lines;
}

/* A cross-dimensional rule: a segmental feature on one lect predicting a
 * suprasegmental value on another (tonogenesis is the type case). Written as a
 * correspondence, never a rewrite -- the conditioned lect *carries* the value
 * where the environment holds, it is not derived from it. The value is a Chao
 * tone letter or a length/stress name, printed the way a segment's
 * suprasegmental is so the two read alike. */
function crossDimCorrespondence(row) { // eslint-disable-line no-unused-vars
  return `${row.conditioned_lect} ${row.dimension}:${row.value}`;
}

function crossDimEnvironment(row) { // eslint-disable-line no-unused-vars
  const bits = describeContext(row.environment);
  /* environment_lect is where the trigger is read; conditioned_lect is where
   * the value lands. When they differ the rule is cross-lect, the case the
   * type example is about, so the lect is always named rather than assumed. */
  return bits ? `where ${row.environment_lect} — ${bits}` : `across ${row.environment_lect}`;
}

/* Whether a rule's environment is the only feature that carves its
 * observations this way; anything above zero is a confound the corpus cannot
 * resolve, and the row has to say so rather than assert one reading. */
function crossDimConfound(row) { // eslint-disable-line no-unused-vars
  const n = row.environment_alternatives || 0;
  return n > 0 ? `${n} other environment${n === 1 ? "" : "s"} fit the same observations` : "";
}

/* Costs are negative; the CLI writes the minus as a real minus sign and the
 * stats line does too, so the page matches rather than showing a hyphen. */
function fmtCost(value) { // eslint-disable-line no-unused-vars
  return value.toFixed(2).replace("-", "−");
}

/* A class's rate and the interval around it, as a percentage a reader can say
 * out loud. The estimate is the conditional probability the count represents --
 * how often, where this class applies, the correspondence is the one taken --
 * and lower/upper bound it at 95%. The method is Wilson on a plain run and
 * bootstrap once resampling is on, which is the difference the interval is here
 * to show; post_selection marks a rate measured on the same data that chose the
 * environment, so it says how pinned the rate is, not whether the split is real. */
function uncertaintyLabel(u) { // eslint-disable-line no-unused-vars
  if (!u) {
    return "";
  }
  const pct = (x) => `${Math.round(x * 100)}%`;
  const unit = u.method === "bootstrap"
    ? `bootstrap, ${Math.round(u.effective_n)} units`
    : "Wilson";
  return `${pct(u.estimate)} of the time · 95% CI ${pct(u.lower)}–${pct(u.upper)} · ${unit}`
    + (u.post_selection ? " · rate given the chosen environment" : "");
}

/* The interval as three positions on a 0-100 track, so a row can draw it. Always
 * on [0, 1], so bars are comparable down the column: a wide one is a rate the
 * corpus barely pins, a narrow one is a rate it is sure of, and bootstrap
 * narrowing a bar is the resampling doing its work in view. */
function ciBar(u) { // eslint-disable-line no-unused-vars
  const clamp = (x) => Math.max(0, Math.min(100, x * 100));
  return { left: clamp(u.lower), right: clamp(u.upper), tick: clamp(u.estimate) };
}

/* What the shuffled-baseline comparison says, read as a whole rather than as a
 * z-score a visitor has to interpret. Only meaningful once permutations have
 * run, which the caller checks with fit.permutation_count. cost_per_segment is
 * the mean alignment cost; lower is a tighter fit, so a model far below its
 * shuffled baseline has found structure the shuffle destroys. */
function baselineReading(fit) { // eslint-disable-line no-unused-vars
  const z = fit.cost_per_segment_z;
  const measured = fit.rules_measured || 0;
  const stood = fit.rules_above_noise || 0;
  const rules = measured === 0
    ? "No conditioned rule to measure against the shuffle."
    : `${stood} of ${measured} conditioned rule${measured === 1 ? "" : "s"} `
      + `clear${stood === 1 ? "s" : ""} what the same search reaches on the shuffle.`;
  /* z is how many baseline standard deviations the real fit sits below its
   * shuffles. A regular relationship runs tens of SDs out; near zero means the
   * pairings carry no more structure than their shuffled alternative. */
  const strength = z <= -8 ? "far below" : z <= -3 ? "below" : "no better than";
  return `Trained cost/segment ${fmtCost(fit.cost_per_segment)} sits ${strength} the shuffled `
    + `baseline (mean ${fmtCost(fit.null_cost_per_segment_mean)} over ${fit.permutation_count} `
    + `shuffles, z = ${z.toFixed(1)}). ${rules}`
    + (z > -3 ? " There is no structure here a shuffle would not also find." : "");
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
    return "The sets fit evenly; no group stands apart. Rows are still ranked worst first, "
      + "so start at the top.";
  }
  if (fraction <= 0.25) {
    return `About ${Math.round(fraction * 100)}% of sets align much worse than the rest: a tail, `
      + "the shape a few mistaken cognate judgements make. Check whether the ones at the top are "
      + "cognate at all.";
  }
  return `About ${Math.round(fraction * 100)}% of the corpus aligns differently from the rest: `
    + "two populations, not a tail. The number gives the shape, not the cause. Borrowing, a change "
    + "that spread through part of the lexicon, or contact could each produce it, and this won't "
    + "separate them.";
}

/* The score the search committed a class on: the improvement in the chosen
 * scorer and the margin over the runner-up. Costs are negative, so a more
 * negative delta is a larger improvement. Present on every decided class. */
function ruleScore(entry) { // eslint-disable-line no-unused-vars
  const bits = [];
  if (typeof entry.delta_bic === "number") {
    bits.push(`ΔBIC ${fmtCost(entry.delta_bic)}`);
  }
  if (entry.score_kind && entry.score_kind !== "corrected_bic"
      && typeof entry.delta_score === "number") {
    bits.push(`Δ${entry.score_kind} ${fmtCost(entry.delta_score)}`);
  }
  if (typeof entry.search_margin === "number") {
    bits.push(`margin ${entry.search_margin.toFixed(2)}`);
  }
  return bits.join(" · ");
}

/* Whether a conditioned class clears its shuffled null. Measured only when the
 * shuffled baseline was run; the "not measured" wording says so rather than
 * implying the rule failed. "Above noise" is the STANDS verdict the CLI prints;
 * "within noise" means a shuffle of the corpus reaches this search margin as
 * often, so the environment is not distinguished from chance. */
function ruleStanding(entry) { // eslint-disable-line no-unused-vars
  const s = entry.standing;
  if (!s || s === "unmeasured") {
    return "standing not measured; turn on the shuffled baseline to test it";
  }
  if (s === "above noise") {
    return "STANDS: its search margin clears a shuffle of the corpus (rank p ≤ 0.05)";
  }
  return "within noise: a shuffle reaches this margin as often, so the environment isn't distinguished";
}

/* The identifiability confound on a conditioned class: >0 means a different
 * neighbour's feature carves the same split and the corpus cannot say which
 * conditions it. The same flag the cross-dimensional row already reads, here
 * for a segment split. Empty when the environment is uniquely identifiable. */
function ruleConfound(entry) { // eslint-disable-line no-unused-vars
  const n = entry.environment_alternatives || 0;
  return n > 0
    ? `${n} other environment${n === 1 ? "" : "s"} carve this split the same way; `
      + "the corpus cannot say which conditions it"
    : "";
}

/* What the held-out prediction says, read as prose rather than as a table of
 * scores. Present only once cross-validation has run, which the caller checks
 * with predictive.status. Two claims are separable and both matter: whether the
 * learned correspondences generalise at all (their top-1 coverage against the
 * best naive baseline), and whether *conditioning* earns its place out of
 * sample (the log-loss gain and the confirmed/not-confirmed verdict). The
 * second is the harder bar, and on the multi-lect layer it is often not
 * cleared even when the first is -- so the two are said separately rather than
 * collapsed into one verdict. */
function predictiveReading(p) { // eslint-disable-line no-unused-vars
  if (!p || p.status === "unmeasured") {
    return "";
  }
  const pct = (x) => `${Math.round(x * 100)}%`;
  const cond = p.conditioned;
  const uncond = p.unconditioned;
  const best = cond.top1_coverage >= uncond.top1_coverage ? cond : uncond;
  const naive = Math.max(
    p.identity.top1_coverage, p.feature_distance.top1_coverage, p.inventory_frequency.top1_coverage);
  const generalises =
    `Trained on part of the corpus and asked for reflexes held out of the rest, the `
    + `correspondences get ${pct(best.top1_coverage)} right on the first guess, against `
    + `${pct(naive)} for the best naive baseline below.`;
  let conditioning;
  if (p.status === "confirmed") {
    conditioning = ` Conditioning earns its place out of sample: held-out log loss falls by `
      + `${p.log_loss_gain.toFixed(3)} when the environments are used.`;
  } else if (p.status === "not_confirmed") {
    conditioning = ` Conditioning does not help out of sample here `
      + `(log-loss change ${p.log_loss_gain.toFixed(3)}): the environments fit the corpus they were `
      + `found on and no further.`;
  } else {
    conditioning = ` Too few groups to test whether conditioning generalises; the coverage is `
      + `descriptive only.`;
  }
  return generalises + conditioning;
}

/* An event's correspondence, with each lect's grapheme set named by the
   features that pick it out when any do. A set no feature names is still an
   event -- see rg_proposed_event_row on why that is ordinary. */
function eventCorrespondence(event) { // eslint-disable-line no-unused-vars
  return event.members
    .map((m) => {
      const set = `${m.lect}:{${m.graphemes.join(",")}}${suprasegmentals(m)}`;
      return m.class_features.length ? `${set}=[${m.class_features.join(" & ")}]` : set;
    })
    .join("  ~  ");
}

/* The shared feature displacement, rendered as a compact rule annotation.
 * Each item says which feature differs and in which direction, relative
 * to the first two member slots (member[0] and member[1]):
 *   from=present, to=absent  => member[0] has it   => "+feature (lect0)"
 *   from=absent,  to=present => member[1] has it   => "+feature (lect1)"
 *   otherwise (valued features)                     => "feature: from->to"
 */
function eventDisplacement(event) { // eslint-disable-line no-unused-vars
  const disp = event.shared_displacement;
  if (!disp || !disp.length) return "";
  const members = event.members || [];
  return disp
    .map((d) => {
      if (d.from === "absent" && d.to === "present") {
        return `+${d.feature}(${members[1] ? members[1].lect : "?"})`;
      }
      if (d.from === "present" && d.to === "absent") {
        return `+${d.feature}(${members[0] ? members[0].lect : "?"})`;
      }
      return `${d.feature}: ${d.from}→${d.to}`;
    })
    .join(", ");
}
