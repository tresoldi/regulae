# The regulae package: a consumer's guide

**Status: DOCUMENTATION.** This document describes the public
API from the perspective of a downstream package that wants to
*consume* its output — historical inference, visualization,
cross-family comparison, whatever.

**Two interfaces, one contract.** The core is a C99 library, and
its output has two faces that carry the same data:

- The **C ABI** — the opaque `rg_multi_model` read through
  `rg_multi_model_*` accessors, declared in `include/regulae.h`.
  This is the authoritative spelling: `train_model` is
  `rg_train_model`, a class row is `rg_multi_class_row`, and so on.
- The **JSON** the core renders (`rg_model_to_json`, the CLI's
  `train --json`), which is the same model as a string. The
  **Python package** is a thin wrapper that trains through the C
  core and parses that JSON into dataclasses (§2); it does no
  modelling of its own, so the JSON schema *is* the Python schema.

This document describes what each field *means*, and what it does
not mean, in a readable dataclass spelling that follows the Python
wrapper. Where a name differs, the C accessor is given alongside.
**Where this document and `include/regulae.h` disagree, the header
is right.**

`RG_ABI_VERSION` is 39. The public surface has moved since this
guide was first written: every published table is handed out whole
rather than through a count/index pair (§4.9); the fields that say
what a search decided moved into one `evidence` member (§4.6); a
class row names its distinct supporting sets (§4.2) and links to
the row that contrasts it (§4.3); a deletion has a row shape (§4.9,
§6); a cross-dimensional rule can be lect-internal (§6); and a
conditioned rule flags when its environment is not identifiable
from the corpus (§4.3, §6). The Python wrapper replaced a full
reimplementation on 2026-08-17 (§2) — the retired code is under
`docs/legacy_python/`.

The reliability fields — `standing`, `search_margin`,
`decision_index`, the contrast counts, `rg_corpus_fit` — are what
decide whether an inference built on this output is defensible.
§4.6 is that section.

The target reader is someone building a separate package (not
modifying `regulae` itself) who needs to know:

- What to import.
- What `train_model` returns and how to read it.
- What each field on the output means, and what it *doesn't* mean.
- Which types are public API in a semver sense and which are
  internal.
- Where the natural handoff points are for the historical
  inference layer (`07_historical_inference.md`) and what
  it should expect.

If you want to know how the alignment search *works* internally
(EM, DP, BIC-gated discovery), read `docs/alignment_details.md`
and the numbered specification documents in the parent
repository. This document is only about the external
contract.

## 1. What regulae is and isn't

**What regulae does**: given an explicit list of cognate sets
across N lects (N ≥ 1), produce a structured description of the
systematic phonological correspondences between those lects.
The output is a set of *synchronic* correspondence classes
— patterns of which segments correspond to which, optionally
conditioned on phonological environment, optionally with
cross-dimensional rules (e.g. tonogenesis) layered on top.

**What regulae does not do**:

- **No cognacy detection.** The user supplies which forms are
  cognate; regulae never decides that for them. A non-cognate pair
  passed into training will be trained on alongside the real
  cognates and its segment pairings will contribute to the
  learned counts. The only post-hoc recourse is the outlier
  diagnostic — `rg_find_cognate_outliers`, and the CLI's
  `regulae outliers` — which z-scores cognate sets by alignment
  cost so a human reviewer can flag suspects. The thin wrapper does
  not surface it; the model JSON's `outliers` section (on `.raw`)
  carries the same ranking.
- **No directionality.** The pairwise alignment and scoring
  functions are symmetric. "Source" and "target" in the API are
  positional labels inside a pair, not a claim that one is the
  ancestor. The segment cost is the geometric mean of the two
  conditional directions, so a model trained on (Latin, Spanish)
  and one trained on (Spanish, Latin) give the same tables read
  backwards. Two caveats: the displacement table describes a
  feature moving from one value to another and so is directional
  by construction, and the alignment DP breaks exact cost ties by
  enumeration order, which the exchange does not preserve — on a
  corpus small enough for a tie to decide an alignment, the two
  directions can still differ.
- **A doublet is data, not an error.** One lect with two
  reflexes in a cognate set is published as one set per
  combination, each with its share of the confidence.
  `rg_corpus_doublet_set_count` says how often that happened.
  A consumer counting cognate sets should expect more of them
  than the source file has, and weights below 1.
- **Intervals say what they were computed from.** Every
  count-bearing row carries `rg_uncertainty_estimate`. The
  default is a Wilson interval whose denominator counts aligned
  positions, which are not independent observations — a
  Latin–Spanish corpus has 413 of them over 97 cognate sets.
  Setting `bootstrap_n` replaces it with a percentile interval
  resampled over the named observation group. `bootstrap_unit=AUTO`
  uses `etymon_group` when supplied and otherwise treats whole
  **cognate sets** as independent; source-publication grouping is
  opt-in. `rg_corpus_fit` publishes the chosen unit, its effective
  count and how many sets lacked each optional label. Measured, the
  resampled interval is *narrower* on
  most rows, not wider: the rate is a ratio whose numerator and
  denominator move together under resampling, which the
  fixed-denominator binomial model does not capture.
  `post_selection` marks a row whose environment was chosen by
  the same data; that interval says how well the rate is pinned
  given the environment, not whether the environment is real.
- **A conditioned class reports how far it stands above the
  search that found it.** `search_margin` is the search charge
  the rule's evidence could carry; `rg_corpus_fit.null_search_margin`
  is what the same search reaches on the corpus shuffled. A rule
  at or under that level is not distinguishable from an artefact
  of looking, however large its count.
- **A conditioned class is a comparison, and reports one.**
  `contrast_count` is the same correspondence where the
  environment does not hold, and `score_kind`/`delta_score` say what the split
  scored. A class with more mass outside its stated environment
  than inside it is not evidence that the environment conditions
  anything, whatever its `count` says. Consumers presenting a
  conditioned rule should carry both numbers or neither.
- **No claim of relatedness.** A trained model is a description of
  whatever correspondences the search found, not evidence that the
  lects are related. The class counts in particular are not that
  evidence and move against it: shuffling a corpus's pairings removes
  every correspondence and raises the counts. `rg_multi_model_fit`
  carries the statistic that does discriminate, and
  `permutation_count` calibrates it against the same corpus shuffled.
  A consumer presenting regulae output as a relatedness finding
  without it is presenting the wrong number.
- **No time.** Lects are referred to by string IDs; there are no
  dates, epochs, or temporal orderings anywhere in the output.
- **No proto-forms.** A `CorrespondenceClass` names which segment
  appears in each participating lect; it does not reconstruct what
  "the proto-segment" was, because
  reconstruction presupposes a topology (see
  `07_historical_inference.md` §4).
- **No phylogeny.** regulae never builds a tree, network, or
  distance matrix. The output has no notion of which lects are
  "closer" than which others.
- **No morphological analysis.** regulae never segments a word.
  It does consume boundaries the caller supplies: they block
  chunk promotion across a seam, and they condition rules
  through `rg_context_spec.morphological` (where the segment
  sits in its morpheme) and `morpheme_index` (which morpheme).
  A corpus without boundaries gets neither axis. Deciding where
  the boundaries are is the job of the package upstream.

These are all **design choices**, not gaps. Each of them is a
presupposition that regulae refuses to smuggle in; they belong in
the layer that consumes regulae's output, where they can be
modelled explicitly as priors or posteriors over separate
variables.

## 2. Installation and import

**C consumers** link the library:

```sh
cmake -S . -B build && cmake --build build
# libregulae.a (or .so), include/regulae.h, and merkmal alongside it
```

`include/regulae.h` is the contract, and `README.md` has a tour of
it. `RG_API` marks the exported surface: anything without it is not
public, and the `*_internal` symbols are hidden by the visibility
preset.

**Python consumers** install the wrapper, which compiles the C core
straight into a CPython extension — so a built wheel needs neither
`libregulae` nor the merkmal Python package:

```sh
pip install -e .          # from the repository root
```

The merkmal C is found at `../merkmal` by default
(`REGULAE_MERKMAL_SOURCE_DIR` overrides it). The wrapper is
deliberately small — it trains and reads JSON, nothing more:

```python
import regulae

model = regulae.train_model(open("cognates.tsv").read(), fmt="tsv")

# The trimmed surface, all parsed from the C core's JSON:
regulae.MultiLectModel           # the output (§4)
regulae.CorrespondenceClass      # a class row (§4.2, §4.3)
regulae.CrossDimensionalRule     # §6
regulae.PairwiseModel            # a pair's tables, incl. gaps (§4.1, §6)
regulae.GapCorrespondence        # a deletion or epenthesis (§6)
regulae.Segment, regulae.Uncertainty, regulae.PredictiveEvidence
regulae.GAP_GRAPHEME             # the "∅" a deleting lect carries
regulae.RegulaeError, regulae.SourceMarkerError
```

The wrapper used to be a full reimplementation of the engine; it
drifted behind the C and was retired to `docs/legacy_python/` on
2026-08-17. The fine-grained primitives it exposed (`score_link`,
`align_forms`, `find_cognate_outliers`, the EM internals) are gone
— they existed only because Python had its own engine. What
remains is the model and the loaders. Any field the dataclasses do
not surface is on `MultiLectModel.raw`, the parsed JSON verbatim.

## 3. The canonical entry point

In Python, `train_model` takes a corpus as text or a path and
returns a `MultiLectModel`:

```python
import regulae

model = regulae.train_model(
    "cognates.tsv",          # a path, or the corpus as a string
    fmt="tsv",               # "wide" | "tsv" | "gled" | "arcaverborum"
    options={"permutation_count": 100},   # optional, passed to the core
)
```

`fmt` selects the loader: `"wide"` is one row per cognate with a
column per lect; `"tsv"` is one row per (cognate, lect);
`"gled"` and `"arcaverborum"` read those dataset formats. `options`
is a flat mapping the C core reads (`rg_train_options_from_json`),
which rejects unknown keys.

In C the same call is `rg_train_model(ctx, cognates, count,
&options, &model)`, over a `rg_cognate_set` array built by one of
the `rg_corpus_parse_*` / `rg_corpus_load_*` loaders. The loaders
are where a corpus is assembled; there is no hand-built dataclass
path in the wrapper, because the wrapper hands the corpus text
straight to the core.

Training is **deterministic**: identical inputs produce
bitwise-identical outputs, cross-process. All internal discovery
loops are sequential-greedy with deterministic tiebreakers,
vocabulary iteration is sorted, and the merkmal dependency is
configured to avoid cross-process drift (§8).

## 4. The output shape: `MultiLectModel`

`MultiLectModel` is a frozen dataclass parsed from the model JSON:

```python
@dataclass(frozen=True)
class MultiLectModel:
    lects:                 tuple[str, ...]
    unconditioned_classes: tuple[CorrespondenceClass, ...]
    conditioned_classes:   tuple[CorrespondenceClass, ...]
    cross_dimensional:     tuple[CrossDimensionalRule, ...]
    pairwise_models:       tuple[PairwiseModel, ...]
    fit:                   dict          # rg_corpus_fit, §4.6
    regulae_version:       str
    raw:                   dict          # the parsed JSON verbatim
```

`.classes` returns the unconditioned rows then the conditioned
ones. Anything not surfaced as a field is on `.raw` — the model
JSON is the whole contract, and `.raw` is that JSON. The C reads
the same data through `rg_multi_model_*` accessors over the opaque
`rg_multi_model`.

### 4.1 `pairwise_models`

A tuple of `PairwiseModel`, one per lect pair that had shared
cognate data. Pairwise training is symmetric (§1), so `(a, b)` and
`(b, a)` are one entry; a pair absent from the tuple means the two
lects shared no cognates. Each carries the pair's `conditioned`
correspondences and its `gaps` (§6). The full per-pair segment,
displacement, tonal and chunk tables are on the C
`rg_pairwise_model` and in `.raw`; the wrapper surfaces the two a
downstream consumer reaches for.

In C this is `rg_multi_model_pair_model_at`, which returns an
`rg_multi_pair_model_row` carrying `lect_a`, `lect_b` and the
model. It is the one table still read a row at a time (§4.9): it
holds each pair's owned strings alongside the published row, so
there is no array of rows to hand out. The row *is* ordered — reconciliation walks lect pairs in
ascending lect-id order, which fixes the direction each pair is
aligned in — but the analysis it holds is not: training A against
B and B against A produce mirror models, and the tests assert it.
Read the pair as a labelled edge, not as a direction of change.

The wrapper's `PairwiseModel` surfaces the two per-pair tables a
downstream consumer usually reaches for: `conditioned`, a tuple of
`ConditionedCorrespondence` (one source grapheme to one target
under one `Context`, §5), and `gaps`, a tuple of `GapCorrespondence`
(a segment answering to nothing, §6). The C `rg_pairwise_model`
carries more, each with its own accessor and each present in the
JSON on `MultiLectModel.raw`:

- `rg_pairwise_model_segment_counts`: the 1-to-1 correspondence
  table, the emission model alignment is scored against.
- `rg_pairwise_model_displacements`: the feature-space signature of
  how segments changed, aggregated across all 1-to-1 links — what
  lets a natural class be found across different grapheme pairs
  (the same displacement vector behind `p~f`, `t~θ`, `k~x`).
- `rg_pairwise_model_chunks`: promoted multi-segment
  correspondences (e.g. Latin `kt → tʃ` in Italian).
- `rg_pairwise_model_tonal_counts`: source tone to target tone,
  for pairs where tone is present.
- `rg_pairwise_model_gap_counts`: the gap table §6 describes.
- `rg_pairwise_model_cross_dimensional_rows`: the pair's
  cross-dimensional rules, lifted into `model.cross_dimensional`.

For most downstream consumers the pairwise models are
**low-level, detail-heavy**. Read `conditioned_classes` and
`unconditioned_classes` first (§4.2, §4.3), and descend into
`pairwise_models` only for a specific pair's tables.

### 4.2 `unconditioned_classes`

A tuple of `CorrespondenceClass`, sorted by count descending, then
by segment tuple. Each class says:

> "Across the corpus, when these N lects have segments at
> corresponding positions within a cognate, this is one of the
> segment tuples that appears, and it happened *count* times."

The class is the **unit of downstream evidence**. It is the
strongest summary regulae produces: it's consolidated across
pairwise training via union-find reconciliation, it has a
count, and every participating lect contributes a segment.

```python
for klass in model.unconditioned_classes:
    print(klass.graphemes)              # {"LATIN": "k", "SPANISH": "tʃ"}
    print(klass.count)                  # aligned positions, e.g. 8.0
    print(len(klass.supporting_cognates))  # distinct sets, §4.2 below
    print(klass.confidence)             # 1.0 for unconditioned classes
```

`segments` is a tuple of `Segment` (`.lect`, `.grapheme`, and a
per-lect `.context` on conditioned classes); `.graphemes` is the
`lect -> grapheme` mapping. A `Segment` whose grapheme is
`GAP_GRAPHEME` marks a lect that deleted the segment the others
keep (§6), so a class can carry a `∅`.

An unconditioned class also carries `suprasegmentals`
(`rg_multi_class_row.suprasegmentals`), a per-segment `tone` /
`length` / `stress` parallel to `graphemes` and part of the class's
outcome identity: the same graphemes under a different tone are two
classes, so a Sinitic or Hmong-Mien tone correspondence set gets its
own row (from ABI 39). Each field is `""` where the segment carries
none, and the JSON omits it there. Conditioned classes leave it unset
— a suprasegmental predicted by an environment is a cross-dimensional
rule (§6), not a class.

**Read `len(supporting_cognates)`, not `count`, for how many
cognate sets a class rests on.** `count` is aligned positions
weighted by cognate confidence, so one word with a geminate or a
repeated segment reaches 2 without the correspondence recurring
anywhere. `supporting_cognates` lists each set once (from ABI 30);
its length is the recurrence a reader wants, and an adjudication
panel repeatedly mistook `count` for it. On the C side this is
`rg_multi_class_row.supporting_cognates` /
`supporting_cognate_count`.

### 4.3 `conditioned_classes`

Same shape as unconditioned, but at least one `Segment` carries a
non-empty `.context` — the conditioning environment under which
that lect's segment appears (`klass.conditioned` is true). A lect
without a specific constraint carries no context.

**One segment tuple may appear in several conditioned classes,
and code that indexes this list by tuple has to expect that.**
Discovery is greedy and each rule is committed against what the
earlier ones left, so a change conditioned by something that is
not a natural class comes out as a decision list: RUKI's *s
retracts after *r*, *u*, *k* and *i*, which is four rules with
one outcome, and no single feature covers the four. Read them in
`evidence.decision_index` order — a later rule refines or applies
within what an earlier one did not settle.

It was not so until 2026-08-17: classes were merged on the tuple
alone, which collapsed the whole list into one row whose
environment was whichever carried the most constraints. On the
four-lect Romance corpus that discarded 32 of the 58 splits the
search had committed, and on Latin/Spanish it is the difference
between 20 conditioned classes and 25. Rows merge now when a
later split's observations are a subset of an earlier row's —
which is what a second *description* of one rule looks like — and
stay apart when the split brings observations no earlier row
has.

Reading a conditioned class:

```python
for klass in model.conditioned_classes:
    print(klass.graphemes)
    print(klass.count)
    print(klass.confidence)          # coverage-based strength in [0, 1]
    for seg in klass.segments:
        if seg.context:              # this lect's conditioning environment
            print(seg.lect, seg.context)
    # The row that holds the pivot's other reflex out of the environment:
    print(klass.contrast_class_id)   # -1 when there is none
```

**A conditioned class's `contrast_count` is ~0 by construction,
and `contrast_class_id` is the number to read instead.** A real
split exists because the pivot takes a *different* reflex out of
the environment, so the same tuple barely recurs there;
`contrast_count` measures that same tuple and is near zero exactly
when the conditioning holds. `contrast_class_id` (from ABI 31) is
the id of the class holding the pivot's *other* reflex — on Verner,
the conditioned `gothic:d ~ pgmc:θ` before a vowel points at
`gothic:d ~ pgmc:d` — and `contrast_alternative_count` is that
reflex's mass in the complement. `-1` where there is none. Look the
id up in `model.classes` (unconditioned first, then conditioned).

A conditioned class also carries **`environment_alternatives`** (ABI 36), the
same identifiability flag the cross-dimensional row has (§6): the count of rival
conditioners at another position the corpus cannot tell this environment from. 0
when the environment is pinned, >0 when a different neighbour's feature carves
the split the same way. On a corpus where the preceding consonant and the
following vowel are confounded it flags the palatalisation rule; on Verner it
reads 0 for the rules committed with the stress environment and >0 for a rule
framed as "preceded by a vowel," which the data cannot separate from "followed
by a stressed syllable." It is 0 on every unconditioned class.

The `confidence` field on conditioned classes is the
**pivot-bucket coverage**: `count / pivot_bucket_size` where
`pivot_bucket_size` is the number of observations of the pivot
lect+grapheme that the class-discovery loop saw. A value of 1.0 means "every
observation of the pivot is in this class" (a strong rule); 0.1
means "only 10% of the pivot's observations landed here" (weak
minority). Use this to sort or filter classes by signal
strength when your downstream stage is sensitive to marginal
evidence.

### 4.4 Provenance: there is no `cognate_corpus`

**The model does not retain the input corpus.** The retired Python
model kept it; `rg_multi_model` does not, and neither does the
wrapper. In C the corpus is a handle the caller already owns
(`rg_corpus`, from one of the loaders) and it outlives the training
call anyway, because the model borrows from it — walking back from
a class to the source forms means keeping that handle. In the
wrapper the corpus was a string the caller passed in; the model
JSON does carry `alignments` and `outliers` sections, reachable on
`MultiLectModel.raw`, but not the corpus itself.

`rg_multi_class_row.supporting_cognates` is the anchor that does
exist: the cognate ids a class was built from, so a consumer can
point at the evidence without matching graphemes back by hand.

Its length is also the number to read when the question is whether
a correspondence recurs. `count` cannot answer that — it is
aligned positions weighted by cognate confidence, so one word with
a geminate reaches 2 without recurring anywhere. M6's adjudicators
made that mistake repeatedly, and a downstream tool ranking rows
by `count` will make it silently. From ABI 30 each set is listed
once; before it, an id appeared once per position, so the list's
length agreed with `count` and answered nothing.

### 4.5 `lects`

The canonical ordered list of lect IDs observed in the corpus
(`MultiLectModel.lects`; `rg_multi_model_lects` in C). First-seen
order — **not** alphabetical. This ordering is stable and
deterministic; downstream packages can rely on it as an index.

### 4.6 What the model says about its own reliability

This is the part a consumer is most likely to miss by reading the
type definitions alone, and the part that decides whether an
inference built on top of regulae is defensible.

**Class counts are not evidence of relatedness, and they move the
wrong way.** Shuffling a corpus's pairings removes every
correspondence there is to find, and greedy splitting over a
large candidate inventory then finds *more* environments in the
noise, not fewer. A downstream stage that ranks language pairs by
how many classes they produce has built a detector for corpus
size. `rg_corpus_fit.cost_per_segment` is the number that
separates signal from noise — strongly negative on real cognates,
near zero on shuffled ones — and its scale depends on the corpus,
so it has to be read against that corpus's own baseline.

`rg_multi_model_fit` returns an `rg_corpus_fit`:

- `cost_per_segment`, and with `permutation_count > 0`, the same
  measure over shuffled trainings: `null_cost_per_segment_mean`,
  `null_cost_per_segment_sd`, and `cost_per_segment_z` between
  them. The baseline is off by default because it costs one full
  training run per shuffle; when it is off, every `null_` field
  is zero and means "not measured", not "zero".
- `null_search_margin` at `null_search_margin_quantile`: the
  search charge a rule has to clear to be saying more than the
  search itself does.
- `rules_above_noise` of `rules_measured`: the multi-lect rules --
  the conditioned classes and the cross-dimensional rules, which
  already include every lect pair's, because the multi-lect table
  is built by lifting them.
- `pairwise_rules_above_noise` of `pairwise_rules_measured`: the
  conditioned correspondences each lect pair carries in its own
  model. Reported separately, and not added to the pair above,
  because they are counted **per pair** -- a rule visible in every
  pair of a four-lect corpus is six here and one there, so merging
  them would make the ratio depend on how many lects the corpus
  samples.

  Do not read the multi-lect pair alone. On the Grassmann fixture
  it says 0 of 4 -- nothing distinguishable from having looked --
  while the rule that *is* Grassmann's Law, Greek `t` answering
  PIE `tʰ` where an aspirate follows, stands in the per-pair
  table. A corpus's only real finding can be in either.
- `inferred_nucleus_form_count` and `syllabified_form_count`:
  forms with no vowel and no syllabic consonant, which were given
  a nucleus so the syllable predicates had something to hold of.
  A syllable-conditioned rule on a corpus with many of these is
  resting on a guess.
- `scored_set_count`, and separately
  `rg_multi_model_unpaired_set_count` — sets that carried fewer
  than two forms and so contributed no correspondence. Worth
  reading as a proportion: a high one means the lect sample, not
  the method, is deciding the result.
- `etymon_group_count`, `source_group_count`, their missing-label counts,
  `bootstrap_unit` and `bootstrap_effective_unit_count`: the dependence
  structure supplied by the caller and the level actually resampled. A
  missing label falls back to that cognate set as its own group; it is never
  inferred from spelling or form similarity.

Every conditioned rule and every conditioned class carries what
the search decided about it, in one `evidence` member:

```c
typedef struct rg_rule_evidence {
    rg_split_scorer  scorer;          /* criterion that selected it */
    double           delta_score;     /* authoritative score */
    double           delta_bic;       /* compatibility alias */
    int              decision_index;  /* where in the decision list */
    double           search_margin;   /* how heavy a charge it carries */
    rg_rule_standing standing;        /* the verdict */
    rg_null_model    standing_null;   /* comparison supporting it */
    rg_predictive_evidence predictive; /* held-out, not in-sample */
} rg_rule_evidence;
```

- `standing` — `above-noise`, `within-noise`, or `unmeasured`
  when no baseline was run. **`unmeasured` is not a pass.** It
  says the comparison was never made.
- `search_margin` — the number that verdict is computed from,
  against `rg_corpus_fit.null_search_margin`.
- `decision_index` — discovery is greedy, so the rules form a
  decision list: a later rule refines what an earlier one left
  unsettled. Published tables are sorted by key so lookups can
  binary-search them, which destroys that order; this preserves
  it. `-1` means the row was not decided by a search.
- `scorer` and `delta_score` — the criterion and its score, negative where the
  split paid for its complexity. `delta_bic` is numerically identical for
  source compatibility, but is not a BIC value when `scorer` names NML or the
  Dirichlet marginal likelihood.
- `predictive` — a separate group-held-out result. `unmeasured` means no
  predictive run was requested; `descriptive_only` means the corpus or the
  exact association could not support confirmation without leakage;
  `confirmed` means conditioning reduced held-out log loss; and
  `not_confirmed` means it did not. This field never changes `delta_score` or
  `standing`.

Set `rg_train_options.predictive_folds` to at least two to run it. Fold units
are connected components under the caller's cognate ids, etymon groups and
source groups, so alternate reflexes and paradigm cells cannot cross the
split. Every fold learns its feature vocabulary, alignments and environment
decision list from training components only. `rg_corpus_fit.predictive`
reports conditioned and unconditioned log loss, top-k coverage, Brier score,
ten-bin calibration error and abstention; sibling fields report identity,
inventory-frequency and feature-distance baselines. Pairwise scores include
both orientations, and three-or-more-lect corpora additionally report
leave-one-lect-out pooling. See `docs/m4_evaluation.md` for the frozen protocol
and the real negative panel.

These four were separate fields on each row until 2026-08-16,
copied into four row types. A C consumer reads
`row->evidence.standing` where it used to read `row->standing`.

`contrast_count` and its denominators stay **on the row**, not in
the evidence, because their shape differs per row type — a
conditioned correspondence contrasts against a total, a
cross-dimensional rule against a source count and a confidence. A
conditioning claim is a comparison, and these are the other side
of it: a rule published without the contrast it was measured
against cannot be read.

On a conditioned multi-lect class, `contrast_count` alone will not
answer "is this real?" — it is the *same* reflex out of the
environment, and that is ~0 exactly when the conditioning holds,
because a real split means the pivot takes a *different* reflex
elsewhere. From ABI 31 the row also carries `contrast_class_id`,
the id of the class holding that different reflex, and
`contrast_alternative_count`, its mass in the complement. On
Verner the conditioned `gothic:d ~ pgmc:θ` (before a vowel) points
at `gothic:d ~ pgmc:d`: the contrast that makes the rule, which
sat in an unrelated row with nothing linking it before. `-1` where
there is no contrast (an unconditioned class, or a complement with
no majority). The id indexes `class_id` — a position in the
unconditioned array, or past its end in the conditioned array.

`uncertainty` also stays on the row. How well a rate is pinned is
a different question from whether the environment is real, and it
is asked of every row including the aggregated tables, which
carry no evidence at all — a segment, displacement, tonal or
chunk row is counted, not decided.

Intervals say what they were computed from.
`rg_uncertainty_estimate.method` distinguishes a closed-form
Wilson interval from a resampled bootstrap one, which answer
different questions, and `post_selection` is set on every row
whose environment was chosen by the same data the interval is
computed from. Such an interval says how well the rate is pinned
*given* that environment, and nothing about whether the
environment is real; `search_margin` against `null_search_margin`
is what answers that.

Finally, the corpus reports how it was built:
`rg_corpus_doublet_set_count` and
`rg_corpus_doublet_expansion_count` say how often a lect
contributed more than one reflex to a set. A doublet is a fact
about a language, not an error in a file, and the corpus carries
one set per combination of reflexes with a share of the
confidence each.

### 4.7 Is the corpus one thing?

`rg_corpus_fit` carries two numbers about the *shape* of the fit
rather than its level. `cost_split_separation` is how far apart
the two sides are, in pooled standard deviations, at the best
two-way split of the per-cognate-set alignment costs;
`cost_split_fraction` is the share of sets on the worse-aligning
side.

A unimodal sample still has a best split, so the number is never
zero and has to be read against something — and read *with* the
fraction. Measured on this repository's fixtures:

| corpus | separation | fraction |
| --- | ---: | ---: |
| real pair corpora | 2.7 – 3.0 | 39–65% |
| `chance` (unrelated lects) | 2.4 | 52% |
| `contaminated` (5 bad judgements in 45) | 7.2 | 11% |
| `contact` (half the wordlist borrowed) | 6.3 | 50% |
| `stratum`, `diffusion` | 23.0, 31.5 | 50% |

A high separation with a *small* fraction is a tail of sets that
do not belong. A high separation at about half is a corpus that
is two populations.

**It is not a borrowing test.** The two highest numbers in that
table are `stratum` and `diffusion`, where every set is cognate
and nothing was borrowed: half the words underwent a change and
half did not. What a high separation says is that the corpus is
not one thing, which is a reason to ask a different question and
not an answer to this one.

### 4.8 Transcription drift between sources

`rg_find_transcription_drift` asks whether two lects in one
corpus were transcribed by sources that disagree about where a
segment ends — one writing `tʃ` where the other writes `t ʃ`,
`tʰ` where the other writes `t h`, `aː` where the other writes
`a a`.

Nothing else catches this. Every grapheme involved is valid IPA,
the corpus loads, `regulae check` finds no unreadable grapheme,
and what training produces is a family of clean, well-supported
correspondences that read as deaffrication, loss of aspiration
and loss of vowel length. The shuffled baseline does not help and
cannot: it separates a pattern from chance, and this pattern is
perfectly systematic, which is what a sound law is.

Each row names the grapheme, what the other lect writes instead,
and two counts. `corroborated / forms` is the evidence: how often
the other lect actually writes the pieces where this one writes
the whole. A ratio near 1 is a transcription difference; a low
one is a sound change — Latin `kʷ` against French `k w` comes
out at 1 of 5, which is *qu* → /k/ and not a convention.

It reports and never refuses, and it is not a verdict. A corpus
can honestly hold one language with affricates and one without.
Only the person who assembled it can tell that from two sources
disagreeing; what this does is say where to look.

### 4.9 How a table is read

Every published table is handed out whole — the rows and how many,
borrowed and valid while the model that owns them lives:

```c
size_t n = 0;
const rg_multi_class_row *rows = rg_multi_model_conditioned_classes(model, &n);
for (size_t i = 0; i < n; i++) {
    printf("%d  %s\n", rows[i].class_id,
           rg_rule_standing_string(rows[i].evidence.standing));
}
```

The tables, and what each yields:

| Accessor | Row |
|---|---|
| `rg_multi_model_lects` | `const char *const *` |
| `rg_multi_model_unconditioned_classes` | `rg_multi_class_row` |
| `rg_multi_model_conditioned_classes` | `rg_multi_class_row` |
| `rg_multi_model_cross_dimensional_rows` | `rg_multi_cross_dimensional_row` |
| `rg_pairwise_model_segment_counts` | `rg_segment_count_row` |
| `rg_pairwise_model_conditioned_segment_counts` | `rg_conditioned_segment_count_row` |
| `rg_pairwise_model_chunks` | `rg_chunk_row` |
| `rg_pairwise_model_cross_dimensional_rows` | `rg_cross_dimensional_row` |
| `rg_pairwise_model_displacements` | `rg_displacement_row` |
| `rg_pairwise_model_tonal_counts` | `rg_tonal_count_row` |
| `rg_pairwise_model_gap_counts` | `rg_gap_count_row` |

Passing `NULL` for the count is allowed; passing a `NULL` model
yields a `NULL` table and a zero count.

`rg_pairwise_model_gap_counts` (ABI 32) is where a deletion or an
epenthesis has a row. The segment tables are one-to-one and cannot
say "this answers to nothing"; the gap table does, keyed by the
grapheme on the side that keeps it, with `deletion` for the
source-to-target loss direction and `count / present_total` the
rate it is dropped. It is a post-EM aggregation, not the scoring
model, so a consumer reads it exactly like the tonal table.

The multi-lect class table states losses too, from ABI 33: a lect
that dropped a segment the others keep appears in the class with
the grapheme `RG_GAP_GRAPHEME` (`"∅"`), so `{french:∅, latin:u,
…}` is French apocope and not a lect missing from the row. A
consumer distinguishing a deletion from an absent lect compares the
grapheme against `RG_GAP_GRAPHEME`. Gaps appear only in
unconditioned classes — a gap conditions nothing, so the
conditioning search never sees one — and `"∅"` is never a scoring
grapheme, so it is never passed to the feature system.

Until 2026-08-16 these were a count function and an index
function each, twenty-two of them, so a consumer wrote a loop
calling a function per row. If you want the old shape back it is
four lines over the new one — `tests/c/table_access.h` in this
repository is exactly that, and is the recommended way to keep
per-row assertions readable:

```c
static inline const rg_multi_class_row *conditioned_at(
    const rg_multi_model *m, size_t i) {
    size_t n = 0;
    const rg_multi_class_row *rows = rg_multi_model_conditioned_classes(m, &n);
    return i < n ? &rows[i] : 0;
}
```

**A table is sorted by a stable key**, so a consumer may
binary-search it. That is not the order the rules were decided
in, which is on each row's `evidence.decision_index` (§4.6).

## 5. The conditioning environment

A conditioning environment is `rg_context_spec` in C. In the JSON —
and so on the wrapper's `Segment.context` and a rule's
`.environment` — it is a plain object with a slot per axis, each
slot a list of `{"feature": ..., "value": ...}` constraints
(`"position"` is a bare string). The slots:

```
position               "initial" | "medial" | "final"
preceding, following   the immediate neighbours
self                   the segment the environment is about
morphological          "initial" | "final" | "internal" | "only"
morpheme_index         "0", "1", ... counted from the start
preceding_at_distance, following_at_distance   [offset, constraint]
somewhere_preceding, somewhere_following       anywhere on that side
same_syllable, next_syllable, previous_syllable
self_stress, preceding_stress, following_stress
```

A slot present with constraints is a **conjunction**: every one
must hold. `{"feature": "front", "value": "+"}` means the relevant
segment has `front` set to `+`. An empty or absent environment is
unconditioned. In C the same structure is walked field by field,
and `rg_context_spec_compare_internal` / `rg_context_spec_is_subset`
are the ordering and lookup primitives.

Three of those slots are newer than the rest of this document and
matter to a consumer.

`self` (C: `rg_context_spec.self`) constrains the segment the
environment is *about*, not a neighbour. A conditioned
correspondence rarely needs it — the segment is already the key —
but a cross-dimensional rule does, and §6 explains why.

`morphological` and `morpheme_index` are no longer reserved. They
are derived from the morpheme boundaries **the caller supplied on
the form**, and never from inference: regulae does not segment
words. A corpus with no boundaries gets no morphological axis at
all, so an absent value means "not asked", not "not conditioned".
`morpheme_index` does not travel between a suffixing language and
a prefixing one, where the same index is a different thing.

A non-empty field is a **conjunction**: every `FeatureConstraint`
in the list must hold for the context to apply.

To tell an unconditioned environment from a conditioned one, check
whether any slot carries a constraint — the total constraint count
across slots is zero for the empty environment, and rises with
specificity. Subset semantics (does a table-entry environment match
a link's environment — every constraint in the entry present in the
link) are the lookup primitive; in C that is
`rg_context_spec_is_subset`.

The long-range slots let the framework express rules like
"a → æ when the *next* syllable has a front vowel" without
requiring the intervening consonants to encode the
triggering feature themselves. They are populated by the
long-range discovery loop that runs after the
immediate-neighbour split stage.

Conditioned correspondences carry `context_is_target`, which says whose
environment the context describes — the source form's or the target's.
Conditioning is discovered from both sides, because a change is only visible
from the side that has the split, and a consumer that ignores the flag will
read half its rules against the wrong form.

## 6. Cross-dimensional rules and gaps

Two row shapes live under this section: a **cross-dimensional
rule**, where a feature conditions a suprasegmental dimension, and
a **gap**, where a segment answers to nothing.

```python
@dataclass(frozen=True)
class CrossDimensionalRule:            # C: rg_cross_dimensional_row
    environment:         dict          # a Context, not one feature at one position
    context_is_target:   bool          # which form the environment is read from
    dimension_from_environment: bool   # ABI 34: tone read from the env's own form
    environment_lect:    str
    conditioned_lect:    str           # == environment_lect when lect-internal
    dimension:           str           # "tone"|"length"|"stress"
    value:               str           # e.g. "4" for tone 4
    position_offset:     int           # signed offset from the link
    count:               float         # observed matches
    source_count:        float         # observations in the environment
    confidence:          float         # count / source_count
    contrast_count:      float         # matches outside the environment
    contrast_confidence: float         # contrast_count / contrast_source_count
    environment_alternatives: int      # ABI 35: rival conditioners the data cannot rule out
    uncertainty:         Uncertainty
    predictive:          PredictiveEvidence
```

**The environment is a whole `Context`, not a single feature at a
single position.** This document described it as an
`src_feature` / `src_position` pair until 2026-08-15, and that
shape could not state the rule the stage exists for. The Middle
Chinese register split conditions the target tone on the
preceding onset's voicing *and* on the source segment's own tone,
and neither predicate alone predicts it above chance: with one
predicate the rule was published at confidence 0.50 and read as a
weak finding rather than half of one. The environment is the same
type a conditioned correspondence carries, so it can name several
predicates, including one about the segment itself via `self`.

A conditioned split is a two-sided statement and both halves are
published: a `"-"` constraint names the complementary
environment, and a consumer that filters to `"+"` will read a
merger as a one-way change.

**`dimension_from_environment` (ABI 34) says whether the rule is
lect-internal.** 0 is the cross-lect rule — one lect's material
predicts the other lect's tone. 1 is tonogenesis proper: an onset
and the tone it conditions in the same language, read from one
form. With `context_is_target` this names the single lect both
sides are read from (the multi-lect JSON exposes it as
`conditioned_lect`, equal to `environment_lect` for a
lect-internal rule). The two are found independently, so a clean
tonogenesis where a lect kept its voicing publishes both readings;
a lect that merged its voicing keeps only the cross-lect one.
Lect-internal rules do not price the alignment — they describe one
lect's structure, not how two forms line up — so they never
perturb the segment tables.

**Do not read `confidence` on its own.** It is
P(value | environment), and a rule holding at 0.9 where the
contrast also holds at 0.9 is the ambient distribution rather
than a conditioning effect. The contrast fields are what make the
row a claim; the search charge `search_margin` on the C row scores
the environment as a whole and is what a rule must clear.

**`environment_alternatives` (ABI 35) says whether the named
conditioner is pinned.** A rule can predict the outcome perfectly
and still not identify its cause: if every voiced onset is also
before a front vowel, "voiced onset → low tone" and "front vowel →
low tone" fit the data equally, and the corpus cannot say which is
the conditioner. This field counts the distinct other features, at
a *different* segment position, that carve the rule's observations
the same way — 0 when the environment is uniquely identifiable, >0
when it is confounded and the stated environment is one of several
the data supports. It fires only on a perfect confound, never on a
partial correlation, and it says nothing about whether the split is
real (that is `search_margin` and the contrast fields) — only
whether its cause is pinned. The M7.2 analyst study
(`docs/m7_analyst_study.md`) found a confounded environment stated
at full confidence is the report's most anchoring line; a consumer
that surfaces `environment_alternatives > 0` beside `confidence` is
warning the reader the report cannot warn them of otherwise.

The rule reads: "where one form satisfies `environment`, the
other form's `dimension` carries `value` at `position_offset`
from the link." That is the tonogenesis signature — onset voicing
predicting tone on the following vowel, for example.

**`context_is_target` says which form the environment is read
from**, and the conditioned dimension is then on the other one.
Both computational orientations of a pair are searched, and they
are different questions: a lect that has merged the conditioning
contrast has nothing to state an environment over, and a lect
with no tone has nothing to condition. An association that holds
both ways is published once per lect, because which lect carries
the environment is part of the claim.

Neither orientation is a direction of change. On the lifted
multi-lect row, `source_lect` and `target_lect` name the order the
pair was trained in and nothing more; the lect the environment
sits in is `context_is_target ? target_lect : source_lect`.

These live on the per-pair model
(`rg_pairwise_model_cross_dimensional_rows`), because discovery
commits them per pair. The multi-lect model lifts every one of
them into a single table
(`rg_multi_model_cross_dimensional_rows`), and the lifted row *is*
the pairwise row plus the pair it was found in:

```c
typedef struct rg_multi_cross_dimensional_row {
    const char *source_lect;
    const char *target_lect;
    rg_cross_dimensional_row rule;   /* the whole pairwise row */
} rg_multi_cross_dimensional_row;
```

So a multi-lect cross-dimensional rule is read as
`row->rule.confidence`, `row->rule.evidence.standing`, and so on.
Until 2026-08-16 the two structs were separate declarations that
happened to share fifteen of seventeen fields in the same order,
and the lift copied them across one at a time.

**Because every pairwise rule is lifted, `rg_corpus_fit`'s
`rules_measured` counts each of them exactly once** — through the
multi-lect table. Counting the per-pair copies as well would count
them twice. The *conditioned correspondences* are the ones not
lifted, which is why they have their own pair of counts (§4.6).

### 6.1 Gaps: a segment answering to nothing

```python
@dataclass(frozen=True)
class GapCorrespondence:               # C: rg_gap_count_row
    grapheme:      str                 # the segment on the side that keeps it
    deletion:      bool                # True: a loss; False: an epenthesis
    count:         float               # times the gap link occurred
    present_total: float               # count / present_total is the drop rate
```

A 1-to-1 table cannot say "this answers to nothing", so a deletion
— the commonest sound change — had no row shape until ABI 32. The
gap table (`PairwiseModel.gaps`, `rg_pairwise_model_gap_counts`)
gives it one, per pair, keyed by the grapheme on the side that
kept it. On real Romance data it recovers French apocope as
`- ~ e`, `- ~ a`, `- ~ o` where Italian keeps the vowel.

At the multi-lect level (from ABI 33) a loss is a class row too: a
lect that dropped a segment the others keep appears in the class
with `GAP_GRAPHEME` (`"∅"`), so `{french:∅, latin:u, …}` is
apocope and not a lect missing from the row. Compare a segment's
grapheme against `GAP_GRAPHEME` to tell a deletion from a lect that
never had the word. Gaps appear only in unconditioned classes — a
gap conditions nothing — and `"∅"` is never handed to the feature
system. What still has no row is a non-one-to-one correspondence
that is not a pure gap (a 2-to-1 fusion); those are chunk-table
rows.

## 7. Public API stability contract

**The stable contract is the C ABI and the JSON it renders.** The
authoritative types are the `rg_*` structs and accessors in
`include/regulae.h`, marked with `RG_API`. The model JSON is the
same data as a string, and the Python wrapper's dataclasses are a
parse of it — so the schema they follow is the C library's, and a
newer core that adds a JSON field does not break an older wrapper
(unknown keys are ignored). The wrapper's own surface — `train_model`,
`segment`, `version`, `MultiLectModel` and the row dataclasses — is
small and stable, but it is a reader of the contract, not the
contract.

**In C the version is `RG_ABI_VERSION`.** Adding a field to a
public struct changes its layout, so it moves the version whether
or not it breaks a source-level consumer. The rule is:
`RG_ABI_VERSION` moves on any exported struct layout, enum,
signature or ownership change, and the reason is recorded in
`docs/c_conversion_roadmap.md`. It is at **36**.

What has landed, most recent first, as a guide to the kind of
break to expect. Every one of them fails a consumer at compile
time rather than silently, which is the intent.

- **39** — `rg_multi_class_row` gained `suprasegmentals`, an array parallel to
  `graphemes` of `rg_suprasegmentals` (`tone`, `length`, `stress`). A tone
  correspondence set is a correspondence set, and now gets a class row: the same
  graphemes under a different tone are two classes. Carried on an unconditioned
  class; NULL on a conditioned one, whose outcome is the segmental split (a
  suprasegmental under an environment is a cross-dimensional rule, §6). The JSON
  segment gains `tone`/`length`/`stress` only where non-empty, so a segmental
  corpus's model is byte-for-byte unchanged. Additive.
- **38** — `rg_bic_config` gained `class_outcome_mode`: how a multi-lect class
  split is charged for its outcome parameters. `PER_SISTER_LECT` (the default)
  prices the split per sister lect, so the charge is bounded by each lect's
  inventory rather than growing with the number of lects; `SISTER_TUPLE`
  reproduces the pre-M11 joint-tuple charge. Affects the multi-lect layer only.
- **36** — `rg_multi_class_row` gained `environment_alternatives`: the same
  identifiability flag, now on a conditioned segment class (§4.3). 0 when the
  environment is pinned, >0 when a different neighbour's feature carves the split
  the same way; always 0 on an unconditioned class. Additive.
- **35** — `rg_cross_dimensional_row` gained `environment_alternatives`: the
  count of rival conditioners at another position the corpus cannot tell the
  committed environment from (§6). 0 when identifiable, >0 when confounded.
  Additive; discovery is unchanged.
- **34** — `rg_cross_dimensional_row` gained `dimension_from_environment`:
  a cross-dimensional rule can now be lect-internal (an onset conditioning the
  tone in the same lect), not only cross-lect (§6). Lect-internal rules are kept
  out of the alignment scoring, so the class and cross-lect tables are unchanged.
- **33** — a multi-lect class states a deletion: a lect that dropped a segment
  the others keep appears with `RG_GAP_GRAPHEME` (`"∅"`), distinguishing a loss
  from a lect absent from the cognate (§6). A redistribution of the unconditioned
  table, not an addition — present-only classes that were losses in disguise
  become gap-bearing.
- **32** — `rg_gap_count_row` and `rg_pairwise_model_gap_counts`: a per-pair gap
  table gives a deletion or epenthesis a row (§6). A post-EM aggregation beside
  displacement and tone, not the scoring model.
- **31** — `rg_multi_class_row` gained `contrast_class_id` and
  `contrast_alternative_count`: a conditioned class links to the row holding the
  pivot's other reflex out of the environment — the comparison `contrast_count`
  cannot state, because it is ~0 when the conditioning is real (§4.3).
- **30** — `rg_multi_class_row.supporting_cognates` lists each cognate set once,
  so its length is a count of distinct sets rather than of aligned positions
  (§4.2). Conditioned classes now carry the list too, having published an empty
  one before.
- **29** — `rg_cross_dimensional_row` searches both computational orientations
  and merges equivalent findings; `rg_corpus_fit` gained lect and pair counts
  and near-duplicate-lect detection.
- **28** — `rg_train_options` gained opt-in grouped predictive-validation
  settings; `rg_rule_evidence` and `rg_corpus_fit` gained predictive evidence;
  `rg_predictive_status`, `rg_predictive_score`,
  `rg_predictive_evidence` and dependency-component observation units are
  public. Training with zero folds preserves the old execution path and
  publishes `unmeasured`.

- **27** — `rg_split_scorer` selects corrected BIC, exact multinomial NML or a
  symmetric-Dirichlet marginal likelihood; `rg_rule_evidence` names its scorer
  and authoritative `delta_score`; `rg_corpus_fit` reports the selected scorer
  removes the un-derived multi-lect small-sample addition, charges the full
  distinct-partition model space and uses a zero score threshold.
- **26** — `rg_cognate_set` gained optional `etymon_group` and
  `source_group`; TSV and wide loader options can name those columns;
  `rg_train_options.bootstrap_unit` selects cognate, etymon or source
  clustering; `rg_uncertainty_estimate` names its observation unit and
  effective count; `rg_rule_evidence` names the null supporting its verdict;
  and `rg_corpus_fit` reports the supplied and effective units.
  The loader owns its copies, while programmatic input remains borrowed for
  training.
- **25** — `rg_find_transcription_drift`,
  `rg_transcription_drift_rows_free`, `rg_transcription_drift_row`
  and `rg_drift_kind` are new (§4.8). Additive, and it moves the
  version because the surface is what the version names.

  Landing beside it, and *not* a compile-time break, is a change a
  consumer has to know about: a segment tuple may now appear in
  more than one conditioned class. Rows used to be merged on the
  tuple alone, which collapsed a decision list — a change
  conditioned by something that is not a natural class is several
  rules with one outcome — into a single row whose environment was
  whichever carried the most constraints. On the four-lect Romance
  corpus that discarded 32 of the 58 splits the search had
  committed. Code that indexes conditioned classes by tuple and
  expects at most one will now silently see only the first;
  `evidence.decision_index` is the order to read them in.
- **24** — `rg_corpus_fit` gained `pairwise_rules_above_noise` and
  `pairwise_rules_measured` (§4.6). Additive, but a struct layout
  change.
- **23** — the four fields saying what a search decided moved into
  one `evidence` member (§4.6); `rg_multi_cross_dimensional_row`
  became the pairwise row plus two lect names (§6); the
  twenty-two per-row table accessors became eleven that hand out
  a table whole (§4.9); `rg_format_multi_model_summary` and
  `rg_format_pairwise_tables` were added, so the CLI's two
  machine-readable renderings are library functions.
- **22** — public booleans became `bool` rather than `int`, and
  every loader now reports its own failure into a caller-supplied
  `rg_load_diagnosis` instead of a process-wide buffer that was
  neither thread-safe nor cleared between calls.

Nothing outside this repository links regulae today, which is why
23 could be as wide as it was. That will stop being true, and the
cost of a break will rise with it.

**Not public API**:

- Anything ending in `_internal`, and anything not declared in
  `include/regulae.h`. The private headers — `internal.h`,
  `environment.h`, `split_search.h`, `model_internal.h`,
  `search_internal.h`, `loader_internal.h`,
  `multilect_internal.h` — are implementation and change without
  notice. The discovery helpers, the per-link context builders and
  the class-level split committer live behind them.
- The specific numeric values of BIC thresholds, dominance
  floors, and similar tuning constants. These are tuned
  empirically and will move.
- The debug/inspection labelling of `rg_model_to_json`: it is a
  single maximum-a-posteriori snapshot, not the ensemble
  interchange format, and the wrapper reads exactly it. Fields may
  be added to the JSON; the wrapper ignores keys it does not know.

When a downstream layer like morphological alignment or the
historical-inference package starts introducing changes that
require breaking the contract, any such break will be
version-gated and announced in `framework/00_overview.md`.

## 8. Determinism and reproducibility invariants

These are **guaranteed**, and each names the thing that checks it.

- `rg_train_model` on the same corpus gives a bitwise-identical
  model, in the same process and across processes, given the same
  merkmal version. Anything iterating a set to produce output
  sorts first, and float sums over a set are taken in sorted key
  order.
- The same holds **across builds**: the native and WebAssembly
  builds must produce byte-identical JSON for the same corpus,
  which the `wasm_smoke` test asserts by training real corpora in
  both. This is what caught a comparator that was not a total
  order, where `qsort` returned equal-comparing rows in different
  orders on the two targets and moved every class id downstream.
- The same holds **across compilers**: the model every corpus in
  the tree trains to is hashed in `testdata/model_hashes.txt` and
  those hashes are identical under GCC and clang.
- Corpus reordering produces classes and pairwise models with the
  same content; only sort order may differ. Multi-lect `class_id`s
  are stable.
- Pairwise training is symmetric: training A against B and B
  against A produce mirror models, asserted in
  `tests/c/test_sound_laws.c`.

Breaking any of these is a bug. `scripts/model_hashes.py` is the
mechanism a consumer can borrow: it trains every corpus through
`regulae train --json` and compares a hash per corpus against a
committed baseline, and `scripts/check.sh` fails when any moves.
That is what makes "this change was a refactor" checkable rather
than asserted — a passing test suite is compatible with a great
many changed models.

Historical inference can rely on all of this for reproducible
posterior computation.

## 9. What the historical-inference layer should consume

`07_historical_inference.md` §6 names four kinds of input from
regulae. In API terms:

| historical-inference input | regulae output |
|---|---|
| multi-lect correspondence classes as likelihood constraints | `MultiLectModel.unconditioned_classes` + `.conditioned_classes` |
| evidence for structured sound change (regularity) | a `CorrespondenceClass` with `.conditioned` true |
| evidence for cluster-level change | the chunk table on a `rg_pairwise_model` (`.raw`) |
| loss as evidence | a class carrying `GAP_GRAPHEME`, and `PairwiseModel.gaps` |
| provenance / exemplar walk-back | `CorrespondenceClass.supporting_cognates` (the cognate ids) |

A natural consumption sketch:

```python
import regulae

# Step 1. Train the alignment layer (this package, today).
align_model = regulae.train_model("walworthpolynesian.tsv", fmt="arcaverborum")

# Step 2. Build the likelihood the historical-inference layer
#         uses. At minimum, each multi-lect class becomes a
#         constraint: "the history H must admit this segment
#         tuple across these lects with approximately this
#         weight".
from hist_inference import likelihood_from_align_model  # future
likelihood = likelihood_from_align_model(align_model)

# Step 3. Combine with a prior over histories and run posterior
#         inference. Outputs: posterior samples of histories,
#         marginal phylogenies, reconstructed proto-forms, etc.
from hist_inference import TypologicalPrior, infer_history
prior = TypologicalPrior(family="Austronesian")
posterior = infer_history(likelihood, prior)

# Step 4. Derived queries are marginals over the posterior.
tree = posterior.marginal_tree()
proto = posterior.reconstruct_at(node=tree.root)
```

None of the `hist_inference.*` names exist yet. They're a
sketch of what the **separate package** will import and what
shape its entry points should have. The design spec in
`07_historical_inference.md` is the authoritative source for
that package's semantics.

## 10. What historical inference should NOT do at the regulae
boundary

Things that belong inside regulae (and not historical inference):

- **Re-doing alignment.** If the historical-inference package
  wants to refine alignments given a hypothesised topology
  (an iterated-alignment loop that feeds committed rules
  back into the DP transition costs), that belongs in
  regulae, not in the consuming layer. Historical inference
  should treat `MultiLectModel` as immutable observed
  evidence.
- **Feature-system changes.** The feature system is fixed by the
  regulae training run. Historical inference doesn't re-project
  segments into a different feature system; if you need one,
  retrain regulae. The default is merkmal's `distinctive`; models
  trained under different systems are not comparable.
- **Cognacy filtering.** If a cognate set looks suspect after
  historical inference runs, the workflow is: flag it, re-read the
  outlier ranking on the regulae model (the JSON's `outliers`, or
  `regulae outliers`), decide by hand what to drop, retrain
  regulae. Historical inference should not silently discard or
  down-weight cognates based on its own fit.

Things that belong in historical inference (and not regulae):

- **Directionality** — who inherited from whom, who borrowed
  from whom.
- **Time** — dates, event rates, calibration.
- **Reconstruction** — building explicit proto-forms at
  internal nodes of a topology.
- **Phylogeny / topology** — building trees, networks, dialect
  continua.
- **Relatedness** as a posterior query (see
  `04_relatedness.md`).

## 11. Worked example: read a trained model

```python
import regulae

model = regulae.train_model("walworthpolynesian.tsv", fmt="arcaverborum")

# Walk the strongest classes.
for klass in model.conditioned_classes[:5]:
    print(f"#{klass.id}  count={klass.count}  conf={klass.confidence:.2f}"
          f"  on {len(klass.supporting_cognates)} sets")
    for seg in klass.segments:
        tag = f" / {seg.context}" if seg.context else ""
        gap = "  (deleted)" if seg.is_gap else ""
        print(f"    {seg.lect}: {seg.grapheme}{tag}{gap}")
    if klass.contrast_class_id >= 0:
        other = next(c for c in model.classes if c.id == klass.contrast_class_id)
        print(f"    vs elsewhere: {other.graphemes}")

# Deletions, per pair.
for pair in model.pairwise_models:
    for gap in pair.gaps:
        if gap.deletion:
            print(f"{pair.source_lect}: {gap.grapheme} lost "
                  f"{gap.count:g}/{gap.present_total:g}")

# Lect-internal tonogenesis, if any.
for rule in model.cross_dimensional:
    if rule.dimension_from_environment:
        print(f"{rule.environment_lect}: own onset -> {rule.dimension} {rule.value}")

# Whether the corpus has signal at all (§4.6), needs permutation_count > 0.
print("cost/segment:", model.fit["cost_per_segment"])
```

### 11.1 The same thing in C

The wrapper reads the JSON the C core renders. A C consumer works
against the model directly; here is what that looks like. It
compiles clean under the warning set `scripts/check.sh` enforces.

```c
#include "regulae.h"
#include <stdio.h>

int main(void) {
    rg_context *ctx = 0;
    rg_corpus *corpus = 0;
    rg_multi_model *model = 0;
    rg_train_options options;
    rg_load_diagnosis diagnosis;
    const rg_multi_class_row *classes;
    const rg_corpus_fit *fit;
    size_t count = 0;
    size_t i;
    char *text;

    if (rg_context_new_builtin(&ctx) != RG_OK) {
        return 1;
    }
    if (rg_corpus_load_tsv("corpus.tsv", 0, &corpus, &diagnosis) != RG_OK) {
        fprintf(stderr, "line %lu: %s\n", (unsigned long)diagnosis.line, diagnosis.message);
        rg_context_free(ctx);
        return 1;
    }

    rg_train_options_init_defaults(&options);
    /* Without this every `standing` is RG_RULE_STANDING_UNMEASURED, which is
     * not a pass -- it says the comparison was never made. Costs one training
     * run per permutation. */
    options.permutation_count = 30;
    if (rg_train_model(ctx, rg_corpus_cognates(corpus),
                       rg_corpus_cognate_count(corpus), &options, &model) != RG_OK) {
        const char *grapheme = 0;
        const char *system = 0;
        rg_context_last_error(ctx, &grapheme, &system);
        fprintf(stderr, "training: %s in %s\n", grapheme ? grapheme : "?", system);
        rg_corpus_free(corpus);
        rg_context_free(ctx);
        return 1;
    }

    /* Does the corpus have signal at all? Class counts do not answer this and
     * move the wrong way; cost_per_segment does. */
    fit = rg_multi_model_fit(model);
    printf("cost/segment %.4f   %lu of %lu rules stand, %lu of %lu per-pair\n",
           fit->cost_per_segment,
           (unsigned long)fit->rules_above_noise, (unsigned long)fit->rules_measured,
           (unsigned long)fit->pairwise_rules_above_noise,
           (unsigned long)fit->pairwise_rules_measured);

    /* A table is handed out whole. It is sorted by key, not by the order the
     * rules were decided in -- that is on each row's evidence. */
    classes = rg_multi_model_conditioned_classes(model, &count);
    for (i = 0; i < count; i++) {
        const rg_multi_class_row *row = &classes[i];
        size_t j;
        printf("#%d  count=%.1f  %s\n", row->class_id, row->count,
               rg_rule_standing_string(row->evidence.standing));
        for (j = 0; j < row->segment_count; j++) {
            printf("    %s: %s\n", row->lect_ids[j], row->graphemes[j]);
        }
    }

    text = rg_format_multi_model(model, 0);   /* human-readable */
    if (text != 0) {
        fputs(text, stdout);
        rg_string_free(text);
    }

    rg_multi_model_free(model);
    rg_corpus_free(corpus);
    rg_context_free(ctx);
    return 0;
}
```

Four things in it are the parts consumers get wrong:

- **`permutation_count` is not set by default**, so `standing`
  reads `unmeasured` on every rule and the fit's `null_` fields are
  zero. `unmeasured` is not a pass.
- **`cost_per_segment`, not class counts**, answers "does this
  corpus have signal" — counts move the wrong way (§4.6).
- **`rg_context_last_error`** names the grapheme a refusal was
  about; the status code alone is unactionable on a real corpus.
- **Everything borrowed is valid only while its owner lives.** The
  rows point into the model, the graphemes into the rows. Free the
  model and they are gone. Only `char *` returned by `rg_format_*`
  and `rg_model_to_json` is caller-owned, and it is freed with
  `rg_string_free`.

The `rg_format_*` and `rg_describe_*` helpers return strings for
human debugging. For machine consumption use `rg_model_to_json`, or
walk the tables directly as above — the CLI's own machine-readable
summary is `rg_format_multi_model_summary`, so a consumer wanting
exactly that output can call it rather than parse the CLI.

## 12. Where to look next

- `framework/00_overview.md` — the whole framework's architecture.
- `framework/03_alignment.md`, `docs/alignment_details.md` — conceptual
  overview of the alignment layer.
- `docs/training_pipeline.md` — staged training rationale.
- `docs/correspondence_discovery.md` — discovery mechanisms.
- `framework/07_historical_inference.md` — the design spec for the
  downstream package. Read alongside this document.
- `framework/04_relatedness.md` — relatedness as a marginal over the
  historical-inference posterior, decomposed into six
  competing causal processes. The "why does regulae stop
  where it stops" argument.
- `README.md` — install instructions, test commands,
  a quick API tour, the list of shipped experiments.
- `experiments/<name>/findings.md` — what each experiment found on
  real data. Written for framework developers, not downstream
  consumers, but they are the closest thing to worked results.
- `docs/architecture_plan.md` — why the modules are shaped the way
  they are, and which decisions were deliberately left open. Read
  it before proposing a change to the published shape.
