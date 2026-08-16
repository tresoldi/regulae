# The regulae package: a consumer's guide

**Status: DOCUMENTATION.** This document describes the public
API from the perspective of a downstream package that wants to
*consume* its output — historical inference, visualization,
cross-family comparison, whatever.

**On names.** The contract below is written in the original
Python spelling (`train_model`, `MultiLectModel`,
`snake_case` fields). The implementation is a C99 core, so the
authoritative spelling of any name is `include/regulae.h`:
`train_model` is `rg_train_model`, `MultiLectModel` is the
opaque `rg_multi_model` read through `rg_multi_model_*`
accessors, `find_cognate_outliers` is
`rg_find_cognate_outliers`, and the loaders are
`rg_corpus_load_*`. What each field *means*, and what it does not
mean, is the point of this document.

This document said until 2026-08-15 that the meanings were
"unchanged", and it was wrong in four places: a cross-dimensional
rule's environment is a whole `Context` rather than one feature at
one position (§6), the model does not retain the input corpus
(§4.4), `Context` gained the segment-itself and morphological
slots (§5), and there is no Python package to install (§2). The
reliability fields — `standing`, `search_margin`,
`decision_index`, the contrast counts, `rg_corpus_fit` — were
never described here at all, and they are what decides whether an
inference built on this output is defensible. §4.6 is that
section. **Where this document and `include/regulae.h` disagree,
the header is right.**

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
  learned counts. The only post-hoc recourse is
  `find_cognate_outliers`, a diagnostic that z-scores cognate
  sets by alignment cost so a human reviewer can flag suspects.
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
  resampled over whole **cognate sets**, which is the unit the
  corpus actually samples, and carries the confidence weighting
  with it. Measured, the resampled interval is *narrower* on
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
  environment does not hold, and `delta_bic` is what the split
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
- **No proto-forms.** A `MultiLectCorrespondenceClass` names
  which segment appears in each participating lect; it does not
  reconstruct what "the proto-segment" was, because
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

**There is no installable Python package today.** The original
implementation is archived under `python/` and is not built,
tested or supported; the Python wrapper over the C core is M6 on
the roadmap and has not been written. A consumer today links the
C library:

```sh
cmake -S . -B build && cmake --build build
# libregulae.a (or .so), include/regulae.h, and merkmal alongside it
```

`include/regulae.h` is the contract, and `README.md` has a tour
of it. The Python spelling below describes the *shape* of what
that API returns, which is what the rest of this document is
about; the archived listing is kept because it reads better than
a list of accessor names, not because it can be imported.

The archived package exposed its public API on `regulae` at the
top level:

```python
from regulae import (
    CognateSet, Form, Segment,       # inputs
    train_model,                     # the canonical entry point
    cognate_sets_from_pairs,         # pair → multi-lect migration helper
    MultiLectModel, LearnedModel,    # outputs
    MultiLectCorrespondenceClass,
    ConditionedCorrespondence,
    CrossDimensionalLink,
    Context, FeatureConstraint,      # conditioning
    load_gled, load_arcaverborum,    # loaders
    find_cognate_outliers,           # diagnostic
)
```

That list was in `python/src/regulae/__init__.py` under
`__all__`. For the C API the equivalent boundary is `RG_API` in
`include/regulae.h`: anything not marked with it is not exported,
and the `*_internal` symbols that carry external linkage for the
library's own use are hidden by the visibility preset.

## 3. The canonical entry point

```python
from regulae import train_model, cognate_sets_from_pairs, CognateSet

# Input: a list of CognateSets.
corpus: list[CognateSet] = ...

# Canonical call. Returns a MultiLectModel for N ≥ 1 lects.
model = train_model(corpus)
```

Two ways to build the `corpus`:

1. **Loaders** — `load_gled(path, family="Indo-European",
   doculects={"LATIN", "SPANISH"})` or
   `load_arcaverborum(path, dataset="walworthpolynesian",
   language_ids={...})` return a `list[CognateSet]` directly.
2. **Hand-built** — construct `CognateSet` objects with
   `cognate_id` and a `forms` mapping `lect_id -> Form`.
3. **Legacy pair helper** — if you have a
   `list[tuple[Form, Form]]` pair corpus,
   `cognate_sets_from_pairs(pairs, ("lect_a", "lect_b"))`
   converts it into the canonical form.

`train_model` is **deterministic**: identical inputs produce
bitwise-identical outputs, cross-process. This is a guaranteed
invariant (covered by `test_invariants.py`). All internal
discovery loops are sequential-greedy with deterministic
tiebreakers, vocabulary iteration is sorted, and the merkmal
dependency is configured to avoid cross-process drift.

## 4. The output shape: `MultiLectModel`

`MultiLectModel` is a frozen dataclass with five fields:

```python
@dataclass(frozen=True)
class MultiLectModel:
    pairwise_models:       Mapping[frozenset[str], LearnedModel]
    unconditioned_classes: tuple[MultiLectCorrespondenceClass, ...]
    conditioned_classes:   tuple[MultiLectCorrespondenceClass, ...]
    cognate_corpus:        tuple[CognateSet, ...]
    lect_ids:              tuple[str, ...]
```

### 4.1 `pairwise_models`

A mapping keyed by `frozenset({lect_a, lect_b})` — explicitly
*not* an ordered tuple, because pairwise training is symmetric
(see §1). A pair missing from the mapping means the two lects
had no shared cognate data in the corpus.

In C this is `rg_multi_model_pair_model_at`, which returns an
`rg_multi_pair_model_row` carrying `lect_a`, `lect_b` and the
model. The row *is* ordered — reconciliation walks lect pairs in
ascending lect-id order, which fixes the direction each pair is
aligned in — but the analysis it holds is not: training A against
B and B against A produce mirror models, and the tests assert it.
Read the pair as a labelled edge, not as a direction of change.

Each value is a `LearnedModel` — the full per-pair training
output for that lect pair. It carries:

- `segment_table`: a `SegmentCorrespondenceTable` of
  `ConditionedCorrespondence → count` entries. Each
  correspondence binds one source grapheme to one target
  grapheme under one `Context`. The context may be empty
  (unconditioned) or carry immediate-neighbour, long-range, or
  syllable-structural constraints (see §5).
- `displacement_dist`: a `DisplacementDistribution` over
  `tuple[FeatureDisplacement, ...]` — the feature-space
  signature of how segments changed, aggregated across all 1-to-1
  links. This is what lets natural classes be discovered across
  different grapheme pairs (e.g. the same displacement vector
  produces `p~f`, `t~θ`, `k~x`).
- `chunk_table`: a `ChunkPhraseTable` of promoted chunk
  correspondences — multi-segment → multi-segment patterns the
  EM loop decided were worth memorising (e.g. Latin `kt → tʃ`
  in Italian).
- `tonal_table`: a `TonalCorrespondenceTable` mapping source tone
  to target tone with counts, for pairs where tones are present.
- `cross_dimensional_table`: a `CrossDimensionalLinkTable` of
  committed cross-dimensional rules where a segmental
  feature at one position predicts a tonal value at another
  (the tonogenesis pattern). Empty by default.

For most downstream consumers the pairwise models are
**low-level, detail-heavy**. The recommended consumption pattern
is to read `conditioned_classes` and `unconditioned_classes`
first (§4.2, §4.3), and only descend into `pairwise_models` when
you need a specific lect pair's full table.

### 4.2 `unconditioned_classes`

A tuple of `MultiLectCorrespondenceClass`, sorted by count
descending, then by segment tuple. Each class says:

> "Across the corpus, when these N lects have segments at
> corresponding positions within a cognate, this is one of the
> segment tuples that appears, and it happened *count* times."

The class is the **unit of downstream evidence**. It is the
strongest summary regulae produces: it's consolidated across
pairwise training via union-find reconciliation, it has a
count, and every participating lect contributes a segment.

```python
for klass in model.unconditioned_classes:
    # A frozen mapping lect_id -> grapheme.
    print(klass.segments)     # e.g. {"LATIN": "k", "SPANISH": "tʃ"}
    print(klass.count)        # e.g. 8.0
    print(klass.confidence)   # 1.0 for unconditioned classes
```

### 4.3 `conditioned_classes`

Same shape as unconditioned, but with a non-None `contexts`
field: a per-lect mapping `lect_id -> Context` describing the
conditioning environment under which that lect's segment
appears. A lect without a specific constraint has the empty
`Context()`.

Reading a conditioned class:

```python
for klass in model.conditioned_classes:
    print(klass.segments)
    print(klass.count)
    print(klass.confidence)   # coverage-based strength in [0, 1]
    for lect, ctx in klass.contexts.items():
        if ctx.constraint_count() > 0:
            # This lect has a specific conditioning environment.
            ...
```

The `confidence` field on conditioned classes is the
**pivot-bucket coverage**: `count / pivot_bucket_size` where
`pivot_bucket_size` is the number of observations of the pivot
lect+grapheme that the class-discovery loop saw. A value of 1.0 means "every
observation of the pivot is in this class" (a strong rule); 0.1
means "only 10% of the pivot's observations landed here" (weak
minority). Use this to sort or filter classes by signal
strength when your downstream stage is sensitive to marginal
evidence.

### 4.4 `cognate_corpus`

**Not on the C model.** The Python model retained the input
corpus for provenance; `rg_multi_model` does not, and there is no
accessor for it. The corpus is a handle the caller already owns
(`rg_corpus`, from one of the loaders), and it has to outlive the
training call anyway, because the model borrows from it. Walking
back from a class to the source forms means keeping that handle,
not asking the model for it.

`rg_multi_class_row.supporting_cognates` is the anchor that does
exist: the cognate ids a class was built from, so a consumer can
point at the evidence without matching graphemes back by hand.

### 4.5 `lect_ids`

The canonical ordered list of lect IDs observed in the corpus.
First-seen order — **not** alphabetical. This ordering is stable
and deterministic; downstream packages can rely on it as an
index.

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

Every conditioned rule and every conditioned class carries four
fields that belong together:

- `standing` — the verdict, `above-noise`, `within-noise`, or
  `unmeasured` when no baseline was run. **`unmeasured` is not a
  pass.** It says the comparison was never made.
- `search_margin` — the number that verdict is computed from,
  against `rg_corpus_fit.null_search_margin`.
- `decision_index` — discovery is greedy, so the rules form a
  decision list: a later rule refines what an earlier one left
  unsettled. Published tables are sorted by key so lookups can
  binary-search them, which destroys that order; this preserves
  it. `-1` means the row was not decided by a search.
- `contrast_count` and `delta_bic` — a conditioning claim is a
  comparison, and these are the other side of it. A rule
  published without the contrast it was measured against cannot
  be read.

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

## 5. The conditioning environment: `Context`

```python
@dataclass(frozen=True)
class Context:                                # C: rg_context_spec
    position:             str | None = None   # "initial"|"medial"|"final"
    preceding:            tuple[FeatureConstraint, ...] = ()
    following:            tuple[FeatureConstraint, ...] = ()
    self_:                tuple[FeatureConstraint, ...] = ()
    # Morphological, from boundaries the caller supplied:
    morphological:        str | None = None   # "initial"|"final"|"internal"|"only"
    morpheme_index:       str | None = None   # "0", "1", ... counted from the start
    # Long-range fields:
    preceding_at_distance: tuple[tuple[int, FeatureConstraint], ...] = ()
    following_at_distance: tuple[tuple[int, FeatureConstraint], ...] = ()
    somewhere_preceding:   tuple[FeatureConstraint, ...] = ()
    somewhere_following:   tuple[FeatureConstraint, ...] = ()
    same_syllable:         tuple[FeatureConstraint, ...] = ()
    next_syllable:         tuple[FeatureConstraint, ...] = ()
    previous_syllable:     tuple[FeatureConstraint, ...] = ()
    # Stress, which is a dimension rather than a feature of a neighbour:
    self_stress:           tuple[FeatureConstraint, ...] = ()
    preceding_stress:      tuple[FeatureConstraint, ...] = ()
    following_stress:      tuple[FeatureConstraint, ...] = ()
```

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
in the tuple must hold for the context to apply. A
`FeatureConstraint("front", "+")` means "the relevant segment
has the feature `front` set to `+`".

Subset semantics are provided by `Context.is_subset_of`: a table
entry context matches a link context iff every constraint in the
table entry is present in the link.

Two methods you'll probably want:

- `ctx.constraint_count()` → total number of constraints across
  all slots. Useful for "is this an unconditioned context?"
  (count == 0) or sorting by specificity.
- `ctx.is_subset_of(other_ctx)` → does this context match the
  other? (You probably won't need this directly; it's the
  internal lookup primitive.)

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

## 6. Cross-dimensional rules: `CrossDimensionalLink`

```python
@dataclass(frozen=True)
class CrossDimensionalLink:            # C: rg_cross_dimensional_row
    source_environment:  Context       # not one feature at one position
    tgt_dimension:       str           # "tone"|"length"|"stress"
    tgt_value:           str           # e.g. "4" for tone 4
    tgt_position_offset: int           # signed offset from the link
    count:               float         # observed matches
    src_count:           float         # observations in the environment
    confidence:          float         # count / src_count
    contrast_count:      float         # matches outside the environment
    contrast_src_count:  float         # observations outside it
    contrast_confidence: float         # contrast_count / contrast_src_count
    delta_bic:           float         # score for the environment
    decision_index:      int           # where in the decision list (§4.6)
    search_margin:       float         # search charge the evidence carries
    standing:            str           # the verdict (§4.6)
    uncertainty:         UncertaintyEstimate
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

**Do not read `confidence` on its own.** It is
P(value | environment), and a rule holding at 0.9 where the
contrast also holds at 0.9 is the ambient distribution rather
than a conditioning effect. The contrast fields are what make the
row a claim; `delta_bic` scores the environment as a whole and is
negative for every published row.

The rule reads: "where the source form satisfies
`source_environment`, the target form's `tgt_dimension` carries
`tgt_value` at `tgt_position_offset` from the link." That is the
tonogenesis signature — onset voicing predicting tone on the
following vowel, for example.

These live on the per-pair model
(`rg_pairwise_model_cross_dimensional_row_at`), because discovery
commits them per pair. The multi-lect model lifts them with
explicit `source_lect` / `target_lect` labels
(`rg_multi_model_cross_dimensional_row_at`), and the lifted row
carries the same fields.

## 7. Public API stability contract

The following types are **public API** and their field
layout is stable:

- `CognateSet`, `Form`, `Segment`, `Lect`
- `MultiLectModel`, `LearnedModel`
- `MultiLectCorrespondenceClass`, `ConditionedCorrespondence`
- `Context`, `FeatureConstraint`, `FeatureDisplacement`
- `CrossDimensionalLink`
- `train_model`, `align_forms`, `find_cognate_outliers`
- `load_gled`, `load_arcaverborum`, `load_cognates_from_tsv`,
  `cognate_sets_from_pairs`

**New fields may be added** to the frozen dataclasses as
new capabilities land — this is how
`LearnedModel.cross_dimensional_table` and the long-range
slots on `Context` were added without breaking prior
consumers. Existing fields will not be renamed, removed, or
have their semantics changed.

**In C, that promise is `RG_ABI_VERSION`, and it is weaker.**
Adding a field to a public struct changes its layout, so it moves
the ABI version whether or not it breaks a source-level consumer.
The rule is: `RG_ABI_VERSION` moves on any exported struct
layout, enum, signature or ownership change, and the reason is
recorded in `docs/c_conversion_roadmap.md`. It is at **22**.

Two changes landed at 22, and are the kind to expect: public
booleans became `bool` rather than `int`, and every loader now
reports its own failure into a caller-supplied
`rg_load_diagnosis` instead of a process-wide buffer that was
neither thread-safe nor cleared between calls. Both break a
consumer at compile time rather than silently, which is the
intent.

**Not public API**:

- Anything starting with `_` in any module. Names like the
  internal discovery helpers, the per-link context builder,
  the class-level split committer — these are implementation
  and will change without notice.
- The specific numeric values of BIC thresholds, dominance
  floors, and similar tuning constants. These are tuned
  empirically and will move.
- Intermediate outputs like `PatternHypothesis` from
  `find_residual_patterns`. The diagnostic itself is public
  but its internal candidate enumeration can change.

When a downstream layer like morphological alignment or the
historical-inference package starts introducing changes that
require breaking the contract, any such break will be
version-gated and announced in `framework/00_overview.md`.

## 8. Determinism and reproducibility invariants

These are **guaranteed** and covered by `test_invariants.py`:

- `train_model(corpus) == train_model(corpus)` bitwise, in the
  same process.
- `train_model(corpus) == train_model(corpus)` bitwise, across
  separate Python processes, given the same merkmal version.
- Corpus reordering (shuffling `list[CognateSet]`) produces
  classes and pairwise models with the same content; only sort
  order and internal provenance tuples may differ. Multi-lect
  class `class_id`s are stable.
- Per-lect first-seen ordering for `cognate_sets_from_pairs`:
  `cognate_sets_from_pairs(pairs, ("A", "B"))` trains the
  pair model in the A→B direction, regardless of which lect
  appeared first in `pairs`.

Breaking any of these is considered a bug. Historical inference
can rely on them for reproducible posterior computation.

## 9. What the historical-inference layer should consume

`07_historical_inference.md` §6 names four kinds of input from
regulae. In API terms:

| historical-inference input | regulae output |
|---|---|
| multi-lect correspondence classes as likelihood constraints | `MultiLectModel.unconditioned_classes` + `.conditioned_classes` |
| evidence for structured sound change (regularity) | `MultiLectCorrespondenceClass.contexts` non-None |
| evidence for cluster-level change | `LearnedModel.chunk_table` entries in `pairwise_models` |
| provenance / exemplar walk-back | `MultiLectModel.cognate_corpus` + `MultiLectCorrespondenceClass.supporting_cognates` |

A natural consumption sketch:

```python
from regulae import train_model, load_arcaverborum

# Step 1. Train the alignment layer (this package, today).
corpus = load_arcaverborum("...", dataset="walworthpolynesian", language_ids=...)
align_model = train_model(corpus)

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
- **Feature-system changes.** The `feature_system` string on
  `LearnedModel` is fixed by the regulae training run. Historical
  inference doesn't re-project segments into a different feature
  system; if you need one, retrain regulae. The default is
  merkmal's `distinctive`; every model records which system it was
  trained under, and models trained under different systems are not
  comparable.
- **Cognacy filtering.** If a cognate set looks suspect after
  historical inference runs, the workflow is: flag it, re-run
  `find_cognate_outliers` on the regulae model, decide by hand
  what to drop, retrain regulae. Historical inference should
  not silently discard or down-weight cognates based on its
  own fit.

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
from regulae import (
    train_model, load_arcaverborum, format_multi_lect_model,
    describe_multi_lect_class,
)

corpus = load_arcaverborum(
    "/path/to/arcaverborum/cldf.csv",
    dataset="walworthpolynesian",
    language_ids={"walworthpolynesian_Hawaiian",
                  "walworthpolynesian_Samoan",
                  "walworthpolynesian_Tongan"},
)
model = train_model(corpus)

# Headline: print a readable summary.
print(format_multi_lect_model(model, top_conditioned=10))

# Walk the classes programmatically.
for klass in model.conditioned_classes[:5]:
    print(f"#{klass.class_id}  count={klass.count}  conf={klass.confidence:.2f}")
    for lect, grapheme in sorted(klass.segments.items()):
        ctx = klass.contexts[lect] if klass.contexts else None
        tag = f" / {ctx}" if ctx and ctx.constraint_count() > 0 else ""
        print(f"    {lect}: {grapheme}{tag}")

# Drill into a specific lect/grapheme for everything regulae
# committed about it (unconditioned + conditioned + evidence).
print(describe_multi_lect_class(model, "walworthpolynesian_Hawaiian", "ʔ"))

# Flag cognate sets that look suspect under the current model.
for report in find_cognate_outliers(corpus, model, top_k=10):
    print(f"z={report.z_score:+.2f}  {report.cognate_id}")
```

The public `format_*` and `describe_*` helpers are designed for
human debugging, not for machine consumption — they return
strings. Downstream packages should walk the frozen dataclasses
directly.

## 12. Where to look next

- `framework/00_overview.md` — the whole framework's architecture.
- the `framework/03_alignment.md` doc, `docs/alignment_details.md` — conceptual
  overview of the alignment layer.
- `docs/training_pipeline.md` — staged training rationale.
- `docs/correspondence_discovery.md` — discovery mechanisms.
- `07_historical_inference.md` — the design spec for the
  downstream package. Read alongside this document.
- `04_relatedness.md` — relatedness as a marginal over the
  historical-inference posterior, decomposed into six
  competing causal processes. The "why does regulae stop
  where it stops" argument.
- `README.md` — install instructions, test commands,
  a quick API tour, the list of shipped experiments.
- `experiments/SUMMARY*.md` — development-history notes
  tracking what each implementation round changed on real
  data. These are written for framework developers, not
  downstream consumers.
