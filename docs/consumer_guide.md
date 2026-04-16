# The regulae package: a consumer's guide

**Status: DOCUMENTATION.** This document describes the public
API of `src/regulae/` from the perspective of a downstream
package that wants to *consume* its output — historical
inference, visualization, cross-family comparison, whatever.

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
  ancestor. A `LearnedModel` trained on (Latin, Spanish) and one
  trained on (Spanish, Latin) produce symmetric tables up to
  relabeling.
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
- **No morphology.** The alignment is purely phonological.
  Morpheme boundaries are captured from loaders if present in
  the source data, stored on `CognateSet.morpheme_boundaries`,
  and currently not consumed (reserved for a future
  morph-aware extension).

These are all **design choices**, not gaps. Each of them is a
presupposition that regulae refuses to smuggle in; they belong in
the layer that consumes regulae's output, where they can be
modelled explicitly as priors or posteriors over separate
variables.

## 2. Installation and import

```sh
pip install -e path/to/src/regulae
pip install -e path/to/merkmal  # required dependency
```

The package exposes its public API on `regulae` at the top level:

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

The complete export list is in `src/regulae/src/regulae/__init__.py`
under `__all__`. Names not in that list (internal helpers,
underscore-prefixed functions) are **not** public API — they can
change between releases without notice.

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

The input corpus retained on the model for provenance. If you
want to walk back from a class to the source forms (e.g. to
display a few exemplar words, or to recompute statistics), this
is the anchor.

### 4.5 `lect_ids`

The canonical ordered list of lect IDs observed in the corpus.
First-seen order — **not** alphabetical. This ordering is stable
and deterministic; downstream packages can rely on it as an
index.

## 5. The conditioning environment: `Context`

```python
@dataclass(frozen=True)
class Context:
    position:             str | None = None   # "initial"|"medial"|"final"
    preceding:            tuple[FeatureConstraint, ...] = ()
    following:            tuple[FeatureConstraint, ...] = ()
    morphological:        str | None = None   # reserved
    # Long-range fields:
    preceding_at_distance: tuple[tuple[int, FeatureConstraint], ...] = ()
    following_at_distance: tuple[tuple[int, FeatureConstraint], ...] = ()
    somewhere_preceding:   tuple[FeatureConstraint, ...] = ()
    somewhere_following:   tuple[FeatureConstraint, ...] = ()
    same_syllable:         tuple[FeatureConstraint, ...] = ()
    next_syllable:         tuple[FeatureConstraint, ...] = ()
    previous_syllable:     tuple[FeatureConstraint, ...] = ()
```

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

## 6. Cross-dimensional rules: `CrossDimensionalLink`

```python
@dataclass(frozen=True)
class CrossDimensionalLink:
    src_feature:         FeatureConstraint
    src_position:        str                  # "relative_-1"|"relative_0"|...
    tgt_dimension:       str                  # "tone"|"length"|"stress"
    tgt_value:           str                  # e.g. "4" for tone 4
    tgt_position_offset: int                  # signed offset from link
    count:               float                # observed matches
    src_count:           float                # observations where src feature held
    confidence:          float                # count / src_count
```

Lives on `LearnedModel.cross_dimensional_table.entries`, not on
multi-lect classes — the cross-dimensional discovery loop
commits rules per-pair only. The multi-lect model surfaces
these via its own lifted
`cross_dimensional_table: MultiLectCrossDimensionalLinkTable`,
which carries the same rules with explicit `src_lect` /
`tgt_lect` labels.

The rule reads "when the source form has `src_feature` at the
position given by `src_position` (relative to a link), the
target form's `tgt_dimension` carries `tgt_value` at
`tgt_position_offset` from the link". This is the tonogenesis
signature: segmental voicing at position -1 predicts tone on
position 0, for example.

`src_count` and `count` are the denominator and numerator of
the confidence. `confidence` is the fraction of matches among
the observations where the source feature held.

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
  system; if you need one, retrain regulae.
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
