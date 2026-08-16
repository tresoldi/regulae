# Training Pipeline: Design and Rationale

This document explains the *why* of the training pipeline — why it
is staged, why stages are ordered the way they are, what invariant
each stage establishes, and why certain non-obvious design choices
(the log-normalizer offset, the two cost scales, BIC gates at every
commit) are in the code.

The implementation is the C core: `src/model.c` drives the pairwise
pipeline, `src/model_context.c`, `src/model_chunks.c` and
`src/model_crossdim.c` are the discovery stages, `src/split_search.c` is
the split search they share, and `src/multilect_*.c` is the multi-lect
layer. The path given here until 2026-08-16 pointed at the archived
Python tree under `python/`, which is not built or supported.

Conceptual overview in `framework/03_alignment.md`, technical detail in
`docs/alignment_details.md`, discovery mechanisms in
`docs/correspondence_discovery.md`.

## Why a staged pipeline

A naive end-to-end EM loop over cognate data would try to learn
everything simultaneously: segment correspondences, chunk phrases,
context conditioning, tonal patterns, cross-dimensional rules. This
does not work. Three concrete reasons:

1. **Context conditioning needs stable segment-level correspondences.**
   A greedy context split wants to partition observations of `k → ?`
   by whether the following segment is `[+front]`. If segment-level
   EM hasn't converged yet, the observations are noisy and the split
   commits on artifacts. Segment-level EM must run to near-convergence
   first so the context-discovery stage sees a clean distribution.

2. **Chunk promotion traps segment-level EM.** A chunk like `kt → tt`
   committed early steals observations from the `k → t` and `t → t`
   cells that would otherwise drive segment-level convergence.
   Promoting chunks after segment-level EM has converged keeps the
   segment table honest.

3. **Cross-dimensional rules are residuals.** Tonogenesis is, by
   definition, the part of the target distribution that is not
   explained by segmental correspondences. Discovering it requires a
   baseline: a model that already explains everything it can segmentally
   and tonally, so that what is left is genuinely unexplained. Running
   the stage before the segmental baseline has converged produces
   spurious rules that are really just under-trained segment
   correspondences. The same reasoning runs inside the stage, where each
   committed rule retires the evidence it accounts for before the next
   environment is tested.

So stages are ordered by what they consume. Each stage either
converges on its own objective (segment EM, chunk promotion) or runs
one-pass aggregation of residuals from the previous stages.

## The stages

All stages are private functions in `training.py`. Names used here
match the code.

### 1. Segment-level EM — `_segment_em`

**What it does.** Iterative E/M loop: align every form pair under the
current model, then re-estimate the segment correspondence table from
the 1-to-1 links in those alignments. Chunks and gaps do not
contribute to counts at this stage (chunks are handled later, gaps
are not modelled).

**What it produces.** A `SegmentCorrespondenceTable` with
Dirichlet-smoothed counts keyed by `ConditionedCorrespondence(src,
tgt, context=Context())`. Context is empty at this stage — everything
is unconditioned.

**Convergence criterion.** Relative change in total corpus cost below
`convergence_eps`, or `max_iter` reached. Total cost is summed over
per-pair alignment costs under the current model.

**Why Dirichlet smoothing.** The Dirichlet prior is derived from the
merkmal feature distance: at zero observations, the posterior
collapses to the merkmal prior, so the model degrades gracefully to
prior-only scoring. As observations accumulate, the posterior
concentrates on the data-driven correspondences. This gives one of the
framework's load-bearing properties: the model does something sensible
on small corpora and learns more from large ones without a phase
transition.

### 2. Displacement aggregation — `_displacement_aggregation`

**What it does.** One-pass re-alignment of the corpus with the
post-EM model, counting every 1-to-1 link's `FeatureDisplacement`
vector.

**What it produces.** A `DisplacementDistribution`: counts of
displacement vectors over the corpus.

**Why this layer exists.** Displacement-level regularities generalize
across natural classes in ways that segment-pair counts cannot. The
classic case: once the Dirichlet table has seen `p → f`, `t → θ`,
`k → x`, the displacement distribution has three observations of
`{continuant: − → +}` on voiceless stops — and if a new pair contains
`kʷ`, the model predicts `kʷ → xʷ` even though that specific pair
has zero observations. Without displacement aggregation, the model
can only memorize pairs it has seen.

### 3. Chunk promotion — `_chunk_promotion`

**What it does.** Walks the training alignments, collects candidate
multi-segment chunks, and runs a greedy BIC loop: at each step, pick
the candidate whose promotion gives the largest cost reduction, and
commit it if the reduction beats the length-proportional BIC penalty
and the minimum-observation floor.

**What it produces.** A `ChunkPhraseTable` mapping
`(source_chunk, target_chunk)` tuples to learned costs.

**BIC formulation.** `ΔBIC = −2·(log_L_new − log_L_old) + k·ln(N)`
where `k = max(src_len, tgt_len)` rather than the textbook `k = 1`.
Why length-proportional? A single-segment chunk is indistinguishable
from a correspondence table entry, so `k = 1` would admit every
single-pair chunk that happens to appear more than once. With
`k = max(len)`, longer chunks have to clear a proportionally higher
bar, which is what prevents runaway promotion of spurious long
chunks.

**Min observation floor.** BIC alone admits singleton chunks in small
corpora. An explicit `MIN_CHUNK_OBSERVATIONS = 2` floor rejects them.

**The two-cost-scales issue.** See the separate section below. Short
version: chunk costs stored in the phrase table must be offset for
the DP, but the BIC comparison uses unoffset `−log P`.

### 4. Context discovery — `_context_discovery`

**What it does.** For each source grapheme with at least two observed
targets, runs a greedy feature-based split search. Candidate splits
come from the contrastive vocabulary derived from the corpus's merkmal
feature system, on the immediate preceding and following segments,
plus word position. A split is committed when the configured score is negative,
up to `MAX_SPLIT_DEPTH` levels.
Splitting a pooled `K`-outcome multinomial in two adds `K−1` parameters,
so the default corrected-BIC complexity charge is `(K−1)·ln(N)`, not a
fixed `ln(N)`. Exact NML and a symmetric-Dirichlet marginal likelihood use
the same search through `split_score.c`.

**What it produces.** New entries in the segment correspondence table
with non-empty `Context` fields. The unconditioned entries from stage
1 are kept as fallbacks: any link whose context doesn't match a
conditioned entry falls through to the unconditioned version.

**Why feature-based and not grapheme-based.** Splitting on "the
following segment is `/i/`" is specific and fragile; splitting on
"the following segment is `[+front]`" generalizes across natural
classes. The vocabulary is contrastive in the inventory rather than a
hand-written list. Exact duplicate partitions count once in the adaptive
search charge, so synonymous feature encodings cannot change selection.

**Why corrected BIC remains the default.** BIC has the
complexity penalty built in. Mutual information does not penalize
splits by how many parameters they add, so it would over-commit on
small data. M3 selected the full model-space charge, `γ = 1`, and a zero
threshold after the earlier half charge admitted a conditioned class in the
pre-existing unrelated-lect restraint fixture. Corrected BIC survived the
recorded comparison with exact NML and Dirichlet marginal likelihoods; see ADR
0001.

**Why after chunk promotion.** See the design choice discussion
at the top — chunks are a kind of compressed correspondence, and
context-splitting on already-promoted chunks would re-learn the same
pattern in two registers. Running context discovery after chunk
promotion means context splits operate only on what's still
expressed at the 1-to-1 level.

Multi-segment link observations are decomposed into 1-to-1
equivalents via a sub-alignment with `max_chunk_size=1`, so that
patterns expressed as chunks by the main search (like `sk → ʃ`) still
contribute to context discovery.

### 5. Tonal aggregation — `_tonal_aggregation`

**What it does.** One pass: re-align, then count each 1-to-1 link's
tonal correspondence. A link where both segments are untoned
contributes nothing; a link where either carries a tone value
contributes to a `TonalCorrespondence(src_tone, tgt_tone)` entry.

**What it produces.** A `TonalCorrespondenceTable`. Inert on
non-tonal corpora (empty table → zero cost contribution at scoring
time).

**Why a separate table.** Tones are suprasegmental — they belong to
a dimension parallel to segment features, not a sub-field of the
segment. A separate table keeps the key spaces clean and lets tonal
scoring be additive with segmental scoring without any cross-keying.
It also makes the next stage (cross-dimensional discovery) tractable:
"tonogenesis" is literally "a segmental feature correlates with a
tonal outcome", and that correlation can only be tested when the two
dimensions live in separate tables.

### 6. Cross-dimensional discovery — `discover_cross_dimensional_rows`

**What it does.** Walks the 1-to-1 links once, recording for every toned
target segment which of the candidate environments (source feature ×
source offset) hold at its source position. It then commits environments
one at a time, best first, testing each against what earlier rules have
left unexplained, up to `_CROSS_DIM_MAX_ITERATIONS`.

An environment is committed only if its complement is attested and the
target distribution differs between the two by more than the extra
parameters cost under BIC; a value inside it is reported only if the
environment raises that value relative to the contrast and the rise
passes its own 2×2 BIC test. Both sides of the split are published, the
complement under `source_value = "-"`.

**What it produces.** A `CrossDimensionalLinkTable` the alignment cost
consults. Every row carries the contrast it was measured against, because
a conditional probability without its baseline is not evidence of
conditioning. See `docs/correspondence_discovery.md` for the criterion
and for what the original Python design had that this does not.

**Where it is charged, and why that is a stage-order fact.** As of
2026-08-16 the DP charges for these rules while it searches, rather than
the rules re-scoring an alignment already chosen. The adjustment is local
to a DP transition — its source predicate reads the source form at the
link's start and its target value the target form at that position plus
the rule's offset, and both forms are fixed input — so the DP can price
it, and the alignment returned is now the one minimising the function
that reports its cost.

The consequence lands on the *next* stage. Long-range context discovery
runs after this one and re-aligns, so it now sees alignments chosen in
knowledge of the cross-dimensional rules. That is a change in what that
stage is shown, not only in what a caller is told an alignment cost, and
it is why stage order is load-bearing here as elsewhere. Measured over
every corpus in the tree it moved two — both tonal, both improving their
`cost_per_segment` — and left every `testdata/soundlaws/` fixture
bit-identical.

**Why the environment is tested against its complement rather than
scored on its own.** The stage answers "does this feature condition this
dimension", and conditioning is a comparison. Until 2026-08-14 it was
answered with `P(value | environment) >= 0.5` and no comparison at all,
which committed predicates true of the whole corpus, predicates that
partitioned the corpus without moving anything, and — on a two-valued
dimension — both values for the same environment at once.

**Why the greedy loop tests against the residue.** Correlated framings
of one fact would otherwise commit separately: `consonant`, `sonorant`,
`nasal` and `voiced` at the same offset can all be true of the same
onset, and four rows describing one finding read as four findings.
Retiring the evidence a committed rule accounts for makes the redundant
framings fail their own test, which is more honest than filtering them
by name afterwards.

**Why a support floor in addition to BIC.** On corpora with many rare
tonal outcomes, BIC's Dirichlet-smoothed LLR on a 1/N-support rule
scores slightly negative — passing BIC but committing noise. The
floors are calibrated to reject this class of commits without
blocking genuine low-count rules.

**Why dual-framing dedup.** The greedy commit loop naturally emits
rules in multiple positional framings of the same underlying pattern
(e.g. `voiced@−1 → tone=4@+0` and `voiced@0 → tone=4@+1` describe the
same rule shifted by one link position). The dedup step collapses
these to the smallest-`src_offset` ("leftmost") framing. Without
dedup, a clean tonogenesis fixture would commit four rules when two
are warranted.

**Joint predictors.** A rule can optionally carry a second predictor
(`src_feature_2`, `src_position_2`) for cases where no single feature
is enough — the cross-source-dimension case where voicing AND source
tone jointly condition a target tone. Joint rules pay an extra BIC
parameter penalty so they only win when a single-predictor rule
genuinely cannot explain the data.

### 7. Long-range context discovery — `_long_range_discovery`

**What it does.** A second pass of context discovery, but using a
different candidate set: distance-bounded predicates (offsets
{2, 3} in either direction), existential predicates ("some segment
somewhere before/after the link"), and syllable-structural predicates
(same-syllable, next-syllable, previous-syllable).

**What it produces.** Additional entries in the segment correspondence
table whose `Context` fields populate the long-range slots. Existing
immediate-neighbour entries stay in place; long-range entries are
more specific overlays, and the scoring-time lookup picks the most
specific match.

**Why separate from immediate-neighbour context discovery.**
Immediate-neighbour context discovery enumerates about fourteen
candidates per split step; long-range enumerates about fifty-four
(six features × nine slots). Keeping them separate lets the scorer charge each
loop for its own distinct observed partitions and lets the long-range loop keep
its higher support floor without suppressing immediate-neighbour signal.

**Calibrated gates.** Long-range uses the same zero score threshold as
immediate-neighbour discovery because the score now prices the distinct
partitions searched. It retains a minimum split observation count of 5 (vs. 2)
and a trimmed feature inventory that drops tautological predicates
(`vowel`/`consonant`) on syllable-structural slots where they apply
by definition. Without these, the larger candidate space surfaces
chance partitions on small corpora.

**Why 1-to-1 only.** Multi-segment link observations are not
decomposed into 1-to-1 equivalents here (unlike stage 4): chunks
span multiple positions and the "same/next/previous syllable"
predicates become ambiguous over a span. Chunks are silently
skipped.

## Confidence weighting

`CognateSet.confidence ∈ [0, 1]` threads through every stage above
as an evidence weight. A set with `confidence = c` contributes `c`
to every count it would otherwise add by one: segment EM counts,
displacement counts, chunk candidate counts, context-split and
long-range observation masses, tonal counts, cross-dimensional BIC
(`ln N` and per-corpus cost), and multi-lect class aggregation.
`c = 0` is equivalent to excluding the set from training — the set
is retained in the corpus for auditability but contributes nothing
to any table.

**Threshold semantics.** The minimum-observation constants
(`MIN_CHUNK_OBSERVATIONS = 2`, `MIN_SPLIT_OBSERVATIONS = 3`,
`_LONG_RANGE_MIN_SPLIT_OBS = 5`) now compare against weight *sums*
rather than integer cardinalities. On uniform-confidence corpora
the behaviour is unchanged. On mixed-confidence corpora this means
two cognate sets at `confidence = 0.75` sum to `1.5` and will not
clear the `MIN_CHUNK_OBSERVATIONS` floor — which is the intended
semantics: the evidence is downweighted and so is the threshold it
has to clear.

## Multi-lect training

When training on cognate sets (N lects rather than 2), the pipeline
runs per-pair first, then reconciles. The detail lives in
`docs/correspondence_discovery.md` under "Multi-lect training".
Short version:

1. **Per-pair pass.** Run the full seven-stage pipeline on every
   directed pair of lects that shares enough cognate data.
2. **Reconciliation.** Union-find over per-pair 1-to-1 link positions
   produces N-way correspondence classes. Each class is a tuple of
   one segment per participating lect at corresponding positions,
   with observation counts aggregated across the corpus.
3. **Class-level context discovery.** A multi-lect version of stage 4,
   iterating over pivot lects and committing splits at the class
   level. It uses the same categorical scorer and a sample-size-scaled
   min-commit floor. M3 removed the un-derived AICc-shaped addition.
4. **Cross-dimensional rule lifting.** Per-pair cross-dimensional
   commits are surfaced at the multi-lect level with explicit
   `src_lect`/`tgt_lect` labels. No new discovery runs at this
   level — rules inherit the consolidation filters from the per-pair
   pass.

## The two cost scales

A non-obvious technical invariant that is load-bearing in scoring
and BIC.

**Naive scoring** of a segment pair under the Dirichlet posterior
gives `cost = −log P(t|s)`. On a log-probability scale this doesn't
balance against the gap cost: at zero observations, the merkmal
prior gives a non-zero `−log P` even for the identity pair `s → s`,
so insertions look artificially cheap compared to keeping the segment.

**Fix.** Subtract `log Z(s) = log Σ_t' exp(−τ · d(s, t'))` — the
log-normalizer of the prior — from the segment cost. This offset
makes the segment cost zero for the identity pair under the prior,
which aligns the scale with the gap-cost constants and recovers the
"initial model is behaviorally identical to prior-only scoring"
property. `log_normalizers` are cached on `SegmentCorrespondenceTable`
and applied at scoring time in `scoring.py`.

**Second problem.** The log-Z offset breaks BIC for chunk promotion:
under the offset scale, compositional scores are artificially cheap,
so every chunk looks worse than its compositional equivalent and
nothing ever gets promoted.

**Second fix.** Chunk promotion computes BIC with the unoffset raw
`−log P` scale but stores the promoted cost on the offset scale (by
subtracting `log Z` before writing). The two scales live side by side:
the DP and scoring use the offset scale, BIC comparisons during
promotion use the unoffset scale.

This is the most subtle invariant in the training pipeline. If you
touch `_chunk_promotion` or `_total_corpus_cost`, check you haven't
mixed the scales.

## Re-alignment between stages

Stages 2, 4, 5, 6, 7 all re-run alignment on the corpus with the
current model before doing their own work. This is correct: each
stage's observations have to reflect the best alignment under the
current model, not a stale alignment under the pre-previous-stage
model. The alignments are cheap relative to the training work
itself, and the `align_corpus` function is memoized over the batch.

The only stage that re-aligns *during* its own work is stage 1
(segment-level EM) — that's what makes it EM rather than a one-pass
count.

## Evaluation nulls are not interchangeable

The production fit baseline is a **pairing shuffle**: it retains every lect's
wordlist and set-membership pattern while breaking which forms answer one
another, then retrains the whole pipeline. It supports the corpus-fit z-score
and, today, the published per-rule standing threshold.

M2's evaluation harness adds two conditioned-selection nulls without changing
that default. A **within-source-bucket outcome shuffle** retains the source
segment, its environment and weights while permuting target outcomes inside
the source bucket. A **parametric unconditioned null** samples those outcomes
from the fitted unconditioned bucket distribution. They ask how often adaptive
environment selection invents a conditioned association when the
correspondence bucket itself is retained. Their fixed-position implementation
is deliberately limited to clean equal-length synthetic forms; it is an
evaluation instrument, not another production verdict hidden in the model.

The null name appears in JSON for both corpus fit and conditioned standing.
`docs/m2_evaluation.md` records the factorial snapshot and permutation-count
convergence.

## What's not in the pipeline (and why)

Things a new reader might expect but won't find:

- **No explicit alignment quality threshold.** There's no
  "convergence-failed-on-this-pair, drop it" gate. Pairs that align
  badly get included and contribute noisy observations. The user is
  expected to run `find_cognate_outliers` post-hoc to triage outlier
  pairs manually.
- **No iterated-DP pass after cross-dimensional commits.** The
  cross-dimensional overlay is currently a post-hoc additive
  correction at scoring time, not a factor inside the DP. This means
  the DP picks alignments without knowing the rules will fire; the
  rules re-score those alignments after the fact. The alternative
  (bake the overlay into the DP cost function) is a substantial
  refactor and has not been done.
- **No joint training of displacement and segment tables.** Stage 2
  is a one-pass aggregation, not an EM loop. This is fine as long as
  segment-level EM has converged first — the displacement counts
  are then just a feature-level view of the same data — but it does
  mean the displacement distribution cannot refine the segment table.
- **No end-to-end gradient or differentiable objective.** The
  pipeline is built out of classical estimation primitives
  (Dirichlet update, BIC, mutual information, permutation null).
  Nothing is trained by gradient descent. This is by design: the
  estimation procedures are explicable and the BIC gates are
  interpretable, which matters for a tool historical linguists are
  supposed to trust.
