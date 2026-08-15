# Correspondence Discovery: Design and Rationale

How the `regulae` pipeline discovers each kind of sound
correspondence — from raw segment pairs to context-conditioned
splits, tonal correspondences, cross-dimensional rules, long-range
predicates, and multi-lect reconciliation. This document covers
the *discovery mechanisms*; the *pipeline ordering* rationale lives
in `docs/training_pipeline.md`.

Implementation: `src/regulae/src/regulae/training.py` (discovery),
`src/regulae/src/regulae/scoring.py` (scoring overlay),
`src/regulae/src/regulae/anomaly.py` (residual-MI engine).

## 1. Segment correspondences via Dirichlet update

The foundation of the whole system. Every pair of cognate forms
contributes 1-to-1 link observations to a
`SegmentCorrespondenceTable` keyed by
`ConditionedCorrespondence(src, tgt, context)`.

### The prior

The prior is derived from the merkmal feature system: for every
source grapheme `s`, the prior assigns mass to each possible target
`t` proportional to `exp(−τ · d(s, t))`, where `d` is the merkmal
feature distance and `τ` (the concentration hyperparameter) controls
how sharply the prior favours similar segments. This is a structured
prior, not a uniform one — it says "nearby segments in feature space
are more likely correspondents," which is a weak typological bet.

### The update

Standard Dirichlet-multinomial: at each EM step, the counts from
the alignment E-step are added to the prior pseudo-counts. The
posterior probability of `s → t` given context `c` is
`P(t | s, c) = (count(s,t,c) + α(s,t)) / (Σ_t' count(s,t',c) + Σ_t' α(s,t'))`.

### Scoring in both directions at once

`P(t | s, c)` is a directional quantity, and a cost built from it
alone makes the analysis depend on which lect the corpus happened to
name first: the two directions carry different denominators, so
aligning A against B and B against A cost different amounts, and every
class reconciled from those alignments inherited the difference.
regulae takes no view on which lect is ancestral, so a score that
takes one is making the claim by accident.

The segment cost is therefore the geometric mean of the two
conditionals — half the surprisal of seeing `t` given `s`, plus half
of seeing `s` given `t`:

```
cost_seg = −½ log P(t | s, c) − ½ log P(s | t, c)
```

Both directions share the observed count; only the denominators and
the prior differ. The reverse prior needs no table of its own —
`α(s,t)` is the concentration times a softmax over merkmal distances
from `s`, and the distance is symmetric, so `α(t,s)` is the same
number rescaled by the ratio of the two partition functions. It is
read straight out of the prior table rather than derived, because an
`exp` of a difference of logs is the one step here whose last bit
moves between C libraries, and the published tables have to agree bit
for bit between the native and the WebAssembly build.

The reverse denominator needs the mass answering to `t` as a *target*,
which is why `rg_segment_count_row` publishes `target_total` alongside
`source_total`.

This makes the score symmetric. It does not yet make a whole training
run symmetric: the displacement term describes a feature moving from
one value to another and is directional by construction, and the
alignment DP resolves exact cost ties by enumeration order, which the
exchange does not preserve.

### Graceful degradation

At zero observations, the posterior equals the prior, so the model
scores exactly like merkmal. As data accumulates, the posterior moves
toward the empirical distribution. This gives one of the framework's
design invariants: "the initial model is behaviourally identical to
prior-only scoring."

### Why not train the prior itself?

The prior is fixed (derived from merkmal, not learned). This is
deliberate: the prior embeds feature-geometry knowledge that is
cross-linguistically stable and shouldn't be washed out by a single
corpus's idiosyncrasies. If the prior were also learned, a
heavily-biased corpus could distort it and degrade performance on
out-of-sample pairs.

## 2. Feature-displacement distribution

After segment-level EM converges, a one-pass re-alignment produces
a count of feature-displacement vectors: tuples of
`FeatureDisplacement(feature, from_value, to_value)` aggregated
across all 1-to-1 links.

### Generalisation across natural classes

The displacement distribution's purpose is to generalise beyond
the specific segment pairs the corpus happened to contain. If the
corpus contains `p → f`, `t → θ`, `k → x`, the displacement
distribution has three observations of `{continuant: − → +}` on
voiceless stops. When a new form contains `kʷ`, the model predicts
`kʷ → xʷ` through the displacement term even though no specific
`kʷ → xʷ` observation exists. Without this layer, the model is
limited to memorised pairs.

### Scoring integration

The layered cost of a 1-to-1 link is:

```
cost = w_seg · cost_seg + w_disp · cost_disp + w_tone · cost_tone
```

where `cost_seg` is the Dirichlet-posterior segment cost (offset by
`log Z`; see `docs/training_pipeline.md`), `cost_disp` is the
displacement-level `−log P` (offset by `log V_eff`), and
`cost_tone` is the tonal correspondence cost (see below). All three
are in nats, and the weights (`segment_weight`, `displacement_weight`,
`tone_weight`) control their relative influence.

## 3. Context-conditioned splits

After the unconditioned segment table is converged and chunks have
been promoted, context discovery splits the distribution of targets
for each source grapheme by phonological context.

### Feature inventory

Candidates are built from `rg_context_feature_names` in `src/context.c`
— 27 names projected out of merkmal's system — applied to the
preceding and following segment, plus word position and stress. That
is 52 feature candidates, 3 positions and the stress axes: about 55
per immediate split step, and about 135 for the long-range pass, which
also ranges over distance, existential and syllable-relative slots.

Two things follow, and both matter more than the list itself.

First, the vocabulary is a **projection, and a lossy one**. Merkmal's
`distinctive` system carries far more than 27 names over its base
inventory, and what survives the projection is a manner-and-place
vocabulary with a Eurocentric shape. Not present, and therefore not
expressible as a conditioning environment: tone of any kind, rounding,
vowel nasalisation (`nasal` here matches nasal *stops*), lateral,
trill, tap, retroflex, ejective, implosive, click, breathy, creaky,
ATR, pharyngealisation, syllabicity. A change conditioned by rounding
— the organising fact of Turkic and Uralic vowel harmony, and this
repo ships a Turkish–Azerbaijani corpus — is invisible to discovery
however regular it is. Widening this is the open question in the
handoff note; it cannot be done without also facing the calibration
problem below, since a wider inventory is a wider argmax.

Second, an earlier version of this section claimed the inventory was
~14 candidates and argued that adding `{lateral, retroflex, labial}`
would "widen the candidate space without adding discriminative power".
The inventory has since grown to 52 and includes labial. The argument
was never measured, and the effect it hand-waved at — that a wider
candidate space makes a spurious partition more likely to cross BIC —
is real, is not priced by BIC, and is what the permutation baseline
exists to expose.

### Why BIC, not mutual information or chi-square

BIC has the complexity penalty built in: committing a context split
costs `k · ln(N)` parameters, so splits survive only when the
likelihood gain exceeds the penalty. Mutual information or
chi-square would require a separate significance threshold and an
adjustment for multiple comparisons. BIC does both in one formula.

The `DELTA_BIC_THRESHOLD = −1.0` safety buffer is tuned empirically:
splits with `ΔBIC ∈ (−1, 0)` on tested corpora were uniformly
spurious (same dominant target, slightly different minority mass).

### Subset-match scoring

When multiple conditioned correspondences match a link's context,
the scoring function picks the most specific one (the one with the
highest `constraint_count`). This creates an implicit specificity
hierarchy without an explicit tree structure: the unconditioned entry
is the fallback, any conditioned entry that matches is preferred,
and a doubly-conditioned entry is preferred over a singly-conditioned
one.

### Multi-segment link decomposition

Multi-segment links (chunks) from the main alignment are decomposed
into 1-to-1 equivalents via a sub-alignment with `max_chunk_size=1`.
This recovers segment-level observations from patterns the main
search expressed as chunks (e.g. `sk → ʃ` packaged as a 2-to-1
chunk). Without decomposition, context discovery would be blind to
any source grapheme that the main search absorbed into a chunk.

## 4. Tonal correspondences

A separate `TonalCorrespondenceTable` counts `(src_tone, tgt_tone)`
pairs across 1-to-1 links where at least one segment carries a tone
annotation.

### Why a separate table

Tone is a suprasegmental dimension — it belongs to a different
tier from segmental features. Embedding tonal information in the
segment key would conflate the two dimensions and make the segment
table sparse (every toned segment would split into `N_tones`
variants). A separate table keeps the segmental model dense and
lets the tonal model grow independently.

### Inertness on non-tonal data

If no segment in the corpus carries a tone annotation, the tonal
table stays empty and contributes zero cost at scoring time. This
means tonal scoring is fully backward-compatible: non-tonal
corpora train and score identically whether or not the tonal
infrastructure exists.

## 5. Cross-dimensional rules

A `CrossDimensionalLink` encodes a rule of the form
"segmental feature F at source position p predicts suprasegmental
value V at target position q." The canonical instance is tonogenesis:
voicing on a consonant predicts tone on the following vowel.

### What has to be true before a rule is committed

A cross-dimensional rule is a claim about a conditioned split, so it is tested
the way a conditioned split has to be tested. Three things must hold, and each
of them rejects a class of statement that looks like a rule and is not.

**The environment must have a complement.** "The preceding segment is a
consonant" is not an environment on a corpus of CV syllables; it is a
description of the corpus. A predicate true of every observation partitions
nothing, and a rule conditioned on it says only what the ambient distribution
already said. Both sides must reach `_CROSS_DIM_MIN_RULE_COUNT`.

Positions that do not exist are excluded from both sides rather than counted as
the complement. Word-initial position is not a voiceless onset, and a rule
stated over the union of "after a voiceless segment" and "at the left edge"
cannot be read as either. This is why the Middle Chinese voicing rule in
`experiments/mandarin_historical/` is visible at all: with edges folded into the
complement it is diluted below its own threshold.

**The environment must move the distribution.** The target dimension is modelled
once over all observations, and again separately inside and outside the
environment; the second model is accepted only when its likelihood gain beats
what its extra parameters cost under BIC. This is the criterion, and the code
shape, that context discovery already uses for segmental splits — the same
question is being asked, so it gets the same test.

**The value must be the thing that moved.** An environment passing its own test
says the distribution changed, not which value changed, and on a dimension with
several values most of them did not. Each candidate value is therefore tested on
its own 2×2 table — value against not-value, inside against outside — with its
own BIC penalty, and a value is reported only if it is *raised* inside the
environment relative to the contrast. Requiring a rise is what stops a
two-valued dimension from publishing both of its values for the same
environment, which is a frequency table pretending to be a pair of rules.

### Both halves of a split are findings

A conditioned split is a two-sided statement. "Voiced onsets give tone 4" is
half of a tonogenesis; "voiceless onsets give tone 1" is the other half, and a
report carrying only the first describes a conditioned merger as though it were
a one-way change. The complementary environment is published as a rule in its
own right, under `source_value = "-"`.

### Greedy commit against the residue

Environments are committed one at a time, best ΔBIC first, and each committed
rule retires the observations it accounts for. What a rule mispredicts stays
live, so a later rule can still explain the residue — which is the whole point
of running this stage after the segmental and tonal baselines.

Without the retirement step every correlated framing of one fact commits
separately. On a corpus of CV syllables, `consonant`, `sonorant`, `nasal` and
`voiced` at the same offset are four descriptions of the same onset, and a
reader given all four has no way to tell they are one finding. Retiring the
evidence is what makes the second-best framing fail its own test, which is more
honest than filtering it afterwards by name.

`_CROSS_DIM_MAX_ITERATIONS` bounds the loop. Typical corpora exit after one or
two rounds, when nothing left unexplained passes.

### Both sides of a pair

A sound change is conditioned by the environment it happened in, and that
environment lives in the ancestor — which regulae refuses to identify. So
conditioning is looked for from both sides of every pair.

The target-side pass reads the same alignments the model produced, and records
each link from the target's point of view: the target grapheme becomes the
thing being conditioned, the source grapheme the outcome, and the target form's
context the environment. Everything downstream is unchanged, and the roles are
put back when the row is written. `rg_conditioned_segment_count_row` carries
`context_is_target` to say which form a rule's environment belongs to, and the
scorer matches each rule against that form's context.

This is not a refinement. A change is only *visible* from the side that has the
split: where the daughter reflects a conditioned change, the ancestor's segment
answers to two daughter segments, and from the daughter's side each segment has
one source and there is nothing to condition at all. Latin rhotacism read from
the Latin side is "r answers to s before a vowel", which is true and is not the
law; read from the other side it is /s/ becoming /r/ *between* vowels.

### One search, two bars

Immediate neighbours and long-range axes are searched by separate stages, and
that stays: which kind of predicate *opens* a split is a linguistic decision,
and the pipeline's stage order is load-bearing. But once a split is committed,
refinement within it considers every axis, because a conditioning environment is
not obliged to be built out of one kind of predicate. Grassmann's Law is
word-initial *and* followed somewhere by an aspirate; assimilation is regularly
"before X" *and* "after Y".

Each candidate carries its own bar — minimum observations, ΔBIC threshold,
dominance — rather than the search carrying one. There are a handful of
immediate neighbours and around a hundred long-range and existential axes, and
the more candidates a search considers the likelier one of them fits an outcome
by chance, so the same evidence has to buy less from the larger pool. The best
split is then the candidate that clears *its own* bar by the most, not the one
with the lowest raw score, which would let the larger pool win on volume alone.

### Support floors

- `_CROSS_DIM_MIN_RULE_COUNT = 3` — minimum absolute support, on each side of
  the contrast and on the rule itself. BIC alone admits `count=1/N` noise on
  corpora with many rare outcomes.
- `_CROSS_DIM_MIN_RULE_CONFIDENCE = 0.0` — off by default, and deliberately.
  A fixed conditional probability is not a measure of conditioning: with two
  possible values 0.5 is chance, with ten it is overwhelming evidence, and a
  floor that rejects a value occurring at 0.4 against a base rate of 0.05
  discards exactly the conditioned splits this stage exists to find. It remains
  available for suppressing weak rules in a report.

## Reading a model against its own noise

Nothing in a model says whether the corpus had any signal in it. Class counts
do not: they rise when the signal is removed. So every trained model carries a
fit summary (`rg_multi_model_fit`, printed by `train --human`, exported as
`fit` in `--json`):

- `cost_per_segment` — the mean alignment cost per segment over every cognate
  set the model can score, the same quantity `rg_find_cognate_outliers`
  z-scores across sets. This is the number that behaves: strongly negative on
  real cognates, near zero on shuffled ones.
- The shuffled baseline, when `permutation_count > 0`. Each run rebuilds the
  corpus with every lect's wordlist intact, every set's size and lect
  membership intact, and only the pairing permuted — Fisher–Yates per lect, so
  no set can end up with two forms of one lect — then trains on it. The
  reported `cost_per_segment_z` is how many baseline standard deviations the
  real corpus sits below its own noise.

The baseline is off by default because each shuffle costs a full training run.
Ten shuffles of a 97-set two-lect corpus take about six seconds. It is seeded
from `permutation_seed` and is exactly reproducible: a fit statistic a reader
cannot recompute is not a statistic.

What the numbers look like, at ten shuffles:

| corpus | cost/segment | baseline | z | classes (real → shuffled) |
| --- | --- | --- | --- | --- |
| `rhotacism` | −1.82 | −0.58 | −14.8 | 20/3 → 58/12 |
| `grimm` | −1.50 | −0.50 | −13.3 | 35/7 → 63/11 |
| `verner` | −1.83 | −0.89 | −17.9 | 23/5 → 79/23 |
| `latin_spanish` | −1.93 | −0.58 | −28.2 | 62/25 → 134/41 |
| unrelated pseudo-words | −0.19 | −0.28 | **+1.9** | 78/34 → 75/34 |

The last row is the point. Thirty-four conditioned classes, each printed with a
stated environment, from data with no relationship in it — and the fit
statistic says so plainly while the counts do not.

This is a corpus-level statement, not a per-rule one. It answers "is there
anything here", which is the question that has to be answered first. It does
not certify any individual environment, and a rule's own evidence — its count,
its contrast, its interval — remains the reader's business.

### What the original design had and this does not

The Python implementation ranked candidates by residual mutual information
against a permutation null (shuffle the target assignments, recompute, 100×)
before the BIC loop saw them, supported joint predictors (`src_feature_2`), and
ran a post-commit pass to collapse dual positional framings of one rule. None
of that is ported.

The claim that "the permutation null and the BIC gate answer the same question
by different means, and the loop above keeps the cheaper one" stood here until
2026-08-15 and was wrong. They answer different questions. BIC prices
*parameters*: one added term against the likelihood it buys. The permutation
null prices the *search* — and the search is large, since `find_best_split`
takes the argmax over ~55 immediate or 135 long-range candidates, greedily,
recursively. A gate that charges for one parameter while the argmax ranges over
a hundred is not measuring what it appears to measure.

Measured: permuting a corpus's pairings — every wordlist intact, only which
form answers to which destroyed — *raises* the class counts on every corpus
tried. Latin–Spanish goes from 62 unconditioned and 25 conditioned to about 133
and 41; Proto-Polynesian–Hawaiian from 23 and 7 to about 75 and 34. A corpus of
unrelated pseudo-words yields 78 and 34 with no relationship in it at all.

The null is back, as a calibration layer rather than a candidate filter — see
"Reading a model against its own noise" below. Dual-framing dedup is
unnecessary here: the second framing of a committed rule has no unexplained
evidence left to justify it. Joint predictors are a real gap, tracked in
`c_conversion_roadmap.md` — `rg_cross_dimensional_row` cannot represent a rule
conditioned on two predicates, so a change like the full Middle Chinese
register split, which needs voicing *and* source tone together, is out of reach.

### What a row reports

Every row carries the contrast it was measured against: `count`, `source_count`
and `confidence` describe the environment, `contrast_count`,
`contrast_source_count` and `contrast_confidence` describe everywhere else, and
`delta_bic` scores the environment as a whole. Reading `confidence` alone will
mislead — a rule holding at 0.9 where the contrast also holds at 0.9 is not a
rule — which is why the human formatter prints both.

### Scoring overlay

At alignment-cost time, committed cross-dimensional rules are
evaluated as an additive overlay on top of segmental + tonal scoring.
For each 1-to-1 link, each rule checks whether its source predicate
holds; if so, it computes a Dirichlet-smoothed log-likelihood ratio
(match vs. mismatch) and applies it as a positive or negative cost
adjustment. Rules that explain a real correspondence produce negative
adjustments (lower cost), making the alignment cheaper on matching
data and more expensive on mismatching data.

The overlay is *not* inside the DP: the search picks alignments
without knowing rules will fire, and the rules re-score afterwards.
This is a provisional design — integrating the overlay into the DP
cost function would be more principled but requires substantial
refactoring.

## 6. Long-range context predicates

The long-range context discovery stage mirrors immediate-neighbour
context discovery but operates on a different candidate set:

### Three predicate families

1. **Distance-bounded** — `preceding_at_distance(d)` and
   `following_at_distance(d)` for offsets {2, 3}. Offset 1 is the
   immediate neighbour, already handled by stage 4.
2. **Existential** — `somewhere_preceding` / `somewhere_following`:
   "some segment strictly before/after the link satisfies this
   constraint." Position-agnostic.
3. **Syllable-structural** — `same_syllable`, `next_syllable`,
   `previous_syllable`: constraints on segments in the same or
   adjacent syllable as the link position. Populated only for 1-to-1
   links — chunks span multiple positions and the syllable notion
   becomes ambiguous.

### Syllabification

Syllable-structural predicates depend on syllable breaks, which are
computed by a minimal sonority-based syllabifier
(`regulae.syllabification`): the max-onset principle under the
sonority sequencing principle. Language-agnostic and simple by
design — users with language-specific phonotactics should populate
`Form.syllable_breaks` externally, and the module respects
pre-populated values.

### Calibrated thresholds

The larger candidate space (~54 per step) requires tighter thresholds
than the immediate-neighbour case:

- `ΔBIC ≤ −5.0` (vs. −1.0)
- Minimum 5 observations per split branch (vs. 2)
- Feature inventory: `{front, back, close, open, voiced, voiceless}`
  — dropped `vowel` and `consonant` because they are tautological on
  syllable-structural slots (every next syllable has a vowel).

### Scoring integration

Long-range conditioned entries live in the same segment
correspondence table as immediate-neighbour entries — they just
have non-empty long-range `Context` fields. At scoring time,
subset matching picks the most specific entry that fires, so a
long-range entry overrides a less-specific immediate-neighbour
entry or the unconditioned fallback.

## 7. Multi-lect training

When training on cognate sets with more than two lects, the pipeline
extends through three additional stages.

### Per-pair pass

The full seven-stage pairwise pipeline runs on every directed pair of
lects that shares enough cognate data. The "canonical pair ordering"
derives from `_collect_lect_ids`, which alphabetises lect IDs and
iterates `combinations(lect_ids, 2)`.

### Union-find reconciliation

Per-pair 1-to-1 link positions are reconciled into N-way
correspondence classes via a union-find structure. Two position pairs
`(lect_A, pos_i)` and `(lect_B, pos_j)` that appear in the same
1-to-1 link of a pairwise alignment are in the same class. Transitive
closure over all pairs produces the multi-lect classes.

Each class is a `MultiLectCorrespondenceClass`: a mapping from
lect ID to grapheme, with an observation count aggregated across
the corpus. These are the unconditioned multi-lect classes.

### Class-level context discovery

Multi-lect context discovery mirrors per-pair context discovery
(stage 4) but operates at the class level. It iterates over pivot
lects: for each pivot, it gathers all observations where that lect
participates, and runs greedy BIC splits exactly as in the pairwise
case.

Two adaptive knobs:

- **AICc correction.** The standard BIC formula assumes large `N`.
  On small multi-lect corpora (typical for under-documented
  language families), the AICc correction penalises model complexity
  more heavily, preventing over-splitting.
- **Min-commit floor scaling.** The minimum observation count for
  a split branch scales with the corpus size: `floor = max(2, total × scale)`.
  This prevents committing a split on one or two observations in a
  10-pair corpus, while still allowing it in a 200-pair corpus.

Pivot-lect deduplication merges classes that were discovered through
multiple pivots but describe the same conditioning pattern. Two
conditioned classes are duplicates when they carry the same segment
tuple and equivalent contexts.

### Cross-dimensional rule lifting

Per-pair cross-dimensional commits are surfaced at the multi-lect
level by iterating the canonical pair ordering and wrapping each
`CrossDimensionalLink` with `src_lect`/`tgt_lect` labels. No new
discovery runs at this level — rules inherit the consolidation
filters (support floors, dual-framing dedup) from the per-pair pass.

## Threshold calibration summary

Thresholds throughout the pipeline are not arbitrary constants.
They were calibrated on a combination of synthetic fixtures
(tonogenesis, umlaut, harmony) and real-data experiments
(Latin–Spanish, Latin–French, Proto-Polynesian–Hawaiian, Old English–
Modern English, GLED Romance, GLED Polynesian). The calibration
principle is conservative: suppress false positives (spurious
commits on noise) even at the cost of missing marginal true
positives, because a linguist can more easily identify a missing
rule than diagnose a spurious one.

Key calibration points:

| Threshold | Value | Scope | Why this value |
|-----------|-------|-------|----------------|
| `DELTA_BIC_THRESHOLD` | −1.0 | immediate-neighbour splits | Rejects near-zero splits that are spurious |
| `_LONG_RANGE_DELTA_BIC_THRESHOLD` | −5.0 | long-range splits | Larger candidate space needs tighter margin |
| `MIN_SPLIT_OBSERVATIONS` | 2 | immediate-neighbour splits | Allows minority targets (e.g. 3 obs. of `k → θ` vs. 11 `k → k`) |
| `_LONG_RANGE_MIN_SPLIT_OBS` | 5 | long-range splits | Prevents chance partitions in large candidate space |
| `MAX_SPLIT_DEPTH` | 3 | both context loops | Caps runaway multi-level splits |
| `_CROSS_DIM_MIN_RULE_COUNT` | 3 | cross-dimensional commit | Rejects 1/N noise rules |
| `_CROSS_DIM_MIN_RULE_CONFIDENCE` | 0.0 | cross-dimensional commit | Off: a fixed fraction does not measure conditioning |
| `_CROSS_DIM_MAX_ITERATIONS` | 5 | cross-dimensional loop | Safety valve; typical corpora exit after 1–2 |
| `_CROSS_DIM_DELTA_BIC_THRESHOLD` | −1.0 | cross-dimensional commit | The environment, and each value, must beat its parameters |
| `MIN_CHUNK_OBSERVATIONS` | 2 | chunk promotion | Rejects singleton chunks BIC can't filter |
