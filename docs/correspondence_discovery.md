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

The candidate feature set is small and linguistically motivated:

```
following: vowel+, front+, back+, close+, open+
preceding: vowel+, front+, back+, voiced+, voiceless+, consonant+
position:  initial, medial, final
```

About 14 candidates per split step. This is not laziness — it is a
calibrated design decision. Adding more features (e.g.,
`{lateral, retroflex, labial}`) would widen the candidate space
without adding discriminative power on the corpora tested, while
increasing the chance that a random feature partition crosses BIC by
chance.

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

### Discovery via anomaly detection

The anomaly detection engine (`regulae.anomaly`) computes residual
mutual information between every candidate `(source_feature,
source_offset, target_value, target_offset)` pair across the 1-to-1
links in the training alignments. "Residual" means conditional on
what the model already explains — so a feature pair that is already
captured by the segment or tonal table will not surface as an anomaly.

Each candidate is tested against a permutation null (shuffle target
assignments, recompute MI, repeat 100×). Candidates whose observed
MI ranks above the 95th percentile of the null distribution become
`PatternHypothesis` objects passed to the BIC commit loop.

### BIC commit loop

The commit loop evaluates each hypothesis by:

1. Building a trial `CrossDimensionalLink` from the hypothesis.
2. Re-scoring the entire training corpus under a trial model that
   includes the candidate rule.
3. Computing `ΔBIC = −2 · (trial_cost − baseline_cost) + k · ln(N)`.
4. Accepting the best candidate whose `ΔBIC` beats the threshold.
5. Repeating on the updated model.

This is sequential-greedy with re-ranking: the expensive permutation
null recomputation happens at most once per iteration. In practice
the loop terminates after 1–2 iterations on typical corpora.

### Support floors

BIC alone admits `count=1/N`-style noise rules on corpora with many
rare tonal outcomes. Explicit floors reject these:

- `_CROSS_DIM_MIN_RULE_COUNT = 3` — minimum absolute support.
- `_CROSS_DIM_MIN_RULE_CONFIDENCE = 0.5` — minimum conditional
  probability of the target value given the source predicate.

### Dual-framing dedup

The greedy commit naturally emits rules in multiple positional
framings of the same underlying pattern (e.g.,
`voiced@−1 → tone=4@+0` and `voiced@0 → tone=4@+1`). A post-commit
dedup step collapses these to the canonical framing (smallest
`src_offset`). Two rules are duals iff they share the source feature,
target dimension, target value, and the delta between target and
source offsets.

### Joint predictors

An optional second predictor (`src_feature_2`, `src_position_2`)
handles the cross-source-dimension case: voicing AND source tone
jointly predict a target tone. Joint rules pay `k = 2` in the BIC
formula (instead of `k = 1`) and must strictly outperform either
component predictor alone — a non-interaction guard that rejects
"passenger" predictors where one component does all the work.

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
| `_CROSS_DIM_MIN_RULE_CONFIDENCE` | 0.5 | cross-dimensional commit | Rejects low-confidence rules |
| `_CROSS_DIM_MAX_ITERATIONS` | 5 | cross-dimensional loop | Safety valve; typical corpora exit after 1–2 |
| `MIN_CHUNK_OBSERVATIONS` | 2 | chunk promotion | Rejects singleton chunks BIC can't filter |
