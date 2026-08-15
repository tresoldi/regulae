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

There is no list of feature names in regulae. A grapheme's features are
whatever the merkmal system in use reports for it, and the searchable
vocabulary is derived from the corpus's own inventory by
`rg_feature_vocabulary_build_internal`, on two tests:

- **Contrastive.** Some segment in the corpus carries the predicate and some
  does not. A predicate nothing carries is dead weight; one everything carries
  partitions nothing, which is a documented way to commit a rule on no
  evidence.
- **Distinct.** No two predicates that separate this corpus's inventory
  identically both survive. merkmal's vocabulary is not orthogonal — `vowel`,
  `vocoid` and `syllabic` coincide in most corpora — and keeping every synonym
  widens the argmax without widening what can be found. Equivalence is a fact
  about the corpus: features that coincide in Latin come apart in a language
  contrasting syllabic consonants, and there they are kept apart.

Ties are broken toward the name claiming *less* — `coronal` before `alveolar`,
`labial` before `bilabial` — because when a corpus cannot distinguish two
environments the weaker report is the honest one. That preference is a
readability table over the categorical systems' names and decides nothing
about what is findable; a system whose vocabulary it does not cover loses
readability and no capability.

In practice this lands at 22–31 predicates for the fixtures here.

#### Any system, including the valued ones

merkmal's systems report two shapes. The categorical ones — `distinctive`
(the default), `descriptive`, `broad` — name the features a segment *has*:
`bilabial`, `nasal`. The valued ones — `phoible`, the `pbase-*` family — name
every feature with its value: `anterior=+`, `approximant=-`,
`advancedTongueRoot=.`. Splitting on the sign reads both as the same
(feature, value) pair, so one code path serves all of them.

On a valued system this also gives negative environments for nothing: "not
anterior" is `anterior=-`, which is in the data rather than something the
candidate list has to invent, and the contrastive filter drops it where the
corpus does not use it.

The same rounding-conditioned change, found under five systems:

| system | environment reported |
| --- | --- |
| `distinctive` / `descriptive` / `broad` | `fol[rounded=+]` |
| `phoible` | `fol[labial=+]` |
| `pbase-spe` | `fol[round=+]` |
| `pbase-hc` | `fol[labial=+]` |

Under the hand-written list this replaced, the three valued systems found
**zero** conditioned classes and said nothing about it.

#### Why it is not a fixed list

It was one until 2026-08-15: 27 names, hand-picked, by someone writing about
Latin. No rounding, no vowel nasalisation, no lateral, trill, tap or retroflex,
no ejective, implosive or click, no breathy or creaky, no syllabicity, and no
vowel height between close and open; affricates received no manner feature at
all. `testdata/soundlaws/rounding_harmony.tsv` — twenty-four regular instances,
environments differing in rounding alone — produced nothing under it. A
vocabulary that fits one family is a claim about the others.

### Reorderings

A transposition is one event and the alignment search is monotone, so the
search's natural output for it is two correspondences running in opposite
directions — /s/ answering /k/ and /k/ answering /s/. Nothing about the
segments changed; the order did.

A span whose target is its own segments in another order is recognised as such
(`rg_link_is_reordering_internal`) and handled in three places:

- **Scoring.** The span costs what the reordering costs — each segment against
  the one it actually answers to — rather than what pretending each position
  substituted for the one below it would cost. Scoring it positionally prices
  metathesis out of the search.
- **Reconciliation.** Union-find binds the positions that answer to each other,
  which for a reordering is not the diagonal. Binding it positionally makes the
  two segments members of each other's class in both directions.
- **The chunk row.** `rg_chunk_row.reordering` says the promotion was a
  reordering, so the model records that something happened where the segment
  table shows only identities.

**Distance.** Adjacent transposition fits inside a chunk. Anything wider does
not: a monotone search can only express it as one link covering everything
between the two segments that moved, and Spanish *milagro* against Latin
*miraculo* needs five. So a span up to `RG_MAX_REORDER_SPAN` enters the search
as one extra candidate when — and only when — the two sides really are the same
segments in a different order. That test is cheap and fails on the first
grapheme they do not share. It is one candidate rather than a raised chunk
limit, which would admit every ragged span of that width as well.

**What is not published.** A long-distance reordering is recovered — the
segments correspond to themselves, and the classes come out right — but it is
only written into the model as a row if its span also clears chunk promotion,
and a seven-segment chunk rarely pays for its parameters. So the analysis has
it and the tables do not. Closing that needs a row type of its own, gated on
its own regularity rather than on the chunk test.

### Intervals, and what they are computed from

Every count-bearing row carries an interval on the rate the count
represents. Two things about it are easy to misread, so both are now
stated on the row itself.

**What the denominator counts.** The default Wilson interval's `n` is
aligned positions, and positions from one word pair are not
independent observations: Latin–Spanish has 413 of them over 97
cognate sets, 4.3 per set. Setting `bootstrap_n` replaces the closed
form with a percentile interval resampled over whole **cognate sets** —
the unit the corpus actually samples — which handles the clustering
and the confidence weighting together, and needs no realignment
because `class_positions` already records which cognate each
reconciled position came from.

The measured effect is the opposite of the obvious guess. Resampling
sets gives *narrower* intervals on most rows: 229 of 240 classes across
six corpora, mean width 0.049 → 0.041. The rate is a ratio whose
numerator and denominator move together under a set-level resample,
and the Wilson model — fixed denominator, independent Bernoulli trials
in the numerator — does not capture that. It is conservative here, not
optimistic.

**Whether the environment was chosen from the same data.**
`post_selection` is set on every conditioned row. The interval then
says how well the rate is pinned *given* the environment, and nothing
about whether the environment is real; that is what `search_margin`
against `rg_corpus_fit.null_search_margin` is for. Two numbers
answering two questions, each labelled.

Intervals are published on every row that carries one. Six of the eight
row types computed one and surfaced it nowhere until 2026-08-15.

### Cross-dimensional rules, and why one predicate is not enough

Does something about the source form condition a suprasegmental value on the
target? The claim is a conditioned split and is tested as one: the complement
must be attested, and modelling the dimension separately inside and outside must
beat modelling it once, under BIC plus the search charge.

**The environment is an `rg_context_spec`** — the same type a conditioned
correspondence uses. It has to be, because one predicate is not always enough.
The Middle Chinese register split conditions the target tone on the preceding
onset's voicing **and** on the source segment's own tone:

| source tone | voiced onset | voiceless onset |
| --- | --- | --- |
| 1 | → 1 | → 1 |
| 2 | → **4** | → **2** |

Neither half predicts anything alone. Voicing alone reported this at confidence
0.50, which reads as a weak finding rather than half of one, and the source's
own tone was not a predicate at all — the vocabulary was thirteen hardcoded
segmental features. Both are now searched, and the rule comes back as
`pre[voiced:+] self[tone:2] -> tone=4` at confidence 1.00, with its complement.

**Predicates** come from the same corpus-derived vocabulary as context
conditioning, plus every suprasegmental value the corpus carries, at each of
three positions, in both polarities. `self` is the segment the rule is about;
suprasegmentals are features named `tone`, `length` and `stress`.

**Pairs are searched, and paid for.** P predicates give P(P-1)/2 pairs and P is
in the hundreds, so conjunctions are formed among the best sixteen singles by
ΔBIC, and the whole candidate count enters the split penalty as
`γ · 2 · ln(m)`, the same charge the context splitter pays.

Three rules keep the output readable, each of which was a real failure first:

- **A conjunction publishes only the side it holds on.** The complement of
  "voiced and source tone 2" is not "voiceless and not source tone 2", and a
  context cannot say *not (A and B)*. The other half is reachable as its own
  conjunction, because negation is a predicate. A single predicate still
  publishes both sides, where the complement is one predicate.
- **An environment that raises several values has not determined the
  outcome**, so its members stay live for a narrower environment to explain.
  Retiring them is what stopped the register split being found: source tone 2
  raises both tone 2 and tone 4, and consuming both left the conjunction with
  nothing to work on.
- **Equivalent and redundant environments are skipped.** Different predicates
  often carve one corpus identically, and committing each in turn republishes
  one finding as several. A narrower environment inside one that already
  determined its outcome is skipped too — but only inside a *determined* one,
  because refining an undetermined environment is the whole point.

**The target dimension is not only tone.** The scorer has handled stress and
length as targets since the port and this stage proposed neither, so
compensatory lengthening and stress shifts were unreachable however regular.
All three are searched now, and all three are suppliable: `tone`, `stress` and
`length` columns in the long format, `<lect>_tone`, `<lect>_stress` and
`<lect>_length` in the wide.

Length is the one with two spellings, and they mean different things. Written
into the grapheme as `aː`, merkmal reads it as its own segment carrying the
`long` feature — the right shape where length is contrastive, and what the
syllable predicates use. Written in the `length` column it is a dimension of
the vowel, which is what a *lengthening* is: `p a t a` answering `p a t a` with
the vowel long is one correspondence and a rule about it, where `a` answering
`aː` is two different vowels and no rule at all.

### Morphological environments

A morpheme boundary is not a sound, and a change that respects one is not
stating a phonological fact. Latin rhotacism is the standard case: intervocalic
/s/ became /r/ inside a morpheme and did not across a compound seam, where it
stands between the same two vowels. Any environment stated in features alone is
wrong on half of the data.

Two axes carry it, both derived in `search.c` from the `morpheme_breaks` the
caller supplies on the form:

- `morphological` — where the segment sits in its own morpheme: `initial`,
  `final`, `internal`, `only`.
- `morpheme_index` — which morpheme, counted from the start of the word: `0`,
  `1`, … A rule stated in it does not travel between a suffixing language and a
  prefixing one, where the same index is a different thing.

Both conjoin with the feature predicates through the ordinary refinement
machinery, so "intervocalic and morpheme-initial" is one environment rather
than two rules.

**regulae does not segment words and will not start.** The boundaries are
input. A corpus that carries none gets no morphological axis at all — the
candidate list is built from the values actually observed, so the axis costs
nothing where the data is silent, which is most corpora. Morphological analysis
belongs to the package upstream of this one.

Boundaries reach the model from the wide loader's `<lect>_breaks` column and,
since 2026-08-15, the long loader's `breaks` column.

### Pricing the search, not only the parameter

Widening the vocabulary widens the argmax, and BIC does not price an argmax. It
charges for one added term against the likelihood it buys; the term that
survives a split step is the best of *m* of them, and the maximum of a hundred
candidates clears its bar by chance far more often than one does.

So the split penalty carries `γ · 2 · ln(m)` alongside it — the standard
extended-BIC shape for a large model space, in the same currency as the BIC
term. `bic.search_penalty_gamma` defaults to 0.5, which is the largest fixed
value at which no sound law in `testdata/soundlaws/` is lost: at 0.75 the
multi-lect stage stops committing on the palatalization corpus, at 1.0 lenition
finds only two of its three stops.

On Grassmann's law this takes the output from five conditioned classes — one
the law, four environments fitting the same partition — to one, correctly
stated as *an aspirate somewhere later*.

#### The margin a rule clears, and the level noise reaches

Every conditioned rule publishes `search_margin`: the γ its evidence could
carry and still commit. It is comparable across corpora, and it is comparable
against the same number measured on the corpus **shuffled** — which
`--permutations` reports as `null_search_margin`, the p95 of what the search
reaches in data with no correspondences left in it.

That comparison is the per-rule verdict, and it agrees with linguistic
expectation wherever the answer is known:

| corpus | strongest rule | noise reaches | reading |
| --- | --- | --- | --- |
| `rounding_harmony` | 7.75 | 0.62 | the law towers over its noise |
| `place_assimilation` | 5.77 | 1.68 | clear |
| `rhotacism` | 3.11 | 1.07 | clear |
| `grimm` (unconditioned) | 0.91 | 0.79 | near-noise, which is correct |
| `verner` | 1.54 | 1.80 | **below** — see below |
| unrelated pseudo-words | 1.48 | 1.15 | indistinguishable, correctly |

Verner is the instructive one. Its stress-conditioned rules are real and its
corpus is forty sets, and on forty sets the search finds artefacts stronger
than the law. The margin does not hide that; it reports it.

#### Tuning the charge from the corpus

`tune_search_penalty` (CLI `--tune-search`, requires `--permutations`) sets γ to
`null_search_margin` instead of the default: what the shuffles reached becomes
what a rule has to beat. The shuffled runs are trained with **no** charge at
all, because what they measure is how high an unpriced search can reach.

This buys precision with recall, and the trade is steep. Conditioned classes,
default charge against tuned:

| corpus | default | tuned |
| --- | --- | --- |
| unrelated pseudo-words | 26 | 2 |
| `grimm` (unconditioned) | 4 | 0 |
| `ppn_hawaiian` | 4 | 0 |
| `rhotacism` | 3 | 3 |
| `rounding_harmony` | 1 | 1 |
| `place_assimilation` | 2 | 2 |
| `verner` | 5 | **0** |
| `lenition` | 4 | 1 |
| `latin_spanish` | 20 | 3 |

The first three rows are the case for it: noise collapses, an unconditioned law
stops being given conditioning, and a corpus whose rules sit below its own
noise says so. The `verner` row is the case against: a real law, entirely
suppressed, on a corpus too small for it to stand above the search. It is off
by default for that reason, and it is a per-corpus decision a reader can make
from the margins the default run already prints.

### What a committed split publishes

Every conditioned row carries the comparison that produced it, not just its own
side of it: `contrast_count` (the same correspondence in the observations where
the environment does not hold), `contrast_total` (that side's denominator, on
the pairwise row), and `delta_bic` (what the split scored). The complement is
already in hand at commit time — it is `best_no` — so this costs nothing beyond
carrying it.

This is what makes a row readable. `latin:s ~ old_latin:s` before a vowel, with
count 6, looks like a rule until the complement shows 26 of the same
correspondence outside that environment. It cleared the BIC gate; it is still
not a statement about conditioning, and no reader could have seen that from the
count.

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
computed by a minimal sonority-based syllabifier: the max-onset
principle under the sonority sequencing principle. Language-agnostic
and simple by design — supply your own with a `syllables` column
(long format) or `<lect>_syllables` (wide), and they are respected
unchanged. Until 2026-08-15 the docs told users to do that and no
loader could read the column.

The scale is stop < fricative < nasal < liquid < glide < vowel, and it
reads three things that matter typologically:

- **A segment marked `syllabic` is a nucleus**, whatever its manner.
  Without that test a syllabic consonant was a nucleus only when its
  manner happened to clear the peak threshold, so `l̩` and `r̩` were and
  `n̩`, `m̩` and `s̩` were not — a distinction with nothing behind it.
- **Clicks and implosives are stops.** merkmal says so with the
  features `click` and `implosive`; the scale did not test them, so
  they fell through to the unknown score, which is the *nasal* value.
  Every click sat above every fricative in the hierarchy, in exactly
  the languages that have clicks.
- **A form with no nucleus of its own** — no vowel, no syllabic
  consonant — is still given one, because the predicates need
  something to hold of, but the model counts it.
  `rg_corpus_fit.inferred_nucleus_form_count` says how many forms that
  was, and the human report prints it. Vowelless words are real and
  their analyses differ; making the guess silently was the problem.

Two properties of the syllable itself join its feature union, so they
conjoin with segment predicates through the ordinary machinery:
`syllable_shape` (`open`/`closed`) and `syllable_nucleus`
(`long`/`short`). They are named for what they measure rather than for
*heavy* and *light*, which are language-particular verdicts — CVC is
heavy in Latin and is not in every quantity system.

A caveat worth stating: in a corpus of segments, syllable shape is
usually *also* describable segmentally, and the search often prefers
the segmental description because it reaches it first. A vowel
lengthening in an open syllable comes back as
`fol[consonant:+] fol@2[vowel:+]`, which is the same partition. The
shape predicate is a generalisation over that, not new evidence, and
it wins only where no bounded segmental description coincides with
it.

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
