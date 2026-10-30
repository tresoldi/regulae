# Ensembles, soft cognacy, partial cognacy and lost-trigger traces

**Status (2026-10-30):** the ensemble, temperature calibration, soft cognacy and the lost-trigger record and per-morpheme partial cognacy are implemented (`rg_train_ensemble`, `rg_train_ensemble_soft`, `rg_calibrate_alignment_temperature`, `rg_ensemble_to_json`, `rg_multi_model_lost_triggers`, `rg_form.word`; CLI `regulae train --ensemble N [--alignment-temperature calibrate] [--cognacy-pairs pairs.tsv]`; Python `regulae.train_ensemble`; the TSV loader's `morpheme_cognates` column), on branch `claude/ensembles`.
This is Stage C of `tresoldi/framework` `roadmap_beyond_columns.md`. The
constitution requires regulae's egress to historia to be *ensembles or weighted
samples* (§5.2). A single best-fit system is tier C0, which allows no claim
export; C1, an ensemble, is the minimum (§7.2). Today regulae publishes one
model, and its bootstrap reweights that model's fixed table: it never realigns
or re-searches. This document says what an ensemble member is and what is
exported.

## Decisions (author, 2026-10-30)

1. **A member refits everything and samples its alignments.** Each member
   resamples the observation groups, retrains the whole pipeline (alignment,
   environments, classes), and draws its alignments from the alignment
   distribution instead of taking the single best one.
2. **The weights are a Bayesian bootstrap.** Dirichlet(1, …, 1) weights are
   drawn over the observation groups for each member, and members are equally
   weighted. No likelihood is invented.
3. **Soft cognacy enters the ensemble now.** Given pairwise cognacy weights
   (cognator's `q(pairs)`), each member samples a cognate partition.
4. **In this round, besides the ensemble:** the interchange export
   (`regulae_package`), per-morpheme partial cognacy, and a typed lost-trigger
   trace record.

## One member

For member *m*, with its own seed:

1. **Partition.**
   * Caller-supplied cognate sets are used as given, and the member records
     them as `condition_on`.
   * With pairwise weights, every edge *(a, b)* within a meaning is kept with
     probability *w(a, b)*, independently. The connected components become
     the member's cognate sets.
   * Several forms of one lect in a component are doublets. The corpus already
     handles these by splitting confidence across combinations.
   * Edge independence is an assumption and is stated as such.
2. **Weights.**
   * Draw *g ~ Dirichlet(1, …, 1)* over the observation groups, using the
     same units as `bootstrap_unit`.
   * A set's weight is its confidence × *G·g*, where *G* is the number of
     groups. The weights then have mean 1, so evidence floors and the BIC
     *N* keep their scale on average.
   * Weights can exceed 1. Member training lifts the [0, 1] check for them,
     and only for them. Dirichlet weights are never exactly 0, so the
     pairwise code's "0 means 1" default is never reached.
   * The NML scorer refuses non-integer weights. An ensemble under NML is
     therefore refused with a clear status, not silently changed.
3. **Alignments.**
   * Every call of the alignment DP in the member's training draws a path
     from *p(path) ∝ exp(−cost(path)/T)*, by forward filtering (log-sum-exp
     over the same moves) and backward sampling. The default is *T* = 1.
   * The costs are the DP's own pseudo-surprisals. Their `−log Z` offset
     cancels in the normalisation.
   * The draw for a pair is seeded from the member seed and the two forms, so
     it is reproducible and does not depend on call order. As *T* → 0 the draw
     tends to the best path.
   * EM under sampled alignments is Monte Carlo EM.
4. **Train.** The member runs the normal pipeline on its partition, its
   weights and its alignment draws.

## The ensemble and its summary

* **Matching members.** Members are matched on content, never on `class_id`,
  which is only a rank inside one model. A class is identified by its tuple
  of (lect, grapheme, tone, length, stress). A conditioned row is identified
  by that tuple plus its pivot lect and environment.
* **Summaries.** For every class or row seen in any member: its *inclusion
  probability* (the share of members containing it), its rate (mean and 95%
  percentile interval over the members containing it), and its standing in
  each member where one was measured.
* **Pairwise tables.** The per-pair segment correspondences are summarised
  the same way.

## The interchange export

`regulae_package` (framework `interchange.md` §3.5):
`role: "q_synchronic"`, the producer pin, the feature system, members with
weights and seeds, the per-member partition (form → class), the per-pair
tables, the contexts, and the summaries above, with `approx_hint: "C1"`. It is
a separate JSON writer from the single-model export, which keeps saying it is
not claim-capable.

## Partial cognacy per morpheme

A form may carry one cognate id per morpheme, aligned with its morpheme breaks.
A morpheme-level set aligns only the span of the shared morpheme in each form.
Environments are still read from the whole word, so a morpheme-initial segment
after a vowel in the preceding morpheme is still intervocalic. Forms gain an
optional alignment span, which defaults to the whole form. This is an ABI
change.

## Lost-trigger traces

A conditioned class whose environment holds on the pivot lect can be checked at
the aligned positions of each sister lect. If the predicate is absent there, the
sister lost the trigger, or never had it. The record carries:

* the class tuple, the environment and the pivot lect;
* the sister lect;
* how often the predicate holds at the sister's aligned positions;
* the support.

It is a typed trace for historia's latent-segment proposals (framework
`08_latent_elements.md` §6). It is not a claim about direction.

## First measurements (Latin–Spanish, `testdata/corpora/real_latin_spanish.tsv`, 103 sets, 20 members)

| alignment temperature | classes seen | unconditioned with inclusion ≥ 0.9 | unconditioned < 0.2 | the single model's 6 conditioned rows: inclusion |
|---|---|---|---|---|
| 0 (best path, resampling only) | 280 | 59 | 10 | 0.30, 0.65, 0.45, 0.25, 0.60, 0.10 |
| 0.25 | 283 | 54 | 22 | 0.20, 0.65, 0.50, 0.10, 0.55, 0.00 |
| 1 | 370 | 60 | 106 | 0.15, 0.25, 0.10, 0.00, 0.15, 0.10 |

* **Resampling alone unsettles every environment.** None of the single
  model's conditioned rows survives in more than 65% of members even at
  temperature 0. The single-model export does not show this; the ensemble is
  the first place it is visible.
* **Read by correspondence, the conditioning is stable** (`summary.conditioned_correspondences`,
  temperature 0). *o ~ o* is conditioned in 100% of members, *u ~ o* in 95%,
  *p ~ b* in 75% and *k ~ θ* in 65%. The environment that conditions each
  varies between near-equivalent predicates (`consonant` and `consonantal`).
* **Temperature 1 is too hot for these costs.** The DP costs are
  pseudo-surprisals, not a normalised likelihood. At T = 1 the sampled
  alignments add about a hundred classes that appear in under 20% of members
  (alignment noise such as *latin:e ~ spanish:r*). The temperature is
  therefore calibrated (next section).
* **Cost:** 30 s (T = 0) to 60 s (T = 1) for 20 members in a release build.

## Temperature calibration (author's decision, 2026-10-30)

`rg_calibrate_alignment_temperature` trains with group-held-out predictive
folds at each temperature on a grid (0, 0.05, 0.1, 0.25, 0.5, 1). Each
temperature is averaged over sampling seeds, and the one with the lowest mean
held-out log loss of conditioned reflexes is kept, ties going to the colder
temperature. Held-out reflexes are always scored on the cheapest alignment
(`best_path_options` in `predictive.c`), so only the training changes with the
temperature and the comparison is fair. `--alignment-temperature calibrate`
runs it before an ensemble, and the scores go into the package under
`ensemble.temperature_calibration`.

**Latin–Spanish** (5 folds, 3 seeds per temperature):

| T | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|---|---|
| held-out log loss | **0.769** | 0.772 | 0.773 | 0.793 | 0.857 | 0.978 |

**Romance, 4 lects** (`real_romance_4lect.tsv`, same grid; T = 0 is
deterministic, so it is run once):

| T | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|---|---|
| held-out log loss | **1.051** | 1.058 ± 0.009 | 1.063 ± 0.008 | 1.060 ± 0.013 | 1.115 ± 0.031 | 1.391 ± 0.039 |

On both corpora held-out loss rises with temperature, so the calibrated
temperature is 0. On Romance, 0.05–0.25 sit within about two seed SDs of
each other but all above 0. The 4-lect run took about 2.5 hours in a release
build. On this corpus, single sampled alignments make the model predict held-out
reflexes worse. The ensemble's uncertainty then comes from resampling and,
with soft cognacy, from the partition. A likely reason is that each training
run sees one path per pair. Expected counts over all paths (soft EM, from the
same forward pass plus a backward one) would use alignment ambiguity without
that noise, and are the natural next step if alignment uncertainty is wanted
in the counts.

## Soft cognacy, as built

`rg_soft_corpus_load` reads a forms table (`form_id lect_id meaning segments`,
with the TSV loader's optional tone, stress and breaks) and a pairs table
(`form_a form_b weight`). `rg_train_ensemble_soft` samples one partition per
member, as described under "One member", and the observation groups are the
meanings. When calibration is asked for, it runs on the partition of edges with
weight ≥ 0.5. The package says `condition_on: sampled_partitions`, and each
member lists its `cognate_sets` (the form ids of each set).

**Romance check:** the 4-lect corpus (381 forms) was given its true pairs at
weight 0.9 and 30 decoy edges between meanings at 0.3. Ten members at
temperature 0 took 68 s. They held 86–93 sets each (decoys sometimes merge two
meanings), with mean set size 3.96 (at most 4, one form per lect).

## Lost-trigger traces, as built (ABI 48)

`rg_multi_model_lost_triggers` returns one `rg_lost_trigger_row` per
(conditioned class, pivot environment, sister lect), computed in training
from the positions the model already keeps for each class (`src/traces.c`).
Only positions where the environment holds on the pivot are read: a class
merged from two pivots carries a context from each, and a position need
satisfy only one. At each such position the environment is tested at the
sister's aligned segment in the sister's own form; `tested` and `holds` are
the cognate-confidence weights, and a sister gap goes to `untestable`. The
model export carries the rows under `lost_triggers` (with `holds_rate`). The
package matches them across members on the conditioned class's content, the
pivot and the sister, and summarises the rate under
`summary.lost_triggers`.

**Law of palatals in miniature** (`tests/c/test_lost_triggers.c`). Greek and
Latin keep *e*. Sanskrit palatalises *k* before *e* and *i* and then merges
*e* into back *ɑ*. The class *k ~ k ~ c* before a front vowel, on the Greek
pivot, holds at every Latin position (1.0) and at half the Sanskrit ones
(0.5): the *ci* words keep the trigger, the *cɑ* words lost it. The same
corpus also traces *ɑ ~ e* after a palatal on the Sanskrit pivot at 0.0 in
Greek. The record is not directional, and says so.

**Latin–Spanish** (`real_latin_spanish.tsv`, single model). Six conditioned
classes trace. *l ~ ʎ* before a voiced segment (*collum > cuello*,
*capillus > cabello*, *stella*, *caballus*, 4 sets) holds at 0.0 in Spanish:
the trigger is the second half of the Latin geminate, which Spanish lost.
*p ~ b* and *t ~ d* after a vowel hold at 1.0, since lenition left its
trigger in place. *u ~ o* after a short-nucleus syllable holds at 0.91.
Whether Spanish vowels fail "voiced" here is a property of the feature system
in use, and a reader of the trace has to check the environment it names.

## Partial cognacy, as built (ABI 48)

`rg_form` gains `word` and `word_offset`. A form with `word` set is one
morpheme of that word: its `segments` are the span's, the alignment DP runs
over the span alone, and every environment is built on the whole word at the
span's offset. That covers the DP's link contexts, the per-position contexts
class discovery and the predictive folds read, and the cross-dimensional
neighbour reads (`model_crossdim.c`, `search_cost.c`). A span that does not
match its word segment for segment, or runs past its end, is refused with
`RG_ERR_INVALID_ARGUMENT`. `word` NULL is the old behaviour, so every corpus
hashes as before. Callers that fill an `rg_form` field by field must now zero
it; the in-tree tests that did not were fixed.

The TSV loader reads an optional `morpheme_cognates` column: one id per
morpheme of the row's `breaks`, with `-` for a morpheme that has no set. Such a
row's morphemes go to the sets they name, the row's own `cognate_id` may be
empty and is not used as a set, and the whole word is kept by the corpus for
the spans to point at. A count that does not match the morphemes is a parse
error with the line.

**Check** (`testdata/corpora/partial_cognacy.tsv`,
`tests/c/test_partial_cognacy.c`). There are 12 compounds whose second
morpheme starts with *p*, and lect *b* voices that *p* after the first
morpheme's vowel. There are also 12 simplex words whose *p* is word-initial
and stays *p*. Read alone, the span's *p* has no left neighbour and no
boundary, so it cannot be told from the simplex *p*. Read in its word, the
model finds *a:p ~ b:b* conditioned and states it as morpheme-initial. Its
rivals are the readings that cross the boundary: preceding vowel, preceding
sonorant, morpheme index 1, and an open previous syllable.

**Open:**
* Morpheme sets cut from one word are not independent, but each is its own
  observation group in the bootstrap. A consumer that wants them resampled
  together can tie them with `etymon_group`.
* The wide, GLED and soft loaders do not yet read per-morpheme ids.
