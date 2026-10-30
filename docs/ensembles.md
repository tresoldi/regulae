# Ensembles, soft cognacy, partial cognacy and lost-trigger traces

**Status (2026-10-30): the ensemble is implemented** (`rg_train_ensemble`, `rg_ensemble_to_json`, `regulae train --ensemble N`, `regulae.train_ensemble`); soft cognacy, partial cognacy and the lost-trigger record are next, on branch `claude/ensembles`.
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
  (alignment noise such as *latin:e ~ spanish:r*). The temperature therefore
  needs a principled choice, which is open; the default is still 1.
* **Cost:** 30 s (T = 0) to 60 s (T = 1) for 20 members in a release build.
