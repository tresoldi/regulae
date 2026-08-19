# regulae

Pairwise and multi-lect phonological alignment for the new
historical linguistics framework.

The implementation is a C99 core library with a C ABI. It was
ported from a Go implementation, which served as a frozen
executable reference until merkmal 1.0 changed the feature
system out from under it; the original Python is archived under
`python/`. See `docs/c_conversion_roadmap.md` for the conversion
state.

## What this package does

Given user-supplied cognate sets — pairs of forms in the
two-lect case, N-way tuples of forms in the multi-lect case —
`regulae` produces a structured model of the systematic
phonological correspondences between the lects involved.
Nothing more. It does not infer cognates, propose proto-forms,
build phylogenies, or assign dates to changes: those are jobs
for a downstream historical-inference package that consumes
this one's output.

What you get back from a training run is a multi-lect model
carrying:

- **Per-pair learned models** for every directed pair of lects
  with enough shared cognate data. Each carries an
  EM-trained segment correspondence table (Dirichlet-smoothed,
  with context-conditioned splits discovered automatically), a
  feature-level displacement distribution, a phrase table of
  memorised multi-segment correspondences, and, where
  applicable, tonal and cross-dimensional tables.
- **Unconditioned multi-lect correspondence classes** — tuples
  of one segment per participating lect at corresponding
  positions, with observation counts aggregated across the
  corpus via union-find reconciliation.
- **Conditioned multi-lect classes** carrying per-lect
  phonological context environments, discovered via a shared
  categorical split criterion at the class level. Corrected BIC is
  the M3-selected default; exact NML and a Dirichlet marginal
  likelihood are available for controlled comparisons.
- **Multi-lect cross-dimensional rules** where a segmental
  feature on one lect conditions a suprasegmental value on
  another (the classic case being tonogenesis: voicing on the
  source-lect initial predicts tone on the target-lect
  vowel).

## Capabilities

- **0-to-1, 1-to-many, many-to-many links.** The alignment DP
  handles cluster collapses, vowel splits, and cluster
  insertion or deletion, not just diagonal 1-to-1 alignments.
- **Chunk memoization.** Multi-segment correspondences that
  beat a BIC promotion test land in a phrase table and are
  used directly by subsequent alignments.
- **Reorderings.** A span whose target is its own segments in
  another order is read as a transposition, not as a pile of
  substitutions: each segment corresponds to itself and the
  chunk row records that the order changed. Long-distance
  transposition too — the search takes a wider span when, and
  only when, the two sides really are a permutation.
- **Context conditioning.** Immediate-neighbour splits
  (preceding / following segment feature constraints, word
  position) are discovered automatically by a greedy
  criterion-driven search over a vocabulary derived from the corpus
  itself. regulae holds no list of feature names: a segment's
  features are whatever the merkmal system in use reports, so
  the categorical systems and the valued ones (`phoible`,
  `pbase-*`) both work. A fixed list fits the language its
  author had in mind and silently loses the rest.
- **A verdict per rule.** Every conditioned rule reports how
  heavy a search charge its evidence carries, and — with
  `--permutations` — whether that clears what the same search
  reaches on the corpus shuffled. `--tune-search` turns the
  comparison into a gate. The default categorical charge prices
  all `K−1` parameters added when a `K`-outcome distribution is
  split. On unrelated wordlists from one inventory
  (`testdata/restraint/chance.tsv`) that correction leaves no
  selected conditioned correspondence; the shuffled runs still
  measure how high an unpriced search can reach.
- **A decision list, not a bag of rules.** Discovery is greedy,
  so a later rule refines what an earlier one left unsettled.
  That order is published and the reports render it. Every committed
  rule publishes the contrast it was measured against — the same
  correspondence where the environment does not hold — and the
  named-criterion score, because a conditioning claim without its
  complement cannot be read.
- **Long-range conditioning.** A second split loop looks for
  conditioning environments beyond the immediate neighbours —
  distance-bounded, existential (*somewhere preceding /
  following*), and syllable-structural. Captures umlaut,
  harmony, and related phenomena.
- **Tonal correspondences** and **cross-dimensional rules**
  (segmental features predicting suprasegmental values, with a
  shared categorical-score gate, joint-predictor support, and a
  non-interaction guard).
- **Multi-lect reconciliation.** Union-find over pairwise
  alignment positions, yielding multi-lect classes that bind
  N lects in one row.
- **Non-cognate outlier diagnostics** via
  `rg_find_cognate_outliers`.
- **A fit statistic, and a baseline to read it against.** Every model
  carries `rg_multi_model_fit`: the mean alignment cost per segment,
  and — with `--permutations <n>` — the same statistic over trainings
  on a corpus whose pairings have been shuffled. This measures
  whether the supplied pairings contain more cross-lect structure
  than their shuffled alternative; it is not a genealogical
  verdict. Measured with twelve shuffles on two wordlists drawn
  from one inventory with no history between them: 79
  unconditioned and zero conditioned classes, against a shuffled
  80 and 5.8, and `z = -0.1`. A corpus with a regular surface
  relationship is tens of standard deviations from its shuffles.
- **Determinism.** Same input, same output across runs and
  processes.

## Where this sits against lingrex CoPaR

The field's tool for cross-lect correspondence is lingrex's **CoPaR**
(correspondence-pattern recognition): it clusters the columns of aligned
cognate sets into correspondence patterns and imputes a missing reflex from
them. regulae overlaps with it — both take expert cognate sets, both align
across lects, and both can predict a held-out reflex — and stating the overlap
is the point of this note. What regulae adds is the part a historical linguist
reads a correspondence for: the **environment** each one is conditioned on, the
**decision list** that orders the conditioned rules, and a **standing verdict**
for each against a per-pivot null. A pattern table says which sounds correspond;
regulae says where, in what order the conditions were found, and whether the
conditioning survives its own shuffle.

The multi-lect half of that is recent. Before M11 (see
`docs/multilect_hardening_plan.md`) regulae's conditioned-class discovery
decayed as lects were added: the outcome charge grew with the number of
languages, so a rule recovered at two lects was lost at five, and only the
pairwise stage held at every arity. Since M11 the split is scored per sister
lect, and a conditioned correspondence is recovered at every arity
(`testdata/restraint/arity_2 … arity_5`); the multi-lect class table is a
first-class output rather than a lossy lift of the pairs.

`scripts/benchmark_cloze.py` runs a leave-one-reflex-out cloze — hold out one
lect's reflex in a cognate set, predict it from the rest — for both tools on
shared Lexibank families. It is a reproducible harness, not a leaderboard: the
two predict different units (regulae scores a held-out reflex position by
position, CoPaR imputes an alignment site) and an exact whole-reflex match is a
harsh metric, so the numbers indicate rather than rank. On a Tukanoan-area
Arawakan sample regulae recovers about two-thirds of held-out reflex positions;
run the script for the CoPaR side and other families. The comparison is there
for the boundary above — regulae is not a better pattern recogniser, it is a
correspondence model that also states where each correspondence holds and
whether it stands.

## Dependencies

`regulae` is built on [merkmal](../merkmal) for phonological
feature lookups and segment distances, used through its public
C API only. CMake finds an installed `merkmal::merkmal`, or
falls back to a sibling checkout at `../merkmal`
(override with `-DREGULAE_MERKMAL_SOURCE_DIR=...`).

## Build and test

```sh
cmake -S . -B build/c && cmake --build build/c
ctest --test-dir build/c --output-on-failure
```

Sanitizer builds use `-DREGULAE_ENABLE_SANITIZER=address` (or
`undefined`).

Three corpus suites sit behind `ctest`, and they ask three different
questions. Each has a README explaining how to add to it.

| suite | question | test |
| --- | --- | --- |
| [`testdata/soundlaws/`](testdata/soundlaws/) | can regulae find a change the field settled long ago? | `sound_laws` |
| [`testdata/restraint/`](testdata/restraint/) | can it decline to find one that is not there? | `restraint` |
| [`testdata/diagnostics/`](testdata/diagnostics/) | can it tell you your corpus is wrong? | `diagnostics` |

A tool that passes the first and fails the second is worse than useless,
because a spurious conditioned rule is formatted exactly like a real one.
`restraint/` carries two lects with no history between them, a change spread
over the lexicon rather than over an environment, a borrowed stratum, and a
ladder that measures the evidence floor — about eight examples of a change and
eight counterexamples, below which the search stays quiet rather than guessing.
That figure is for a trigger in the immediate neighbour; a trigger a syllable
away needs five examples a side, not three, and the `distant_*` ladder pins it.

## Command line

```sh
regulae train <corpus.tsv>              # machine-readable model summary
regulae train --human <corpus.tsv>      # readable report
regulae train --permutations 20 <corpus.tsv>   # calibrate against shuffled data
regulae train --predictive-folds 5 <corpus.tsv> # group-held-out reflex evidence
regulae train --pairwise <corpus.tsv>   # per-pair learned tables
regulae align <corpus.tsv>              # alignments, prior-only
regulae align --model <corpus.tsv>      # alignments under the trained model
regulae outliers --top-k 10 <corpus.tsv>
```

`--format tsv|wide|gled|arcaverborum` selects the input format
(default `tsv`). The generic TSV reader expects `cognate_id`,
`lect_id` and `segments` columns, with optional `confidence`,
`breaks` (morpheme boundaries), `syllables`, `etymon_group` and
`source_group`; a cognate set
takes the lowest confidence any of its rows reports.

`etymon_group` names paradigm cells or records that repeat one lexical
history; `source_group` names a publication or transcription batch. regulae
never infers either from identifiers or forms. The bootstrap uses etymon groups
automatically when they are supplied and otherwise reports that cognate sets
are being treated as independent. Source-level resampling is explicit because
collapsing a whole publication to one draw is usually too coarse.

Tone, stress and length can each be given per segment — `tone`,
`stress` and `length` columns in the long format, `<lect>_tone`,
`<lect>_stress` and `<lect>_length` in the wide. Length written
into the grapheme as `aː` is a different claim: merkmal reads
that as its own segment with the `long` feature, which is right
where length is contrastive. The column says the vowel *has* a
length, which is what a lengthening changes.

A lect may appear twice in one cognate set. That is a doublet —
two reflexes of one etymon — and it is 3.2% of cognate-set
members across the Lexibank datasets with expert judgements.
The corpus reads it as one set per combination of reflexes,
each carrying its share of the confidence, so both reflexes are
counted and neither is a second vote.

The `wide` reader takes the shape linguistic data is usually
written in — one row per cognate, one column per lect, holding
whole unsegmented words:

```
gloss   latin   spanish   latin_breaks   spanish_breaks
father  pater   padre     -              -
woman   femina  ember     -              3
```

Words are segmented through merkmal, so multi-codepoint
graphemes survive: `pʰ`, `t͡ʃ`, `kʷ` and a base plus combining
diacritic each stay one segment, where splitting on characters
would break them. The tie bar is what marks an affricate as one
segment: untied `tʃ` is read as two, which is what the
transcription says. A `<lect>_breaks` column supplies morpheme
boundary indices.

Tone written on the word is carried through as the segment's own
dimension rather than as part of the grapheme: `ma³³` gives `m`
and `a` bearing `³³`, and a tone spelled as a separate token —
how CLDF wordlists publish it — attaches to the segment before
it. This reads Chao's superscript digits, which is what the
field writes; an ASCII `1` is not tone notation and is still
reported as an unknown grapheme. A `<lect>_tone` column can
annotate tone per segment instead, and overrides what the word
carries.

Stress is carried the same way. The IPA marks in a word are lifted
onto the syllable nucleus — `ˈpater` accents the first syllable,
`paˈter` the second — and a `<lect>_stress` column annotates it per
segment for corpora that record it separately. Leaving the mark in
the grapheme would make a stressed segment a different segment from
its unstressed self, which splits every correspondence it takes part
in and puts stress out of reach of the conditioning it drives.

## Run experiments

Each experiment under `experiments/` is a curated corpus and a
`findings.md` interpreting what training it produced. Run one
through the CLI:

```sh
regulae train --human --format wide experiments/latin_spanish/cognates.tsv
regulae train --human --format wide experiments/oe_english/cognates.tsv
regulae train --human --format wide experiments/ppn_hawaiian/cognates.tsv
```

The `gled_*` and `arcaverborum_polynesian` experiments read
external corpora and need the data file supplied.

## Quick API tour

```c
#include "regulae.h"

rg_context *ctx = NULL;
rg_corpus *corpus = NULL;
rg_multi_model *model = NULL;
rg_train_options options;

rg_context_new_builtin(&ctx);                 /* owns the merkmal bridge */

/* The last argument is where a failed load says what went wrong and on which
 * line. Pass NULL to decline it. */
rg_load_diagnosis why;
if (rg_corpus_load_tsv("cognates.tsv", NULL, &corpus, &why) != RG_OK) {
    fprintf(stderr, "line %lu: %s\n", (unsigned long)why.line, why.message);
    return 1;
}
rg_train_options_init_defaults(&options);

/* Multi-lect training: the canonical entry point. */
rg_train_model(ctx, rg_corpus_cognates(corpus),
               rg_corpus_cognate_count(corpus), &options, &model);

/* Read the reconciled classes. A table is handed out whole, and the rows are
 * borrowed from the model. */
size_t count = 0;
const rg_multi_class_row *classes = rg_multi_model_unconditioned_classes(model, &count);
for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < classes[i].segment_count; j++) {
        printf("%s%s:%s", j ? " ~ " : "",
               classes[i].lect_ids[j], classes[i].graphemes[j]);
    }
    printf("  count=%.0f\n", classes[i].count);
}

char *text = rg_format_multi_model(model, NULL);   /* caller-owned */
puts(text);
rg_string_free(text);

rg_multi_model_free(model);
rg_corpus_free(corpus);
rg_context_free(ctx);
```

Every call that can fail returns `rg_status`; output pointers
become caller-owned only on `RG_OK`. Accessors return borrowed
pointers valid while the owning handle lives. Loaders are
provided for generic TSV, GLED and arcaverborum data
(`rg_corpus_load_tsv`, `rg_corpus_load_gled`,
`rg_corpus_load_arcaverborum`), and `rg_corpus_from_pairs`
lifts a directed pairwise corpus into cognate sets.

`include/regulae.h` is the API contract; `RG_ABI_VERSION`
moves on any layout, signature or ownership change. It is at
**28**. The most recent breaks, and `docs/consumer_guide.md` §7
has the full list with reasons:

- Each published table is handed out whole — the rows and a count
  — rather than through a count function and an index function.
- What a search decided about a rule (scorer and delta-score, decision index,
  search margin, standing) is one `evidence` member on the row
  rather than four separate fields.
- `rg_corpus_fit` reports the per-pair verdict separately from the
  multi-lect one, because the two are counted differently.
- Predictive validation is a separate, opt-in evidence layer. Set
  `predictive_folds` or pass `--predictive-folds <n>` to select environments
  inside dependency-group training folds and score unseen reflexes. It never
  changes an in-sample split score into a predictive claim.

## Documentation

- **Scientific roadmap** at
  [`docs/surface_relationship_roadmap.md`](docs/surface_relationship_roadmap.md)
  — the product boundary, statistical correction, evaluation programme and
  milestone gates for a calibrated surface relationship model. Its domain
  terms are fixed in [`CONTEXT.md`](CONTEXT.md).
- **Guide** at [`docs/GUIDE.md`](docs/GUIDE.md) — what the output means and
  how to read a conditioned environment. Also the source for the in-page
  walkthrough.
- **What regulae reads** at [`docs/capabilities.md`](docs/capabilities.md) —
  generated: which capabilities are implemented, and which corpora the loaders
  can currently express. A corpus it cannot read says something about the input
  path, not the engine.
- **What it does on real data** at
  [`docs/family_survey.md`](docs/family_survey.md) — a recorded run over forty-odd
  published wordlists, one or two per family, from Indo-European to Pama-Nyungan.
  Regenerate with `scripts/families.py` against your own Lexibank clone; nothing
  is vendored, so it is not checked by CI.
- **Conversion state** at `docs/c_conversion_roadmap.md`
  (milestones, intentional deviations, why parity ended) and
  `docs/c_conversion_handoff.md`.
- **Design documents** at `docs/training_pipeline.md`,
  `docs/correspondence_discovery.md`, and
  `docs/alignment_details.md` — rationale for the staged
  training pipeline, discovery mechanisms, and data types
  (language-agnostic; written against the original design).
- **Consumer guide** at [`docs/consumer_guide.md`](docs/consumer_guide.md) —
  the external contract: what each published field means, what it does not
  mean, how to read a table, and the ABI stability rules. Start here if you
  are building on regulae rather than changing it.
- **Why the modules are shaped this way** at
  [`docs/architecture_plan.md`](docs/architecture_plan.md) — the structural
  pass of 2026-08-15/16, what each phase was allowed to change, and the
  decisions deliberately left open.
- **Static analysis** at `docs/static_analysis.md` — the clang-tidy baseline
  and why each suppression exists.
- The original Python implementation is archived under
  `python/` for reference. It is not built, tested or supported, and the
  tutorials under `docs/tutorials/` were written against it — see that
  directory's README.

## Licence

MIT, matching [merkmal](../merkmal), which regulae is built on. See
[`LICENSE`](LICENSE).

The vendored copy of cJSON in `third_party/cjson/` is also MIT and keeps its
own licence file; it is regulae's only dependency other than merkmal.

## What is not in this package

- Cognate detection. Cognate sets are user input.
- Historical inference. Reconstruction, phylogeny, dating.
- Morphological segmentation. Boundaries are user input. Given
  them, regulae will condition rules on them — Latin rhotacism
  applies inside a morpheme and not across a compound seam, and
  that is expressible — but it will not decide where they are.
- Directionality. The alignment search is symmetric.
