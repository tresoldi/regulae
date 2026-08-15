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
  phonological context environments, discovered via BIC-gated
  context splitting at the class level.
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
  BIC-driven search over a vocabulary derived from the corpus
  itself. regulae holds no list of feature names: a segment's
  features are whatever the merkmal system in use reports, so
  the categorical systems and the valued ones (`phoible`,
  `pbase-*`) both work. A fixed list fits the language its
  author had in mind and silently loses the rest.
- **A margin per rule, and the level noise reaches.** Every
  conditioned rule reports how heavy a search charge its
  evidence could carry; `--permutations` reports the same
  number measured on the corpus shuffled. A rule at or under
  that level was findable in data with no correspondences in
  it. `--tune-search` turns the comparison into a gate. Every committed
  rule publishes the contrast it was measured against — the same
  correspondence where the environment does not hold — and the
  ΔBIC it scored, because a conditioning claim without its
  complement cannot be read.
- **Long-range conditioning.** A second split loop looks for
  conditioning environments beyond the immediate neighbours —
  distance-bounded, existential (*somewhere preceding /
  following*), and syllable-structural. Captures umlaut,
  harmony, and related phenomena.
- **Tonal correspondences** and **cross-dimensional rules**
  (segmental features predicting suprasegmental values, with a
  BIC-gated search, joint-predictor support, and a
  non-interaction guard).
- **Multi-lect reconciliation.** Union-find over pairwise
  alignment positions, yielding multi-lect classes that bind
  N lects in one row.
- **Non-cognate outlier diagnostics** via
  `rg_find_cognate_outliers`.
- **A fit statistic, and a baseline to read it against.** Every model
  carries `rg_multi_model_fit`: the mean alignment cost per segment,
  and — with `--permutations <n>` — the same statistic over trainings
  on a corpus whose pairings have been shuffled. This is the only
  thing in the output that answers "is there a relationship here at
  all". The class counts do not: shuffling removes every
  correspondence there is to find and the counts go **up**.
- **Determinism.** Same input, same output across runs and
  processes.

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

## Command line

```sh
regulae train <corpus.tsv>              # machine-readable model summary
regulae train --human <corpus.tsv>      # readable report
regulae train --permutations 20 <corpus.tsv>   # calibrate against shuffled data
regulae train --pairwise <corpus.tsv>   # per-pair learned tables
regulae align <corpus.tsv>              # alignments, prior-only
regulae align --model <corpus.tsv>      # alignments under the trained model
regulae outliers --top-k 10 <corpus.tsv>
```

`--format tsv|wide|gled|arcaverborum` selects the input format
(default `tsv`). The generic TSV reader expects `cognate_id`,
`lect_id` and `segments` columns, with optional `confidence`,
`breaks` (morpheme boundaries) and `syllables`; a cognate set
takes the lowest confidence any of its rows reports.

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
rg_corpus_load_tsv("cognates.tsv", NULL, &corpus);
rg_train_options_init_defaults(&options);

/* Multi-lect training: the canonical entry point. */
rg_train_model(ctx, rg_corpus_cognates(corpus),
               rg_corpus_cognate_count(corpus), &options, &model);

/* Read the reconciled classes. Rows are borrowed from the model. */
for (size_t i = 0; i < rg_multi_model_unconditioned_class_count(model); i++) {
    const rg_multi_class_row *row = rg_multi_model_unconditioned_class_at(model, i);
    for (size_t j = 0; j < row->segment_count; j++) {
        printf("%s%s:%s", j ? " ~ " : "", row->lect_ids[j], row->graphemes[j]);
    }
    printf("  count=%.0f\n", row->count);
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
moves on any layout, signature or ownership change.

## Documentation

- **Guide** at [`docs/GUIDE.md`](docs/GUIDE.md) — what the output means and
  how to read a conditioned environment. Also the source for the in-page
  walkthrough.
- **What regulae reads** at [`docs/capabilities.md`](docs/capabilities.md) —
  generated: which capabilities are implemented, and which corpora the loaders
  can currently express. A corpus it cannot read says something about the input
  path, not the engine.
- **Conversion state** at `docs/c_conversion_roadmap.md`
  (milestones, intentional deviations, why parity ended) and
  `docs/c_conversion_handoff.md`.
- **Design documents** at `docs/training_pipeline.md`,
  `docs/correspondence_discovery.md`, and
  `docs/alignment_details.md` — rationale for the staged
  training pipeline, discovery mechanisms, and data types
  (language-agnostic; written against the original design).
- **Consumer guide** at `docs/consumer_guide.md`.
- The original Python implementation is archived under
  `python/` for reference.

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
