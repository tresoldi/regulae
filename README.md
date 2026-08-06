# regulae

Pairwise and multi-lect phonological alignment for the new
historical linguistics framework.

The implementation is a C99 core library with a C ABI. A Go
implementation is kept alongside it as a frozen executable
reference the C port is diffed against, and is removed once the
Python wrapper, WebAssembly path and migration tests are in
place; the original Python is archived under `python/`. See
`docs/c_conversion_roadmap.md` for the conversion state.

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
- **Context conditioning.** Immediate-neighbour splits
  (preceding / following segment feature constraints, word
  position) are discovered automatically by a greedy
  BIC-driven search over a feature vocabulary.
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
`undefined`). `scripts/parity.sh` diffs the C build against the
frozen Go reference over the corpora in `testdata/parity/`; it
needs the Go reference, which needs `../merkmal/go` restored as
described in `docs/c_conversion_handoff.md`.

## Command line

```sh
regulae train <corpus.tsv>              # machine-readable model summary
regulae train --human <corpus.tsv>      # readable report
regulae train --pairwise <corpus.tsv>   # per-pair learned tables
regulae align <corpus.tsv>              # alignments, prior-only
regulae align --model <corpus.tsv>      # alignments under the trained model
regulae outliers --top-k 10 <corpus.tsv>
```

`--format tsv|gled|arcaverborum` selects the input format
(default `tsv`). The generic TSV reader expects `cognate_id`,
`lect_id` and `segments` columns, with an optional
`confidence`; a cognate set takes the lowest confidence any of
its rows reports.

## Run experiments

Each experiment is a standalone `package main` under
`experiments/` that trains a model on a specific corpus and
prints a human-readable report. These still run against the Go
reference; the same corpora are exercised through the C build
by `scripts/parity.sh` and `testdata/parity/real_*.tsv`.

```sh
cd experiments/latin_spanish && go run .
cd experiments/oe_english   && go run .
cd experiments/ppn_hawaiian && go run .
# tonal / cross-dimensional fixtures:
cd experiments/tone_chinese_like_clean && go run .
# long-range context fixtures:
cd experiments/umlaut_synthetic  && go run .
cd experiments/harmony_synthetic && go run .
```

Each experiment directory carries a `findings.md` with the
interpretation of its output. The `gled_*` and
`arcaverborum_polynesian` experiments read external corpora
and expect a data-file path (`go run . /path/to/data`).

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

- **Conversion state** at `docs/c_conversion_roadmap.md`
  (milestones, parity results, intentional deviations) and
  `docs/c_conversion_handoff.md`.
- **Design documents** at `docs/training_pipeline.md`,
  `docs/correspondence_discovery.md`, and
  `docs/alignment_details.md` — rationale for the staged
  training pipeline, discovery mechanisms, and data types
  (language-agnostic; written against the original design).
- **Consumer guide** at `docs/consumer_guide.md`.
- The original Python implementation is archived under
  `python/` for reference.

## What is not in this package

- Cognate detection. Cognate sets are user input.
- Historical inference. Reconstruction, phylogeny, dating.
- Morphological segmentation (a reserved slot exists on
  `CognateSet` but the pipeline is purely phonological).
- Directionality. The alignment search is symmetric.
