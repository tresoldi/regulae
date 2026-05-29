# regulae

Pairwise and multi-lect phonological alignment for the new
historical linguistics framework — a Go implementation.

## What this package does

Given user-supplied cognate sets — pairs of forms in the
two-lect case, N-way tuples of forms in the multi-lect case —
`regulae` produces a structured model of the systematic
phonological correspondences between the lects involved.
Nothing more. It does not infer cognates, propose proto-forms,
build phylogenies, or assign dates to changes: those are jobs
for a downstream historical-inference package that consumes
this one's output.

What you get back from a training run is a `MultiLectModel`
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
- **Non-cognate outlier diagnostics** via `FindCognateOutliers`.
- **Determinism.** Same input, same output across runs and
  processes.

## Dependencies

`regulae` is built on the Go port of
[merkmal](../merkmal/go) for phonological feature lookups and
segment distances; the dependency is wired via a `replace`
directive in `go.mod`.

## Build and test

```sh
go build ./...
go test ./...
go vet ./...
```

## Run experiments

Each experiment is a standalone `package main` under
`experiments/` that trains a model on a specific corpus and
prints a human-readable report:

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

```go
import regulae "github.com/tresoldi/regulae"

// Build a corpus of cognate sets (or use a loader).
corpus := regulae.CognateSetsFromPairs(pairs, [2]string{"latin", "spanish"}, "ls")

// Multi-lect training (the canonical entry point):
model, err := regulae.TrainModel(corpus, regulae.DefaultTrainOptions())
if err != nil { /* e.g. *regulae.UnknownGraphemeError */ }
fmt.Println(regulae.FormatMultiLectModel(model, 20, 20))

// Drill into one pair's learned model:
if pm, ok := model.PairwiseModel("latin", "spanish"); ok {
    fmt.Println(regulae.FormatModel(pm, regulae.DefaultFormatModelOptions()))
}

// Post-hoc outlier check:
reports, _ := regulae.FindCognateOutliers(corpus, model, 10, 0)
for _, r := range reports {
    fmt.Printf("  z=%+.2f  %s\n", r.ZScore, r.CognateID)
}

// Drill into one lect's behaviour:
fmt.Println(regulae.DescribeMultiLectClass(model, "latin", "w"))
```

Loaders are provided for generic TSV, GLED, and arcaverborum
data: `LoadCognatesFromTSV`, `LoadGLED`, `LoadArcaverborum`.

## Documentation

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
