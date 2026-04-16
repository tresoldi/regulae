# regulae

Pairwise and multi-lect phonological alignment for the new
historical linguistics framework.

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
  BIC-driven search over a feature vocabulary. Refinements
  are allowed within committed splits.
- **Long-range conditioning.** A second split loop looks for
  conditioning environments that sit beyond the immediate
  neighbours — distance-bounded (*n* positions away),
  existential (*somewhere preceding / following*), and
  syllable-structural (*same / next / previous syllable*).
  Captures umlaut, harmony, and related phenomena.
- **Tonal correspondences.** Source → target tone mappings
  aggregated across 1-to-1 links.
- **Cross-dimensional rules.** Segmental features predicting
  suprasegmental values (or vice versa), committed via a
  BIC-gated search with support and confidence floors. Joint
  predictors (conjunctions of two source features) are
  supported with a non-interaction guard that prevents
  "passenger" commits.
- **Multi-lect reconciliation.** Union-find over pairwise
  alignment positions, yielding multi-lect classes that bind
  N lects in one row rather than forcing you to read five
  pairwise reflexes separately.
- **Non-cognate outlier diagnostics.** A post-hoc
  `find_cognate_outliers` helper ranks cognate sets by
  alignment-cost z-score under the trained model. The
  framework never silently drops suspect cognates; the
  diagnostic is a tool for the user to audit their own
  corpus.
- **Determinism.** Same input, same output, bitwise, across
  runs and across processes.

## Install

```sh
pip install -e .
# merkmal must be installed (sibling directory):
pip install -e ../merkmal
```

## Run tests

```sh
python -m pytest -q
```

Current test count: **427 passing, 2 xfailed** (pre-existing).

## Run experiments

Each experiment is a standalone script under `experiments/`
that trains a model on a specific corpus and prints a
human-readable report:

```sh
# Pair experiments:
python experiments/latin_spanish/run_experiment.py
python experiments/latin_french/run_experiment.py
python experiments/latin_italian/run_experiment.py
python experiments/oe_english/run_experiment.py
python experiments/ppn_hawaiian/run_experiment.py

# Multi-lect experiments:
python experiments/gled_romance/run_experiment.py
python experiments/gled_polynesian/run_experiment.py
python experiments/arcaverborum_polynesian/run_experiment.py

# Tonal and cross-dimensional fixtures:
python experiments/tone_synthetic/run_experiment.py
python experiments/tone_yoruba_like/run_experiment.py
python experiments/tone_vietnamese_like/run_experiment.py
python experiments/tone_chinese_like/run_experiment.py
python experiments/tone_chinese_like_clean/run_experiment.py
python experiments/tone_3way_synthetic/run_experiment.py

# Long-range context fixtures:
python experiments/umlaut_synthetic/run_experiment.py
python experiments/harmony_synthetic/run_experiment.py
```

Each experiment directory carries a `findings.md` with the
interpretation of its output and notes on what the framework
recovered versus what it missed.

## Quick API tour

```python
from regulae import (
    CognateSet,
    Form,
    Segment,
    train_model,
    cognate_sets_from_pairs,
    find_cognate_outliers,
    format_multi_lect_model,
    load_gled,
    load_arcaverborum,
)

# Multi-lect training from a loader:
corpus = load_gled(
    "/path/to/gled.tsv",
    family="Indo-European",
    doculects={"LATIN", "SPANISH", "FRENCH_2"},
)
model = train_model(corpus)
print(format_multi_lect_model(model, top_conditioned=20))

# Post-hoc outlier check:
for report in find_cognate_outliers(corpus, model, top_k=10):
    print(f"  z={report.z_score:+.2f}  {report.cognate_id}")

# Drill into one lect's behaviour:
from regulae import describe_multi_lect_class
print(describe_multi_lect_class(model, "LATIN", "w"))
```

The pair-based input form is still accepted as a convenience
for two-lect corpora, with a deprecation warning:

```python
# Pair input (emits DeprecationWarning):
pairs = [(latin_form, spanish_form), ...]
legacy_model = train_model(pairs)  # returns LearnedModel

# Recommended form:
cogs = cognate_sets_from_pairs(pairs, ("latin", "spanish"))
multi = train_model(cogs)
legacy_model = multi.pairwise_models[frozenset({"latin", "spanish"})]
```

## Documentation

- **Tutorials** at `docs/tutorials/` — a five-part series for
  historical linguists. Read these first if you are new to
  the package.
- **Consumer guide** at `docs/consumer_guide.md` — the
  authoritative external contract for downstream consumers of
  a trained model.
- **Design documents** at `docs/training_pipeline.md`,
  `docs/correspondence_discovery.md`, and
  `docs/alignment_details.md` — rationale for the staged
  training pipeline, discovery mechanisms, and data types.
- **Framework-level context** in the `framework/` sibling
  repository — lect theory, asymmetry, historical inference
  design. Read when making cross-package design decisions.

## What is not in this package

- Cognate detection. Cognate sets are user input.
- Historical inference. Reconstruction, phylogeny, dating,
  relatedness-as-posterior all belong to a separate
  downstream package.
- Morphological segmentation. A reserved slot exists on
  `CognateSet` for future morph-aware alignment, but the
  current pipeline is purely phonological.
- Directionality. The alignment search is symmetric. Who
  inherited from whom, who borrowed from whom, is not a
  question `regulae` answers.
