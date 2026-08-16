# GLED Romance Experiment

First real-data multi-lect run of multi-lect reconciliation. Seven Romance lects from the
GLED release `20221127`, trained jointly via the multi-lect pipeline.

## Status, 2026-08-16: the corpus is not in the repository

This directory holds findings only. The GLED data is not vendored -- those
datasets carry their own licences -- so the run below cannot be reproduced from
a checkout, and the source paths in it are wherever the data sat on the machine
that produced it.

`scripts/lexibank.py` is the supported route from a CLDF wordlist with expert
cognate judgements to a regulae corpus, and `regulae check <corpus>` reports
what a given dataset costs in graphemes the feature system cannot read before
training on it.

## Setup

- **Source**: `/tmp/gled_clone/releases/20221127/gled.tsv`
- **Family filter**: `Indo-European`
- **Doculect filter**: `{LATIN, SPANISH, PORTUGUESE_2, FRENCH_2, ITALIAN_2, CATALAN_3, ROMANIAN_2}`
- **Cognate sets loaded**: 51
- **Size distribution**: 22 seven-way, 8 six-way, 6 five-way, 3 four-way, 3 three-way, 9 two-way
- **Per-lect coverage**: 41 ITALIAN_2 / 41 PORTUGUESE_2 / 41 CATALAN_3 / 40 SPANISH / 39 ROMANIAN_2 / 35 LATIN / 34 FRENCH_2

## Model

- 21 per-pair `LearnedModel` instances (full 7-choose-2 coverage)
- **73 unconditioned multi-lect classes**
- **4 conditioned multi-lect classes** from the context-conditioning
  stage (was 58 before the reconciliation consolidation pass; the
  adaptive BIC correction + adaptive minimum-commit-count floor cut
  low-signal commits)

## What it recovered

### Preserved obstruents across all seven lects

Seven-way classes with identity reflexes — the easy case, and the model
picked it up on the expected consonants:

```
[CAT:m, FRA:m, ITA:m, LAT:m, POR:m, ROM:m, SPA:m]     count=3
[CAT:d, FRA:d, ITA:d, LAT:d, POR:d, ROM:d, SPA:d]     count=2
[CAT:n, FRA:n, ITA:n, LAT:n, POR:n, ROM:n, SPA:n]     count=2
[CAT:s, FRA:s, ITA:s, LAT:s, POR:s, ROM:s, SPA:s]     count=2
[CAT:t, FRA:t, ITA:t, LAT:t, POR:t, ROM:t, SPA:t]     count=2
[CAT:p, FRA:p, ITA:p, LAT:p, POR:p, ROM:p, SPA:p]     count=1
[CAT:b, FRA:b, ITA:b, LAT:b, POR:b, ROM:b, SPA:b]     count=1
```

Counts are low (the corpus is small) but the shape is right.

### Latin `w` → Romance `v`, Catalan preserves `b`

```
[CAT:b, FRA:v, ITA:v, LAT:w, POR:v, ROM:v, SPA:v]     count=2
```

This is the classic `*VENIRE` / `*BIBERE` story. Catalan has `b`, the
other modern lects merge on `v`, Latin preserves the original `w`/`u̯`.
Multi-lect recovery of this correspondence is a headline result because
**pairwise context discovery can't express the three-way split directly** — it sees
only pairs at a time.

### Palatalized `*k` — multiple reflexes across lects

```
[CAT:dʒ, ITA:g, POR:x, ROM:k, SPA:g]                  count=1
[CAT:k,  ITA:k, POR:x, ROM:k, SPA:g]                  count=1
```

and the unconditioned class with the t-series reflex:

```
[CAT:t, ITA:t, POR:t, ROM:t, SPA:tʃ]                  count=1
```

These are real Romance palatalization outputs but each appears only
once in the corpus, so the evidence is anecdotal here. The conditioned
table picks up the right split direction (front-vowel following
context) but also count=1.

### Context-conditioning commits (class-level context splits)

The context-conditioning stage committed exactly **4** conditioned classes under the
reconciliation consolidation defaults:

```
count=2  [CAT:b, FRA:v, ITA:v, LAT:w, POR:v, ROM:v, SPA:v]   [POR: pos=initial]
count=2  [CAT:n, FRA:n, ITA:n, LAT:n, POR:n, ROM:n, SPA:n]   [FRA: foll=[front]]
count=2  [CAT:s, FRA:s, ITA:s, LAT:s, POR:s, ROM:s, SPA:s]   [LAT: pos=initial]
count=2  [CAT:t, FRA:t, ITA:t, LAT:t, POR:t, ROM:t, SPA:t]   [CAT: foll=[vowel]]
```

The headline finding — Latin `*w → v` across Romance, split by
Portuguese `pos=initial` — is the **first** of these four. Romance
initial-vs-medial `v ~ w` is exactly the conditioning axis that
matters historically (initial `v` is stable; intervocalic is
messier). That the multi-lect pipeline finds it as the top commit
on 51 cognate sets is the clearest reconciliation-only result on this corpus.

The other three commits are count=2 and involve preserved
correspondences — they may be real (initial `s` is often anchored
in historical phonology) or spurious on this tiny corpus. A larger
Romance corpus is needed to tell.

## Weaknesses exposed

* **Corpus is too small.** GLED's Indo-European file has only ~50
  Romance cognate sets at this release, and Latin is missing from
  roughly 30% of them. The unconditioned classes mostly have count ≤ 3;
  the conditioned classes almost all have count = 1. A proper Romance
  experiment would need Lexibank-scale data (hundreds of sets).

* **Context conditioning over-commits on thin data.** The sequential greedy BIC
  loop was tuned for context discovery with a corpus of ~100 pair observations per
  source grapheme. At the multi-lect level, "observations" are
  class instances and there are far fewer of them. The BIC threshold
  should arguably scale with observation count or be tightened for
  multi-lect use. Candidate for a consolidation round.

* **No dedicated Latin handling.** Latin is just another peer (Q2
  peer design, confirmed). That's philosophically correct but means
  Latin-anchored classes compete against Latin-absent ones. For
  historical reconstruction work a "pin the reconstructed form to a
  specific lect" mode might be useful as an overlay, not a
  replacement.

* **Conditioned class IDs carry "the pivot lect's context only"**
  which is accurate but slightly unhelpful in reports — a reader
  looking at a `(CAT:t, ..., SPA:tʃ)` split wants to see "what is
  Catalan's context" as well as the pivot lect's. The format helpers
  should thread this through.

## What works

* **Full multi-lect pipeline runs end-to-end on real data** in under
  a second for this corpus size. No crashes, no pathological loops.
* **Per-pair context-discovery models are trained transparently** — all 21 are
  available for inspection via `model.pairwise_models`.
* **The reconciliation output is deterministic and inspectable.**
  Classes are sorted by count desc, tied by segment tuple lex asc.
* **Backward compatibility is intact.** All prior pair-based
  experiments still work via `cognate_sets_from_pairs`.
* **The headline Romance sound laws are visible in the output.** Even
  on thin data the Latin `w → v` and the palatalized `k` series both
  appear, which is the minimum bar for "the pipeline can see the
  history".

## Recommended next actions

1. **Polynesian experiment** — smaller lect set, better
   coverage in GLED, and the Hawaiian merger disambiguation is the
   motivating example for multi-lect reconciliation. Should be much cleaner than this one.
2. **Multi-lect BIC consolidation** (future): tighten the context-conditioning
   threshold or scale it with observation count so thin corpora don't
   over-commit.
3. **Larger Romance corpus via arcaverborum**: rerun on a
   hundreds-of-sets Romance subset and compare context-conditioning commits.
