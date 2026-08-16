# morph_boundary_synthetic — findings

Synthetic fixture validating the morpheme-boundary chunk filter.

## Status, 2026-08-16: this corpus does not load

`regulae train --format wide experiments/morph_boundary_synthetic/cognates.tsv`
refuses it:

```
regulae: training: "+" is CLDF/CLTS markup, not a transcribed sound.
The gap is in the source data, not in the feature system.
```

The corpus writes the morpheme boundary twice: inline in the transcription
(`pat+a`) and as a segment index in the `proto_breaks` / `derived_breaks`
columns (`3`). The loaders read the columns and refuse the inline `+`, which is
the documented behaviour -- CLDF/CLTS markup is refused separately from an
unknown grapheme precisely so a reader can tell a gap in the source data from a
gap in the feature system. `docs/capabilities.md` lists this corpus under what
the loaders cannot yet express, with `+` named as the blocker.

The results below were produced by a per-experiment Python driver that parsed
the inline notation itself. That tooling is archived under `python/` and is not
built or supported, so this document records a real result that no shipped
command reproduces.

Making it reproducible means dropping the inline `+` and relying on the
`_breaks` columns the corpus already carries. That is a change to the fixture,
not to this note, and has not been made.

## Setup

24 cognate pairs, 16 stem+suffix and 8 monomorphemic controls.

The stem-suffix pairs follow the rule
``stem-final t doubles when a suffix vowel follows``:

    pat+V → pattV   (stem `pat`, suffix V)
    Cat+V → CattV   (analogous for several stem-onset Cs)

The boundary is at position 3 in both forms — between stem `Cat`
and the geminate-onset suffix `tV`. The 8 control pairs
(monomorphemic CVC, identity correspondences) provide the
unconditioned baseline so segment EM has clean signal.

## What the model recovered

Trained twice on the same data:

**Without boundaries** — the chunk extractor promotes 10 chunks,
including 6 that span the morphological seam:

    (at, att), (t, tt), (ta, tta), (ti, tti), (to, tto), (tu, ttu)

These are spurious "phonological" chunks that actually encode
morphology — the gemination rule applied across the boundary.

**With boundaries** — the chunk extractor rejects every candidate
whose source or target span crosses a morpheme break, leaving 5
chunks. All survivors either sit fully within one morpheme or
start exactly at the boundary (e.g., `(i, ti)` describes the
suffix-initial gemination at the morphological seam, but does not
cross it).

The 6 rejected chunks are exactly the morphology-in-disguise ones
that an opaque chunk table would otherwise pass downstream as if
they were phonological correspondences.

## Why this fixture matters

Chunks that straddle morpheme boundaries are a known source of
false phonological-rule reports:

- Latin-Spanish chunk audit (Theme 1 reassessment) showed that
  ~8 of 35 promoted chunks were inflectional-ending reductions
  like `(ate, ad)`, `(ire, ir)`, `(are, ar)`, `(ere, er)` — all
  crossing the stem/suffix seam.
- Romance reanalysis routinely needs to factor out morphological
  alternations before extracting sound correspondences.
- Bantu and Semitic morphology make boundary-crossing chunks
  pervasive without the filter.

This fixture pins the mechanism on a controlled signal so the
behavior on real data (covered by ``test_morpheme_boundaries.py``
on Latin-Spanish) is anchored.

## Design notes

The boundary check is *strict*: a break at exactly the chunk's
start or end position is on the edge and not "inside" the chunk.
This means a chunk that BEGINS at the boundary (e.g., the
gemination `(V, tV)` starting at the suffix onset) is still
allowed. It describes a real morphological seam phenomenon and
should be visible to downstream consumers — the filter only
removes chunks that span across.

User-supplied only: there is no auto-segmentation. Forms without
``morpheme_breaks`` behave exactly as before.
