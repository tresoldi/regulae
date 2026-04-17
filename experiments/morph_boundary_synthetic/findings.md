# morph_boundary_synthetic — findings

Synthetic fixture validating the morpheme-boundary chunk filter.

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
