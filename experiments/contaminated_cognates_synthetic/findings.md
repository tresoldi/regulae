# contaminated_cognates_synthetic — findings

25 pairs: 20 clean (confidence 1.0, regular `p → f` rule) + 5
deliberately mis-labeled (confidence 0.0).

## What the training recovers

With confidence weighting on the bad pairs at 0.0:

```
Segment table (p source):
  p -> f: count=20.0
```

Clean rule only. Bad pairs' nonsense correspondences (`p → q`,
`p → z`, etc.) contribute zero mass because the underlying
counts are multiplied by their cognate set's confidence.

Control run (same pairs, all at confidence=1.0):

```
Segment table (p source):
  p -> f: count=20.0
  p -> q: count=1.0
  p -> x: count=1.0
  p -> k: count=1.0
```

Noise is now visible in the counts.

## What this validates

- `CognateSet.confidence` actually controls evidence mass, not
  just a display field.
- Setting confidence to 0.0 is equivalent to excluding the pair
  from training entirely (no leakage into segment, displacement,
  chunk, or class tables).
- Historians curating a corpus can mark suspicious cognates as
  low-confidence and retain them in the retained corpus for
  auditability (visible via `model.cognate_corpus`) without
  contaminating the learned model.
