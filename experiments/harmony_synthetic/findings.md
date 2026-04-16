# Synthetic harmony fixture — Findings

**Purpose**: validate long-range context discovery's `_phase5_long_range_discovery`
on a `previous_syllable`-conditioned rule, complementing the
`next_syllable` coverage of `umlaut_synthetic/`.

## Setup

- **45 pairs** organised as:
  - 15 harmony-firing: proto `CV1CV2` with V1 ∈ {u, o}, V2 = a
    → derived `CV1CV2` with V2 = o. The `a → o` backing
    fires iff the previous syllable has a back vowel.
  - 15 non-firing: proto `CV1CV2` with V1 ∈ {i, e}, V2 = a →
    derived unchanged.
  - 5 neutral: proto `CaCa` → derived unchanged.
  - 10 controls with V2 ≠ a (rule doesn't apply).

Again, the intervening consonant doesn't encode V1's backness,
so immediate-context discovery cannot capture the pattern on immediate neighbours.

## Expected long-range discovery output

A committed conditioned correspondence on source `a` carrying a
`previous_syllable` long-range constraint. Either framing is
valid:

- `a → o / previous_syllable=[back:+]` (the rule-firing side)
- `a → a / previous_syllable=[front:+]` (the complement — "a
  stays a iff previous vowel is front")

The two partitions are equivalent on this fixture because
every non-back vowel is front and every back vowel triggers
harmony. The greedy loop commits whichever predicate comes
first in the candidate enumeration and meets BIC.

## Actual long-range discovery output

Long-range discovery commits:

```
a → o / _ prev_syl=[back:+]: 15
```

on the 45-pair corpus. On the smaller 25-pair test-suite
variant (15 hit + 10 front-controls), it commits the mirror
framing:

```
a → a / _ prev_syl=[front:+]: 10
```

Both are correct and both fully partition the data with 100%
within-group dominance.

## Validation status

- Long-range discovery commits exactly the expected pattern and
  no other long-range entries.
- BIC, dominance, and min-count thresholds all admit the
  commit.
- The `previous_syllable` slot is exercised end-to-end —
  search-time context construction populates the field, the
  training observation pipeline carries it through
  `_compute_single_position_context`, the Phase 5 long-range
  candidate enumeration emits `previous_syllable` predicates,
  and the scoring overlay matches the committed entry on
  alignment.

## Known limitation

The greedy loop commits one framing per partition. If the
corpus expressed the rule as two mutually exclusive contexts
(e.g., if there were also `a → some_third_outcome / prev_syl
neither-front-nor-back`), the loop would need multiple commits
to cover them — this works because the outer while loop
re-runs on the residuals after each commit, but depth is
capped at `MAX_SPLIT_DEPTH * 4 = 12` iterations.
