# Synthetic umlaut fixture — Findings

**Purpose**: validate long-range context discovery's `_phase5_long_range_discovery`
on a rule-explicit synthetic corpus with a clean long-range
conditioning pattern.

## Setup

- **40 pairs** organised as:
  - 10 umlaut-firing: proto `CaCV` with V ∈ {i, e} → derived
    `CæCV`. The a → æ fronting fires iff the next syllable has a
    front vowel.
  - 10 non-firing: proto `CaCV` with V ∈ {o, u} → derived
    unchanged.
  - 20 controls with V1 ∈ {i, u}, where no source `a` is
    present (rule cannot apply).

The rule depends on a feature (`next_syllable=[front:+]`) that
is **not** captured by immediate-neighbour context: the
intervening consonant (from `{p, t, k, b, d, g, m, n}`) is a
plain stop or nasal and doesn't encode V2's frontness.
Immediate-context discovery alone therefore cannot recover this pattern.

## Expected long-range discovery output

- At least one committed conditioned correspondence
  `a → æ` whose context includes
  `next_syllable=[front:+]`, covering ~10 observations.

## Actual long-range discovery output

Long-range discovery commits exactly one entry:

```
a → æ / _ next_syl=[front:+]: 10
```

Covering all 10 umlaut-firing observations. No other long-range
entries are committed (no spurious commits on the 30
non-umlaut observations).

Immediate-context discovery commits an unconditioned `a → æ / _: 10` entry (the
default when looking only at immediate neighbours, since V2 is
one syllable away and invisible to the immediate-context
predicates), but long-range discovery refines this into the
single rule the signal actually carries.

## Validation status

- Long-range discovery commits exactly the expected rule and no
  other long-range entries.
- Dominance filter (≥60% of YES observations share target)
  correctly admits this commit: 10/10 observations map to `æ`.
- BIC threshold (-5.0) correctly admits this commit.
- Tight feature inventory (no `vowel` or `consonant` in
  long-range enumeration) does not block recovery — the rule
  fires on `next_syllable=[front:+]`, a feature that stays in
  the inventory.

## Why the dual framing wasn't committed

The greedy loop commits the single best split per iteration on
the remaining observations. After committing
`a → æ / next_syl=[front:+]` the remaining observations are
homogeneous (`a → a` on the 10 non-firing pairs plus any
residuals), and no further long-range predicate beats ΔBIC=-5.0
on them. The dual framing `a → a / next_syl=[back:+]` is not
committed because the residuals no longer need a conditioning.
This is the correct behaviour for long-range context discovery — one rule per
partition, not a full decision tree.
