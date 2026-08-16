# Clean Chinese-like Tonogenesis Fixture — Findings

**Purpose**: validate the cross-dimensional discovery commit
loop on a rule-explicit synthetic corpus. The sibling
`tone_chinese_like/` fixture is the negative case (noisy, mixed
signal, expected to commit very little under BIC); this clean
fixture is the positive case.

## Status, 2026-08-16: this does not reproduce as the corpus loads

**The rules below are real and the engine still finds them — but not on this
corpus in the direction it is read.** Loaded as a wide TSV, the pair is
analysed `cantonese > mandarin`, because reconciliation walks lect pairs in
ascending lect-id order and `cantonese` sorts first. The fixture encodes the
opposite: *mandarin's* onset voicing determines *cantonese's* tone. Asked
whether anything about cantonese predicts mandarin's tone, the answer is no —
mandarin's tone is the free variable here, evenly spread over four values — so
the stage commits nothing, correctly.

Rename the columns so the source lect sorts first and the fixture produces
exactly what this document claims:

```
  #0 a_mandarin>z_cantonese  pre[voiced:-] -> tone=¹¹@+0  count=40 conf=1.00 vs 0.00 elsewhere
  #0 a_mandarin>z_cantonese  pre[voiced:+] -> tone=⁴⁴@+0  count=40 conf=1.00 vs 0.00 elsewhere
```

Two counts of 40 at confidence 1.00 against 0.00 elsewhere, which is the whole
of what the fixture encodes.

**Cross-dimensional discovery looks from one side of a pair only.** Conditioned
correspondences look from both — that is what `context_is_target` is for, and
the header comment on `rg_conditioned_segment_count_row` argues the case at
length: a change is only visible from the side that has the split, and which
side that is depends on which lect happens to sort first. The same argument
applies here and has not been applied. The sibling synthetic fixtures
(`umlaut_synthetic`, `harmony_synthetic`, `length_conditioned_synthetic`) all
still reproduce precisely because their rules come back as *target-side*
conditioned correspondences.

This was true before the 2026-08-15/16 structural work and is unchanged by it:
the pre-pass build produces the same zero. The notation also moved — a
cross-dimensional environment is a whole `rg_context_spec` now, printed
`pre[voiced:+]`, not the `voiced=+@relative_-1` pair this document was written
against.

Nothing here has been changed to match the engine, because the engine is what
looks wrong.

## Setup

- **80 pairs** generated deterministically from a grid:
  - 40 voiceless-initial pairs: initial ∈ `{p, t, k, f, s}`, vowel
    ∈ `{i, a, u, o, e}`, source tone ∈ `{¹¹, ²², ³³, ⁴⁴}`. Target
    tone is always **`¹¹`** regardless of the source tone.
  - 40 voiced-initial pairs: initial ∈ `{b, d, g, m, n}`, vowel
    ∈ `{i, a, u, o, e}`, source tone ∈ `{¹¹, ²², ³³, ⁴⁴}`. Target
    tone is always **`⁴⁴`**.

  The four tones were written `1`–`4` until 2026-08-14. They are
  arbitrary labels — the fixture attaches no pitch to them — but an
  ASCII digit is not tone notation, so no loader could read them and
  the corpus was reachable only through the direct API. They are now
  four distinct Chao level tones, which is an injective relabelling
  and changes nothing the fixture tests.
- **Rule by construction**: the voicing of the source initial
  consonant perfectly determines the target vowel's tone.
  Source tones are irrelevant to the target — this is a
  deliberate simplification of the real Mandarin/Cantonese
  tonogenesis pattern, designed so that a single pair
  (src_feature, tgt_value) carries all the predictive
  information. A more complex fixture (where the source tone
  also matters) would require joint-predictor anomaly
  detection, which is out of cross-dimensional discovery scope.

## Expected cross-dimensional discovery output

- At least one committed `CrossDimensionalLink` with
  `src_feature="voiced"` and `tgt_value="⁴⁴"`, confidence 1.0.
- A matching rule for the complementary environment, `→ tone=¹¹`.
- The context-discovery tonal correspondence table should also show
  the source-tone-to-target-tone mappings, independent of the
  cross-dimensional discovery overlay.

## Actual cross-dimensional discovery output

Two rules, which is the whole of what the fixture encodes:

```
voiced=+@relative_-1 → tone=⁴⁴@+0   count=40  conf=1.00  vs 0.00 elsewhere
voiced=-@relative_-1 → tone=¹¹@+0   count=40  conf=1.00  vs 0.00 elsewhere
```

A conditioned split is a two-sided statement, and both halves are findings, so
the complementary environment is published in its own right under
`source_value = "-"`. The reference reported the same two facts as four rows,
carrying each of them in both of its positional framings.

The `vs 0.00 elsewhere` is the part worth reading. Each tone is not merely
frequent in its environment; it does not occur outside it. That is what makes
this a conditioned split rather than a description of a skewed corpus.

### What this fixture caught

Between 2026-08-14, when the corpus first became loadable, and later the same
day, this stage committed **18** rules here rather than 2. The two above were
among them at confidence 1.00; the other sixteen sat at exactly 0.50, in
contradictory pairs:

```
consonant=+@relative_-1 → tone=¹¹@+0   count=40  conf=0.50
consonant=+@relative_-1 → tone=⁴⁴@+0   count=40  conf=0.50
```

Every source segment in this corpus is a consonant, so `consonant=+` predicts
nothing, and splitting 80 observations evenly across two outcomes is what a
predicate carrying no information looks like. They committed because the gate
was `confidence >= 0.5` with no contrast set: on a two-valued dimension an even
split clears one half of the threshold from either side.

Three things were wrong with that gate and all three are now fixed — the
environment must have an attested complement, the split must beat its
parameters under BIC against that complement, and a reported value must be
raised relative to the contrast and pass its own test. The reasoning is in
[`docs/correspondence_discovery.md`](../../docs/correspondence_discovery.md).

This is not the fixture failing. It is the fixture doing its job, one day after
the corpus it needed could be read.

The context-discovery tonal table also captures the raw tonal mappings:

```
Tonal correspondences (8):
  ¹¹ → ¹¹: 10
  ²² → ²²: 10    (voiceless: all preserve)
  ³³ → ³³: 10
  ⁴⁴ → ⁴⁴: 10
  ¹¹ → ⁴⁴: 10
  ²² → ⁴⁴: 10    (voiced: all map to ⁴⁴)
  ³³ → ⁴⁴: 10
  ⁴⁴ → ⁴⁴: 10
```

The cross-dimensional rules and the context-discovery tonal table are
**additive** — both contribute to alignment cost. The
`confidence=1.0` rules still carry non-zero adjustments because
the base rate of each target tone in the tonal table is 50%
(40/80 are tone 1, 40/80 are tone 4). The Dirichlet-smoothed
LLR for a rule that perfectly predicts tone 4 vs a base rate of
0.5 is:

```
pos_adj = -(log(1.0 - ε) - log(0.5 + ε)) ≈ -0.669 nats per match
neg_adj = -(log(ε) - log(0.5 - ε))       ≈ +3.045 nats per miss
```

(ε is the Dirichlet smoothing correction.)

Aggregate effect: 40 matches × -0.669 nats = -26.76 nats per rule.
Passes BIC with a wide margin.

## Validation status

✅ Cross-dimensional discovery commits at least one voicing→tone rule with
   confidence >= 0.8. Both of them, at 1.00.
✅ Every committed rule is semantically meaningful: two rules, both
   voicing→tone, and nothing else.
✅ The rules are deterministic under repeated training.
✅ No duplicates.

## Known limitations surfaced by this fixture

- **Aggregate cost figures below are stale.** They were computed when
  four rules were committed rather than two, and against the ASCII
  tone labels. The mechanism they describe is unchanged.
- **Real-data Mandarin/Cantonese rule is more complex**. The
  actual Middle Chinese tonogenesis mapping is
  `voiced+src_tone → voiced_register_with_shifted_tone`, which
  requires a joint predictor (both voicing AND source tone).
  Cross-dimensional discovery's anomaly detection only enumerates
  single-feature predictors, so on the sibling `tone_chinese_like/`
  fixture the signal is too fragmented to pass BIC. Joint-predictor
  enumeration is explicitly out of the current cross-dimensional
  discovery scope — it belongs in a future extension.
- **Dirichlet smoothing constant** (α=1.0) bakes in a specific
  cost magnitude. On this fixture with a 50% base rate, a
  perfectly-predictive rule's adjustment is only -0.669 nats
  per match, not -log(0.5) = -0.693 as an unsmoothed LLR
  would give. The difference is negligible and the test suite
  doesn't pin the exact value.
