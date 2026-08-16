# stress_conditioned_synthetic — findings

Synthetic fixture validating stress-conditioned context discovery.

## Status, 2026-08-16: this corpus does not load

`regulae train --format wide experiments/stress_conditioned_synthetic/cognates.tsv`
refuses it:

```
regulae: training: unknown grapheme "-" in feature system "distinctive".
```

The corpus writes syllable boundaries as `-` inside the transcription
(`ˈpe-ta`), which no feature system covers. `docs/capabilities.md` lists this
corpus under what the loaders cannot yet express, with `-` named as the blocker.

The results below came from a per-experiment Python driver -- the `parse_form`
this document mentions -- which stripped the separator and attached the stress
mark itself. That tooling is archived under `python/` and is not built or
supported, so this document records a real result that no shipped command
reproduces.

The engine's stress conditioning is not in question: `testdata/soundlaws/graded_3_stress.tsv`
exercises it through the loaders and `tests/c/test_sound_laws.c` asserts it.
What is missing is a syllable-boundary notation the loaders accept -- the wide
loader takes a `<lect>_syllables` column for exactly this.

## Setup

36 CVCV cognate pairs with two stress-conditioned 1-to-1 rules:

    /e/ → /ɛ/ / [stress:+]   (stressed /e/ lowers)
    /o/ → /ɔ/ / [stress:+]   (stressed /o/ lowers)
    /e/ → /e/, /o/ → /o/     (unstressed: preserved)

22 pairs have the stressed vowel in position 2 (triggers the rule);
14 pairs have the stress in the second syllable (no firing).

Stress is marked with the IPA primary-stress symbol ``ˈ`` in the
TSV. The experiment's parser (`parse_form`) attaches the stress to
the **nucleus vowel** of the stressed syllable, not the onset
consonant, so the resulting `Segment(stress="+")` sits on the
vowel where historical linguists expect it.

## What the model recovered

With stress predicates in the context-split candidate inventory:

    e → ɛ / self_stress=[stress:+]    count=12
    o → ɔ / self_stress=[stress:+]    count=10

Both rules committed cleanly, matching the generating process. A
small amount of noise survives in the stressed-e→e and
stressed-o→o entries (counts 1 and 2) — residual observations where
the aligner didn't land the stress mark exactly where the
ground-truth generator did, not misdiscoveries.

No chunks were promoted: the context split fully explains the
signal and BIC rejected the chunk alternatives.

## Why this fixture matters

Stress conditioning is the signature of many historical sound
changes:

- Romance open-vowel diphthongization (stressed ĕ → je, ŏ → we in
  Spanish and Italian).
- Vulgar Latin vowel-length merger to quality (long ē → close e,
  short ĕ → open ɛ, both only in stressed syllables).
- Germanic pre-Grimm accent-driven shifts.
- Catalan and some Portuguese dialects: stressed-vowel lowering.
- Russian vowel reduction (the opposite asymmetry — unstressed
  reduce, stressed preserve).

Before this round, the framework could not condition correspondences
on stress because ``Segment.stress`` was defined but never consumed
by the split-discovery layer. The fixture pins the commitment.

## Design notes

The rule ``e → je`` (true Romance diphthongization) is a 1-to-many
correspondence that lives in the chunk table, not in the
context-conditioned segment table. For the segment-level
stress-conditioned rule to surface cleanly, the fixture uses
vowel lowering (1-to-1) instead. The chunk table and context
splits handle different slices of the correspondence space —
see ``docs/training_pipeline.md``.

The companion test in ``tests/test_stress_conditioning.py`` pins
the commitment at the given counts.
