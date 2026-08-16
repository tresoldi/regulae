# mandarin_historical — findings

40 Middle Chinese → Mandarin cognates, illustrating the classic
tonogenesis by loss of voicing distinction on obstruent onsets.

## Status, 2026-08-16: the headline rule does not reproduce as the corpus loads

**The voicing tonogenesis below is not what the engine currently commits on
this corpus.** Loaded as a wide TSV it produces two rules, and they are
source-tone-to-target-tone, not voicing-to-tone:

```
  #3 mandarin>middle_chinese  self[tone:-] -> tone=1@+0  count=10 conf=0.43 vs 0.00 elsewhere
  #3 mandarin>middle_chinese  self[tone:3] -> tone=2@+0  count=8  conf=0.89 vs 0.04 elsewhere
```

Note the direction: `mandarin > middle_chinese`. Reconciliation walks lect
pairs in ascending lect-id order and `mandarin` sorts before `middle_chinese`,
so the stage is asking what about Mandarin predicts Middle Chinese — the
reverse of the historical claim, and the direction in which the rule is not
there to find.

Rename the lects so Middle Chinese sorts first and the phenomenon comes back,
as one of thirteen committed rules:

```
  #3 a_mc>z_man  pre[voiced:+] self[tone:-] -> tone=2@+0  count=7 conf=0.50 vs 0.00 elsewhere
```

A voiced preceding onset, on a syllable carrying no tone of its own, predicting
Mandarin tone 2. That is the Middle Chinese voiced-onset tonogenesis this
document is about. It is now a joint predicate rather than the single
`voiced=+@relative_-1` below, which is why its confidence reads 0.50 against a
contrast of 0.00 rather than 1.00 — the environment names two things.

See `experiments/tone_chinese_like_clean/findings.md` for the same problem on
the clean fixture, and for why it is a property of the stage rather than of
either corpus: cross-dimensional discovery looks from one side of a pair only,
while conditioned correspondences look from both.

Unchanged by the 2026-08-15/16 structural work — the pre-pass build produces
the same two rules. Nothing here has been rewritten to match the engine.

## Correspondences recovered

The cross-dimensional discovery layer commits:

```
voiced=+@relative_-1 -> tone=2@+0    count=6/6  conf=1.00
```

This is the **Middle Chinese voiced-onset → Mandarin tone-2
tonogenesis**: the framework recovered from real historical
data the rule that MC tone-1 (píng) syllables with voiced
onsets (b, d, g, dz, dʑ) became Mandarin tone 2 after the
voicing distinction was lost on obstruents.

A second rule is also committed:

```
tone=2@relative_+1 -> tone=3@+1      count=5/6  conf=0.83
```

This reflects the MC tone-2 (shǎng) → Mandarin tone-3 redirection,
captured as a cross-dimensional rule because tone changes on the
target encode a categorical shift from the source tone.

Additional recovered patterns (segment table):

- **MC voiced obstruents merged to voiceless aspirated/unaspirated**
  in Mandarin: `b → f`, `d → t`, `ɡ → k`.
- **Final stops lost**: `-p`, `-t`, `-k` endings of MC drop.
- **Final -m merged to -n**: `sam → san` "three".
- **Palatalisation before front vowels**: `kj- → tɕ-`.

## What this validates

- The cross-dimensional discovery layer recovers the voicing
  tonogenesis on **real historical data**, not just synthetic
  fixtures — this was the chief motivating case for the
  framework.
- Tonal correspondences propagate correctly through training
  when the tone is attached to the nucleus vowel via
  `Segment.tone`.
- Complex consonant inventories (MC palatals `tɕ`, `dʑ`,
  retroflexes `ʂ`, `ɻ`) work with merkmal's default system.

## Sources

Middle Chinese reconstructions after Baxter (simplified to avoid
the `H`/`X` grave-tone diacritics). Mandarin transcriptions in
phonemic IPA. Tone values `1/2/3/4` on the nucleus vowel.
Retroflex apical vowels (the `sī`, `shī`, `rì` class) are
approximated as `ɯ`, `ʂɯ`, `ʐɯ` for merkmal compatibility —
a widely-used simplification in Chinese comparative work.
