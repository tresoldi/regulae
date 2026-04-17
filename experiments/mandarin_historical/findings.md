# mandarin_historical — findings

40 Middle Chinese → Mandarin cognates, illustrating the classic
tonogenesis by loss of voicing distinction on obstruent onsets.

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
