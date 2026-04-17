# arabic_hebrew — findings

32 Semitic cognates inherited from Proto-Semitic.

## Correspondences recovered

- **Arabic ḍād (`dˤ`) ↔ Hebrew ṣade (`tsˤ`)**: committed as
  `dˤ → tsˤ` (2 observations). Classic Proto-Semitic *sˤ/*dˤ
  reflex merger in Hebrew.
- **Arabic s (sīn) ↔ Hebrew ʃ (shin)** in cognates like
  `sinn ~ ʃen` "tooth", `samaːʔ ~ ʃamajim` "heaven". Captured
  in the segment table.
- **Arabic χ ↔ Hebrew ħ**: `ʔaχ ~ ʔaħ` "brother" (and sister).
- **Arabic θ ↔ Hebrew ʃ**: `θalaːθa ~ ʃaloʃ` "three",
  `iθnaːn ~ ʃnajim` "two".
- **Pharyngeals ʕ, ħ** preserved in both languages.
- **Vowel reduction**: Arabic CvC roots with full vowels
  correspond to Hebrew's segolate patterns (`kalb → kelev`,
  `bajt → bajit`).

## What this validates

- The framework handles the full Semitic consonant inventory
  (pharyngeals, emphatics/pharyngealized, uvulars, glottals)
  without special configuration.
- Non-Indo-European, non-Austronesian data produces sensible
  segment tables and splits.
- Segolate vowel insertion in Hebrew surfaces as chunks
  (`bajt ~ bajit`, `kalb ~ kelev`).

## Sources

Cognates selected from standard Semitic comparative reference
(Klein's etymological dictionary, Huehnergard's Proto-Semitic
paradigms). Hebrew transcribed in modern Israeli pronunciation
with ṣade as /tsˤ/ (preserving the emphatic feature for
correspondence visibility). Non-cognates with the same gloss
(different roots) were explicitly excluded — the point is the
cognate signal, not the translation equivalence.
