# finnish_estonian — findings

40 Finnish ↔ Estonian cognates from basic vocabulary (body parts,
kin terms, numerals, core verbs).

## Correspondences recovered

Headline per-source distributions from `format_model` (intervals
Wilson, n varies per source):

- `k → k` (15), `k → g` (2) — consonant gradation trace
- `i → e` (3) alongside `i → i` (10) — Estonian vowel lowering
  in final syllables
- `æ → ɑ` (7), `æ → æ` (3) — vowel-harmony flip where Estonian
  drops front-back distinction in the second syllable
- `e → eː` (2), `e → e` (10), `e → i` (1) — compensatory
  lengthening in monosyllables like `keːl` from `kieli`
- `l → l` (11), `l → d` (1) — consonant gradation
- Context-conditioned splits emerge (visible with
  `format_model(..., annotate_chunks=True)`)

## What this validates

- Vowel harmony as a long-range conditioning axis is detectable
  on real Finnic data (not just synthetic fixtures).
- Vowel length from merkmal's `long` feature is carried through
  without special casing.
- The chunk table promotes legitimate CV clusters (e.g., `ie → eː`
  monophthongisation) while boundaries on verb infinitives keep
  `-da`, `-dä`, `-la` from leaking into the segment table.

Morpheme-break annotations are on the 5 verbs ending in
`-da`/`-dä` where the boundary is transparent (infinitive suffix
following the stem).

## Sources

Cognates drawn from standard Finnic comparative literature
(Itkonen, Hakulinen) and cross-checked against Wiktionary
Finnic tables. Forms transcribed in phonemic IPA following
Finnish/Estonian orthography conventions (näär → næː, öö → øː,
etc.).
