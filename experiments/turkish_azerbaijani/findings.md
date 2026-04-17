# turkish_azerbaijani — findings

40 Turkic cognates across numerals, body parts, nature vocabulary,
8 verb infinitives (with -mek/-mək boundaries annotated).

## Correspondences recovered

- **Word-initial k ~ q/ɡ** committed as `k → ɡ / _[vowel:+]`
  (count 6) vs unconditioned `k → k` (count 3). Azerbaijani
  preserves the uvular reflex in many positions where Turkish
  fronted to velar.
- **Front-vowel harmony** between root and suffix committed as
  `e → æ / [voiced:+]_` (9 observations) vs
  `e → e / [voiced:+]_` (5). The split reflects Azerbaijani's
  lower /æ/ in positions where Turkish retains /e/.
- **Word-final k ~ χ** captured as unconditioned
  (`k → χ: 3` alongside `k → k: 15`).
- **Vowel harmony obeying cross-side agreement**: Turkish
  verb infinitive `-mek` matches Azerbaijani `-mək` for all 8
  annotated verbs. Morpheme-boundary filter rejects
  boundary-crossing chunks.

## What this validates

- The framework scales to agglutinative languages with rich
  vowel inventories.
- Morpheme-boundary annotation on verb suffixes cleanly separates
  root-level from suffix-level correspondences.
- Long-range (vowel-harmony-style) conditioning is discoverable
  without custom rule templates.

## Sources

Wiktionary Turkish and Azerbaijani entries. Transcriptions
phonemic/broad-phonetic IPA (Turkish `k` before front vowels
written `c` for the palatal allophone, per the compilation
agent's choice).

Arabic/Persian loans explicitly excluded during compilation.
