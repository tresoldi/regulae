# Family survey

<!-- Recorded by scripts/families.py. Not checked by CI: the numbers
     depend on which revision of the Lexibank clone produced them. -->

43 published wordlists with expert cognate judgements over segmented forms, 5 lects each (the most widely attested), trained with default options.

There is no score here and there could not be. Nobody has a gold standard for *which conditioned environments are real in Dravidian*, and a number claiming there is would be inventing one. What the survey reports is the shape of the answer, across enough families that the shape itself says something.

## What to read off it

**Class counts scale with the corpus, not with the discovery.** They rise with the number of lects and the number of sets, and they rise on shuffled data too. Nothing in this repository reports them as a result, and the column is here so that the point is visible rather than asserted.

**Tone.** 6 of the 43 wordlists transcribe tone at all -- most of the world's tone languages are represented here by sources that do not write it -- and 5 of those 6 yielded cross-dimensional rules. Until 2026-08-16 the number was zero, on every one of them: the CLDF `Segments` column writes Chao tone as a token of its own (`tʰ u ⁵¹`) and the pre-segmented loader kept it as a segment, so no segment carried a tone and the stage that exists for tonogenesis had nothing to read.

Still silent, with tone in the transcription and no cross-dimensional rule out of it: `mannburmish`. Worth a look rather than an explanation -- zero is a legitimate answer, and it is also what the directionality limitation recorded in `experiments/README.md` produces, since the stage searches one side of a lect pair and the side is decided by which lect id sorts first.

**2 wordlists contain forms with no nucleus** -- no vowel and no syllabic consonant -- which are given one so the syllable predicates have something to hold of. A syllable-conditioned rule on those corpora rests on that guess. The column is `guessed/syllabified`.

## The run

| dataset | family | sets | classes | conditioned | cross-dim | nucleus guessed | cost/seg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `starostinpie` | Indo-European | 185 | 344 | 43 | 0 | 0/549 | -2.029 |
| `saenkoromance` | Romance | 131 | 280 | 44 | 0 | 0/504 | -1.841 |
| `syrjaenenuralic` | Uralic | 174 | 335 | 36 | 0 | 0/462 | -1.720 |
| `zhivlovobugrian` | Ob-Ugric | 118 | 244 | 25 | 0 | 0/485 | -2.192 |
| `savelyevturkic` | Turkic | 321 | 593 | 75 | 0 | 0/1301 | -2.413 |
| `hruschkaturkic` | Turkic | 209 | 365 | 49 | 0 | 0/928 | -2.060 |
| `robbeetstriangulation` | Transeurasian | 334 | 515 | 76 | 0 | 0/1218 | -2.659 |
| `leekoreanic` | Koreanic | 266 | 457 | 59 | 0 | 0/1005 | -2.294 |
| `hattorijaponic` | Japonic | 214 | 331 | 50 | 0 | 0/896 | -2.386 |
| `leejaponic` | Japonic | 216 | 246 | 58 | 0 | 0/948 | -2.416 |
| `leeainu` | Ainu | 254 | 288 | 67 | 0 | 0/986 | -2.289 |
| `liusinitic` | Sinitic | 466 | 701 | 85 | 138 | 0/1852 | -2.889 |
| `hsiuhmongmien` | Hmong-Mien | 58 | 77 | 8 | 30 | 0/290 | -2.199 |
| `mannburmish` | Burmish | 568 | 1124 | 116 | 0 | 0/2092 | -2.508 |
| `yanglalo` | Loloish | 1055 | 1481 | 92 | 55 | 0/4871 | -2.694 |
| `luangthongkumkaren` | Karenic | 406 | 432 | 81 | 105 | 0/1719 | -3.177 |
| `sagartst` | Sino-Tibetan | 234 | 423 | 24 | 1 | 0/652 | -1.399 |
| `bodtkhobwa` | Kho-Bwa | 1013 | 1339 | 265 | 0 | 0/3834 | -3.298 |
| `sidwellbahnaric` | Bahnaric | 238 | 465 | 62 | 0 | 0/770 | -2.101 |
| `deepadungpalaung` | Palaungic | 131 | 282 | 32 | 0 | 0/471 | -2.225 |
| `nagarajakhasian` | Khasian | 192 | 379 | 54 | 0 | 0/607 | -1.897 |
| `dunnaslian` | Aslian | 157 | 262 | 20 | 0 | 0/399 | -2.172 |
| `dravlex` | Dravidian | 223 | 457 | 40 | 0 | 0/731 | -1.767 |
| `kitchensemitic` | Semitic | 134 | 281 | 43 | 0 | 1/344 | -1.694 |
| `felekesemitic` | Ethiosemitic | 212 | 483 | 70 | 0 | 0/638 | -2.312 |
| `ratcliffearabic` | Arabic | 89 | 191 | 11 | 0 | 0/408 | -2.276 |
| `gravinachadic` | Chadic | 78 | 146 | 12 | 0 | 0/185 | -1.960 |
| `grollemundbantu` | Bantu | 119 | 328 | 28 | 0 | 0/435 | -2.001 |
| `walworthpolynesian` | Polynesian | 357 | 452 | 80 | 0 | 0/1265 | -1.830 |
| `blustaustronesian` | Austronesian | 162 | 251 | 33 | 0 | 0/402 | -1.345 |
| `abvdphilippines` | Austronesian | 537 | 639 | 15 | 0 | 2/1631 | -1.448 |
| `smithborneo` | Austronesian | 911 | 1123 | 155 | 0 | 0/2555 | -1.980 |
| `robinsonap` | Alor-Pantar | 237 | 441 | 50 | 0 | 0/681 | -1.300 |
| `mcelhanonhuon` | Huon | 168 | 298 | 48 | 0 | 0/487 | -1.882 |
| `bowernpny` | Pama-Nyungan | 78 | 214 | 30 | 0 | 0/191 | -1.537 |
| `utoaztecan` | Uto-Aztecan | 195 | 376 | 22 | 0 | 0/485 | -0.899 |
| `wichmannmixezoquean` | Mixe-Zoquean | 135 | 289 | 33 | 0 | 0/467 | -1.771 |
| `mattercariban` | Cariban | 104 | 229 | 20 | 0 | 0/360 | -1.154 |
| `constenlachibchan` | Chibchan | 141 | 140 | 15 | 0 | 0/321 | -1.638 |
| `chaconarawakan` | Arawakan | 161 | 344 | 21 | 0 | 0/566 | -1.197 |
| `crossandean` | Andean | 197 | 186 | 31 | 0 | 0/722 | -2.366 |
| `kesslersignificance` | control | 152 | 247 | 34 | 0 | 0/372 | -1.559 |
| `bdpa` | many | 531 | 499 | 37 | 0 | 0/2654 | -2.377 |

## Not surveyed

Each row is a thing to fix rather than a dataset to drop. A grapheme the feature system does not cover is one line in merkmal, and `docs/grapheme_coverage.md` has the standing measurement of how many tokens each one costs; a corpus that runs long is a corpus to sample; a dataset with cognate judgements but no segmented forms is invisible to every method in this repository and worth saying so about.

| dataset | why |
| --- | --- |
| `meloniromance` | over 150s at 5 lects (5350 cognate sets) |
| `iecor` | grapheme `ɹ̪` not covered |
| `oskolskayatungusic` | grapheme `aːˁ` not covered |
| `houchinese` | grapheme `ɚ` not covered |
| `starostinhmongmien` | grapheme `⁶/` not covered |
| `zhangrgyalrong` | grapheme `H/` not covered |
| `sidwellvietic` | grapheme `mᵊ` not covered |
| `tuled` | grapheme `íì` not covered |
| `oliveiraprotopanoan` | grapheme `∼` not covered |
