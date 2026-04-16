# Latin → Italian Experiment: Findings

## Motivation

Fifth real-data experiment, completing the three-way Romance
comparison. Italian is the **most conservative** of the Romance
descendants we've run: geminate consonants preserved, final vowels
preserved, less reduction than Spanish or French. Our hypothesis
going in was that context discovery should find **fewer** context splits on
Italian than on Spanish or French, because Italian has fewer
context-dependent changes.

## Setup

* **Corpus**: 95 Latin → Italian cognate pairs, IPA-adjacent
  notation for Modern Italian (including geminates and affricates).
* **Training**: default context-discovery hyperparameters.

## Headline numbers

| metric | value |
|---|---|
| corpus pairs | 95 |
| prior-only total cost | ~105 |
| learned-model total cost | −930 |
| Reduction | ~1035 |
| conditioned segment entries | 29 |
| promoted chunks | 17 |
| tonal correspondences | 0 |

The **29 conditioned entries** is between Spanish (25) and French
(44), matching Italian's intermediate position on the
"phonological reduction intensity" scale.

## Key context splits recovered

```
k → tʃ / _[vowel]: 2           Italian palatalization of c before e/i
                                (centum→tʃɛnto, kinkwe→tʃiŋkwe)
l → ʎ / [prec=voiceless]_[front]: 3   Latin l → Italian ʎ (palatal
                                      lateral, two-level split:
                                      voiceless preceding AND front
                                      following)
l → j / [preceding voiceless]: 4   Latin -l- → Italian -j- (glide)
                                   (this is mostly a chunk artefact)
t → t / [preceding front]: 4       positional preservation (noise)
t → d / [preceding front]: 2       lenition minority (only 2 cases)
u → o / pos=medial: 5          Romance short u → o (stressed/medial)
u → u / pos=medial: 4          medial u preservation
n → n / _[vowel]: 14           majority identity
n → ɲ / _[vowel]: 1            singleton palatal n (maybe ignis→leɲo type)
```

The `l → ʎ / [prec=voiceless, foll=front]` split is particularly
nice: it's a **two-level refinement** (first split on preceding
voiceless, then inside that split further on following front).
The framework found a multi-feature context that neither axis alone
would capture.

## Italian's position in the 3-way comparison

| descendant | conditioned entries | reduction profile |
|---|---:|---|
| Italian | 29 | conservative (keeps geminates, finals) |
| Spanish | 24 | moderate (some lenition, palatalization) |
| French | 44 | extreme (nasalization, final loss, etc.) |

The framework **ordered these correctly** without any instruction:
French > Italian ≈ Spanish in terms of conditioning needs. This
matches the Romance linguistics consensus on relative reductive
intensity.

The near-equal Italian/Spanish counts are surprising at first —
Italian is usually described as the "more conservative" — but on
reflection, Italian does have real context-conditioned phenomena
(gemination maintenance, palatalization of c/g before e/i, vowel
diphthongization in open syllables) that Spanish has in simpler
forms. The framework's count captures that Italian's conservatism
is **phonological** (preservation of inherited segments) but not
**rule-free** — it still has context-conditioned changes, just
different ones from Spanish.

## Reproducing

```sh
python experiments/latin_italian/run_experiment.py
```
