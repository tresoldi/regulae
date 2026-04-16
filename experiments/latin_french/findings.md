# Latin → French Experiment: Findings

## Motivation

Fourth real-data experiment. After Latin → Spanish (Romance,
moderate reduction), the third major Romance descendant gives us
an intra-family comparison: French has the most extreme reductive
changes in the Romance family — heavy nasalization, loss of most
final consonants, aggressive palatalization, vowel system overhaul.

If context discovery recovered splits well on Spanish, does it recover a
*different* (more extensive) set on French? And does it correctly
surface the French-specific phenomena (nasalization, final-consonant
loss, h aspirated, etc.)?

## Setup

* **Corpus**: 96 Latin → French cognate pairs, manually curated,
  using IPA-adjacent notation for Modern French. A few non-strict
  cognates kept (e.g., `wood → bwa` from Germanic *boscu*), because
  this is about training behaviour rather than cognate purity.
* **Training**: default context-discovery hyperparameters.

## Headline numbers

| metric | value |
|---|---|
| corpus pairs | 96 |
| prior-only total cost | ~70 |
| learned-model total cost | −773 |
| Reduction | ~830 |
| conditioned segment entries | ~44 |
| promoted chunks | 26 |
| tonal correspondences | 0 |

The **44 conditioned entries** is the highest of any real-data
experiment, about 1.8× Latin-Spanish. Consistent with the French
reductive profile being the most context-heavy of the three Romance
pairs we've run.

## Key context splits recovered

```
k → ʃ / _[front]: 6         THE palatalization: c before front V → ch/ʃ
                            (kaput→ʃɛf, kane→ʃjɛ̃, kantare→ʃɑ̃te, etc.)
k → s / _[front]: 2         alternative palatalization outcome
                            (centum→cent, kinkwe→sɛ̃k)
k → ʃ / [preceding vowel]: 2   intervocalic variant
g → w / [preceding vowel]: 3   Latin intervocalic g → French w
                               (like in "regn → rwa"-style reductions)
l → j / [preceding back]: 3    Latin l → French palatal j
                               (Gallo-Romance palatalization)
```

All real French sound laws. The framework also recovered many minor
conditioned patterns — some of them are noise given the small
corpus, but the framework correctly surfaces conditioning on
features that matter (front vs back vowels, voicing, position).

## What didn't work cleanly

A large number of the conditioned entries are for the source
segments `a` and `e` — these are the most variable in French
(multiple outcomes depending on syllable position, stress, nasality),
and the splits the framework committed don't always correspond to
linguistically canonical descriptions. For example:

```
a → ə / [prec=voiceless]: 3
a → a / [prec=voiceless]: 3
a → j / [prec=voiceless]: 2
```

Three different outcomes under the same context — which means the
context alone doesn't fully predict the outcome, and the framework
is conditioning on a feature that doesn't capture the real
conditioning (stress, probably). Stress isn't part of the context-discovery
model. This is consistent with the "prosodic context needed" note in
the overall context-discovery findings doc.

## French-specific observations

* The framework **does not** surface a clean `VN → Ṽ` (vowel
  nasalization) rule at the segment level, because the alignment
  search expresses cases like `ventu → vɑ̃` as multi-segment chunks,
  not as 1-to-1 with context. The sub-alignment decomposition in
  the consolidation round catches some of this but not all, because
  the 1-to-1 sub-alignment may choose `v~v e~∅ n~ɑ̃` or similar
  reshufflings that aren't linguistically transparent.

* **Final-consonant loss** shows up as gap-type chunk correspondences
  rather than context-conditioned segment splits. This is expected:
  gap events aren't part of the segment correspondence table.

## Reproducing

```sh
python experiments/latin_french/run_experiment.py
```
