# Linguistic challenge corpora

These corpora ask whether a statistically clean correspondence can be read as
historical evidence. The repeated-etymon and taxon-sampling commitments are
mandatory regressions; the confounding and external real-negative cases remain
research probes whose current behavior has to be discussed before a later
implementation decision is made.

Regenerate the synthetic files with:

```sh
python3 scripts/linguistic_probes.py
```

## Taxon sampling changes the verdict

`taxon_sampling_2lect.tsv`, `taxon_sampling_3lect.tsv` and
`taxon_sampling_4lect.tsv` contain the same 32 cognate sets and the same exact
conditioned innovation: ancestor /p/ answers innovator /f/ before /i/, with 16
examples and 16 counterexamples. The three-lect corpus adds an unchanged
conservative sister. The four-lect corpus adds a second, identical innovative
sister.

At snapshot `5de30f8`, with ten permutations:

| lect sample | rule margin | shared null margin | multi-lect verdict | pairwise above/measured |
| --- | ---: | ---: | --- | ---: |
| ancestor + innovator | 5.47 | 3.17 | above noise | 1/1 |
| + conservative sister | 5.47 | 6.49 | within noise | 0/2 |
| + second innovator | 5.47 | 7.72 | within noise | 0/4 |

Nothing about the innovation or its lexical evidence changes. What changes is
the number of pairwise searches whose maximum feeds the shared null threshold.
The probe therefore separates evidence for a rule from the multiplicity induced
by taxon sampling. A future verdict may correct for that multiplicity, but it
must not silently turn one attested innovation into stronger or weaker evidence
because near-identical sisters were sampled.

## Repeated paradigm cells are not independent etyma

`repeated_etymon.tsv` has sixteen cognate-set identifiers but only two lexical
histories: eight cells repeat one changing stem and eight repeat one unchanged
stem. Its `etymon_group` column now makes those two units explicit. The model
may still describe the surface association from all cells, but an automatic
bootstrap reports two effective etymon units rather than sixteen independent
histories.

When the column is absent, the fit report says that cognate sets are being
treated as independent. It never guesses grouping from a shared prefix. The
probe and the taxon-sampling trio are mandatory regressions in
`tests/c/test_evaluation_m2.c`.

## A winning feature is not necessarily an identified cause

`confounded_conditioning.tsv` has /p ~ f/ before /i/ and /p ~ p/ before /a/.
The change is therefore equally describable by height, frontness, vowel
identity and several correlated feature predicates. The current search prints
`following[close:+]`, but the corpus cannot establish that closeness rather than
frontness is the historical conditioning factor.

`deconfounded_conditioning.tsv` holds height and rounding variation on both
sides while frontness alone partitions the outcomes. It is the companion that
can identify frontness. These corpora should eventually test that equivalent
analyses are exposed as equivalent on the first and that the isolated feature
is recovered on the second; exact agreement with one preferred printed feature
name is not enough.

## A real negative control without vendoring its data

Kessler's *The Significance of Wordlists* dataset contains 200 meaning-matched
forms for eight languages and is available as a versioned CC-BY CLDF dataset.
The generator can deliberately pair two unrelated wordlists by meaning while
ignoring the expert cognate assignments:

```sh
python3 scripts/linguistic_probes.py \
  --kessler ~/lexibank_clone/kesslersignificance \
  --pair Hawaiian,Navajo --out /tmp/hawaiian-navajo.tsv
build/c/regulae train --json --permutations 50 /tmp/hawaiian-navajo.tsv
```

This is intentionally bad cognate input and therefore outside regulae's claim
to infer cognacy. It is nevertheless the right real control for the stronger
documentation claim that the corpus fit statistic answers whether a
relationship exists. On the reviewed checkout, ten permutations gave
`z=-5.47`, 10/107 multi-lect conditioned rows above noise and 13/416 pairwise
rows above noise; twenty permutations gave `z=-4.08`. Related English--German
and Latin--French controls were much farther from their shuffles (`-36.60` and
`-20.56` with ten permutations), so the statistic is informative but not a
categorical relationship verdict at the demonstrated sample sizes. Fifty
permutations put Hawaiian--Navajo at `z=-4.58`; the ten-, twenty- and
fifty-shuffle values also show that a small permutation count does not yet give
a stable z-score in this difficult real null.

Source: Brett Kessler, *The Significance of Wordlists* (2001), via the
[SequenceComparison CLDF dataset](https://github.com/SequenceComparison/kesslersignificance).
Keep its version, licence and citation with any recorded run; the derived data
are deliberately not committed here.
