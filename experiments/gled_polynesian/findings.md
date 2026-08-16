# GLED Polynesian Experiment

Second multi-lect reconciliation real-data run. Six Polynesian lects from GLED release
`20221127`, trained jointly. The original motivation was to
demonstrate **Hawaiian merger disambiguation** — the case where
pairwise models can't see what a third lect makes obvious. The
result is mixed: the pipeline runs cleanly and recovers recognizable
Polynesian correspondences, but the particular merger story we
expected doesn't manifest on this corpus, and we'll explain why.

## Status, 2026-08-16: the corpus is not in the repository

This directory holds findings only. The GLED data is not vendored -- those
datasets carry their own licences -- so the run below cannot be reproduced from
a checkout, and the source paths in it are wherever the data sat on the machine
that produced it.

`scripts/lexibank.py` is the supported route from a CLDF wordlist with expert
cognate judgements to a regulae corpus, and `regulae check <corpus>` reports
what a given dataset costs in graphemes the feature system cannot read before
training on it.

## Setup

- **Source**: `/tmp/gled_clone/releases/20221127/gled.tsv`
- **Family filter**: `Austronesian`
- **Doculect filter**: `{HAWAIIAN_2, MAORI, SAMOAN, TAHITIAN, RAROTONGAN, RAPA_NUI}` (six Polynesian lects)
- **Cognate sets loaded**: 44
- **Size distribution**: 23 six-way, 12 five-way, 2 four-way, 1 three-way, 6 two-way
- **Per-lect coverage**: 40 RAPA_NUI / 38 HAWAIIAN_2 / 38 SAMOAN / 37 MAORI / 34 RAROTONGAN / 34 TAHITIAN

## Model

- 15 per-pair `LearnedModel` instances
- **93 unconditioned multi-lect classes**
- **9 conditioned multi-lect classes** from context conditioning (was 244 before
  the reconciliation consolidation pass; the pivot-lect dedup + adaptive BIC
  correction + adaptive minimum-commit-count floor compressed the
  output by ~96% without losing any headline result)

## What it recovered

### The core Polynesian consonant correspondences

```
count=7  [6-way] HAWAIIAN_2:k MAORI:t RAPA_NUI:t RAROTONGAN:t SAMOAN:t TAHITIAN:t
count=9  [6-way] HAWAIIAN_2:ɐ MAORI:ɐ RAPA_NUI:ɐ RAROTONGAN:ɐ SAMOAN:ɐ TAHITIAN:ɐ
count=8  [6-way] HAWAIIAN_2:i MAORI:i RAPA_NUI:i RAROTONGAN:i SAMOAN:i TAHITIAN:i
count=4  [6-way] HAWAIIAN_2:u MAORI:u RAPA_NUI:u RAROTONGAN:u SAMOAN:u TAHITIAN:u
count=3  [6-way] HAWAIIAN_2:o MAORI:o RAPA_NUI:o RAROTONGAN:o SAMOAN:o TAHITIAN:o
```

The `*t → Hawaiian k` reflex is the headline: all five other lects
preserve `t`, and Hawaiian gives `k`. This is a completely valid
six-way correspondence class. Pairwise context discovery would have rediscovered
it 15 times (once per pair); multi-lect reconciliation reports it as **one class**.

### Liquid split

```
count=3  [6-way] HAWAIIAN_2:l MAORI:r RAPA_NUI:r RAROTONGAN:r SAMOAN:l TAHITIAN:r
```

The `*l/*r` contrast in Proto-Polynesian merges differently in
different daughters. Hawaiian and Samoan choose `l`; the eastern
Polynesian lects choose `r`. This is a clean two-way split
captured as a single multi-lect class.

### Velar nasal with Tahitian loss

```
count=2  [6-way] HAWAIIAN_2:n MAORI:ŋ RAPA_NUI:ŋ RAROTONGAN:ŋ SAMOAN:ŋ TAHITIAN:ʔ
```

Hawaiian has fronted `ŋ → n`, Tahitian has lost it to a glottal
stop, and the other four preserve `ŋ`. Exactly the kind of
asymmetric multi-way split that motivated multi-lect reconciliation.

### Proto `*f` diversity

```
count=1  [6-way] HAWAIIAN_2:h MAORI:h RAPA_NUI:h RAROTONGAN:ʔ SAMOAN:f TAHITIAN:h
count=1  [6-way] HAWAIIAN_2:h MAORI:h RAPA_NUI:h RAROTONGAN:ʔ SAMOAN:s TAHITIAN:h
```

The `*f` reflex shows the textbook outcomes: Samoan keeps `f` (and
sometimes has `s`), most others spirantize to `h`, Rarotongan goes
all the way to glottal `ʔ`.

## What it did NOT recover: the Hawaiian merger story

The multi-lect reconciliation spec's motivating example was "Hawaiian k reflects both `*t`
and `*k` in Proto-Polynesian; a third lect (Samoan or Tongan) should
let reconciliation distinguish the two cases". In this experiment **the merger
story does not manifest**, and the reason is instructive:

### Hawaiian isn't actually merged for `*t`/`*k` in the usual way

In the conventional reconstruction, Proto-Polynesian `*t → Haw k`
and `*k → Haw ʔ`. So Hawaiian `k` is the **unambiguous** reflex of
`*t`. Hawaiian `ʔ` is the unambiguous reflex of `*k`. There is no
merger **of Hawaiian k** in this pattern — it comes from `*t` and
only from `*t`.

The merger is really on the Samoan/Tahitian side: `*k → Samoan ʔ`,
and Tahitian likewise has `ʔ` for multiple sources. But the corpus
has too few `*k`-cognates in the subset we loaded to exhibit
`Samoan:ʔ` as a distinct correspondence class. Across all 93
unconditioned classes, there is **zero** Samoan `ʔ`. This is a
coverage problem in the 44-set GLED Polynesian subset, not a failure
of the pipeline.

### What the pipeline would do on a better corpus

If the corpus had, say, 10 clear `*k` cognates, we would expect two
distinct classes with Hawaiian `ʔ`:
- one with Samoan `ʔ`, Tahitian `ʔ`, Maori `k`, Rarotongan `k`
  (the `*k → Haw ʔ` reflex)
- everything else

and Samoan `ʔ` itself would be the pivot whose sister tuples would
differ between classes corresponding to different proto-segments.

We could **verify** this by running multi-lect reconciliation on a Polynesian corpus
containing Tongan (which preserves `*k` as `k`) and which has
enough `*k` cognates — ideally 10+. The GLED Polynesian subset we
used doesn't hit that bar.

## Context conditioning (conditioned classes) after consolidation

With the reconciliation consolidation pass (pivot-lect dedup + adaptive BIC
correction + adaptive min-commit floor), context conditioning now commits
**9 conditioned classes** on this corpus. The top by count:

```
count=7  [Haw:ɐ, Mao:ɐ, Rap:ɐ, Rar:ɐ, Sam:ɐ, Tah:ɐ]
         [Haw: prec=[voiced] | Mao: prec=[voiced] | Rap: prec=[voiced]
          | Rar: prec=[consonant] | Sam: prec=[voiced] | Tah: prec=[voiceless]]

count=5  [Haw:k, Mao:t, Rap:t, Rar:t, Sam:t, Tah:t]
         [Haw: foll=[back] | Mao: foll=[back] | Rap: foll=[back]
          | Rar: foll=[back] | Sam: foll=[back] | Tah: prec=[vowel]]

count=4  [Haw:u, Mao:u, Rap:u, Rar:u, Sam:u, Tah:u]     [Rar: pos=final]
count=3  [Haw:o, Mao:o, Rap:o, Rar:o, Sam:o, Tah:o]     [Haw: pos=medial | Mao: pos=medial | Tah: prec=[voiceless]]
count=2  [Haw:i, Mao:i, Rap:i, Rar:i, Sam:i, Tah:i]     [several per-lect pos=final]
count=2  [Haw:k, Mao:t, Rap:t, Rar:t, Sam:t]            [Rap: prec=[vowel]]
count=2  [Haw:l, Mao:r, Rap:r, Rar:r, Tah:r]            [all pivots: prec=[vowel]]
count=2  [Haw:n, Mao:n, Rap:n, Rar:n, Sam:n, Tah:n]     [Haw: foll=[close]]
count=2  [Mao:u, Sam:u]                                 [Mao: prec=[back]]
```

The headline **`*t → Haw k` 6-way class** is now the dominant
consonantal commit (count=5), and its per-lect merged context
shows all pivots agree that `foll=back` conditions the reflex.
This is genuinely a multi-lect rule: no pairwise model could
express "every daughter lect sees back-vowel following environment
at the same position."

The `*l/*r` split (Haw/Sam `l`, others `r`) survives as a
conditioned class with `prec=[vowel]` on every pivot (count=2).

## Weaknesses exposed

* **Too few `*k` cognates.** The headline merger disambiguation
  story requires at least 10 clearly-reconstructed `*k` cognates,
  and the GLED Polynesian subset has essentially none (no Samoan
  `ʔ` classes at all).
* **Context conditioning over-commits in the same way as Romance.** The BIC
  threshold allows count=1 commits that aren't real rules.
* **Pivot-lect duplication.** Context conditioning emits one class per pivot
  lect even when the underlying (segment tuple, effective context)
  is the same. This inflates `conditioned_classes` by a factor of
  up to N (number of lects). The 244 count here is misleadingly
  large — a merged view would have maybe 30–50 genuinely distinct
  commits.
* **No Tongan.** Adding Tongan would dramatically help with the
  `*k` story because Tongan preserves `*k`. It's not in GLED
  though; would need to come from arcaverborum or a separate load.

## What works

* End-to-end multi-lect reconciliation runs in seconds on this corpus.
* Six-way classes form correctly, including the non-trivial
  Hawaiian `k ← *t` shift and the `*l/*r` split.
* The correspondence `Hawaiian:n ← Maori:ŋ` with Tahitian glottalization
  is clearly visible as a single six-way class — this is a very
  clean reconciliation-only result (no single pairwise model could show all
  six reflexes at once).

## Recommended next actions

1. **Pivot-lect deduplication in context conditioning.** Group conditioned
   classes by segment tuple, then keep the commit with the most
   discriminating context across pivot lects. Would shrink the
   conditioned table to a usable size without losing information.
2. **Bigger Polynesian corpus.** Either load multiple GLED subsets
   together (e.g. include Tongan from another family/filter
   combination) or wait for the arcaverborum loader and
   use a larger Lexibank Polynesian subset.
3. **BIC scaling by observation count.** Tighten the context-conditioning BIC
   threshold on small multi-lect corpora so count=1 commits don't
   land.
