# arcaverborum Polynesian experiment (multi-lect reconciliation follow-up)

This is the follow-up to `experiments/gled_polynesian`. The GLED
run (44 cognate sets, no Tongan) couldn't exhibit the Hawaiian /
Samoan merger disambiguation because the corpus had essentially
no `*k` cognates. This run uses a Walworth-curated Polynesian
wordlist from arcaverborum that includes **Tongan**, which
preserves the Proto-Polynesian `*k` distinction. The merger
disambiguation manifests cleanly.

## Status, 2026-08-16: the corpus is not in the repository

This directory holds findings only. The Arca Verborum data is not vendored -- those
datasets carry their own licences -- so the run below cannot be reproduced from
a checkout, and the source paths in it are wherever the data sat on the machine
that produced it.

`scripts/lexibank.py` is the supported route from a CLDF wordlist with expert
cognate judgements to a regulae corpus, and `regulae check <corpus>` reports
what a given dataset costs in graphemes the feature system cannot read before
training on it.

## Setup

- **Source**: `arcaverborum A-corecog-20251008` (Zenodo DOI
  10.5281/zenodo.17294927), dataset `walworthpolynesian`
- **Lects**: Hawaiian, Samoan, Tongan, Maori, Tahitian, Niuean, Tuvalu
- **Cognate sets loaded**: 329
- **Size distribution**: 59 seven-way, 42 six-way, 32 five-way, 46
  four-way, 51 three-way, 99 two-way
- **Per-lect coverage**: Sam 241 / Niu 201 / Tuv 202 / Haw 190 /
  Ton 189 / Mao 182 / Tah 155

## Model

- 21 per-pair `LearnedModel` instances (full 7-choose-2 coverage)
- **566 unconditioned multi-lect classes**
- **97 conditioned multi-lect classes** from context conditioning

## The headline: Hawaiian / Samoan `*k ↔ *ʔ` merger disambiguation

### Unconditioned signal

```
count=15  [7-way] Haw:k  Mao:t  Niu:t  Sam:t  Tah:t  Ton:t  Tuv:t
           (Proto *t → Hawaiian k — the chain-shifted reflex)

count= 6  [7-way] Haw:ʔ  Mao:k  Niu:k  Sam:ʔ  Tah:ʔ  Ton:k  Tuv:k
           (Proto *k → Hawaiian/Samoan/Tahitian glottal ; Tongan/Maori/
            Niuean/Tuvalu preserve k)
```

These two classes are the two halves of the Polynesian consonant
story:

1. `*t → Haw k` is the chain-shift that fills the "empty" k slot
   after `*k` glottalized away in Hawaiian. No other lect in this
   set turns `*t` into `k`, so the 7-way correspondence is
   unambiguously `*t`.

2. `*k → Haw ʔ` is the glottalization in the Eastern Polynesian
   group (Haw/Sam/Tah). The fact that the k in that class aligns to
   `k` in Tongan (and Mao/Niu/Tuv) is the **merger disambiguation**:
   without Tongan, the pipeline would see only `Haw:ʔ` vs `Sam:ʔ`
   vs `Tah:ʔ` and couldn't tell `*k` from `*ʔ`.

### Diagnostic: Samoan `ʔ` with Tongan `k`

A stricter test of the disambiguation: count how many distinct
classes have **both** Samoan `ʔ` and Tongan `k`. If there are any
such classes at all, the multi-lect pipeline is getting the signal
that the Samoan glottal came from proto `*k`, not from proto `*ʔ`.

```
DIAGNOSTIC: Sam:ʔ + Ton:k classes: 9

count=6  Haw:ʔ Mao:k Niu:k Sam:ʔ Tah:ʔ Ton:k Tuv:k    (7-way)
count=5  Sam:ʔ Ton:k Tuv:k                              (3-way)
count=3  Niu:k Sam:ʔ Ton:k                              (3-way)
count=2  Niu:k Sam:ʔ Ton:k Tuv:k                        (4-way)
count=1  Haw:ʔ Mao:k Sam:ʔ Ton:k                        (4-way)
count=1  Haw:ʔ Niu:k Sam:ʔ Ton:k Tuv:k                  (5-way)
count=1  Mao:k Niu:k Sam:ʔ Tah:ʔ Ton:k                  (5-way)
count=1  Mao:k Niu:k Sam:ʔ Ton:k Tuv:k                  (5-way)
count=1  Sam:ʔ Ton:k                                    (2-way)
```

Nine classes. Total supporting evidence: 6+5+3+2+1×5 = 21
observations where Samoan's glottal is aligned to Tongan's `k`.
This is precisely the `*k → Sam ʔ` reflex that pairwise context discovery
(Samoan ↔ Hawaiian alone, or Samoan ↔ Maori alone) can't
distinguish from `*ʔ → Sam ʔ`.

### Context-conditioning commits

The multi-lect context-conditioning stage commits both halves of the story as
conditioned classes with per-lect context:

```
count=8   [Haw:k, Mao:t, Niu:t, Sam:t, Tah:t, Ton:t, Tuv:t]
           [Haw prec=vowel | Mao prec=vowel | Niu prec=front
            | Sam foll=open | Tah prec=front | Ton foll=front
            | Tuv foll=open]

count=6   [Haw:ʔ, Mao:k, Niu:k, Sam:ʔ, Tah:ʔ, Ton:k, Tuv:k]
           [Haw foll=front | Mao foll=front | Niu foll=front
            | Sam foll=open | Tah foll=open | Ton foll=front
            | Tuv foll=open]
```

Notice that context conditioning picks slightly different splitting environments
on the different pivots — that's expected since each pivot runs its
own split loop on its own local context distribution, and the dedup
pass merges them into one multi-lect class with contexts contributed
by every pivot that found a signal.

## Other Polynesian correspondences recovered

```
count=11  Haw:l Mao:r Niu:l Sam:l Tah:r Ton:l Tuv:l
          (*l/*r split: Eastern Polynesian r vs Western l)

count= 9  Haw:o Mao:o Niu:o Sam:o Tah:o Ton:o Tuv:o   (vowel *o)
count=10  Haw:i Mao:i Niu:i Sam:i Tah:i Ton:i Tuv:i   (vowel *i)
count=19  Haw:a Mao:a Niu:a Sam:a Tah:a Ton:a Tuv:a   (vowel *a)
count=16  Haw:u Mao:u Niu:u Sam:u Tah:u Ton:u Tuv:u   (vowel *u)
```

The vowels are all stable across the seven lects. The `*l/*r`
split partitions the lects into Western (Haw/Niu/Sam/Ton/Tuv with
`l`) and Eastern (Mao/Tah with `r`).

## What works

* The merger disambiguation is present and attested in 9 distinct
  classes totaling 21 observations. The motivating multi-lect reconciliation
  example from the spec is now a **measured result** on real data.
* 329 cognate sets is enough for context conditioning to commit 97 non-trivial
  conditioned classes, including both halves of the `*t`/`*k`
  story.
* All the classic Polynesian vowel and `*l/*r` correspondences
  recover correctly as high-count 7-way classes.
* Tongan as anchor works exactly as predicted: its `k` column is
  the "proto-grapheme indicator" that lets the multi-lect flow
  disambiguate what Samoan / Hawaiian / Tahitian glottal actually
  reflects.

## Weaknesses and open items

* **One dataset per language.** walworthpolynesian is a single
  curated wordlist; merging multiple arcaverborum datasets on
  Glottocode (Option 2.2c from the plan) would give denser
  coverage. Deferred.
* **"Empty context" splits.** Some of the conditioned
  class entries have contexts like `Sam foll=open` when the
  unconditioned class has `Sam:ʔ` for every observation — the
  split is driven by another pivot's context, not Samoan's. This
  is correct per the algorithm (every pivot contributes its own
  local context if it improves BIC) but it's a bit noisy to read.
  The dedup merges them sensibly; the report format could still
  be tightened.
* **No Proto-Polynesian reconstruction.** The pipeline recovers
  the correspondences but doesn't emit a proto-form. Reconstruction
  is out of multi-lect reconciliation scope and belongs to a downstream
  consumer of the multi-lect class table.
