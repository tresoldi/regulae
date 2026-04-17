# length_conditioned_synthetic — findings

Synthetic fixture validating length-conditioned context discovery.

## Setup

30 VCV cognate pairs with a single clean rule:

    b → β / [+long] _

- 12 pairs where V1 is long (aː, eː, iː, oː, uː): /b/ lenites to /β/.
- 12 pairs where V1 is short: /b/ preserved.
- 6 control pairs without /b/ (t instead) for diagnostic contrast.

## What the model recovered

With `long` in the context-split feature inventory (and in the
DP's `_CONTEXT_FEATURES`), the framework commits:

    b → β / [long:+] _    count=12

as a conditioned segment correspondence, exactly the rule generating
the data. A few chunks (`(aːb, aːβ)`, `(eːb, eːβ)`) also get
promoted — redundant with the context split but not harmful;
downstream consumers can filter chunks via `chunk_min_transparency`.

## Why this fixture exists

Before this round, the framework's context-split inventory covered
frontness, voicing, place, manner, but **not length**. A
length-triggered rule would fail to commit: the data supports it,
but no candidate predicate in the enumeration ever tested
`preceding=[long:+]`, so BIC never saw the split.

Adding `long` to:

* `_discovery._SPLIT_FEATURE_INVENTORY` (immediate-neighbour)
* `_discovery._LONG_RANGE_FEATURES` (long-range)
* `search._CONTEXT_FEATURES` (what gets populated into link
  contexts during alignment)

closes the gap. This fixture pins the behavior.

## Relation to real data

Length conditioning is pervasive in historical phonology:

- Compensatory lengthening (loss of a consonant triggers a length
  increase on the neighboring vowel).
- Osthoff's Law (PIE shortening of long vowels before resonant
  clusters).
- Verner-like conditioning on unstressed long vowels in Proto-
  Germanic.
- Classical Latin / Romance: long vowel diachronics often differ
  from short ones, and the framework now distinguishes them by
  grapheme (aː vs a as separate sources) AND by context (features
  of neighboring long segments as a predicate).

The companion test in `tests/test_length_conditioning.py` pins
both behaviors: the commitment test against this fixture, and a
smoke test that OE/ModE training preserves the long/short
distinction as separate source keys.
