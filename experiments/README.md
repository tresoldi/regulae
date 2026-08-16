# Experiments

Real-data experiments on the ``regulae`` package. Each experiment loads
a curated cognate corpus, trains a learned model (pair or multi-lect),
and produces a human-readable report plus a findings writeup.

Experiments are **not tests**. They're open-ended investigations with
narrative findings documents rather than pass/fail assertions.

**Which is why they go stale, and seven of them had.** Nothing runs a findings
document against the engine, so a claim in one can stop being true without
anything failing. An audit on 2026-08-16 checked every experiment that ships a
corpus against the current build. Each affected file carries a
`## Status, 2026-08-16` section at the top saying what reproduces and what does
not; nothing below those sections was rewritten to match the engine.

| directory | status |
|---|---|
| `tone_chinese_like_clean/` | Claims two cross-dimensional rules at confidence 1.00; the engine commits **none** as the corpus loads. |
| `mandarin_historical/` | Claims the Middle Chinese voicing tonogenesis; the engine commits two unrelated tone-to-tone rules instead. |
| `morph_boundary_synthetic/` | Corpus **does not load** -- inline `+` is refused as CLDF/CLTS markup. |
| `stress_conditioned_synthetic/` | Corpus **does not load** -- inline `-` is not a grapheme any feature system covers. |
| `gled_polynesian/`, `gled_romance/`, `arcaverborum_polynesian/` | No corpus in the repository; the data is not vendored. |

The first two share one cause, and it is a property of the engine rather than
of either corpus. **Cross-dimensional discovery looks from one side of a lect
pair only**, and the side is whichever lect id sorts first. Both corpora put the
derived lect first alphabetically -- `cantonese` before `mandarin`, `mandarin`
before `middle_chinese` -- so the stage asks whether the daughter predicts the
ancestor, which is the direction the rule is not in. Rename the lects so the
source sorts first and both produce exactly what their findings claim.

Conditioned correspondences do not have this problem: they are searched from
both sides, which is what `context_is_target` is for, and the header comment on
`rg_conditioned_segment_count_row` argues the case at length -- a change is only
visible from the side that has the split, and which side that is depends on
which lect happened to sort first. The same argument applies to
cross-dimensional rules and has not been applied to them. That the sibling
synthetic fixtures still reproduce is the tell: `umlaut_synthetic`,
`harmony_synthetic` and `length_conditioned_synthetic` all come back as
*target-side* conditioned correspondences.

It is worth stating as what it is. `src/multilect.c` sorts lects into ascending
id order and explains why in a comment ending **"a name is metadata, and no
analysis may turn on it"** -- an invariant added after renaming a lect changed a
quarter of the published classes on real data. Cross-dimensional discovery turns
on it. On `experiments/tone_chinese_like_clean/`, renaming a lect without
changing which one sorts first changes nothing, and renaming one so that the
other sorts first changes the model:

```
original            (cantonese first): 0 rules
cantonese renamed   (cantonese first): 0 rules
mandarin  renamed   (mandarin  first): 2 rules
```

Everything else audited clean, and none of it was affected by the 2026-08-15/16
structural work: the pre-pass build produces the same results.

## Index

**Pair experiments:**

| directory | pair | purpose |
|---|---|---|
| [`latin_spanish/`](latin_spanish/) | Latin -- Spanish | First real-data test. Messy mixed-profile Romance data with many context-conditioned changes. |
| [`ppn_hawaiian/`](ppn_hawaiian/) | Proto-Polynesian -- Hawaiian | Cross-family validation on clean exceptionless mergers, small phoneme inventory. |
| [`oe_english/`](oe_english/) | Old English -- Modern English | Third family (Germanic). Tests chain shifts (Great Vowel Shift) rather than mergers. |
| [`latin_french/`](latin_french/) | Latin -- French | Extreme reductive Romance. Tests whether context discovery finds *different* conditioning profile than Spanish (same family). |
| [`latin_italian/`](latin_italian/) | Latin -- Italian | Conservative Romance. Completes the 3-way Romance comparison (French extreme, Spanish moderate, Italian conservative). |
| [`finnish_estonian/`](finnish_estonian/) | Finnish -- Estonian | Uralic. Vowel harmony (long-range), vowel length, consonant gradation. |
| [`turkish_azerbaijani/`](turkish_azerbaijani/) | Turkish -- Azerbaijani | Turkic. Vowel harmony, agglutinative morphology (verb infinitives annotated), k/q alternation. |
| [`arabic_hebrew/`](arabic_hebrew/) | Arabic -- Hebrew | Semitic (non-IE/non-Austronesian). Pharyngeals, emphatics, ḍād/ṣade correspondence. |
| [`georgian_svan/`](georgian_svan/) | Georgian -- Svan | Kartvelian. Ejectives, consonant clusters, Geo-t/Svan-šd correspondence. |
| [`mandarin_historical/`](mandarin_historical/) | Middle Chinese -- Mandarin | **Real-data tonogenesis.** Recovers MC voiced-onset → Mandarin tone-2 as a cross-dimensional rule. |
| [`swahili_zulu/`](swahili_zulu/) | Swahili -- Zulu | Bantu. Noun-class prefixes (boundaries annotated), implosives, spirantization. |
| [`navajo_chipewyan/`](navajo_chipewyan/) | Navajo -- Chipewyan | Athabaskan. Complex ejectives, lateral fricatives, tones (parser extracts combining acute/grave into `Segment.tone`). |

**Multi-lect experiments:**

| directory | lects | data source | purpose |
|---|---|---|---|
| [`gled_romance/`](gled_romance/) | 7-way Romance | GLED 20221127 | First multi-lect real-data run. 51 cognate sets. Thin but recovers the Latin `*w -> v` split and the palatalized `*k` series. |
| [`gled_polynesian/`](gled_polynesian/) | 6-way Polynesian | GLED 20221127 | 44 sets, no Tongan. Recovers `*t -> Haw k` chain shift, `*l/*r` liquid split, and the Hawaiian `n / Maori ng / Tahitian glottal stop` velar nasal split. |
| [`arcaverborum_polynesian/`](arcaverborum_polynesian/) | 7-way Polynesian | arcaverborum CoreCog | **Hawaiian merger disambiguation as a measured result.** 329 sets including Tongan. 9 distinct multi-lect classes have `Samoan:glottal stop` coinciding with `Tongan:k` (21+ supporting observations), demonstrating recovery of the Proto-Polynesian `*k vs *glottal stop` distinction that pairwise training cannot see. |

**Synthetic and semi-synthetic tonal experiments:**

| directory | type | purpose |
|---|---|---|
| [`tone_synthetic/`](tone_synthetic/) | synthetic | 30 toy pairs with clean known tonal shifts |
| [`tone_yoruba_like/`](tone_yoruba_like/) | semi-synthetic | 84 Yoruba-style pairs with tone-lowering pattern |
| [`tone_vietnamese_like/`](tone_vietnamese_like/) | semi-synthetic | 74 pairs with 6-tone inventory |
| [`tone_chinese_like/`](tone_chinese_like/) | semi-synthetic | ~70 pairs designed to expose tonogenesis (cross-dimensional discovery motivation) |
| [`tone_chinese_like_clean/`](tone_chinese_like_clean/) | semi-synthetic | Cleaned version with unambiguous tonogenesis signal |
| [`tone_3way_synthetic/`](tone_3way_synthetic/) | synthetic | 3-way tonal correspondence fixture |

**Synthetic segmental experiments:**

| directory | type | purpose |
|---|---|---|
| [`umlaut_synthetic/`](umlaut_synthetic/) | synthetic | Vowel fronting conditioned by following-syllable vowel (long-range context) |
| [`harmony_synthetic/`](harmony_synthetic/) | synthetic | Vowel harmony conditioning (long-range context) |
| [`length_conditioned_synthetic/`](length_conditioned_synthetic/) | synthetic | Length-conditioned consonant lenition (`b → β / [long:+]_`) |
| [`stress_conditioned_synthetic/`](stress_conditioned_synthetic/) | synthetic | Stress-conditioned vowel lowering (`e → ɛ / [stress:+]`) |
| [`morph_boundary_synthetic/`](morph_boundary_synthetic/) | synthetic | Morpheme-boundary chunk filter (rejects stem-suffix straddles) |
| [`contaminated_cognates_synthetic/`](contaminated_cognates_synthetic/) | synthetic | Confidence-weighted training suppresses mis-labeled pairs |

Each experiment directory contains:

* `cognates.tsv` -- the corpus (pair experiments only; multi-lect
  experiments load from external GLED or arcaverborum files)
* `run_experiment.py` -- script that loads, trains, and reports
* `findings.md` -- narrative writeup of what the experiment revealed

## Running an experiment

```sh
cd /home/tiagot/nas-dev/new_chl/regulae
source ~/.venvs/new_chl/bin/activate
python experiments/<dir>/run_experiment.py
```

Each experiment is deterministic -- same corpus, same output.
