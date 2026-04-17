# navajo_chipewyan — findings

32 Athabaskan cognates compiled from ASJP, Wiktionary Proto-
Athabaskan reconstructions, Alderete's phonology work, Krauss
(1964) tables, Carleton's Chipewyan-Apachean wordlist, and
McDonough's IPA transcription of Krauss (2005).

## Correspondences recovered

- **Proto-Athabaskan *s ↔ Chipewyan θ**: committed as
  `s → θ` (2 observations). Classic Dëne Sųłiné interdental
  shift.
- **Proto-Athabaskan *ʃ ↔ Chipewyan s**: `s → tʃʰ / _[front:+]`
  and `s → ð / _[front:+]` committed as conditioned splits.
- **Chipewyan vowel lowering**: `i → e` (4 observations) —
  Chipewyan lowers some Proto-Athabaskan high vowels.
- **Chipewyan rounding shift**: `oː → u` (3 observations).
- **Vowel length**: Navajo preserves more long vowels
  (`aː → a: 5`, `iː → i: 2`). Chipewyan shortens widely; the
  displacement distribution shows `[long: P->A]: 12` as the
  second-most-common feature shift.
- **Consonant lenition**: `d → r` in intervocalic position.
- **Context-conditioned split**: `d → t / [front:+]_` captured
  as a conditioned correspondence.

## Tones

Both languages are tonal. The extracted tones are represented
as `Segment.tone = "H"` (from combining acute) or `"L"` (from
combining grave). The tonal correspondence table captures:

```
H -> H: 14
L -> L: 10
H -> L: 3
L -> H: 1
```

The tone polarity is mostly preserved, with a few cases of
H↔L flip (consistent with the well-known tone-polarity
reversal in constricted-vowel stems in Proto-Athabaskan
reconstruction).

The cross-dimensional discovery commits a tone rule:

```
tone=L@relative_+1 -> tone=L@+1   count=11/14  conf=0.79
```

## What this validates

- The framework handles a complex inventory of ejectives,
  aspirated stops, and lateral fricatives (ɬ, tɬʰ).
- Nasalized vowels (transcribed with combining tilde) are
  read correctly by the parser.
- Tones are correctly extracted from combining diacritics and
  routed through the tonal correspondence layer.
- Tonal correspondences with polarity flips are visible in
  the tone table without polluting segment-level counts.

## Transcription / parser notes

The parse_form function in this experiment:

1. Applies NFD then recomposes nasalised vowels (base +
   combining tilde) into multi-char graphemes.
2. Extracts combining acute / grave as tones (H/L).
3. Appends length mark ``ː`` to the segment's grapheme so
   merkmal sees the long form.

The agent-produced TSV was cleaned up: ogonek-nasals (į, ǫ, ų, ę)
converted to combining-tilde nasals (ĩ, õ, ũ, ẽ) to match merkmal's
grapheme inventory. The ``*_breaks`` columns from the agent output
were stripped because they contained IPA repeats rather than
integer break positions — the forms are bare stems without
transparent morphology, so no boundaries are annotated.

## Sources

Agent compilation cited: ASJP Navajo/Chipewyan wordlists,
Wiktionary Proto-Athabaskan reconstruction pages, Carleton
Chipewyan-Apachean wordlist, Alderete's Tahltan affricate
paper, Wikipedia Proto-Athabaskan and Chipewyan articles,
McDonough's Krauss 2005 tonogenesis transcription.
