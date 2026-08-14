# Grapheme coverage against Lexibank

What the feature system cannot read, measured over the 92 Lexibank datasets
that carry both expert cognate judgements and CLTS-segmented forms. Generated
by `regulae check` over corpora built with `scripts/lexibank.py`; regenerate it
after any merkmal upgrade.

**3,470,833 segment tokens, 4,097 distinct types. 97.45% of tokens resolve.**

Of the 88,504 tokens that do not, 85,887 are source markup rather than
transcription — `+` alone accounts for 78,864 across 71 datasets — and regulae
now reports those as `RG_ERR_SOURCE_MARKER`, which is a different statement
from "this sound is not covered". That leaves **2,617 tokens across 95 types**
where a transcription cannot be read.

The categories below decide who owns each fix. Only the last is merkmal's.

### mis-encoding / editorial — 5 types, 1,640 tokens

| token | status | tokens | datasets |
| --- | --- | ---: | --- |
| `∼` | unknown grapheme | 1,053 | blumpanotacana, northperulex, oliveiraprotopanoan … |
| `→` | unknown grapheme | 400 | starostinhmongmien |
| `←` | unknown grapheme | 107 | starostinhmongmien |
| `*` | unknown grapheme | 65 | mcd |
| `ε` | unknown grapheme | 15 | walkerarawakan |

### IPA merkmal does not cover — 74 types, 478 tokens

| token | status | tokens | datasets |
| --- | --- | ---: | --- |
| `ɚ` | unknown grapheme | 49 | bdpa, houchinese |
| `ɹ̪` | parse error | 28 | iecor |
| `t͡ʂ` | unknown grapheme | 27 | bdpa |
| `aˁ` | unknown grapheme | 26 | oskolskayatungusic, savelyevturkic |
| `t̪ʲʰ` | parse error | 25 | iecor |
| `b̄` | unknown grapheme | 25 | peirosst |
| `xˁ` | unknown grapheme | 22 | bdpa |
| `t̪ˠʰ` | parse error | 18 | iecor |
| `aːˁ` | unknown grapheme | 17 | oskolskayatungusic |
| `iˁ` | unknown grapheme | 16 | oskolskayatungusic, savelyevturkic |
| `pˢ` | unknown grapheme | 12 | hattorijaponic, peirosaustroasiatic, sidwellbahnaric … |
| `ᶯɖr` | unknown grapheme | 12 | mcd |
| `d̄` | unknown grapheme | 12 | peirosst |
| `ł` | unknown grapheme | 11 | bdpa, peirosst |
| `œʏ` | unknown grapheme | 10 | bdpa |
| `ɯˁ` | unknown grapheme | 10 | savelyevturkic |
| `eˁ` | unknown grapheme | 9 | savelyevturkic |
| `iɚ` | unknown grapheme | 8 | bdpa |
| … | | | 56 more |

### cover symbol — 8 types, 330 tokens

| token | status | tokens | datasets |
| --- | --- | ---: | --- |
| `R` | unknown grapheme | 127 | mcd |
| `T` | unknown grapheme | 104 | mcd |
| `S` | unknown grapheme | 77 | mcd |
| `L` | unknown grapheme | 12 | mcd |
| `V` | unknown grapheme | 5 | mcd |
| `K` | unknown grapheme | 2 | mcd |
| `Z` | unknown grapheme | 2 | mcd |
| `C` | unknown grapheme | 1 | mcd |

### ambiguity or markup — 7 types, 112 tokens

| token | status | tokens | datasets |
| --- | --- | ---: | --- |
| `ᴀ̃/ã̱` | unknown grapheme | 41 | houchinese |
| `ᴀ/a̱` | unknown grapheme | 28 | houchinese |
| `<???>` | unknown grapheme | 25 | mcd |
| `ks/kˢ` | unknown grapheme | 12 | peirosaustroasiatic |
| `ouɚ/oɚ` | unknown grapheme | 4 | bdpa |
| `!/∼` | unknown grapheme | 1 | blumpanotacana |
| `ə́˞/ɚ` | unknown grapheme | 1 | simsrma |

### orthography, not IPA — 1 types, 57 tokens

| token | status | tokens | datasets |
| --- | --- | ---: | --- |
| `qu` | unknown grapheme | 57 | pila |

## What this asks of merkmal

The last category is the one to act on, and it is small: under a thousand
tokens, but they block eleven datasets outright, because one unreadable token
in one form refuses the whole training run.

Two of them are systematic rather than incidental:

- **Pharyngealisation.** `aˁ`, `iˁ`, `eˁ`, `ɯˁ`, `øˁ`, `xˁ`, `aːˁ` — a whole
  series, in Tungusic, Turkic and BDPA. The diacritic is standard IPA and the
  pattern is regular, so this looks like one missing rule rather than seven
  missing graphemes.
- **Dental and secondary-articulation stacking.** `ɹ̪`, `t̪ʲʰ`, `t̪ˠʰ` come back as
  `parse error` rather than `unknown grapheme`, which says the resolver got
  part way and then refused a combination.

The rest are individually small but unsurprising: `ɚ` is standard IPA for a
rhotic schwa, and `t͡ʂ` is a tie-barred retroflex affricate, both of which a
reader would expect to resolve.

## What this does not ask of merkmal

The other categories are regulae's problem or the dataset's, and are handled in
`scripts/lexibank.py`:

- **Mis-encodings.** `∼` is U+223C TILDE OPERATOR, a mathematical symbol
  standing in for a tilde; `ε` is Greek epsilon for IPA `ɛ`; `→` and `←` are
  editorial arrows. Guessing the intent would be inventing data.
- **Ambiguity.** `ks/kˢ` and `ᴀ̃/ã̱` are a source saying it could not decide
  between two readings. Picking one silently would be worse than refusing.
- **Cover symbols.** `*R`, `*T`, `*S`, `*L` in the Micronesian comparative
  dictionary are archiphonemes: a reconstructed segment specified only partly.
  This is real and long-standing notation, and regulae has no way to represent
  an underspecified segment — a gap in regulae's data model, not in merkmal's
  inventory.
