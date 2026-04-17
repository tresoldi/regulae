# swahili_zulu — findings

30 Bantu cognates descended from Proto-Bantu.

## Correspondences recovered

- **Proto-Bantu *p → Zulu pʰ (aspirated)**: committed as
  `p → pʰ` with traces in feature displacements. Swahili
  preserves plain /p/.
- **Proto-Bantu *b → Zulu ɓ (implosive)**: `b → ɓ / _[open:+]`
  committed as a conditioned split (2 observations). Swahili
  preserves plain /b/.
- **Proto-Bantu *k before /i/ → Zulu /ʃ/**: the spirantisation
  `kumi ~ iʃumi` "ten" surfaces in the segment table.
- **Class-1 prefix m- ↔ umu-**: Swahili `mtu` vs Zulu `umuntu`.
  The morpheme-boundary annotation keeps these from being
  promoted as boundary-crossing chunks.
- **Class-5 dʒ- ↔ izi-/ili-/it-**: `dʒino ~ iziɲo` "tooth",
  `dʒitʃo ~ iliso` "eye", `dʒiwe ~ itʃe` "stone".

## What this validates

- The framework handles Bantu-specific implosives (ɓ),
  prenasalised initials (mb, nd, mv as two-segment), and the
  noun-class prefix morphology via explicit boundary annotation.
- Spirantisation before high front vowels as a conditioning
  rule is detected where data supports it.
- Arabic loans (explicitly filtered at compile time) don't leak
  correspondence patterns that contradict Proto-Bantu reflexes.

## Sources

Compiled by delegated agent from Wiktionary Swahili/Zulu entries
citing BLR3 (Bantu Lexical Reconstructions v3, Guthrie, Meeussen)
and the Proto-Bantu Wikipedia page.

Arabic loanwords explicitly excluded: samaki "fish", damu "blood",
sita "six", saba "seven", kitabu "book", rafiki "friend".
Non-cognates with same gloss also excluded.
