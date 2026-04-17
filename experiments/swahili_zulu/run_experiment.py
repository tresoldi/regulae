"""Swahili → Zulu cognate alignment experiment.

30 Bantu cognates descended from Proto-Bantu. Exercises:

- **Class-1 prefix**: Swahili m- ↔ Zulu umu- (augment + mu-):
  mtu ~ umuntu, mdomo ~ umlomo, mti ~ umutʰi.
- **Class-5**: Swahili ji- (/dʒi-/) ↔ Zulu i-(li-): dʒino ~ iziɲo,
  dʒitʃo ~ iliso, dʒiwe ~ itʃe.
- **Spirantization before high front vowel**: Proto-Bantu *k → /k/
  in Swahili but /ʃ/ in Zulu: kumi ~ iʃumi "ten".
- **Proto-Bantu *p**: retained plain in Swahili, aspirated in Zulu:
  -pa ~ -pʰa "give", kifua ~ isifuɓa "chest".
- **Proto-Bantu *d**: lenites to /l/ in both: -dɪmi → ulimi ~ ulwimi
  "tongue"; mdomo ~ umlomo "mouth".
- **Implosives in Zulu**: ɓaɓa "father", ɓona "see".
- **Class-9 nasal prefix**: preserved as homorganic /mb/, /nd/ in
  Swahili; full iN- prefix with augment in Zulu.

Morpheme boundaries annotated on noun class prefixes where the
analysis is uncontroversial.

Explicitly excluded: Arabic loans (samaki "fish", damu "blood",
sita "six", saba "seven"). Non-cognates with same gloss (jua
"sun" vs ilanga, kichwa "head" vs ikhanda).

Compiled from Wiktionary Swahili/Zulu/Proto-Bantu entries
(citing BLR3, Guthrie, Meeussen).
"""

from __future__ import annotations

from pathlib import Path

from regulae import (
    Form,
    Segment,
    cognate_sets_from_pairs,
    format_model,
    train_model,
)


_MULTI = ("tʃ", "dʒ", "tʰ", "pʰ", "kʰ", "aː", "eː", "iː", "oː", "uː")


def parse_form(ipa: str) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    i = 0
    while i < len(ipa):
        hit = False
        for cluster in _MULTI:
            if ipa[i : i + len(cluster)] == cluster:
                segments.append(Segment(cluster))
                i += len(cluster)
                hit = True
                break
        if not hit:
            segments.append(Segment(ipa[i]))
            i += 1
    return tuple(segments)


def load_corpus(tsv_path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with tsv_path.open() as f:
        header = f.readline().strip().split("\t")
        assert header[:3] == ["gloss", "swahili", "zulu"]
        has_breaks = header[3:] == ["swahili_breaks", "zulu_breaks"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gloss, s, z = parts[0], parts[1], parts[2]
            s_breaks: tuple[int, ...] = ()
            z_breaks: tuple[int, ...] = ()
            if has_breaks and len(parts) >= 5:
                if parts[3] != "-":
                    s_breaks = tuple(int(x) for x in parts[3].split(","))
                if parts[4] != "-":
                    z_breaks = tuple(int(x) for x in parts[4].split(","))
            src = Form(lect_id="swahili", segments=parse_form(s), morpheme_breaks=s_breaks)
            tgt = Form(lect_id="zulu", segments=parse_form(z), morpheme_breaks=z_breaks)
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Swahili → Zulu cognate pairs.")
    cognate_corpus = cognate_sets_from_pairs(pairs, ("swahili", "zulu"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"swahili", "zulu"})]
    print(format_model(trained, top_segments=20))


if __name__ == "__main__":
    main()
