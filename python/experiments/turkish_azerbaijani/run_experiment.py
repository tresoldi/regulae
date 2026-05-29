"""Turkish → Azerbaijani cognate alignment experiment.

40 basic-vocabulary cognates (body parts, numerals, nature,
verbs) compiled from Wiktionary. Exercises:

- Word-initial k ~ q (Turkish k ~ Azerbaijani ɡ/q — preserved
  uvular reflex in Azerbaijani).
- Word-final k ~ χ (Turkish `ajak` ~ Azerbaijani `ajaχ`).
- ğ fortition: Turkish silent ``ː`` (long vowel from lost ğ)
  ~ Azerbaijani retained [ɣ].
- Vowel harmony between root and verb suffix.
- Morpheme boundaries for the -mek/-mək infinitive suffix
  are annotated on the 8 verbs.
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


_MULTI = ("tʃ", "dʒ", "aː", "eː", "iː", "oː", "uː", "øː", "yː", "æː", "əː", "ɑː")


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
        assert header[:3] == ["gloss", "turkish", "azerbaijani"]
        has_breaks = header[3:] == ["turkish_breaks", "azerbaijani_breaks"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gloss, t, a = parts[0], parts[1], parts[2]
            t_breaks: tuple[int, ...] = ()
            a_breaks: tuple[int, ...] = ()
            if has_breaks and len(parts) >= 5:
                if parts[3] != "-":
                    t_breaks = tuple(int(x) for x in parts[3].split(","))
                if parts[4] != "-":
                    a_breaks = tuple(int(x) for x in parts[4].split(","))
            src = Form(lect_id="turkish", segments=parse_form(t), morpheme_breaks=t_breaks)
            tgt = Form(lect_id="azerbaijani", segments=parse_form(a), morpheme_breaks=a_breaks)
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Turkish → Azerbaijani cognate pairs.")
    print()
    cognate_corpus = cognate_sets_from_pairs(pairs, ("turkish", "azerbaijani"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"turkish", "azerbaijani"})]
    print(format_model(trained, top_segments=20))


if __name__ == "__main__":
    main()
