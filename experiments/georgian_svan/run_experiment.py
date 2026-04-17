"""Georgian → Svan cognate alignment experiment.

30 Kartvelian cognates verified against Klimov's Etymological
Dictionary of the Kartvelian Languages (1998), Tuite's Svan
grammar, and Wiktionary Kartvelian entries.

Correspondences validated:

- Georgian /t/ ~ Svan /ʃd/ cluster: atʰi ~ jeʃd "ten",
  tʼili ~ ʃdim "louse", datʼvi ~ dəʃdw "bear". Classic
  Kartvelian t : Zan t : Svan šd correspondence.
- Georgian /e/ ~ Svan /i/: deda ~ di "mother", ena ~ nin "tongue".
- Georgian /s/ ~ Svan /ʃ/: sva ~ ʃwe "drink".
- Ejectives preserved in both branches: tsʼida ~ tsʼid "dirt",
  tʼba ~ tʼob "lake", mkʼerdi ~ mətʃʼed "chest".
- Svan cluster-initial w-: otʰxi ~ woʃtʰxw "four", xutʰi ~ woxuʃd
  "five".

Shows that the framework handles complex ejective-rich consonant
inventories without special casing.
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


_MULTI = (
    "tʼ", "pʼ", "kʼ", "tsʼ", "tʃʼ", "tʂʼ", "qʼ",
    "tʰ", "pʰ", "kʰ", "tsʰ", "tʃʰ",
    "tʃ", "dʒ", "ts", "dz",
)


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
        assert header[:3] == ["gloss", "georgian", "svan"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gloss, g, s = parts[0], parts[1], parts[2]
            src = Form(lect_id="georgian", segments=parse_form(g))
            tgt = Form(lect_id="svan", segments=parse_form(s))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Georgian → Svan cognate pairs.")
    cognate_corpus = cognate_sets_from_pairs(pairs, ("georgian", "svan"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"georgian", "svan"})]
    print(format_model(trained, top_segments=20))


if __name__ == "__main__":
    main()
