"""Arabic → Hebrew cognate alignment experiment.

32 cognates inherited from Proto-Semitic. Exercises:

- **Consonant correspondence systems**:
  - Arabic θ (thāʾ) ↔ Hebrew ʃ (shin): ``θalaːθa ~ ʃaloʃ``
  - Arabic dˤ (ḍād) ↔ Hebrew tsˤ (ṣade): ``ʔardˤ ~ ʔeretsˤ``
  - Arabic χ (khāʾ) ↔ Hebrew ħ (ḥet): ``ʔaχ ~ ʔaħ``
  - Arabic s (sīn) ↔ Hebrew ʃ (shin) in some contexts:
    ``sinn ~ ʃen``, ``samaːʔ ~ ʃamajim``
  - Pharyngeals ʕ, ħ are preserved in both languages.

- **Vowel reductions and epenthesis**:
  - Arabic CvC roots retain full vowels; Hebrew often reduces
    to segolate patterns (``kalb → kelev``, ``bajt → bajit``).

- **Emphatic consonants** (pharyngealized) are preserved
  systematically: Arabic dˤ, sˤ, tˤ, ðˤ all have Hebrew
  correspondents though the specific mapping involves
  mergers (Arabic dˤ/ðˤ both → Hebrew tsˤ).

These are the classic Proto-Semitic consonant correspondences;
a reliable recovery of the consonantal system is a strong
validation of the framework on non-Indo-European/non-Austronesian
data.

Hebrew forms are typically biblical/classical readings with
modern Israeli Hebrew phonological realization (tsˤ for ṣade).
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
    "aː", "eː", "iː", "oː", "uː",
    "tsˤ", "ts", "dˤ", "tˤ", "sˤ", "zˤ", "ðˤ",
    "tʃ", "dʒ",
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
        assert header[:3] == ["gloss", "arabic", "hebrew"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gloss, ar, he = parts[0], parts[1], parts[2]
            src = Form(lect_id="arabic", segments=parse_form(ar))
            tgt = Form(lect_id="hebrew", segments=parse_form(he))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Arabic → Hebrew cognate pairs.")
    cognate_corpus = cognate_sets_from_pairs(pairs, ("arabic", "hebrew"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"arabic", "hebrew"})]
    print(format_model(trained, top_segments=20))


if __name__ == "__main__":
    main()
