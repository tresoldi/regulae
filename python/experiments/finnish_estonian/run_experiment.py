"""Finnish → Estonian cognate alignment experiment.

**DATA**: 40 basic-vocabulary cognates from Swadesh-style items
(body parts, kin terms, numerals, core verbs).

**Correspondences validated:**

- Finnish final -a / -ä preserved or lost to match Estonian's
  shorter stem (e.g., kieli → keel, silmä → silm).
- Vowel harmony in Finnish: back vowels (ɑ, o, u) vs front
  vowels (æ, ø, y) cluster by root (visible in the segment
  table's src distribution).
- Finnish consonant gradation: /hd/ → /h/ in some cognates
  (syödä → süüa = /syødæ/ → /syːɑ/).
- Length correspondences: Finnish /iː/ → Estonian /iː/, etc.
- Vowel length is both preserved and newly introduced
  (compensatory lengthening after coda loss, e.g., kieli → keːl).

Morpheme-break annotations are provided on verb infinitives where
the -da/-dä suffix is transparent.

Runs as an ``experiment``, not a ``test``. The test suite in
``tests/test_diverse_corpora.py`` verifies training completes
without errors and the model discovers at least a few segment-level
correspondences.
"""

from __future__ import annotations

from pathlib import Path

from regulae import (
    Form,
    Segment,
    align_forms,
    alignment_cost,
    cognate_sets_from_pairs,
    format_model,
    train_model,
)


_MULTI = (
    "ɑː", "eː", "iː", "oː", "uː", "æː", "øː", "yː",
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
        assert header[:3] == ["gloss", "finnish", "estonian"]
        has_breaks = header[3:] == ["finnish_breaks", "estonian_breaks"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gloss, fin, est = parts[0], parts[1], parts[2]
            fin_breaks: tuple[int, ...] = ()
            est_breaks: tuple[int, ...] = ()
            if has_breaks and len(parts) >= 5:
                if parts[3] != "-":
                    fin_breaks = tuple(int(x) for x in parts[3].split(","))
                if parts[4] != "-":
                    est_breaks = tuple(int(x) for x in parts[4].split(","))
            src = Form(
                lect_id="finnish",
                segments=parse_form(fin),
                morpheme_breaks=fin_breaks,
            )
            tgt = Form(
                lect_id="estonian",
                segments=parse_form(est),
                morpheme_breaks=est_breaks,
            )
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Finnish → Estonian cognate pairs.")
    print()

    cognate_corpus = cognate_sets_from_pairs(pairs, ("finnish", "estonian"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"finnish", "estonian"})]

    print(format_model(trained, top_segments=20))

    print()
    print("Source segments with competing target correspondences:")
    by_src: dict[str, list[tuple[str, float]]] = {}
    for cc, c in trained.segment_table.counts.items():
        if cc.context.constraint_count() != 0:
            continue
        by_src.setdefault(cc.src, []).append((cc.tgt, c))
    for src, tgts in sorted(by_src.items(), key=lambda kv: -sum(c for _, c in kv[1]))[:10]:
        tgts = sorted(tgts, key=lambda x: -x[1])
        parts = ", ".join(f"{t}×{c:.0f}" for t, c in tgts[:3])
        total = sum(c for _, c in tgts)
        extra = f" (+{len(tgts)-3} more)" if len(tgts) > 3 else ""
        if total >= 2:
            print(f"  {src!r}: {parts}{extra}  [total {total:.0f}]")


if __name__ == "__main__":
    main()
