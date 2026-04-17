"""Synthetic contaminated-cognates fixture.

**DATA**: 25 pairs total. 20 clean cognates with the regular
correspondence ``p → f`` (at confidence 1.0); 5 deliberately
mis-labeled pairs with nonsensical correspondences (at confidence
0.0).

With ``confidence=0.0`` on the contaminated pairs, training must
ignore their contribution to all counts, so the recovered model
looks exactly like training on the clean 20 pairs alone.

Expected output:

- The ``p → f`` correspondence dominates the segment table.
- No noise from the contaminated pairs shows up in counts,
  displacement distribution, or chunk table.
- A control run where the bad pairs are included at confidence
  1.0 shows noise in the counts, confirming the difference.

This is the companion to the confidence-weighting tests: the
synthetic fixture lets a user visually see that the bad pairs are
indeed "inspected but not learned from" (Theme 0 framing).
"""

from __future__ import annotations

from pathlib import Path

from regulae import (
    CognateSet,
    Form,
    Segment,
    train_model,
)


def _parse(ipa: str) -> tuple[Segment, ...]:
    return tuple(Segment(c) for c in ipa)


def load_corpus(path: Path) -> list[CognateSet]:
    corpus: list[CognateSet] = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "proto", "derived", "confidence"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            gloss, proto, derived, conf = parts
            src = Form(lect_id="proto", segments=_parse(proto))
            tgt = Form(lect_id="derived", segments=_parse(derived))
            corpus.append(
                CognateSet(
                    cognate_id=gloss,
                    forms={"proto": src, "derived": tgt},
                    confidence=float(conf),
                )
            )
    return corpus


def load_corpus_all_equal(path: Path) -> list[CognateSet]:
    """Control: load the same corpus with confidence=1.0 on every
    pair (i.e., the contaminated pairs treated as real data)."""
    out = []
    for cs in load_corpus(path):
        out.append(
            CognateSet(
                cognate_id=cs.cognate_id,
                forms=cs.forms,
                confidence=1.0,
            )
        )
    return out


def _summarise(model) -> None:
    pair = model.pairwise_models[frozenset({"proto", "derived"})]
    print("Segment table (p source):")
    items = [
        (cc, c)
        for cc, c in pair.segment_table.counts.items()
        if cc.src == "p" and cc.context.constraint_count() == 0
    ]
    for cc, c in sorted(items, key=lambda x: -x[1]):
        print(f"  p -> {cc.tgt}: count={c:.1f}")


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    corpus = load_corpus(tsv)
    control = load_corpus_all_equal(tsv)

    print(f"Loaded {len(corpus)} cognates (5 contaminated at confidence=0).")
    print()

    print("=== With confidence weighting (bad pairs at 0.0) ===")
    m = train_model(corpus)
    _summarise(m)
    print()

    print("=== Control: all pairs at confidence=1.0 ===")
    m_ctrl = train_model(control)
    _summarise(m_ctrl)


if __name__ == "__main__":
    main()
