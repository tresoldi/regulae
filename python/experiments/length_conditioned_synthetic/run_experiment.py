"""Synthetic length-conditioned fixture.

**DATA**: 30 VCV pairs with a single clean length-conditioned rule:

    b -> β  when the preceding vowel is long (aː, eː, iː, oː, uː)
    b -> b  when the preceding vowel is short

The corpus has three classes:

- 12 pairs where V1 is long and C2 = b: V2 → β
- 12 pairs where V1 is short and C2 = b: V2 preserved as b
- 6 control pairs without /b/ (t instead)

Expected output:

- A committed conditioned correspondence
  ``b → β`` with context ``preceding=[long:+]``.
- Validation: ``long`` in the feature inventory is what makes this
  commit possible; without it, the fixture would have to rely on
  per-vowel splits (``preceding=[aː]``, ``preceding=[eː]``, ...)
  which the framework cannot enumerate.

Runs as an ``experiment``, not a ``test``: produces a human-readable
report. The test suite pins the behavior in
``tests/test_length_conditioning.py``.
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


_MULTI_GRAPHEMES = ("aː", "eː", "iː", "oː", "uː")


def parse_form(ipa: str) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    i = 0
    while i < len(ipa):
        hit = False
        for cluster in _MULTI_GRAPHEMES:
            if ipa[i : i + len(cluster)] == cluster:
                segments.append(Segment(grapheme=cluster))
                i += len(cluster)
                hit = True
                break
        if not hit:
            segments.append(Segment(grapheme=ipa[i]))
            i += 1
    return tuple(segments)


def load_corpus(path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "proto", "derived"], header
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, proto, derived = parts
            src = Form(lect_id="proto", segments=parse_form(proto))
            tgt = Form(lect_id="derived", segments=parse_form(derived))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} length-conditioned synthetic pairs.")
    print("Rule: b → β / [+long] _")
    print()

    cognate_corpus = cognate_sets_from_pairs(corpus, ("proto", "derived"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"proto", "derived"})]

    print(format_model(trained, top_segments=20))

    print()
    print("Length-conditioned segment entries:")
    for cc, count in trained.segment_table.counts.items():
        ctx = cc.context
        is_length_conditioned = any(
            fc.feature == "long"
            for fc in (*ctx.preceding, *ctx.following)
        )
        if not is_length_conditioned:
            continue
        parts = []
        if ctx.preceding:
            parts.append(
                "prec="
                + ",".join(f"{fc.feature}:{fc.value}" for fc in ctx.preceding)
            )
        if ctx.following:
            parts.append(
                "foll="
                + ",".join(f"{fc.feature}:{fc.value}" for fc in ctx.following)
            )
        print(f"  {cc.src} -> {cc.tgt}: count={count:.1f}  [{' | '.join(parts)}]")


if __name__ == "__main__":
    main()
