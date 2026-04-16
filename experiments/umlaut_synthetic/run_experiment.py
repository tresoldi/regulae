"""Synthetic umlaut fixture (long-range context discovery validation).

**DATA**: 40 CVCV pairs with a single deliberately clean umlaut rule:

    a -> æ  when the next syllable's vowel is a front vowel (i, e)

The corpus has three classes of 10 and 20 pairs:

- 10 pairs with V1 = a and V2 ∈ {i, e}: V1 → æ (the umlaut firing)
- 10 pairs with V1 = a and V2 ∈ {o, u}: V1 preserved (no firing)
- 20 control pairs with V1 ∈ {i, u} (rule doesn't apply)

Expected output:

- A committed conditioned correspondence
  ``a → æ`` with context ``next_syllable=[front:+]``
  (and its dual framing ``a → a`` for the non-firing case may or
  may not be committed depending on entropy).
- Validation: the signal is single-source, single-outcome,
  exclusively long-range; immediate-neighbour context discovery
  cannot capture it because C2 is a plain consonant from
  ``{p, t, k, b, d, g}`` that doesn't encode the V2 feature.
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


def parse_form(ipa: str) -> tuple[Segment, ...]:
    return tuple(Segment(grapheme=c) for c in ipa)


def load_corpus(path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "proto", "derived"]
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
    print(f"Loaded {len(corpus)} synthetic umlaut pairs.")
    print()
    print("Rule: a → æ when next syllable's vowel is front (i, e).")
    print("      Context discovery cannot capture this (the intervening C")
    print("      doesn't encode the triggering V2 feature).")
    print()

    cognate_corpus = cognate_sets_from_pairs(corpus, ("proto", "derived"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"proto", "derived"})]

    print(format_model(trained, top_segments=20))

    print()
    print("Long-range conditioned entries in detail:")
    for cc, count in trained.segment_table.counts.items():
        ctx = cc.context
        lr_total = (
            len(ctx.preceding_at_distance)
            + len(ctx.following_at_distance)
            + len(ctx.somewhere_preceding)
            + len(ctx.somewhere_following)
            + len(ctx.same_syllable)
            + len(ctx.next_syllable)
            + len(ctx.previous_syllable)
        )
        if lr_total == 0:
            continue
        parts = []
        for name, slot in [
            ("pre@", ctx.preceding_at_distance),
            ("fol@", ctx.following_at_distance),
            ("s_pre", ctx.somewhere_preceding),
            ("s_fol", ctx.somewhere_following),
            ("same_syl", ctx.same_syllable),
            ("next_syl", ctx.next_syllable),
            ("prev_syl", ctx.previous_syllable),
        ]:
            for item in slot:
                if hasattr(item, "feature"):
                    parts.append(f"{name}={item.feature}:{item.value}")
                else:
                    parts.append(f"{name}={item[0]}:{item[1].feature}:{item[1].value}")
        print(f"  {cc.src} -> {cc.tgt}: {count}  [{', '.join(parts)}]")


if __name__ == "__main__":
    main()
