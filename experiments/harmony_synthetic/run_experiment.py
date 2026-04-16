"""Synthetic vowel-harmony fixture (long-range context discovery validation).

**DATA**: 45 CVCV pairs with one deliberately clean progressive
harmony rule:

    a -> o  when the previous syllable's vowel is back (u, o)

The corpus has three classes:

- 15 pairs with V1 ∈ {u, o} and V2 = a: V2 → o in the derived form
  (the rule firing)
- 15 pairs with V1 ∈ {i, e} and V2 = a: V2 preserved (control)
- 5 pairs with V1 = a, V2 = a: V2 preserved (control)
- 10 controls with V2 ≠ a (the rule doesn't apply to them)

Expected output:

- A committed conditioned correspondence
  ``a → o`` with context ``previous_syllable=[back:+]``.
- The dual framing ``a → a`` for the non-firing case is not
  required; long-range discovery's greedy loop commits the single
  highest-entropy-reducing split per source.
- Context discovery alone cannot capture this (the intervening
  consonant doesn't encode V1's backness).

This fixture validates the ``previous_syllable`` long-range slot,
complementing the ``next_syllable`` coverage of
``experiments/umlaut_synthetic/``.
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
    print(f"Loaded {len(corpus)} synthetic harmony pairs.")
    print()
    print("Rule: a → o when previous syllable's vowel is back (u, o).")
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
