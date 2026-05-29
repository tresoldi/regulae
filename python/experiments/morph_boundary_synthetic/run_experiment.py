"""Synthetic morpheme-boundary fixture.

**DATA**: 16 stem+suffix pairs and 8 monomorphemic controls.

The stem-suffix pairs follow the pattern ``Cat+V → Cad+V``:
intervocalic-like /t/ → /d/ at the stem/suffix boundary. The
``+`` mark in the TSV indicates the morpheme boundary (after the
stem, before the suffix vowel).

The control pairs (no ``+``, identity) are monomorphemic CVC
words; the model should learn ``t → t`` and ``p → p`` etc. as
unconditioned correspondences.

The experiment trains the same data twice:

1. **Without** boundaries — the chunk-promotion stage is free to
   memorise the stem-suffix boundary as a chunk like
   ``(ata, ada)``, lumping the morphological seam into a
   phonological correspondence.
2. **With** boundaries supplied on each Form's
   ``morpheme_breaks`` — the chunk extractor rejects any
   candidate whose source or target span crosses the boundary,
   so morphology-like chunks no longer enter the chunk table.

Compare the two chunk tables to see the boundaries earning their
keep.
"""

from __future__ import annotations

from pathlib import Path

from regulae import (
    Form,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


def _parse_form(lect: str, ipa: str, breaks_field: str) -> Form:
    # Drop "+" marks; positions count over the filtered segment list.
    segments = tuple(Segment(c) for c in ipa if c != "+")
    if breaks_field == "-":
        return Form(lect_id=lect, segments=segments)
    breaks = tuple(int(x) for x in breaks_field.split(","))
    return Form(lect_id=lect, segments=segments, morpheme_breaks=breaks)


def load_corpus(
    path: Path, *, with_boundaries: bool
) -> list[tuple[Form, Form]]:
    pairs: list[tuple[Form, Form]] = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "proto", "derived", "proto_breaks", "derived_breaks"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            _, proto, derived, pb, db = parts
            if not with_boundaries:
                pb = "-"
                db = "-"
            src = _parse_form("proto", proto, pb)
            tgt = _parse_form("derived", derived, db)
            pairs.append((src, tgt))
    return pairs


def _summarise_chunks(model) -> str:
    pair = model.pairwise_models[frozenset({"proto", "derived"})]
    if not pair.chunk_table.entries:
        return "  (none promoted)"
    rows = []
    for (s, t), cost in sorted(pair.chunk_table.entries.items(), key=lambda kv: kv[1]):
        ss = "".join(x.grapheme for x in s)
        tt = "".join(x.grapheme for x in t)
        rows.append(f"  ({ss}, {tt}): cost={cost:.3f}")
    return "\n".join(rows)


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    pairs_no_bounds = load_corpus(tsv, with_boundaries=False)
    pairs_with_bounds = load_corpus(tsv, with_boundaries=True)
    print(f"Loaded {len(pairs_no_bounds)} pairs.")
    print()

    model_no = train_model(cognate_sets_from_pairs(pairs_no_bounds, ("proto", "derived")))
    print("Without morpheme boundaries — chunks promoted:")
    print(_summarise_chunks(model_no))
    print()

    model_yes = train_model(cognate_sets_from_pairs(pairs_with_bounds, ("proto", "derived")))
    print("With morpheme boundaries — chunks promoted:")
    print(_summarise_chunks(model_yes))
    print()

    pair_no = model_no.pairwise_models[frozenset({"proto", "derived"})]
    pair_yes = model_yes.pairwise_models[frozenset({"proto", "derived"})]
    dropped = set(pair_no.chunk_table.entries) - set(pair_yes.chunk_table.entries)
    print(f"Boundary-rejected chunks ({len(dropped)}):")
    for s, t in sorted(dropped, key=lambda k: ("".join(x.grapheme for x in k[0]), "".join(x.grapheme for x in k[1]))):
        ss = "".join(x.grapheme for x in s)
        tt = "".join(x.grapheme for x in t)
        print(f"  ({ss}, {tt})")


if __name__ == "__main__":
    main()
