"""Synthetic stress-conditioned fixture (vowel lowering under stress).

**DATA**: 36 CVCV pairs with a clean stress-conditioned 1-to-1 rule:

    e → ɛ / [stress:+] (stressed /e/ lowers to /ɛ/)
    o → ɔ / [stress:+] (stressed /o/ lowers to /ɔ/)
    e → e,  o → o     (unstressed: preserved)

The corpus has three classes:

- 12 pairs with stressed e in the first syllable → derived has ɛ
- 10 pairs with stressed o in the first syllable → derived has ɔ
- 14 pairs with the stress on the second syllable and e/o preserved

The IPA primary-stress mark ``ˈ`` in the TSV is parsed into
``Segment.stress = "+"`` on the following segment; ``-`` marks
syllable boundaries (dropped from the segmentation).

The test uses vowel lowering rather than Romance-style
diphthongization (/e/→/je/) because the latter is a 1-to-many
correspondence that lives in the chunk table, not in the
context-conditioned segment table. The stress conditioning
mechanism fits 1-to-1 correspondences cleanly.

Expected output:

- Committed conditioned correspondences
  ``e → ɛ / self_stress=[stress:+]``
  ``o → ɔ / self_stress=[stress:+]``
- Unstressed contexts keep ``e → e`` and ``o → o`` as
  unconditioned fallbacks.
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


_VOWELS = frozenset("aeiouɛɔɐəɨɯæøyɪʊɑɒ")


def parse_form(ipa: str) -> tuple[Segment, ...]:
    """Parse an IPA string with primary-stress marks into Segments.

    Following IPA convention, ``ˈ`` marks the onset of a stressed
    syllable. We attach the stress to the **vowel** of that
    syllable (the syllable nucleus), not the onset consonant, so
    the split-discovery layer can learn rules like
    ``e → ɛ / self_stress=[stress:+]`` that reflect the
    phonological convention.

    - ``ˈ`` triggers stress-pending; the next *vowel* segment gets
      ``stress="+"``.
    - ``-`` is a syllable boundary; dropped from the segmentation.
    - Any other character is a plain grapheme.
    """
    segments: list[Segment] = []
    i = 0
    pending_stress: str | None = None
    while i < len(ipa):
        ch = ipa[i]
        if ch == "ˈ":
            pending_stress = "+"
            i += 1
            continue
        if ch == "-":
            i += 1
            continue
        if ch in _VOWELS and pending_stress is not None:
            segments.append(Segment(grapheme=ch, stress=pending_stress))
            pending_stress = None
        else:
            segments.append(Segment(grapheme=ch))
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
    print(f"Loaded {len(corpus)} stress-conditioned synthetic pairs.")
    print("Rules: stressed e → je, stressed o → wo; unstressed preserved.")
    print()

    cognate_corpus = cognate_sets_from_pairs(corpus, ("proto", "derived"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"proto", "derived"})]

    print(format_model(trained, top_segments=20))

    print()
    print("Stress-conditioned segment entries:")
    for cc, count in trained.segment_table.counts.items():
        ctx = cc.context
        has_stress_constraint = (
            any(fc.feature == "stress" for fc in ctx.self_stress)
            or any(fc.feature == "stress" for fc in ctx.preceding_stress)
            or any(fc.feature == "stress" for fc in ctx.following_stress)
        )
        if not has_stress_constraint:
            continue
        parts = []
        if ctx.self_stress:
            parts.append(
                "self="
                + ",".join(f"{fc.feature}:{fc.value}" for fc in ctx.self_stress)
            )
        if ctx.preceding_stress:
            parts.append(
                "prec_s="
                + ",".join(f"{fc.feature}:{fc.value}" for fc in ctx.preceding_stress)
            )
        if ctx.following_stress:
            parts.append(
                "fol_s="
                + ",".join(f"{fc.feature}:{fc.value}" for fc in ctx.following_stress)
            )
        print(f"  {cc.src} -> {cc.tgt}: count={count:.1f}  [{' | '.join(parts)}]")


if __name__ == "__main__":
    main()
