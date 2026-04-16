"""Synthetic tonal correspondence experiment.

30 hand-constructed pairs with known tonal shifts: every source syllable
has tone ``55`` or ``11`` (high or low), and every target has tone ``33``
(mid). The expected correspondences are:

    tone 55 → tone 33  (H → M shift)
    tone 11 → tone 33  (L → M shift)

The segments are unchanged between source and target (identity
correspondences for consonants and vowels). This isolates the tonal
dimension: every segment-level correspondence is identity, and all the
"interesting" signal is in the tones.

The parser reads the tone digits as a suprasegmental annotation on
the segment preceding them.
"""

from __future__ import annotations

from pathlib import Path

from regulae import (
    Form,
    Segment,
    align_forms,
    alignment_cost,
    cognate_sets_from_pairs,
    format_alignment,
    format_model,
    train_model,
)


def parse_toned_form(ipa: str) -> tuple[Segment, ...]:
    """Parse a form like ``pa55`` into segments, attaching tone digits
    to the preceding vowel.

    Tone notation: a run of digits (e.g. ``55``, ``33``, ``11``) at the
    end of the form attaches to the last vowel in the preceding text.
    """
    # Split into a segmental part and a trailing tone string.
    i = len(ipa)
    while i > 0 and ipa[i - 1].isdigit():
        i -= 1
    segmental = ipa[:i]
    tone = ipa[i:] if i < len(ipa) else None

    # Build one Segment per character; attach the tone to the last vowel.
    vowels = set("aeiouɛɔə")
    segments: list[Segment] = []
    tone_attached = False
    chars = list(segmental)
    for idx in range(len(chars) - 1, -1, -1):
        # Walk backwards so we can find the last vowel.
        if not tone_attached and chars[idx] in vowels and tone:
            segments.insert(0, Segment(grapheme=chars[idx], tone=tone))
            tone_attached = True
        else:
            segments.insert(0, Segment(grapheme=chars[idx]))
    if tone and not tone_attached:
        # No vowel found, attach to the last segment.
        last = segments[-1] if segments else None
        if last is not None:
            segments[-1] = Segment(grapheme=last.grapheme, tone=tone)
    return tuple(segments)


def load_corpus(path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "src", "tgt"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, src_ipa, tgt_ipa = parts
            src = Form(lect_id="src", segments=parse_toned_form(src_ipa))
            tgt = Form(lect_id="tgt", segments=parse_toned_form(tgt_ipa))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} synthetic toned pairs.")

    # Sanity-check: inspect the first few parses.
    print()
    print("Sample parses:")
    for gloss, src, tgt in labeled[:3]:
        src_desc = " ".join(
            f"{s.grapheme}[T={s.tone}]" if s.tone else s.grapheme for s in src.segments
        )
        tgt_desc = " ".join(
            f"{s.grapheme}[T={s.tone}]" if s.tone else s.grapheme for s in tgt.segments
        )
        print(f"  [{gloss}] {src_desc}  ~  {tgt_desc}")

    print()
    print("Training learned model...")
    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(corpus, ("src", "tgt"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"src", "tgt"})]
    print()

    print("=" * 70)
    print("Trained model summary")
    print("=" * 70)
    print(format_model(trained))

    print()
    print("=" * 70)
    print("Headline numbers")
    print("=" * 70)
    m2_cost = sum(alignment_cost(align_forms(s, t)) for s, t in corpus)
    m3_cost = sum(
        alignment_cost(align_forms(s, t, model=trained), model=trained)
        for s, t in corpus
    )
    print(f"Prior-only total cost: {m2_cost:.2f}")
    print(f"Learned-model total cost: {m3_cost:.2f}")
    print(f"Reduction:     {m2_cost - m3_cost:.2f}")

    print()
    print("=" * 70)
    print("Sample alignments")
    print("=" * 70)
    for gloss, src, tgt in labeled[:4]:
        print()
        print(f"[{gloss}]")
        print(format_alignment(align_forms(src, tgt, model=trained)))


if __name__ == "__main__":
    main()
