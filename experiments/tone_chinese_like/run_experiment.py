"""Chinese-like synthetic tonal experiment (Mandarin ↔ Cantonese style).

**DATA CAVEAT**: not authoritative Chinese cognate data. Synthetic
fixture using simplified pinyin-like romanizations with tone digits
for both "Mandarin" (4 tones) and "Cantonese" (6 tones). The forms
and tonal correspondences were constructed from memory and general
knowledge of the sound-correspondence patterns between the two
varieties, not from a cognate database.

The purpose of this experiment is to deliberately trigger the kinds
of failure modes cross-dimensional discovery will need to handle: Cantonese tones do not
derive cleanly from Mandarin tones; they trace back to Middle
Chinese through a separate path, so some "correspondences" are
really tonogenesis or substratum effects that cannot be captured
as segment-to-segment tonal mappings.

Expected outcome: context discovery will recover some regular tone
correspondences (the residual shared inheritance) but also surface
many competing correspondences with no clear conditioning — material
for cross-dimensional discovery.
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
    i = len(ipa)
    while i > 0 and ipa[i - 1].isdigit():
        i -= 1
    segmental = ipa[:i]
    tone = ipa[i:] if i < len(ipa) else None

    vowels = set("aeiouɛɔəɪʊ")
    segments: list[Segment] = []
    tone_attached = False
    chars = list(segmental)
    for idx in range(len(chars) - 1, -1, -1):
        if not tone_attached and chars[idx] in vowels and tone:
            segments.insert(0, Segment(grapheme=chars[idx], tone=tone))
            tone_attached = True
        else:
            segments.insert(0, Segment(grapheme=chars[idx]))
    if tone and not tone_attached and segments:
        last = segments[-1]
        segments[-1] = Segment(grapheme=last.grapheme, tone=tone)
    return tuple(segments)


def load_corpus(path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "mandarin", "cantonese"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, mn, yc = parts
            src = Form(lect_id="mandarin-like", segments=parse_toned_form(mn))
            tgt = Form(lect_id="cantonese-like", segments=parse_toned_form(yc))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Mandarin/Cantonese-like toned pairs.")
    print()
    print("NOTE: synthetic fixture. Designed to expose cross-dimensional")
    print("(tonogenesis) failures in purely tone-to-tone tonal machinery.")
    print()

    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(
        corpus, ("mandarin-like", "cantonese-like")
    )
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[
        frozenset({"mandarin-like", "cantonese-like"})
    ]
    print()
    print(format_model(trained, top_segments=20))

    print()
    print(f"Prior-only total cost:  {sum(alignment_cost(align_forms(s, t)) for s, t in corpus):.2f}")
    print(
        "Learned-model total cost:",
        f"{sum(alignment_cost(align_forms(s, t, model=trained), model=trained) for s, t in corpus):.2f}",
    )


if __name__ == "__main__":
    main()
