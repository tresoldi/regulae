"""Vietnamese-like synthetic tonal experiment.

**DATA CAVEAT**: not authoritative Vietnamese dialect data. This is a
synthetic fixture with tonal shifts designed to exercise the
context discovery machinery on a larger tonal inventory
(Vietnamese-style six tones).
The tonal mappings embedded in the corpus follow the pattern
"certain source tones lower their tone number by 1 in the target"
(partial merger profile). Real Vietnamese dialectal data should
replace this fixture for linguistic conclusions.
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
        assert header == ["gloss", "hanoi", "saigon"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, hn, sg = parts
            src = Form(lect_id="hanoi-like", segments=parse_toned_form(hn))
            tgt = Form(lect_id="saigon-like", segments=parse_toned_form(sg))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Vietnamese-like toned pairs.")
    print()
    print("NOTE: synthetic fixture, not real dialectal data.")
    print()

    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(corpus, ("hanoi-like", "saigon-like"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"hanoi-like", "saigon-like"})]
    print()
    print(format_model(trained))

    print()
    print(f"Prior-only total cost:  {sum(alignment_cost(align_forms(s, t)) for s, t in corpus):.2f}")
    print(
        "Learned-model total cost:",
        f"{sum(alignment_cost(align_forms(s, t, model=trained), model=trained) for s, t in corpus):.2f}",
    )


if __name__ == "__main__":
    main()
