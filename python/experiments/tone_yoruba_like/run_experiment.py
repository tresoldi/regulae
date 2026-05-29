"""Yoruba-like synthetic tonal experiment.

**DATA CAVEAT**: the corpus in this directory is NOT authoritative
Yoruba cognate data. I am not a Yoruba specialist and cannot construct
reliable published dialect data from memory. The corpus is a
synthetic "Yoruba-style" test fixture: lexemes plausibly resembling
Yoruba basic vocabulary, with a designed tonal correspondence
(Standard tone N → dialect tone N-1, i.e., every tone lowers one
level). It exercises the tonal machinery on a larger and more
realistic vocabulary than the pure synthetic experiment.

Real dialectal Yoruba data from published sources should replace
this fixture before drawing linguistic conclusions.
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
    """Parse a form with trailing tone digits.

    Tone notation: digits at the end of the form attach to the last
    vowel as a tone marker. Multiple successive vowel+tone groups are
    not supported; for this fixture we use one tone per word.
    """
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
        assert header == ["gloss", "standard", "ekiti"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, std, dia = parts
            src = Form(lect_id="standard", segments=parse_toned_form(std))
            tgt = Form(lect_id="ekiti-like", segments=parse_toned_form(dia))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Yoruba-like toned pairs.")
    print()
    print("NOTE: this corpus is a synthetic Yoruba-style fixture, not")
    print("authoritative data. See the module docstring for details.")
    print()

    print("Training learned model...")
    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(corpus, ("standard", "ekiti-like"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"standard", "ekiti-like"})]
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
    for gloss, src, tgt in labeled[:5]:
        print()
        print(f"[{gloss}]")
        print(format_alignment(align_forms(src, tgt, model=trained)))


if __name__ == "__main__":
    main()
