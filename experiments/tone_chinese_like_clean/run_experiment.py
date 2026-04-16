"""Clean Chinese-like synthetic tonogenesis fixture (cross-dimensional discovery validation).

**DATA**: synthetic 72-pair corpus with a deliberately clean
tonogenesis rule. The original ``tone_chinese_like`` corpus was
designed to expose cross-dimensional discovery failure modes on
noisy real-ish data; this redone fixture is the positive
counterpart, used to validate the cross-dimensional commit loop
on data where the signal is unambiguous.

Rule by construction:

    voiceless initial consonant → target tone is preserved
    voiced/sonorant initial consonant → target tone = source tone + 3

Equivalence classes (6 total, 12 pairs each):

    voiceless × {1, 2, 3} → voiceless × {1, 2, 3}
    voiced    × {1, 2, 3} → voiced    × {4, 5, 6}

Inventories:

    voiceless initials: {p, t, k, f, s}
    voiced    initials: {b, d, g, m, n}
    vowels:            {i, a, u, o, e}

Expected output from a successful run:

- At least one committed :class:`CrossDimensionalLink` whose
  ``src_feature`` is voicing-related (``voiced``,
  ``voiceless``, ``sonorant``, or ``nasal``) and whose
  ``confidence`` is at or near 1.0.
- The tonal correspondence table should ALSO show the 6
  tone-to-tone mappings, since context discovery is orthogonal
  to cross-dimensional discovery. Both tables coexist (their
  outputs are additive).
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
    """Split off trailing tone digits and attach the tone to the
    last vowel. Same convention as ``tone_chinese_like``."""
    i = len(ipa)
    while i > 0 and ipa[i - 1].isdigit():
        i -= 1
    segmental = ipa[:i]
    tone = ipa[i:] if i < len(ipa) else None

    vowels = set("aeiou")
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
    print(f"Loaded {len(corpus)} synthetic clean Mandarin/Cantonese-like pairs.")
    print()
    print("NOTE: rule-explicit synthetic fixture. Designed to validate")
    print("cross-dimensional discovery commit loop on clean signal.")
    print()

    cognate_corpus = cognate_sets_from_pairs(
        corpus, ("mandarin-like", "cantonese-like")
    )
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[
        frozenset({"mandarin-like", "cantonese-like"})
    ]
    print(format_model(trained, top_segments=20))

    print()
    print("Cross-dimensional rules in detail:")
    for idx, rule in enumerate(trained.cross_dimensional_table.entries):
        from regulae import describe_cross_dimensional_rule
        print(describe_cross_dimensional_rule(trained, idx))
        print()

    print()
    print(
        f"Prior-only total cost:  "
        f"{sum(alignment_cost(align_forms(s, t)) for s, t in corpus):.2f}"
    )
    print(
        "Learned-model total cost:",
        f"{sum(alignment_cost(align_forms(s, t, model=trained), model=trained) for s, t in corpus):.2f}",
    )


if __name__ == "__main__":
    main()
