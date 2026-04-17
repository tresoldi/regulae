"""Middle Chinese → Mandarin cognate alignment experiment.

40 cognates illustrating the Middle-Chinese-to-Mandarin sound
changes, including the classic tonogenesis via voicing of the
onset.

**Correspondences validated:**

- **Tone redistribution by onset voicing** (the cross-dimensional
  rule the framework is designed to recover):
  - MC tone 1 (píng) + voiceless onset → Mandarin tone 1
  - MC tone 1 (píng) + voiced onset → Mandarin tone 2
  - MC tone 2 (shǎng) → Mandarin tone 3
  - MC tone 3 (qù) → Mandarin tone 4
- **Voicing merger**: MC voiced obstruents (b, d, g, dz, dʑ) lost
  in Mandarin; they become aspirated voiceless in level tones,
  unaspirated voiceless in non-level tones.
- **Final stops lost**: MC -p, -t, -k all lost; the checked (rù)
  tone is redistributed to the other four tones based on onset.
- **Final -m merged to -n**: MC sam → san.
- **Palatalization of velars before front vowels**: MC kj- → tɕ-.

The tones are encoded on the vowel nucleus; the framework's
cross-dimensional discovery should recover the voicing-to-tone
rule from the data.

Reconstructions use simplified Baxter-style Middle Chinese IPA.
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


# Multi-char graphemes that appear in the IPA. Order matters: longer first.
_MULTI = (
    "tsʰ", "tʃʰ", "tɕʰ", "tʰ", "pʰ", "kʰ",
    "ɻɻ",  # Mandarin "r" + vowel: two separate segments actually
    "tɕ", "dʑ", "tʃ", "dʒ", "ts", "dz",
    "aː", "eː", "iː", "oː", "uː",
)


def parse_form(ipa: str) -> tuple[Segment, ...]:
    """Parse IPA without tone marks into Segments."""
    segments: list[Segment] = []
    i = 0
    while i < len(ipa):
        hit = False
        for cluster in _MULTI:
            if ipa[i : i + len(cluster)] == cluster:
                segments.append(Segment(cluster))
                i += len(cluster)
                hit = True
                break
        if not hit:
            segments.append(Segment(ipa[i]))
            i += 1
    return tuple(segments)


_VOWELS = frozenset("aeiouɑɒɔɛɪɨɯʊʉyøəɚɻɿ")


def parse_form_with_tone(ipa: str, tone: str) -> tuple[Segment, ...]:
    """Parse an IPA string and attach ``tone`` to the first vowel-like
    segment (the nucleus). For Mandarin/MC, one tone per syllable."""
    bare = parse_form(ipa)
    if tone == "-":
        return bare
    out: list[Segment] = []
    applied = False
    for seg in bare:
        # Vowel-like: any segment whose grapheme starts with a vowel character.
        g = seg.grapheme
        if not applied and any(ch in _VOWELS for ch in g):
            out.append(Segment(grapheme=g, tone=tone))
            applied = True
        else:
            out.append(seg)
    if not applied:
        # Fallback: no vowel found; attach to the last segment.
        out = list(bare[:-1]) + [Segment(grapheme=bare[-1].grapheme, tone=tone)]
    return tuple(out)


def load_corpus(tsv_path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with tsv_path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "middle_chinese", "mc_tone", "mandarin", "md_tone"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            gloss, mc, mc_t, md, md_t = parts
            src = Form(lect_id="middle_chinese", segments=parse_form_with_tone(mc, mc_t))
            tgt = Form(lect_id="mandarin", segments=parse_form_with_tone(md, md_t))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Middle Chinese → Mandarin cognate pairs.")
    cognate_corpus = cognate_sets_from_pairs(pairs, ("middle_chinese", "mandarin"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"middle_chinese", "mandarin"})]
    print(format_model(trained, top_segments=20))

    print()
    print("Cross-dimensional rules (tonogenesis):")
    for rule in trained.cross_dimensional_table.entries:
        print(
            f"  {rule.src_feature.feature}={rule.src_feature.value}"
            f"@{rule.src_position} -> {rule.tgt_dimension}={rule.tgt_value}"
            f"@+{rule.tgt_position_offset}  "
            f"count={rule.count}/{rule.src_count}  conf={rule.confidence:.2f}"
        )


if __name__ == "__main__":
    main()
