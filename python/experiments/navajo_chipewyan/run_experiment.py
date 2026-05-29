"""Navajo → Chipewyan (Dëne Sųłiné) cognate alignment experiment.

32 Athabaskan cognates compiled from ASJP, Wiktionary Proto-
Athabaskan reconstructions, Alderete's Tahltan/Athabaskan
phonology work, Krauss (1964) reconstruction tables, Carleton's
Chipewyan-Apachean wordlist, and McDonough's IPA transcription
of Krauss 2005 tonogenesis tables.

**Correspondences validated:**

- **Proto-Athabaskan *ts ↔ Chipewyan tθ / θ**: the classic
  Dëne Sųłiné interdental split.
  - ``tsíːʔ ~ tθí`` "head"
  - ``tsʼìn ~ tθʼə̀né`` "bone"
  - ``tsòːʔ ~ θù`` "tongue"
  - ``tsʰé ~ θé`` "stone"
  - ``sṍːʔ ~ θə̃́`` "star"
- **Proto-Athabaskan *ʒ ↔ Chipewyan ð**: ``tsìɬ ~ ðèɬ`` "mountain".
- **Word-final glottal preserved in Navajo, usually lost in
  Chipewyan**: Navajo -ʔ endings routinely absent in Chipewyan
  cognates.
- **Tone system**: both languages are tonal. Navajo marks H/L
  with acute/grave accents; Chipewyan likewise. Proto-
  Athabaskan constricted-syllable tonogenesis produces an
  inverse polarity (constricted → Navajo H / Chipewyan L in
  some reconstructions), although this corpus does not
  explicitly tag constriction.

**Transcription choices:**

- Aspirated stops written with ``ʰ`` (pʰ tʰ kʰ).
- Ejectives with ``ʼ`` (tsʼ tʼ kʼ tθʼ).
- Tones not extracted into ``Segment.tone`` in this experiment:
  they are part of the combining-character grapheme (ó, ì,
  etc.). The training pipeline treats them as separate IPA
  segments, which is a simplification — the Athabaskan tone
  system would benefit from explicit tone annotation, but the
  corpus is small enough that bare segmental correspondences
  are the primary signal.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from regulae import (
    Form,
    Segment,
    cognate_sets_from_pairs,
    format_model,
    train_model,
)


# Multi-character graphemes that must be recognised as a single
# segment. Order matters: longer prefixes first.
_MULTI = (
    "tθʼ", "tʃʼ", "tsʼ", "tɬʼ",
    "tθ", "tʃʰ", "tɬʰ", "tsʰ",
    "tʼ", "pʼ", "kʼ",
    "tʰ", "pʰ", "kʰ",
    "dʒ", "dɮ", "dz",
    "aː", "eː", "iː", "oː", "uː",
    "ãː", "ẽː", "ĩː", "õː", "ũː",
    "ã", "ẽ", "ĩ", "õ", "ũ",
)

# Combining tone marks: acute = high, grave = low. Extracted from
# the grapheme and attached to Segment.tone.
_ACUTE = "\u0301"
_GRAVE = "\u0300"


def _normalise(raw: str) -> str:
    """Normalise input: apply NFD so tone marks are separate, then
    recompose nasalisation (combining tilde → precomposed) but
    keep tone marks separate for the parser to extract."""
    decomposed = unicodedata.normalize("NFD", raw)
    # Recompose base + combining tilde (nasalization) but leave
    # acute/grave (tone) decomposed.
    result: list[str] = []
    i = 0
    while i < len(decomposed):
        ch = decomposed[i]
        if i + 1 < len(decomposed) and decomposed[i + 1] == "\u0303":
            # base + combining tilde → precomposed nasalised vowel
            combined = unicodedata.normalize("NFC", ch + "\u0303")
            result.append(combined)
            i += 2
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def parse_form(ipa: str) -> tuple[Segment, ...]:
    """Tokenise an IPA string into segments.

    Multi-character graphemes (affricates, aspirated stops,
    ejectives, long vowels, nasalised vowels) are consumed as
    single units. Combining acute/grave accents are extracted as
    tone values ``"H"``/``"L"`` on ``Segment.tone``. A trailing
    length mark ``ː`` is appended to the segment's grapheme, so
    ``i`` + combining acute + ``ː`` becomes ``Segment("iː", tone="H")``.
    """
    ipa = _normalise(ipa)
    segments: list[Segment] = []
    i = 0
    while i < len(ipa):
        # Emit the next segment (multi-char prefix or single char).
        emitted: str | None = None
        for cluster in _MULTI:
            if ipa[i : i + len(cluster)] == cluster:
                emitted = cluster
                i += len(cluster)
                break
        if emitted is None:
            emitted = ipa[i]
            i += 1
        tone: str | None = None
        # Absorb any trailing diacritics: tone marks, nasalisation
        # (combining tilde), and length. Nasalisation and length
        # attach to the grapheme; tones are extracted separately.
        while i < len(ipa) and ipa[i] in (_ACUTE, _GRAVE, "ː", "\u0303"):
            mark = ipa[i]
            if mark == _ACUTE:
                tone = "H"
            elif mark == _GRAVE:
                tone = "L"
            elif mark == "ː":
                emitted = emitted + "ː"
            elif mark == "\u0303":
                # Combining tilde → append to grapheme so merkmal
                # sees the nasalised form (e.g. 'ə' + '̃' = 'ə̃').
                emitted = emitted + "\u0303"
            i += 1
        segments.append(Segment(grapheme=emitted, tone=tone))
    return tuple(segments)


def load_corpus(tsv_path: Path) -> list[tuple[str, Form, Form]]:
    corpus: list[tuple[str, Form, Form]] = []
    with tsv_path.open() as f:
        header = f.readline().strip().split("\t")
        assert header[:3] == ["gloss", "navajo", "chipewyan"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gloss, nv, ch = parts[0], parts[1], parts[2]
            src = Form(lect_id="navajo", segments=parse_form(nv))
            tgt = Form(lect_id="chipewyan", segments=parse_form(ch))
            corpus.append((gloss, src, tgt))
    return corpus


def main() -> None:
    tsv = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv)
    pairs = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(pairs)} Navajo → Chipewyan cognate pairs.")
    cognate_corpus = cognate_sets_from_pairs(pairs, ("navajo", "chipewyan"))
    multi = train_model(cognate_corpus)
    trained = multi.pairwise_models[frozenset({"navajo", "chipewyan"})]
    print(format_model(trained, top_segments=25))


if __name__ == "__main__":
    main()
