"""Tests for length-conditioned context discovery.

``long`` is a merkmal feature that distinguishes long vowels and
geminate consonants. These tests pin two claims:

1. When the discovery layer's feature inventory includes ``long``,
   a corpus with a length-conditioned rule commits the rule as a
   conditioned correspondence (not as a chunk-only artifact).
2. OE/ModE training preserves the long/short-vowel distinction as
   separate source keys, so the long-vowel reflexes (GVS) and
   short-vowel reflexes live in separate rows of the segment table.
"""

from pathlib import Path

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


def _parse(ipa: str, multi: tuple[str, ...]) -> tuple[Segment, ...]:
    out: list[Segment] = []
    i = 0
    while i < len(ipa):
        hit = False
        for m in multi:
            if ipa[i : i + len(m)] == m:
                out.append(Segment(m))
                i += len(m)
                hit = True
                break
        if not hit:
            out.append(Segment(ipa[i]))
            i += 1
    return tuple(out)


def _synthetic_length_corpus() -> list[CognateSet]:
    multi = ("aː", "eː", "iː", "oː", "uː")
    long_pairs = [("aːba", "aːβa"), ("eːba", "eːβa"), ("iːba", "iːβa"),
                  ("oːba", "oːβa"), ("uːba", "uːβa"),
                  ("aːbo", "aːβo"), ("eːbo", "eːβo"), ("iːbo", "iːβo"),
                  ("oːbe", "oːβe"), ("uːbo", "uːβo"),
                  ("aːbe", "aːβe"), ("eːbe", "eːβe")]
    short_pairs = [("aba", "aba"), ("eba", "eba"), ("iba", "iba"),
                   ("oba", "oba"), ("uba", "uba"),
                   ("abo", "abo"), ("ebo", "ebo"), ("ibo", "ibo"),
                   ("obe", "obe"), ("ubo", "ubo"),
                   ("abe", "abe"), ("ebe", "ebe")]
    controls = [("ata", "ata"), ("aːta", "aːta"), ("eta", "eta"),
                ("eːta", "eːta"), ("iti", "iti"), ("iːti", "iːti")]

    pairs = []
    for proto, derived in long_pairs + short_pairs + controls:
        pairs.append(
            (
                Form("proto", _parse(proto, multi)),
                Form("derived", _parse(derived, multi)),
            )
        )
    return cognate_sets_from_pairs(pairs, ("proto", "derived"))


def test_length_conditioned_rule_is_committed() -> None:
    """COMMITMENT: on a corpus where /b/ weakens to /β/ exclusively
    after long vowels, the discovery layer commits the rule
    ``b → β / [long:+] _`` as a conditioned segment correspondence."""
    corpus = _synthetic_length_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"proto", "derived"})]
    long_conditioned = [
        (cc, c)
        for cc, c in pair.segment_table.counts.items()
        if cc.src == "b"
        and cc.tgt == "β"
        and any(
            fc.feature == "long" and fc.value == "+"
            for fc in cc.context.preceding
        )
    ]
    assert long_conditioned, (
        "expected at least one b→β entry conditioned on preceding [long:+], "
        "got none"
    )
    # The rule should be supported by at least the 12 long-vowel b-pairs.
    total = sum(c for _, c in long_conditioned)
    assert total >= 10


def test_oe_modern_english_preserves_long_short_distinction() -> None:
    """COMMITMENT: OE long vowels (aː, eː, iː, oː, uː, etc.) are
    distinct source keys from their short counterparts after
    training, so each can have its own reflex distribution."""
    tsv = Path(__file__).parent.parent / "experiments/oe_english/cognates.tsv"
    if not tsv.exists():
        import pytest
        pytest.skip("oe_english/cognates.tsv not available")

    multi = (
        "aː", "æː", "ɑː", "eː", "iː", "oː", "ɔː", "uː", "yː",
        "aɪ", "oʊ", "aʊ", "ɔɪ", "eɪ", "tʃ", "dʒ",
    )
    pairs = []
    with tsv.open() as f:
        header = f.readline().strip().split("\t")
        assert header[:3] == ["gloss", "old_english", "modern_english"]
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            _, oe, me = parts[0], parts[1], parts[2]
            pairs.append((
                Form("oe", _parse(oe, multi)),
                Form("me", _parse(me, multi)),
            ))
    corpus = cognate_sets_from_pairs(pairs, ("oe", "me"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"oe", "me"})]
    srcs = set(pair.segment_table.src_totals)
    # Both long and short forms of at least two vowels should appear
    # as separate source keys. OE "oː" shifted via GVS differently
    # from OE "o".
    assert "oː" in srcs and "o" in srcs
    assert "iː" in srcs and "i" in srcs
