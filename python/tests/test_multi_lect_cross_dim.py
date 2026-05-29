"""Tests for : multi-lect cross-dimensional rule lifting.

Validates that ``_lift_cross_dimensional_rules`` surfaces
per-pair :class:`CrossDimensionalLink` commits onto the
:class:`MultiLectModel.cross_dimensional_table` with correct
``src_lect`` / ``tgt_lect`` labels.
"""

from __future__ import annotations

import warnings

import pytest

from regulae import (
    CognateSet,
    CrossDimensionalLink,
    CrossDimensionalLinkTable,
    FeatureConstraint,
    Form,
    LearnedModel,
    MultiLectCrossDimensionalLink,
    MultiLectCrossDimensionalLinkTable,
    MultiLectModel,
    Segment,
    train_model,
)


@pytest.fixture(autouse=True)
def _suppress_pair_deprecation():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    yield


def _form(lect: str, chars: str, *, tone_last: str | None = None) -> Form:
    segs = tuple(Segment(c) for c in chars)
    if tone_last is not None and segs:
        last = segs[-1]
        segs = segs[:-1] + (Segment(grapheme=last.grapheme, tone=tone_last),)
    return Form(lect_id=lect, segments=segs)


def _build_tonogenesis_set(
    cognate_id: str,
    proto_initial: str,
    proto_tone: str,
    daughter_a_tone: str,
    daughter_b_tone: str,
    vowel: str,
) -> CognateSet:
    """One 3-way cognate: each lect has one CV word where the
    vowel carries a lect-specific tone."""
    return CognateSet(
        cognate_id=cognate_id,
        forms={
            "proto": _form("proto", proto_initial + vowel, tone_last=proto_tone),
            "daughter_a": _form(
                "daughter_a",
                proto_initial + vowel,
                tone_last=daughter_a_tone,
            ),
            "daughter_b": _form(
                "daughter_b",
                proto_initial + vowel,
                tone_last=daughter_b_tone,
            ),
        },
    )


def _tonogenesis_corpus() -> list[CognateSet]:
    """80 three-way cognate sets with two different tonogenesis
    outcomes on the same proto-voicing conditioning.

    - daughter_a: voiceless → tone 1, voiced → tone 4
    - daughter_b: voiceless → tone 2, voiced → tone 3

    For each (initial ∈ {p, t, b, d}, vowel ∈ {i, a, u, o, e},
    source_tone ∈ {1..4}), emit one cognate set. That's 80 sets.
    """
    corpus = []
    count = 0
    for initial in ["p", "t", "b", "d"]:
        voiced = initial in ("b", "d")
        for vowel in ["i", "a", "u", "o", "e"]:
            for source_tone in ("1", "2", "3", "4"):
                count += 1
                a_tone = "4" if voiced else "1"
                b_tone = "3" if voiced else "2"
                corpus.append(
                    _build_tonogenesis_set(
                        cognate_id=f"cs-{count:03d}",
                        proto_initial=initial,
                        proto_tone=source_tone,
                        daughter_a_tone=a_tone,
                        daughter_b_tone=b_tone,
                        vowel=vowel,
                    )
                )
    return corpus


def test_multi_lect_cross_dim_table_lifts_per_pair_commits() -> None:
    """COMMITMENT: after training on a clean 3-way tonogenesis
    corpus, the multi-lect model carries a non-empty
    ``cross_dimensional_table`` with lifted entries that name
    both ``src_lect`` and ``tgt_lect`` explicitly."""
    model = train_model(_tonogenesis_corpus())
    assert isinstance(model, MultiLectModel)
    table = model.cross_dimensional_table
    assert isinstance(table, MultiLectCrossDimensionalLinkTable)
    assert len(table.entries) > 0
    # Every entry carries both lect labels and they are drawn from
    # the canonical ``lect_ids`` tuple.
    for rule in table.entries:
        assert isinstance(rule, MultiLectCrossDimensionalLink)
        assert rule.src_lect in model.lect_ids
        assert rule.tgt_lect in model.lect_ids
        assert rule.src_lect != rule.tgt_lect


def test_multi_lect_cross_dim_covers_proto_to_both_daughters() -> None:
    """COMMITMENT: the lift surfaces at least one rule for each
    directed pair (proto → daughter_a) and (proto → daughter_b)
    that predicts the expected daughter-specific tonogenesis
    outcome (tone 4 for daughter_a, tone 3 for daughter_b, from
    a voiced proto initial)."""
    model = train_model(_tonogenesis_corpus())

    # daughter_a should receive at least one high-confidence
    # voiced → tone 4 rule with proto as src_lect.
    a_hits = [
        r for r in model.cross_dimensional_table.entries
        if r.src_lect == "proto"
        and r.tgt_lect == "daughter_a"
        and r.src_feature.feature == "voiced"
        and r.tgt_value == "4"
        and r.confidence >= 0.8
    ]
    assert len(a_hits) >= 1, (
        f"no proto→daughter_a voiced→tone4 rule; "
        f"got {model.cross_dimensional_table.entries}"
    )

    # daughter_b should receive at least one high-confidence
    # voiced → tone 3 rule.
    b_hits = [
        r for r in model.cross_dimensional_table.entries
        if r.src_lect == "proto"
        and r.tgt_lect == "daughter_b"
        and r.src_feature.feature == "voiced"
        and r.tgt_value == "3"
        and r.confidence >= 0.8
    ]
    assert len(b_hits) >= 1, (
        f"no proto→daughter_b voiced→tone3 rule; "
        f"got {model.cross_dimensional_table.entries}"
    )


def test_multi_lect_cross_dim_silent_on_non_tonal_corpus() -> None:
    """COMMITMENT: a 3-way non-tonal corpus produces an empty
    ``cross_dimensional_table`` — no spurious lifted entries."""
    # Trivial 3-way identity corpus.
    corpus = []
    for i, word in enumerate(["pata", "kaka", "mama", "sasa", "dudu",
                               "titi", "nana", "gogo", "bobo", "pipi"]):
        corpus.append(
            CognateSet(
                cognate_id=f"id-{i}",
                forms={
                    "A": _form("A", word),
                    "B": _form("B", word),
                    "C": _form("C", word),
                },
            )
        )
    model = train_model(corpus)
    assert model.cross_dimensional_table.entries == ()


def test_lift_cross_dimensional_rules_helper() -> None:
    """Unit test for ``_lift_cross_dimensional_rules`` on a
    hand-built pair of models."""
    from regulae.training import _lift_cross_dimensional_rules

    rule = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=20.0,
        src_count=20.0,
        confidence=1.0,
    )
    pair_model = LearnedModel.empty()
    from dataclasses import replace
    pair_model = replace(
        pair_model,
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(rule,)),
    )
    pairwise_models = {
        frozenset({"A", "B"}): pair_model,
    }
    table = _lift_cross_dimensional_rules(pairwise_models, ("A", "B"))
    assert len(table.entries) == 1
    lifted = table.entries[0]
    assert lifted.src_lect == "A"
    assert lifted.tgt_lect == "B"
    assert lifted.src_feature == rule.src_feature
    assert lifted.tgt_value == "4"
    assert lifted.count == 20.0
    assert lifted.confidence == 1.0


def test_lift_cross_dimensional_rules_respects_lect_ordering() -> None:
    """COMMITMENT: the lifted ``src_lect``/``tgt_lect`` labels
    follow the canonical ``lect_ids`` tuple ordering (first-seen),
    not an arbitrary order from ``frozenset``."""
    from regulae.training import _lift_cross_dimensional_rules
    from dataclasses import replace

    rule = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=10.0,
        src_count=10.0,
        confidence=1.0,
    )
    pair_model = replace(
        LearnedModel.empty(),
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(rule,)),
    )
    pairwise_models = {frozenset({"X", "Y"}): pair_model}

    # With lect_ids = ("X", "Y"), the lifted rule should be X → Y.
    table_xy = _lift_cross_dimensional_rules(pairwise_models, ("X", "Y"))
    assert table_xy.entries[0].src_lect == "X"
    assert table_xy.entries[0].tgt_lect == "Y"

    # With lect_ids = ("Y", "X"), the lifted rule should be Y → X.
    table_yx = _lift_cross_dimensional_rules(pairwise_models, ("Y", "X"))
    assert table_yx.entries[0].src_lect == "Y"
    assert table_yx.entries[0].tgt_lect == "X"
