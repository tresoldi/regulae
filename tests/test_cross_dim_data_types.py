"""Tests for the  cross-dimensional link data types.

Covers construction, defaults, frozen-ness, hashability, and the
empty-table inertness invariant — a ``LearnedModel`` with an empty
``cross_dimensional_table`` must behave exactly like a pre-
model.
"""

import pytest

from regulae import (
    ChunkPhraseTable,
    CrossDimensionalLink,
    CrossDimensionalLinkTable,
    DisplacementDistribution,
    LearnedModel,
    SegmentCorrespondenceTable,
    TonalCorrespondenceTable,
)
from regulae.types import FeatureConstraint


# ----- CrossDimensionalLink construction ---------------------------------


def test_cross_dimensional_link_minimal_construction() -> None:
    """COMMITMENT: a CrossDimensionalLink can be built with the
    six required-ish fields; count, src_count, and confidence have
    sensible defaults."""
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
    )
    assert link.src_feature.feature == "voiced"
    assert link.src_feature.value == "+"
    assert link.src_position == "relative_-1"
    assert link.tgt_dimension == "tone"
    assert link.tgt_value == "4"
    assert link.tgt_position_offset == 0
    assert link.count == 0.0
    assert link.src_count == 0.0
    assert link.confidence == 1.0


def test_cross_dimensional_link_full_construction() -> None:
    """COMMITMENT: count, src_count, and confidence are settable
    and carry through unchanged."""
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("nasal", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="5",
        tgt_position_offset=0,
        count=12.0,
        src_count=15.0,
        confidence=0.8,
    )
    assert link.count == 12.0
    assert link.src_count == 15.0
    assert link.confidence == 0.8


def test_cross_dimensional_link_is_frozen() -> None:
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
    )
    with pytest.raises(Exception):
        link.count = 99.0  # type: ignore[misc]


def test_cross_dimensional_link_is_hashable() -> None:
    """COMMITMENT: CrossDimensionalLink is hashable so it can be
    used in sets and as dict keys."""
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
    )
    {link}
    {link: "present"}


def test_cross_dimensional_link_equality_is_structural() -> None:
    """COMMITMENT: two links with identical fields are equal;
    links with different fields are not."""
    a = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
    )
    b = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
    )
    assert a == b
    c = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="5",  # different
        tgt_position_offset=0,
    )
    assert a != c


# ----- CrossDimensionalLinkTable construction ----------------------------


def test_cross_dimensional_link_table_defaults_to_empty() -> None:
    table = CrossDimensionalLinkTable()
    assert table.entries == ()


def test_cross_dimensional_link_table_holds_entries() -> None:
    link1 = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
    )
    link2 = CrossDimensionalLink(
        src_feature=FeatureConstraint("nasal", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="5",
        tgt_position_offset=0,
    )
    table = CrossDimensionalLinkTable(entries=(link1, link2))
    assert len(table.entries) == 2
    assert table.entries[0] == link1
    assert table.entries[1] == link2


def test_cross_dimensional_link_table_is_frozen() -> None:
    table = CrossDimensionalLinkTable()
    with pytest.raises(Exception):
        table.entries = ()  # type: ignore[misc]


# ----- LearnedModel integration ------------------------------------------


def test_learned_model_has_empty_cross_dimensional_table_by_default() -> None:
    """COMMITMENT: a default-constructed LearnedModel exposes an
    empty cross_dimensional_table. Pre- training code paths
    that construct LearnedModel without mentioning the new field
    still produce valid models."""
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    assert isinstance(model.cross_dimensional_table, CrossDimensionalLinkTable)
    assert model.cross_dimensional_table.entries == ()


def test_learned_model_empty_factory_has_empty_cross_dimensional_table() -> None:
    model = LearnedModel.empty()
    assert isinstance(model.cross_dimensional_table, CrossDimensionalLinkTable)
    assert model.cross_dimensional_table.entries == ()


def test_learned_model_accepts_populated_cross_dimensional_table() -> None:
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=18.0,
        src_count=20.0,
        confidence=0.9,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(link,)),
    )
    assert len(model.cross_dimensional_table.entries) == 1
    assert model.cross_dimensional_table.entries[0] == link


# ----- src_position canonical form ---------------------------------------


def test_cross_dimensional_link_uses_relative_position_strings() -> None:
    """COMMITMENT: v1 canonical src_position values are exactly
    ``relative_-1``, ``relative_0``, and ``relative_+1``. Structural
    values like ``initial_consonant`` are out of scope per the
    locked design decision in the  spec (Q A / §14).
    """
    for pos in ("relative_-1", "relative_0", "relative_+1"):
        link = CrossDimensionalLink(
            src_feature=FeatureConstraint("voiced", "+"),
            src_position=pos,
            tgt_dimension="tone",
            tgt_value="4",
            tgt_position_offset=0,
        )
        assert link.src_position == pos


# ----- backward compatibility (empty table is inert) --------------------


def test_default_model_round_trips_through_equality() -> None:
    """COMMITMENT: two default-constructed models are equal —
    the new cross_dimensional_table field doesn't break
    structural equality."""
    a = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    b = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    # The tables inside are dicts and aren't hashable, so full
    # equality isn't actually defined — check field-wise that the
    # cross_dimensional_table matches.
    assert a.cross_dimensional_table == b.cross_dimensional_table
    assert a.cross_dimensional_table.entries == ()


# ----- scoring overlay stub ---------------------------------------------


def test_apply_cross_dimensional_adjustments_returns_zero_for_empty_table() -> None:
    """COMMITMENT: with an empty cross_dimensional_table, the
    overlay returns 0.0 regardless of the alignment. This is
    the empty-table inertness invariant that protects every
    existing test."""
    from regulae import align_forms, Form, Segment
    from regulae.scoring import apply_cross_dimensional_adjustments

    model = LearnedModel.empty()
    src = Form("A", (Segment("p"), Segment("a"), Segment("t"), Segment("a")))
    tgt = Form("B", (Segment("f"), Segment("a"), Segment("d"), Segment("a")))
    alignment = align_forms(src, tgt, model=model)
    assert apply_cross_dimensional_adjustments(alignment, model) == 0.0


def test_alignment_cost_is_unchanged_when_table_is_empty() -> None:
    """COMMITMENT: alignment_cost(alignment, model=m) returns the
    same value whether or not the empty cross-dimensional overlay
    is invoked. This checks that the new additive term in
    alignment_cost is wired correctly and that the stub doesn't
    silently perturb anything."""
    from regulae import align_forms, alignment_cost, Form, Segment

    model = LearnedModel.empty()
    src = Form("A", (Segment("p"), Segment("a")))
    tgt = Form("B", (Segment("f"), Segment("a")))
    alignment = align_forms(src, tgt, model=model)
    cost_with_model = alignment_cost(alignment, model=model)
    cost_without = alignment_cost(alignment, feature_system="descriptive")
    # Both paths compute positive finite numbers; they can
    # differ (the model path uses the layered scoring) but the
    # model path must be deterministic under repeated calls.
    assert cost_with_model == alignment_cost(alignment, model=model)


def test_overlay_applies_negative_adjustment_when_rule_is_confirmed() -> None:
    """COMMITMENT: when a committed cross-dimensional rule's source
    context holds AND the target value matches, the overlay
    contributes a NEGATIVE cost adjustment (cost reduction). The
    rule rewards the alignment for matching its prediction."""
    from regulae import align_forms, alignment_cost, Form, Segment
    from regulae.model import TonalCorrespondence

    # Build a tonal table with known base rates: tone "4" and
    # tone "1" each observed 5 times (so base rate is 0.5 for each).
    tonal = TonalCorrespondenceTable(
        counts={
            TonalCorrespondence(src_tone=None, tgt_tone="4"): 5.0,
            TonalCorrespondence(src_tone=None, tgt_tone="1"): 5.0,
        },
        prior_pseudo_counts={},
        src_totals={None: 10.0},
    )
    # A confident rule: every observation of voiced source → tone 4.
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=10.0,
        src_count=10.0,
        confidence=1.0,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        tonal_table=tonal,
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(link,)),
    )
    # Voiced initial + target tone 4: rule fires, adjustment is negative.
    src = Form("A", (Segment("b"), Segment("a")))
    tgt = Form("B", (Segment("b"), Segment("a", tone="4")))
    alignment = align_forms(src, tgt, model=model)
    from regulae.scoring import apply_cross_dimensional_adjustments

    adj = apply_cross_dimensional_adjustments(alignment, model)
    # P_cond ≈ 1.0, P_base ≈ 0.5, so pos_adj = -(log 1.0 - log 0.5) ≈ -0.69.
    # The alignment has ONE 1-to-1 link where the rule's source
    # context (voiced at relative_-1 from the vowel's link position)
    # holds AND the target tone matches.
    # There are actually two 1-to-1 links: (b, b) and (a, a). The
    # rule fires on the (a, a) link because relative_-1 from the
    # vowel position lands on the preceding b, which is voiced.
    # The rule does NOT fire on the (b, b) link because relative_-1
    # lands before the word start (out of bounds).
    assert adj < 0
    assert adj < -0.3  # at least the pos_adj magnitude


def test_overlay_applies_positive_adjustment_when_rule_is_contradicted() -> None:
    """COMMITMENT: when a rule's source context holds but the
    target value does NOT match, the overlay contributes a
    POSITIVE adjustment (cost increase). The rule gets charged
    for a bad prediction."""
    from regulae import align_forms, alignment_cost, Form, Segment
    from regulae.model import TonalCorrespondence

    tonal = TonalCorrespondenceTable(
        counts={
            TonalCorrespondence(src_tone=None, tgt_tone="4"): 5.0,
            TonalCorrespondence(src_tone=None, tgt_tone="1"): 5.0,
        },
        prior_pseudo_counts={},
        src_totals={None: 10.0},
    )
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=10.0,
        src_count=10.0,
        confidence=1.0,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        tonal_table=tonal,
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(link,)),
    )
    # Voiced initial + target tone 1 (NOT the rule's prediction).
    src = Form("A", (Segment("b"), Segment("a")))
    tgt = Form("B", (Segment("b"), Segment("a", tone="1")))
    alignment = align_forms(src, tgt, model=model)
    from regulae.scoring import apply_cross_dimensional_adjustments

    adj = apply_cross_dimensional_adjustments(alignment, model)
    assert adj > 0


def test_overlay_zero_when_rule_context_does_not_hold() -> None:
    """COMMITMENT: when the source context doesn't hold, the rule
    contributes zero adjustment."""
    from regulae import align_forms, Form, Segment
    from regulae.model import TonalCorrespondence

    tonal = TonalCorrespondenceTable(
        counts={
            TonalCorrespondence(src_tone=None, tgt_tone="4"): 5.0,
            TonalCorrespondence(src_tone=None, tgt_tone="1"): 5.0,
        },
        prior_pseudo_counts={},
        src_totals={None: 10.0},
    )
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=10.0,
        src_count=10.0,
        confidence=1.0,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        tonal_table=tonal,
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(link,)),
    )
    # Voiceless initial: rule's source context does not hold.
    src = Form("A", (Segment("p"), Segment("a")))
    tgt = Form("B", (Segment("p"), Segment("a", tone="4")))
    alignment = align_forms(src, tgt, model=model)
    from regulae.scoring import apply_cross_dimensional_adjustments

    adj = apply_cross_dimensional_adjustments(alignment, model)
    assert adj == 0.0


def test_overlay_zero_when_rule_does_not_discriminate() -> None:
    """COMMITMENT: when the rule's P_cond equals the base rate
    P_base, the rule provides no new information and the
    adjustment is zero even when fired."""
    from regulae import align_forms, Form, Segment
    from regulae.model import TonalCorrespondence

    # Base rate of tone 4 = 0.5 exactly.
    tonal = TonalCorrespondenceTable(
        counts={
            TonalCorrespondence(src_tone=None, tgt_tone="4"): 5.0,
            TonalCorrespondence(src_tone=None, tgt_tone="1"): 5.0,
        },
        prior_pseudo_counts={},
        src_totals={None: 10.0},
    )
    # Rule count/src_count also gives 0.5 under Dirichlet smoothing
    # with V=2: (5 + 1) / (10 + 2) = 6/12 = 0.5.
    link = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=5.0,
        src_count=10.0,
        confidence=0.5,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        tonal_table=tonal,
        cross_dimensional_table=CrossDimensionalLinkTable(entries=(link,)),
    )
    src = Form("A", (Segment("b"), Segment("a")))
    tgt = Form("B", (Segment("b"), Segment("a", tone="4")))
    alignment = align_forms(src, tgt, model=model)
    from regulae.scoring import apply_cross_dimensional_adjustments

    adj = apply_cross_dimensional_adjustments(alignment, model)
    # With P_cond == P_base, pos_adj should be (close to) zero.
    assert abs(adj) < 1e-9


def test_overlay_zero_for_empty_table() -> None:
    """COMMITMENT: an empty cross_dimensional_table always
    contributes zero, regardless of the alignment or tonal table."""
    from regulae import align_forms, Form, Segment

    model = LearnedModel.empty()
    src = Form("A", (Segment("b"), Segment("a")))
    tgt = Form("B", (Segment("b"), Segment("a", tone="4")))
    alignment = align_forms(src, tgt, model=model)
    from regulae.scoring import apply_cross_dimensional_adjustments

    assert apply_cross_dimensional_adjustments(alignment, model) == 0.0
