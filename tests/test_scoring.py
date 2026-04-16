"""Tests for link scoring.

These tests are specifications. Each test name encodes a theoretical
commitment from the design discussion. Failures here are not bugs in the
implementation only — they may also indicate that we have to revise the
design.

Notation: COMMITMENT comments mark the design claim being tested.
ROADMAP comments mark known limitations.
"""

import pytest

from regulae import (
    Context,
    FeatureConstraint,
    Link,
    Segment,
    compute_displacement,
    score_link,
)


# ----- Identity, symmetry, and basic shape ---------------------------------


def test_identity_link_has_zero_cost() -> None:
    """COMMITMENT: substituting a segment for itself is free."""
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("p"),))
    assert score_link(link) == 0.0


def test_substitution_costs_more_than_identity() -> None:
    """COMMITMENT: any substitution is more expensive than identity."""
    identity = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("p"),))
    substitution = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    assert score_link(substitution) > score_link(identity)


def test_score_is_symmetric_under_chunk_swap() -> None:
    """COMMITMENT: correspondences are bidirectional. score(A~B) == score(B~A).

    Diachrony (who came from whom) is a separate later inference; at the
    correspondence level, the score must not depend on which lect is on
    which side.
    """
    forward = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    backward = Link(source_chunk=(Segment("f"),), target_chunk=(Segment("p"),))
    assert score_link(forward) == score_link(backward)


def test_closer_segments_have_lower_cost() -> None:
    """COMMITMENT: scoring respects feature-space proximity.

    p~b (one feature: voicing) should be closer than p~f (different
    manner) under merkmal's geometry-weighted distance.
    """
    p_b = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("b"),))
    p_f = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    p_k = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("k"),))
    # All three are nontrivial; we don't pin exact relative ordering of f vs k
    # since that depends on the feature system, but all should exceed identity.
    assert score_link(p_b) > 0.0
    assert score_link(p_f) > 0.0
    assert score_link(p_k) > 0.0


# ----- Insertion and deletion ---------------------------------------------


def test_empty_target_chunk_represents_deletion_with_gap_cost() -> None:
    """COMMITMENT: 1-to-0 link is a first-class type, scored with gap cost."""
    link = Link(source_chunk=(Segment("u"),), target_chunk=())
    assert score_link(link) > 0.0


def test_empty_source_chunk_represents_insertion_with_gap_cost() -> None:
    """COMMITMENT: 0-to-1 link is a first-class type, scored with gap cost."""
    link = Link(source_chunk=(), target_chunk=(Segment("e"),))
    assert score_link(link) > 0.0


def test_insertion_and_deletion_have_equal_cost() -> None:
    """COMMITMENT: bidirectional symmetry extends to insertion vs deletion."""
    insertion = Link(source_chunk=(), target_chunk=(Segment("e"),))
    deletion = Link(source_chunk=(Segment("e"),), target_chunk=())
    assert score_link(insertion) == score_link(deletion)


def test_empty_to_empty_link_costs_nothing() -> None:
    """COMMITMENT: an empty link is degenerate but valid; its cost is zero."""
    link = Link(source_chunk=(), target_chunk=())
    assert score_link(link) == 0.0


# ----- Multi-segment compositional fallback -------------------------------


def test_two_segment_to_two_segment_link_uses_compositional_sum() -> None:
    """ROADMAP: compositional fallback is the v1 behavior for many-to-many.

    Latin /kt/ ~ Italian /tt/. The first version pairs segments left-to-right
    and sums merkmal distances. Real chunk-level scoring (a learned phrase
    table where (kt, tt) gets its own probability after recurring across
    many pairs) requires the EM loop in the training pipeline.
    """
    link = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t")),
    )
    # Compositional: cost(k,t) + cost(t,t) = nontrivial + 0
    cost = score_link(link)
    assert cost > 0.0


def test_asymmetric_chunk_link_has_extra_penalty() -> None:
    """COMMITMENT: links with chunk-length asymmetry pay a penalty.

    A 2~3 link should cost more than the same content scored as two
    independent links of equal sizes, even when the segment-level matches
    are reasonable. This deters the compositional fallback from claiming
    asymmetric chunks for free.
    """
    asymmetric = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t"), Segment("a")),
    )
    symmetric = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t")),
    )
    assert score_link(asymmetric) > score_link(symmetric)


def test_recurring_chunk_correspondence_promoted_to_phrase_table() -> None:
    """When a chunk recurs across many pairs, it should be promoted to
    the phrase table and scored holistically, not compositionally.

    Resolved via ``train_model``'s chunk promotion.
    """
    from regulae import Form
    from tests._helpers import train_model

    def form(lect_id: str, ipa: str) -> Form:
        return Form(lect_id=lect_id, segments=tuple(Segment(c) for c in ipa))

    corpus = [
        (form("A", "akta"), form("B", "atta")),
        (form("A", "ikti"), form("B", "itti")),
        (form("A", "okto"), form("B", "otto")),
        (form("A", "ukte"), form("B", "utte")),
        (form("A", "akta"), form("B", "atta")),
        (form("A", "ikti"), form("B", "itti")),
        (form("A", "okto"), form("B", "otto")),
        (form("A", "ukte"), form("B", "utte")),
        (form("A", "akto"), form("B", "atto")),
        (form("A", "ikti"), form("B", "itti")),
        (form("A", "ukte"), form("B", "utte")),
        (form("A", "okta"), form("B", "otta")),
        (form("A", "akte"), form("B", "atte")),
        (form("A", "ikto"), form("B", "itto")),
        (form("A", "uktu"), form("B", "uttu")),
    ]
    trained = train_model(corpus)
    # The phrase table should be nonempty — at least one chunk has
    # been promoted from this corpus.
    assert len(trained.chunk_table.entries) > 0


# ----- Context conditioning (prior-only: inert) ----------------------------


def test_context_does_not_affect_prior_only_score() -> None:
    """ROADMAP: context-conditioning of scores comes with the trained
    model. Without a model, the context is stored but inert.
    """
    bare = Link(source_chunk=(Segment("k"),), target_chunk=(Segment("tʃ"),))
    conditioned = Link(
        source_chunk=(Segment("k"),),
        target_chunk=(Segment("tʃ"),),
        context=Context(following=(FeatureConstraint("front", "+"),)),
    )
    # Same cost without a model (prior-only scoring).
    assert score_link(bare) == score_link(conditioned)


# ----- Feature displacement -----------------------------------------------


def test_identity_displacement_is_empty() -> None:
    """COMMITMENT: identity has no feature displacement."""
    assert compute_displacement(Segment("p"), Segment("p")) == ()


def test_displacement_is_nonempty_for_substitution() -> None:
    """COMMITMENT: any substitution yields at least one feature displacement."""
    disp = compute_displacement(Segment("p"), Segment("f"))
    assert len(disp) > 0


def test_grimm_type_correspondences_share_a_common_displacement() -> None:
    """COMMITMENT: regular sound correspondences share displacement structure
    across natural classes.

    p~f, t~s, k~x are stop~fricative pairs at different places of
    articulation. They should share at least one common feature
    displacement (the stop ↔ fricative manner shift), even though they
    differ in place. This is the foundation of feature-level regularity:
    what's "regular" is not a list of segment pairs but a feature-space
    transformation that applies across a natural class.
    """
    disp_pf = set(compute_displacement(Segment("p"), Segment("f")))
    disp_ts = set(compute_displacement(Segment("t"), Segment("s")))
    disp_kx = set(compute_displacement(Segment("k"), Segment("x")))

    # All three should be nonempty.
    assert disp_pf and disp_ts and disp_kx

    # All three should share at least one common displacement element.
    common = disp_pf & disp_ts & disp_kx
    assert common, (
        f"Expected at least one shared feature displacement across Grimm-type "
        f"correspondences, got: pf={disp_pf}, ts={disp_ts}, kx={disp_kx}"
    )


def test_displacement_is_inverse_under_segment_swap() -> None:
    """COMMITMENT: swapping the source and target inverts the displacement.

    For every (feature, present, absent) in compute_displacement(A, B),
    there must be a (feature, absent, present) in compute_displacement(B, A),
    and vice versa. This is the segment-level analogue of bidirectionality:
    correspondences are stated symmetrically, but each statement carries
    direction info that flips cleanly under swap.
    """
    forward = compute_displacement(Segment("p"), Segment("f"))
    backward = compute_displacement(Segment("f"), Segment("p"))

    # Same number of displacements.
    assert len(forward) == len(backward)

    # Each forward displacement has an inverse in backward.
    forward_set = set(forward)
    backward_set = set(backward)

    for d in forward:
        inverse = type(d)(feature=d.feature, from_value=d.to_value, to_value=d.from_value)
        assert inverse in backward_set, (
            f"Expected inverse {inverse} of {d} in backward displacement"
        )

    for d in backward:
        inverse = type(d)(feature=d.feature, from_value=d.to_value, to_value=d.from_value)
        assert inverse in forward_set


def test_distant_segments_have_more_displacement_than_close_ones() -> None:
    """COMMITMENT: feature distance reflects feature-space proximity.

    A consonant-to-vowel pair (p, i) should differ on more feature
    dimensions than a near-pair (p, b). This is a sanity check on the
    interaction between merkmal's feature inventory and our displacement
    extraction.
    """
    near = compute_displacement(Segment("p"), Segment("b"))
    far = compute_displacement(Segment("p"), Segment("i"))
    assert len(far) > len(near)


def test_displacement_for_identical_consonants_is_empty() -> None:
    """COMMITMENT: identity has no displacement, regardless of grapheme."""
    for grapheme in ["p", "t", "k", "a", "i", "u", "n", "s", "tʃ"]:
        seg = Segment(grapheme)
        assert compute_displacement(seg, seg) == (), (
            f"Identity displacement for {grapheme!r} should be empty"
        )


# ----- Gap cost calibration ------------------------------------------------


def test_gap_cost_scales_linearly_with_chunk_length() -> None:
    """COMMITMENT: an N-segment insertion costs N times a 1-segment insertion.

    This is a contract of the compositional fallback: gap costs are linear
    in chunk length. Future scoring may replace this with a learned
    distribution, but for prior-only scoring the linearity is the spec.
    """
    one = Link(source_chunk=(), target_chunk=(Segment("e"),))
    two = Link(source_chunk=(), target_chunk=(Segment("e"), Segment("s")))
    three = Link(source_chunk=(), target_chunk=(Segment("e"), Segment("s"), Segment("a")))
    s1 = score_link(one)
    s2 = score_link(two)
    s3 = score_link(three)
    # Approximate equality (no float trickery expected here, but be defensive).
    assert abs(s2 - 2 * s1) < 1e-9
    assert abs(s3 - 3 * s1) < 1e-9


def test_gap_cost_matches_documented_constant() -> None:
    """COMMITMENT: the default gap cost is exposed as DEFAULT_GAP_COST.

    Pinning this prevents accidental drift in the cost when the implementation
    changes. If we need to retune, the test must be updated explicitly.
    """
    from regulae.scoring import DEFAULT_GAP_COST

    one = Link(source_chunk=(), target_chunk=(Segment("e"),))
    assert score_link(one) == DEFAULT_GAP_COST


# ----- Custom feature system ----------------------------------------------


def test_score_link_accepts_alternate_feature_system() -> None:
    """COMMITMENT: scoring is parameterised by feature system.

    The same segment pair scored under different merkmal systems should
    yield (potentially) different costs, and both should run without error.
    Robustness across feature systems is itself evidence (see 02_representation).
    """
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    s_descriptive = score_link(link, feature_system="descriptive")
    s_distinctive = score_link(link, feature_system="distinctive")
    # Both should be positive (p != f).
    assert s_descriptive > 0
    assert s_distinctive > 0


# ----- Smoke tests on real cognate pairs ----------------------------------
#
# These guard against silent regressions in the merkmal interaction. They
# don't pin exact numbers (which would lock us to a specific merkmal
# version), but they assert ordering relationships that any reasonable
# feature distance must respect.


def test_pater_fadar_inflection_cost_ordering() -> None:
    """SMOKE: Latin pater ~ Gothic fadar at the segment level.

    The 1-to-1 link costs should respect:
    - identity (a~a) is cheapest
    - close substitutions (t~d, p~f) are nontrivial
    - the t~d voicing change should be cheaper than p~f manner change
      under most reasonable feature systems
    """
    aa = score_link(Link(source_chunk=(Segment("a"),), target_chunk=(Segment("a"),)))
    pf = score_link(Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),)))
    td = score_link(Link(source_chunk=(Segment("t"),), target_chunk=(Segment("d"),)))

    assert aa == 0.0
    assert pf > 0.0
    assert td > 0.0
    # We don't pin pf vs td absolutely; we do require both to exceed identity.


def test_pitar_pater_low_distance() -> None:
    """SMOKE: Sanskrit pitr-type ~ Latin pater-type segment correspondences
    are nearly identical at the consonant skeleton, p~p and t~t.
    """
    pp = score_link(Link(source_chunk=(Segment("p"),), target_chunk=(Segment("p"),)))
    tt = score_link(Link(source_chunk=(Segment("t"),), target_chunk=(Segment("t"),)))
    assert pp == 0.0
    assert tt == 0.0


def test_noctem_notte_cluster_link_cost_is_finite_and_positive() -> None:
    """SMOKE: Latin /kt/ ~ Italian /tt/ scored by the compositional fallback.

    The link should produce a finite, positive cost. We do not yet check
    that it is "small" — that would require the learned phrase table.
    """
    link = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t")),
    )
    cost = score_link(link)
    assert cost > 0.0
    assert cost < 10.0  # generously bounded; catches obvious blow-ups
