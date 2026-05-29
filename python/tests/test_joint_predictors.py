"""Tests for : joint-predictor cross-dimensional rules.

Validates the data-model extension (optional ``src_feature_2`` /
``src_position_2`` on :class:`CrossDimensionalLink`), the
enumeration of (segmental feature, source tone) joint
candidates, the non-interaction guard that rejects "passenger"
joints, the BIC 2-parameter penalty that keeps singles winning
when they suffice, and the scoring overlay that evaluates both
predicates.
"""

from __future__ import annotations

import warnings

import pytest

from regulae import (
    CrossDimensionalLink,
    CrossDimensionalLinkTable,
    FeatureConstraint,
    Form,
    LearnedModel,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


@pytest.fixture(autouse=True)
def _suppress_pair_deprecation():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    yield


def _toned(word: str) -> Form:
    vowels = set("aeiou")
    tone = None
    if word and word[-1].isdigit():
        tone = word[-1]
        word = word[:-1]
    segs: list[Segment] = []
    tone_attached = False
    for ch in reversed(word):
        if not tone_attached and ch in vowels and tone:
            segs.insert(0, Segment(grapheme=ch, tone=tone))
            tone_attached = True
        else:
            segs.insert(0, Segment(grapheme=ch))
    return Form(lect_id="_", segments=tuple(segs))


def _pair(src: str, tgt: str) -> tuple[Form, Form]:
    return (
        Form(lect_id="src", segments=_toned(src).segments),
        Form(lect_id="tgt", segments=_toned(tgt).segments),
    )


def _pair_model(corpus):
    multi = train_model(cognate_sets_from_pairs(corpus, ("src", "tgt")))
    assert isinstance(multi, MultiLectModel)
    return multi.pairwise_models[frozenset({"src", "tgt"})]


# ----- data model --------------------------------------------------------


def test_cross_dim_link_joint_fields_default_none() -> None:
    """COMMITMENT: single-predictor rules have both joint
    extension fields at None. Existing  constructions
    continue to work."""
    rule = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=40.0,
        src_count=40.0,
        confidence=1.0,
    )
    assert rule.src_feature_2 is None
    assert rule.src_position_2 is None


def test_cross_dim_link_joint_fields_settable() -> None:
    rule = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        src_feature_2=FeatureConstraint("tone", "2"),
        src_position_2="relative_0",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=10.0,
        src_count=12.0,
        confidence=10 / 12,
    )
    assert rule.src_feature_2.feature == "tone"
    assert rule.src_feature_2.value == "2"
    assert rule.src_position_2 == "relative_0"


# ----- scoring overlay ---------------------------------------------------


def test_joint_rule_fires_only_when_both_predicates_hold() -> None:
    """COMMITMENT: a joint rule's adjustment applies only when
    both predicates hold on the source side."""
    from regulae.scoring import _apply_rule_to_link

    # Joint rule: source has voicing at -1 AND tone=2 at 0.
    rule = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        src_feature_2=FeatureConstraint("tone", "2"),
        src_position_2="relative_0",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=10.0,
        src_count=12.0,
        confidence=10 / 12,
    )
    pos_adj = -1.5
    neg_adj = +0.8
    feature_system = "descriptive"

    # Form "ba2" — initial b (voiced), vowel a with tone 2.
    # Link position at the vowel (index 1):
    #   - relative_-1 is the voiced consonant  ✓
    #   - relative_0 is the vowel with tone 2  ✓
    src = _toned("ba2")
    tgt_match = _toned("ba4")   # target vowel has tone 4 (rule target)
    cost = _apply_rule_to_link(
        rule=rule,
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        src_form=src,
        src_pos=1,
        tgt_form=tgt_match,
        tgt_pos=1,
        feature_system=feature_system,
    )
    assert cost == pos_adj

    # Miss on component 1: initial is voiceless.
    src_vl = _toned("pa2")
    cost_vl = _apply_rule_to_link(
        rule=rule,
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        src_form=src_vl,
        src_pos=1,
        tgt_form=tgt_match,
        tgt_pos=1,
        feature_system=feature_system,
    )
    assert cost_vl == 0.0

    # Miss on component 2: wrong tone.
    src_wrong_tone = _toned("ba3")
    cost_wt = _apply_rule_to_link(
        rule=rule,
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        src_form=src_wrong_tone,
        src_pos=1,
        tgt_form=tgt_match,
        tgt_pos=1,
        feature_system=feature_system,
    )
    assert cost_wt == 0.0


# ----- commit loop -------------------------------------------------------


def _joint_signal_corpus() -> list[tuple[Form, Form]]:
    """A corpus where the rule is a JOINT predictor: the target
    tone is 4 only when the source initial is voiced AND the
    source vowel has tone 2. Any other combination keeps source
    tone.

    Neither "voiced alone" nor "tone=2 alone" predicts tone 4
    strongly: out of N observations each predictor alone has
    poor confidence, but the joint is strict.
    """
    pairs = []
    # Voiced + tone 2: target becomes tone 4 (the joint case).
    # 12 pairs — enough to pass BIC.
    for init in ["b", "d", "g", "m"]:
        for vowel in ["a", "i", "u"]:
            pairs.append(_pair(init + vowel + "2", init + vowel + "4"))

    # Voiced + tone 1: target keeps tone 1.
    for init in ["b", "d", "g"]:
        for vowel in ["a", "i", "u", "o"]:
            pairs.append(_pair(init + vowel + "1", init + vowel + "1"))

    # Voiceless + tone 2: target keeps tone 2.
    for init in ["p", "t", "k", "s"]:
        for vowel in ["a", "i", "u"]:
            pairs.append(_pair(init + vowel + "2", init + vowel + "2"))

    # Voiceless + tone 1: target keeps tone 1.
    for init in ["p", "t", "k"]:
        for vowel in ["a", "i", "u", "o"]:
            pairs.append(_pair(init + vowel + "1", init + vowel + "1"))

    return pairs


def test_cross_dim_commits_joint_rule_on_designed_joint_signal() -> None:
    """COMMITMENT: on a corpus designed so that the rule fires
    only when (voicing AND source-tone) jointly hold, cross-dimensional discovery
    commits at least one joint rule with both ``src_feature_2``
    set and high confidence."""
    model = _pair_model(_joint_signal_corpus())
    entries = model.cross_dimensional_table.entries
    assert len(entries) > 0
    joint_rules = [e for e in entries if e.src_feature_2 is not None]
    assert len(joint_rules) >= 1, (
        f"no joint rule committed; got {entries}"
    )
    # The committed joint should be voiced + tone=2 → tone=4 or
    # an equivalent framing.
    matching = [
        r for r in joint_rules
        if r.tgt_value == "4"
        and r.confidence >= 0.8
        and (
            (r.src_feature.feature == "voiced" and r.src_feature_2.feature == "tone")
            or (r.src_feature.feature == "tone" and r.src_feature_2.feature == "voiced")
        )
    ]
    assert len(matching) >= 1, (
        f"no voiced × tone=2 → tone=4 joint; got {joint_rules}"
    )


def test_cross_dim_does_not_commit_joint_when_single_predictor_suffices() -> None:
    """COMMITMENT: on a corpus where a single-predictor rule
    fully explains the signal (voicing alone predicts target
    tone), the BIC 2-parameter penalty keeps the commit loop
    preferring the single-predictor rule. No joint rule is
    committed."""
    # Voiceless → tone 1, voiced → tone 4, irrespective of source
    # tone. The clean-fixture pattern.
    pairs = []
    for src_tone in ["1", "2", "3", "4"]:
        for init in ["p", "t", "k", "s"]:
            for vowel in ["a", "i", "u"]:
                pairs.append(_pair(init + vowel + src_tone, init + vowel + "1"))
        for init in ["b", "d", "g", "m"]:
            for vowel in ["a", "i", "u"]:
                pairs.append(_pair(init + vowel + src_tone, init + vowel + "4"))
    model = _pair_model(pairs)
    joint_rules = [
        e for e in model.cross_dimensional_table.entries
        if e.src_feature_2 is not None
    ]
    assert len(joint_rules) == 0, (
        f"unexpected joint commits on clean single-predictor signal: "
        f"{joint_rules}"
    )


def test_cross_dim_joint_rejects_passenger_predictor() -> None:
    """COMMITMENT: the non-interaction guard filters "passenger"
    joints where one predictor is tautologically implied by the
    other. A corpus where voicing is always tied to a specific
    source tone should not produce joint rules mixing them.
    """
    # All voiced forms have source tone 2; all voiceless have
    # source tone 1. The cross-dim mapping is voicing → target
    # tone (voiced→4, voiceless→1). A joint "voiced+tone2→4"
    # rule would be tautological — the segmental component
    # "voiced" already fully determines the target.
    pairs = []
    for init in ["b", "d", "g", "m"] * 3:
        for vowel in ["a", "i"]:
            pairs.append(_pair(init + vowel + "2", init + vowel + "4"))
    for init in ["p", "t", "k", "s"] * 3:
        for vowel in ["a", "i"]:
            pairs.append(_pair(init + vowel + "1", init + vowel + "1"))
    model = _pair_model(pairs)
    joint_rules = [
        e for e in model.cross_dimensional_table.entries
        if e.src_feature_2 is not None
    ]
    assert len(joint_rules) == 0, (
        f"expected no joint rules on passenger-predictor signal; "
        f"got {joint_rules}"
    )


# ----- dedup and signature -----------------------------------------------


def test_joint_rule_signature_includes_second_feature() -> None:
    """COMMITMENT: ``_rule_signature`` distinguishes joint rules
    from single-predictor rules with the same first feature, so
    the commit loop's iteration-dedup doesn't spuriously skip a
    joint because an earlier iteration committed the single."""
    from regulae.training import _rule_signature

    single = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=20.0,
        src_count=20.0,
        confidence=1.0,
    )
    joint = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        src_feature_2=FeatureConstraint("tone", "2"),
        src_position_2="relative_0",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=12.0,
        src_count=14.0,
        confidence=12 / 14,
    )
    assert _rule_signature(single) != _rule_signature(joint)
