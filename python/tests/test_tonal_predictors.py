"""Tests for : source-tone predictors for cross-dimensional rules.

Validates that the anomaly diagnostic, the cross-dimensional discovery commit loop,
and the scoring overlay all recognise source-tone values as
first-class source-side predictors — not just segmental merkmal
features.

The key test is ``tone_yoruba_like``-style: a fixture where the
rule is "source tone X maps to target tone Y" with no segmental
predictor that would capture the same information. Before 
the enumeration would miss it entirely because source tones
weren't in the candidate pool.
"""

from __future__ import annotations

import warnings

import pytest

from regulae import (
    CrossDimensionalLink,
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
    """Split off a trailing tone digit and attach it to the last
    vowel. Same convention as the tone_* fixtures."""
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


def _pair(src_word: str, tgt_word: str) -> tuple[Form, Form]:
    s = _toned(src_word)
    t = _toned(tgt_word)
    return (
        Form(lect_id="src", segments=s.segments),
        Form(lect_id="tgt", segments=t.segments),
    )


def _tone_shift_corpus() -> list[tuple[Form, Form]]:
    """30 pairs with a clean source→target tone mapping:

    - source tone 1 → target tone 3
    - source tone 2 → target tone 1

    No segmental feature predicts this — the conditioning is
    purely tonal. A single-feature segmental enumeration (pre-
    ) cannot commit any cross-dim rule on this corpus.
    """
    pairs = []
    words1 = ["pa1", "ka1", "ta1", "ba1", "ma1",
              "pi1", "ki1", "ti1", "bi1", "mi1",
              "pu1", "ku1", "tu1", "bu1", "mu1"]
    words2 = ["pa2", "ka2", "ta2", "ba2", "ma2",
              "pi2", "ki2", "ti2", "bi2", "mi2",
              "pu2", "ku2", "tu2", "bu2", "mu2"]
    for w in words1:
        # tone 1 → tone 3 (preserve segments, shift tone)
        tgt = w[:-1] + "3"
        pairs.append(_pair(w, tgt))
    for w in words2:
        tgt = w[:-1] + "1"
        pairs.append(_pair(w, tgt))
    return pairs


def _pair_model(corpus):
    multi = train_model(cognate_sets_from_pairs(corpus, ("src", "tgt")))
    assert isinstance(multi, MultiLectModel)
    return multi.pairwise_models[frozenset({"src", "tgt"})]


def test_cross_dim_commits_tonal_predictor_rule() -> None:
    """COMMITMENT: on a corpus where the cross-dim rule is
    source-tone → target-tone (no segmental conditioning), cross-dimensional discovery
    commits at least one rule whose ``src_feature.feature`` is
    ``"tone"`` and whose ``src_feature.value`` is a source tone
    value from the corpus."""
    model = _pair_model(_tone_shift_corpus())
    entries = model.cross_dimensional_table.entries
    assert len(entries) > 0, "no cross-dim rules committed at all"
    tonal_entries = [
        e for e in entries if e.src_feature.feature == "tone"
    ]
    assert len(tonal_entries) >= 1, (
        f"no tonal source predictors committed; got {entries}"
    )
    # Validate one of the two expected shifts: src tone 1 → tgt 3
    # OR src tone 2 → tgt 1.
    matches = [
        e for e in tonal_entries
        if (e.src_feature.value == "1" and e.tgt_value == "3")
        or (e.src_feature.value == "2" and e.tgt_value == "1")
    ]
    assert len(matches) >= 1, (
        f"no expected tone-shift rule found; got {tonal_entries}"
    )


def test_cross_dim_tonal_rule_high_confidence_on_clean_shift() -> None:
    """COMMITMENT: a clean tonal shift (no noise) produces a rule
    at confidence 1.0."""
    model = _pair_model(_tone_shift_corpus())
    tonal_entries = [
        e for e in model.cross_dimensional_table.entries
        if e.src_feature.feature == "tone"
    ]
    assert any(e.confidence >= 0.95 for e in tonal_entries), (
        f"expected at least one near-1.0-confidence tonal rule; "
        f"got {[(e.src_feature.value, e.tgt_value, e.confidence) for e in tonal_entries]}"
    )


def test_scoring_applies_tonal_predictor_adjustment() -> None:
    """COMMITMENT: a CrossDimensionalLink with src_feature=tone
    applies its adjustment when the source tone matches and
    skips it when it doesn't."""
    from regulae.scoring import _apply_rule_to_link

    src = _toned("pa1")
    tgt_match = _toned("pa3")
    tgt_miss = _toned("pa5")

    rule = CrossDimensionalLink(
        src_feature=FeatureConstraint("tone", "1"),
        src_position="relative_0",
        tgt_dimension="tone",
        tgt_value="3",
        tgt_position_offset=0,
        count=10.0,
        src_count=10.0,
        confidence=1.0,
    )

    # For a CV form like "pa1", the vowel at position 1 carries
    # the tone. Test at position 1 (the vowel) with offset 0.
    pos_adj = -1.5
    neg_adj = +0.8
    cost = _apply_rule_to_link(
        rule=rule,
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        src_form=src,
        src_pos=1,
        tgt_form=tgt_match,
        tgt_pos=1,
        feature_system="descriptive",
    )
    assert cost == pos_adj

    cost_miss = _apply_rule_to_link(
        rule=rule,
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        src_form=src,
        src_pos=1,
        tgt_form=tgt_miss,
        tgt_pos=1,
        feature_system="descriptive",
    )
    assert cost_miss == neg_adj

    # If the source tone doesn't match, the rule doesn't fire at
    # all (returns 0.0).
    src_wrong_tone = _toned("pa2")
    cost_nofire = _apply_rule_to_link(
        rule=rule,
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        src_form=src_wrong_tone,
        src_pos=1,
        tgt_form=tgt_match,
        tgt_pos=1,
        feature_system="descriptive",
    )
    assert cost_nofire == 0.0


def test_anomaly_emits_tonal_hypotheses() -> None:
    """COMMITMENT: ``find_residual_patterns`` surfaces source-tone
    predictors as ``PatternHypothesis`` entries with
    ``src_feature == 'tone'`` and a non-default ``src_value``."""
    from regulae import find_residual_patterns

    corpus = _tone_shift_corpus()
    model = train_model(cognate_sets_from_pairs(corpus, ("src", "tgt")))
    pair_model = model.pairwise_models[frozenset({"src", "tgt"})]
    hypotheses = find_residual_patterns(
        corpus, pair_model, random_seed=0, min_observations=3
    )
    tonal = [h for h in hypotheses if h.src_feature == "tone"]
    assert len(tonal) >= 1
    # Every tonal hypothesis carries a non-"+" src_value (one of
    # the actual source tone values in the corpus).
    for h in tonal:
        assert h.src_value in {"1", "2"}


def test_tonal_predictor_is_not_committed_on_non_tonal_corpus() -> None:
    """COMMITMENT: on a corpus without any source-side tones, no
    tonal predictor rules are committed."""
    corpus = [
        _pair("pata", "pada"),
        _pair("kata", "kada"),
        _pair("tama", "tada"),
        _pair("pima", "pida"),
        _pair("tama", "tada"),
    ] * 3  # inflate to meet min counts
    model = _pair_model(corpus)
    for e in model.cross_dimensional_table.entries:
        assert e.src_feature.feature != "tone"
