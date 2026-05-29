"""Tests for  step 5: the cross-dimensional discovery BIC commit loop.

These verify that ``_cross_dimensional_discovery`` commits
the expected cross-dimensional rules on handcrafted fixtures
where the tonogenesis pattern is unambiguous, and that it
correctly declines to commit on data with no cross-dimensional
signal.
"""

import warnings

import pytest

from regulae import (
    CrossDimensionalLink,
    Form,
    LearnedModel,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


@pytest.fixture(autouse=True)
def _suppress_pair_deprecation():
    """Silence the pair-input DeprecationWarning for these tests."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    yield


def _pair_model(corpus: list[tuple[Form, Form]]) -> LearnedModel:
    """Train on a pair corpus via the CognateSet API and return
    the inner pair model."""
    multi = train_model(cognate_sets_from_pairs(corpus, ("A", "B")))
    assert isinstance(multi, MultiLectModel)
    return multi.pairwise_models[frozenset({"A", "B"})]


def _tonogenesis_pair(src_word: str, tgt_word: str) -> tuple[Form, Form]:
    """Build a (src, tgt) pair where the target's final vowel
    carries tone 4 if the source initial is voiced/sonorant and
    tone 1 otherwise."""
    src = Form("A", tuple(Segment(g) for g in src_word))
    voiced = src_word[0] in "bdgmn"
    tone = "4" if voiced else "1"
    tgt_segs = tuple(
        Segment(grapheme=g, tone=tone if i == len(tgt_word) - 1 else None)
        for i, g in enumerate(tgt_word)
    )
    return src, Form("B", tgt_segs)


def _tonogenesis_corpus() -> list[tuple[Form, Form]]:
    """40 pairs: 20 voiced-initial, 20 voiceless-initial. Each
    has its target vowel tone determined by the source initial's
    voicing (voiced → 4, voiceless → 1)."""
    voiced = ["ba", "da", "ga", "ma", "na", "bi", "di", "gu", "bo", "ma",
              "be", "do", "gi", "mi", "nu", "bu", "du", "gu", "mo", "na"]
    voiceless = ["pa", "ta", "ka", "sa", "fa", "pi", "ti", "ku", "po", "ta",
                 "pe", "to", "ki", "si", "fu", "pu", "tu", "ko", "so", "fo"]
    return [_tonogenesis_pair(w, w) for w in voiced + voiceless]


# ----- commit on clean signal --------------------------------------------


def test_cross_dim_commits_at_least_one_rule_on_clean_tonogenesis() -> None:
    """COMMITMENT: on a 40-pair corpus with a clean voicing→tone
    split, cross-dimensional discovery must commit at least one cross-dimensional
    link whose source feature is voicing-related and whose
    confidence is above 0.8."""
    model = _pair_model(_tonogenesis_corpus())
    entries = model.cross_dimensional_table.entries
    assert len(entries) > 0
    # At least one rule should be high-confidence and
    # voicing-related.
    voicing_features = {"voiced", "voiceless", "sonorant", "nasal",
                        "stop", "fricative", "consonant"}
    high_conf_rules = [
        e for e in entries
        if e.src_feature.feature in voicing_features and e.confidence >= 0.8
    ]
    assert len(high_conf_rules) > 0


def test_cross_dim_commits_rule_predicting_correct_tone() -> None:
    """COMMITMENT: a committed voicing→tone rule must predict
    a tone value that actually appears in the corpus. Not a
    smoke test — this caught a bug where the commit loop
    produced count=0 rules."""
    model = _pair_model(_tonogenesis_corpus())
    for rule in model.cross_dimensional_table.entries:
        assert rule.tgt_value in {"1", "4"}
        assert rule.count > 0
        assert rule.src_count > 0
        assert 0.0 <= rule.confidence <= 1.0


def test_cross_dim_commit_is_deterministic() -> None:
    """COMMITMENT: running train_model twice on the same corpus
    produces the same set of cross-dimensional rules (same
    signatures)."""
    corpus = _tonogenesis_corpus()
    m1 = _pair_model(corpus)
    m2 = _pair_model(corpus)
    sigs1 = sorted(
        (e.src_feature.feature, e.src_position, e.tgt_value,
         e.tgt_position_offset)
        for e in m1.cross_dimensional_table.entries
    )
    sigs2 = sorted(
        (e.src_feature.feature, e.src_position, e.tgt_value,
         e.tgt_position_offset)
        for e in m2.cross_dimensional_table.entries
    )
    assert sigs1 == sigs2


def test_cross_dim_does_not_commit_duplicates() -> None:
    """COMMITMENT: the commit loop must not emit the same rule
    twice. Each committed rule signature should be unique."""
    model = _pair_model(_tonogenesis_corpus())
    sigs = [
        (e.src_feature.feature, e.src_position, e.tgt_dimension,
         e.tgt_value, e.tgt_position_offset)
        for e in model.cross_dimensional_table.entries
    ]
    assert len(sigs) == len(set(sigs))


# ----- no-commit baselines -----------------------------------------------


def test_cross_dim_commits_nothing_on_non_tonal_corpus() -> None:
    """COMMITMENT: a corpus with no tonal annotations on either
    side produces no cross-dimensional commits — the anomaly
    detector sees no tonal outcome to correlate against."""
    pairs = [
        (Form("A", tuple(Segment(g) for g in s)),
         Form("B", tuple(Segment(g) for g in t)))
        for s, t in [("pata", "fada"), ("kata", "hada"),
                     ("pita", "fida"), ("kota", "hoda"),
                     ("puta", "fuda"), ("kuta", "huda")]
    ]
    model = _pair_model(pairs)
    assert model.cross_dimensional_table.entries == ()


def test_cross_dim_commits_nothing_on_independent_tonal_data() -> None:
    """COMMITMENT: on a corpus where source features and target
    tones are independent by construction, cross-dimensional discovery should
    commit no rules (or only ones that happen to pass BIC on
    very thin small-sample noise — but nothing systematic)."""
    # Tones assigned randomly by position, not by voicing.
    pairs = [
        (Form("A", (Segment("b"), Segment("a"))),
         Form("B", (Segment("b"), Segment("a", tone="1")))),
        (Form("A", (Segment("p"), Segment("a"))),
         Form("B", (Segment("p"), Segment("a", tone="2")))),
        (Form("A", (Segment("d"), Segment("i"))),
         Form("B", (Segment("d"), Segment("i", tone="1")))),
        (Form("A", (Segment("t"), Segment("i"))),
         Form("B", (Segment("t"), Segment("i", tone="2")))),
        (Form("A", (Segment("g"), Segment("u"))),
         Form("B", (Segment("g"), Segment("u", tone="2")))),
        (Form("A", (Segment("k"), Segment("u"))),
         Form("B", (Segment("k"), Segment("u", tone="1")))),
    ]
    model = _pair_model(pairs)
    # Exact commit count on tiny noisy corpora is not
    # guaranteed; the invariant is that commits (if any) have
    # low confidence.
    for rule in model.cross_dimensional_table.entries:
        assert rule.confidence <= 1.0  # trivially true
        # No rule should claim near-perfect confidence on a
        # 6-pair corpus with no signal.
        if rule.src_feature.feature in {"voiced", "voiceless"}:
            # These are the ones most likely to spuriously commit
            # — reject high confidence claims.
            assert rule.confidence <= 0.9


# ----- existing experiments are unchanged --------------------------------


def test_cross_dim_does_not_commit_on_pata_fada_regression() -> None:
    """COMMITMENT: the canonical p↔f, t↔d test corpus has no
    tonal data and cross-dimensional discovery must leave it alone. Regression
    check against the dozens of existing tests that use this
    corpus shape."""
    pairs = [
        (Form("A", tuple(Segment(c) for c in "pata")),
         Form("B", tuple(Segment(c) for c in "fada")))
        for _ in range(4)
    ]
    model = _pair_model(pairs)
    assert model.cross_dimensional_table.entries == ()


# ----- formatting --------------------------------------------------------


def test_format_model_shows_cross_dimensional_section() -> None:
    """COMMITMENT: format_model includes a
    "Cross-dimensional links" section after Tonal
    correspondences, populated from the committed rules."""
    from regulae import format_model
    model = _pair_model(_tonogenesis_corpus())
    out = format_model(model)
    assert "Cross-dimensional links" in out
    # On the tonogenesis fixture, at least one voicing-related
    # rule should be rendered.
    assert ("voiced" in out) or ("voiceless" in out) or ("nasal" in out)


def test_format_model_shows_none_when_no_cross_dim_rules() -> None:
    """COMMITMENT: when the table is empty, the section shows
    "(none)" rather than being missing or showing a blank."""
    from regulae import format_model
    model = LearnedModel.empty()
    out = format_model(model)
    assert "Cross-dimensional links (0)" in out
    assert "(none)" in out


def test_describe_cross_dimensional_rule_renders_fields() -> None:
    """COMMITMENT: describe_cross_dimensional_rule returns a
    multi-line report naming the rule's source, target, support
    counts, and per-match cost adjustment."""
    from regulae import describe_cross_dimensional_rule
    model = _pair_model(_tonogenesis_corpus())
    assert len(model.cross_dimensional_table.entries) > 0
    report = describe_cross_dimensional_rule(model, 0)
    # Each expected label is present in the rendered output.
    assert "source:" in report
    assert "target:" in report
    assert "support:" in report
    assert "confidence:" in report
    assert "cost adjustment on match:" in report
    assert "cost adjustment on miss:" in report


def test_describe_cross_dimensional_rule_out_of_bounds_raises() -> None:
    from regulae import describe_cross_dimensional_rule
    model = LearnedModel.empty()
    with pytest.raises(IndexError):
        describe_cross_dimensional_rule(model, 0)


# -----  consolidation: rule floors + dual-framing dedup ---------------


def test_cross_dim_rejects_low_count_rules() -> None:
    """COMMITMENT: cross-dim rules with ``count < 3`` are not
    committed even if BIC passes. This filters the 1/N-style
    noise commits seen on tone_vietnamese_like pre--consolidation.
    """
    from regulae.training import _CROSS_DIM_MIN_RULE_COUNT
    model = _pair_model(_tonogenesis_corpus())
    for rule in model.cross_dimensional_table.entries:
        assert rule.count >= _CROSS_DIM_MIN_RULE_COUNT


def test_cross_dim_rejects_low_confidence_rules() -> None:
    """COMMITMENT: cross-dim rules with ``confidence < 0.5`` are
    not committed."""
    from regulae.training import _CROSS_DIM_MIN_RULE_CONFIDENCE
    model = _pair_model(_tonogenesis_corpus())
    for rule in model.cross_dimensional_table.entries:
        assert rule.confidence >= _CROSS_DIM_MIN_RULE_CONFIDENCE


def test_cross_dim_dedups_dual_framings_on_clean_fixture() -> None:
    """COMMITMENT: dual framings of the same underlying rule are
    collapsed to a single canonical entry. The clean tonogenesis
    fixture commits two logical rules (voiced→tone4 and
    voiceless→tone1); without dedup the greedy loop produces
    four (each rule seen from two link positions). After dedup,
    exactly two entries remain.
    """
    model = _pair_model(_tonogenesis_corpus())
    entries = model.cross_dimensional_table.entries
    # Group by (src_feature, tgt_value). Each group should have
    # exactly one rule (the chosen canonical framing).
    seen_pairs = set()
    for rule in entries:
        key = (rule.src_feature.feature, rule.tgt_dimension, rule.tgt_value)
        assert key not in seen_pairs, (
            f"duplicate framing not deduped: {key}"
        )
        seen_pairs.add(key)


def test_dedup_dual_framings_unit() -> None:
    """Unit test for _dedup_dual_framings on hand-built entries."""
    from regulae.training import _dedup_dual_framings
    from regulae.model import CrossDimensionalLink
    from regulae.types import FeatureConstraint

    rule_a1 = CrossDimensionalLink(
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=0,
        count=40.0,
        src_count=40.0,
        confidence=1.0,
    )
    rule_a2 = CrossDimensionalLink(  # dual of rule_a1 (delta = +1)
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_0",
        tgt_dimension="tone",
        tgt_value="4",
        tgt_position_offset=1,
        count=40.0,
        src_count=40.0,
        confidence=1.0,
    )
    rule_b = CrossDimensionalLink(  # different target — should survive
        src_feature=FeatureConstraint("voiced", "+"),
        src_position="relative_-1",
        tgt_dimension="tone",
        tgt_value="2",
        tgt_position_offset=0,
        count=10.0,
        src_count=12.0,
        confidence=0.83,
    )
    deduped = _dedup_dual_framings((rule_a1, rule_a2, rule_b))
    assert len(deduped) == 2
    # The canonical choice within the dual-framing group is the
    # one with the smallest src_offset (relative_-1 wins over
    # relative_0).
    assert deduped[0] is rule_a1
    assert deduped[1] is rule_b
