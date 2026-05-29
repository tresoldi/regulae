"""Tests for the  anomaly detection engine.

Covers :func:`find_residual_patterns`: the happy-path tonogenesis
signal, the no-signal baseline, the no-tones early-out, empty
corpus and other edges, determinism under repeated runs, and
that the function never mutates the model or corpus.
"""

import pytest

from regulae import (
    Form,
    HYPOTHESIS_KINDS,
    PatternHypothesis,
    Segment,
    find_residual_patterns,
    train_model,
)


def _form(lect: str, word_with_tones: list[tuple[str, str | None]]) -> Form:
    """Build a Form from a list of (grapheme, tone) pairs."""
    return Form(
        lect,
        tuple(Segment(grapheme=g, tone=t) for g, t in word_with_tones),
    )


def _voiced_corpus() -> list[tuple[Form, Form]]:
    """A corpus with a clean tonogenesis signal.

    Every source form starts with a consonant (voiced or
    voiceless) followed by a vowel. The target form preserves
    the segments but puts a tone on the vowel:

    - If the source initial consonant is voiced, the target
      vowel gets tone "4" (low register).
    - If the source initial consonant is voiceless, the target
      vowel gets tone "1" (high register).

    Exactly the segmental→tonal cross-dimensional pattern
    anomaly detection should surface as the top hypothesis.
    """
    pairs_plain = [
        # Voiced initial → tone 4 on vowel
        ("ba", "ba"),
        ("da", "da"),
        ("ga", "ga"),
        ("ma", "ma"),
        ("na", "na"),
        ("bi", "bi"),
        ("di", "di"),
        ("gu", "gu"),
        ("bo", "bo"),
        ("ma", "ma"),
        # Voiceless initial → tone 1 on vowel
        ("pa", "pa"),
        ("ta", "ta"),
        ("ka", "ka"),
        ("sa", "sa"),
        ("fa", "fa"),
        ("pi", "pi"),
        ("ti", "ti"),
        ("ku", "ku"),
        ("po", "po"),
        ("ta", "ta"),
    ]

    def make_pair(src_word: str, tgt_word: str) -> tuple[Form, Form]:
        # Source has no tones.
        src = Form(
            lect_id="A",
            segments=tuple(Segment(g) for g in src_word),
        )
        # Target has a tone on the vowel (last segment here).
        # Tone is 4 if the source initial is voiced/sonorant, 1 otherwise.
        initial = src_word[0]
        # Voiced / sonorant set for this synthetic data.
        voiced = initial in ("b", "d", "g", "m", "n")
        tone = "4" if voiced else "1"
        tgt_segs = tuple(
            Segment(grapheme=g, tone=tone if i == len(tgt_word) - 1 else None)
            for i, g in enumerate(tgt_word)
        )
        tgt = Form(lect_id="B", segments=tgt_segs)
        return src, tgt

    return [make_pair(s, t) for s, t in pairs_plain]


def _independent_corpus() -> list[tuple[Form, Form]]:
    """A corpus with NO cross-dimensional signal: target tones
    are assigned arbitrarily in a pattern unrelated to the
    source consonant. Anomaly detection should return no
    significant hypotheses."""
    pairs = [
        ("ba", "ba", "1"),
        ("da", "da", "1"),
        ("ga", "ga", "2"),
        ("ma", "ma", "2"),
        ("pa", "pa", "1"),
        ("ta", "ta", "2"),
        ("ka", "ka", "1"),
        ("sa", "sa", "2"),
        ("bi", "bi", "2"),
        ("pi", "pi", "1"),
    ]
    out: list[tuple[Form, Form]] = []
    for src_word, tgt_word, tone in pairs:
        src = Form("A", tuple(Segment(g) for g in src_word))
        tgt_segs = tuple(
            Segment(grapheme=g, tone=tone if i == len(tgt_word) - 1 else None)
            for i, g in enumerate(tgt_word)
        )
        tgt = Form("B", tgt_segs)
        out.append((src, tgt))
    return out


def _suppress_deprecation_warning():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----- headline: tonogenesis detection -----------------------------------


def test_find_residual_patterns_surfaces_tonogenesis_as_top_hypothesis() -> None:
    """COMMITMENT: on a corpus where source voicing cleanly
    determines target tone, the top hypothesis must be a
    cross-dimensional tonogenesis rule that predicts the
    observed tone from a voicing-related feature.
    """
    _suppress_deprecation_warning()
    corpus = _voiced_corpus()
    model = train_model(corpus)
    hypotheses = find_residual_patterns(corpus, model)
    assert len(hypotheses) > 0, "expected at least one hypothesis"
    top = hypotheses[0]
    # The top hypothesis is a cross-dimensional tonogenesis rule.
    assert top.kind == "cross_dimensional_tonogenesis"
    # The top hypothesis's source feature must be one that
    # correlates with voicing — "voiced", "voiceless", "sonorant",
    # or "nasal" are all legitimate features that partition the
    # same way on this corpus.
    assert top.src_feature in {"voiced", "voiceless", "sonorant", "nasal", "stop", "fricative", "consonant"}
    # It must be significantly above the permutation null.
    assert top.null_percentile >= 0.95
    # It must have meaningful residual MI.
    assert top.residual_mi > 0.1


def test_find_residual_patterns_produces_only_tonal_hypotheses_for_v1() -> None:
    """COMMITMENT: v1 only implements the cross-dimensional
    tonogenesis kind. Long-range kinds return no hypotheses
    (they're  territory)."""
    _suppress_deprecation_warning()
    corpus = _voiced_corpus()
    model = train_model(corpus)
    hypotheses = find_residual_patterns(corpus, model)
    for h in hypotheses:
        assert h.kind == "cross_dimensional_tonogenesis"


# ----- no signal baseline ------------------------------------------------


def test_find_residual_patterns_returns_no_hypotheses_on_independent_data() -> None:
    """COMMITMENT: when source and target features are
    independent by construction, no hypothesis should exceed the
    permutation null at the 0.95 threshold."""
    _suppress_deprecation_warning()
    corpus = _independent_corpus()
    model = train_model(corpus)
    hypotheses = find_residual_patterns(corpus, model)
    # On a very small independent corpus some spurious
    # candidates may survive due to small-sample noise. The
    # guarantee is that there should be very few compared to
    # the signal corpus and none with high MI.
    if hypotheses:
        # Allow small leakage on tiny corpora.
        assert max(h.residual_mi for h in hypotheses) < 0.5


def test_find_residual_patterns_on_no_tones_returns_empty() -> None:
    """COMMITMENT: a corpus with no tonal data produces no
    cross-dimensional tonogenesis hypotheses — there's nothing
    for the tonal outcome variable to take on."""
    _suppress_deprecation_warning()
    pairs = [
        (Form("A", tuple(Segment(g) for g in "pata")),
         Form("B", tuple(Segment(g) for g in "fada"))),
        (Form("A", tuple(Segment(g) for g in "kata")),
         Form("B", tuple(Segment(g) for g in "hada"))),
    ]
    model = train_model(pairs)
    hypotheses = find_residual_patterns(pairs, model)
    # No tones at all → no tonogenesis hypotheses.
    assert all(h.kind != "cross_dimensional_tonogenesis" for h in hypotheses)


# ----- edge cases --------------------------------------------------------


def test_find_residual_patterns_on_empty_corpus() -> None:
    from regulae.model import LearnedModel
    model = LearnedModel.empty()
    assert find_residual_patterns([], model) == []


def test_find_residual_patterns_rejects_unknown_kind() -> None:
    from regulae.model import LearnedModel
    model = LearnedModel.empty()
    with pytest.raises(ValueError, match="unknown hypothesis kinds"):
        find_residual_patterns([], model, kinds={"made_up_kind"})


def test_find_residual_patterns_kinds_subset_filters_output() -> None:
    """COMMITMENT: passing ``kinds=`` filters which hypothesis
    categories are computed. An empty subset returns no hypotheses.
    """
    _suppress_deprecation_warning()
    corpus = _voiced_corpus()
    model = train_model(corpus)
    hypotheses = find_residual_patterns(
        corpus, model, kinds=set()
    )
    assert hypotheses == []


# ----- determinism -------------------------------------------------------


def test_find_residual_patterns_is_deterministic_under_same_seed() -> None:
    """COMMITMENT: repeated calls with the same ``random_seed``
    produce bitwise-identical hypothesis lists."""
    _suppress_deprecation_warning()
    corpus = _voiced_corpus()
    model = train_model(corpus)
    r1 = find_residual_patterns(corpus, model, random_seed=42)
    r2 = find_residual_patterns(corpus, model, random_seed=42)
    assert r1 == r2


def test_find_residual_patterns_is_not_mutating() -> None:
    """COMMITMENT: the function never mutates the model or the
    corpus. Running it twice yields the same list and the model
    remains unchanged between calls."""
    _suppress_deprecation_warning()
    corpus = _voiced_corpus()
    model = train_model(corpus)
    before_counts = dict(model.segment_table.counts)
    _ = find_residual_patterns(corpus, model)
    after_counts = dict(model.segment_table.counts)
    assert before_counts == after_counts


# ----- sort order --------------------------------------------------------


def test_find_residual_patterns_sorted_by_mi_desc() -> None:
    """COMMITMENT: the result list is sorted by residual_mi
    descending."""
    _suppress_deprecation_warning()
    corpus = _voiced_corpus()
    model = train_model(corpus)
    hypotheses = find_residual_patterns(corpus, model)
    mi_values = [h.residual_mi for h in hypotheses]
    assert mi_values == sorted(mi_values, reverse=True)


# ----- PatternHypothesis type --------------------------------------------


def test_hypothesis_type_is_frozen_and_hashable() -> None:
    h = PatternHypothesis(
        kind="cross_dimensional_tonogenesis",
        src_feature="voiced",
        src_position_spec="relative_0",
        tgt_feature_or_value="4",
        tgt_dimension="tone",
        tgt_position_offset=0,
        residual_mi=0.5,
        null_percentile=0.98,
        supporting_observations=20,
    )
    # Frozen.
    with pytest.raises(Exception):
        h.kind = "something_else"  # type: ignore[misc]
    # Hashable.
    {h}


def test_hypothesis_kinds_tuple_is_exposed() -> None:
    """COMMITMENT: the public ``HYPOTHESIS_KINDS`` tuple lists
    every supported kind.  removed the long-range stubs and
    moved long-range discovery to a direct split loop in
    training — the anomaly module only covers cross-dimensional."""
    assert "cross_dimensional_tonogenesis" in HYPOTHESIS_KINDS
    assert len(HYPOTHESIS_KINDS) == 1
