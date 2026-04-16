"""Tests for  diagnostic helpers.

Covers :func:`find_cognate_outliers`. The framework does not
automatically detect non-cognates ( Q5 answer: explicit cognate
IDs are trusted input), but this helper is a post-hoc ranking tool
that lets the user spot pairs whose alignment cost is anomalous.
"""

import pytest

from regulae import (
    CognateOutlierReport,
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    find_cognate_outliers,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def _regular_corpus() -> list[CognateSet]:
    """A tight corpus where p↔f and t↔d and a↔a are the rule."""
    pairs = [
        ("pata", "fada"),
        ("pita", "fida"),
        ("pota", "foda"),
        ("puta", "fuda"),
        ("peta", "feda"),
        ("pat", "fad"),
        ("pit", "fid"),
        ("pot", "fod"),
    ]
    return [
        CognateSet(
            cognate_id=f"r{i}",
            forms={"A": _form("A", a), "B": _form("B", b)},
        )
        for i, (a, b) in enumerate(pairs)
    ]


# ----- empty / trivial corpora ------------------------------------------


def test_find_outliers_empty_corpus_returns_empty_list() -> None:
    from regulae import LearnedModel
    # A throwaway model; input corpus is empty.
    empty = MultiLectModel.empty()
    assert find_cognate_outliers([], empty) == []


def test_find_outliers_single_cognate_has_zero_z_score() -> None:
    """COMMITMENT: with only one cognate set, there is no
    distribution to compare against and z_score is 0.0."""
    corpus = _regular_corpus()[:1]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    reports = find_cognate_outliers(corpus, model)
    assert len(reports) == 1
    assert reports[0].z_score == 0.0


def test_find_outliers_detects_planted_non_cognate() -> None:
    """COMMITMENT: on a corpus dominated by a regular p↔f
    correspondence, an explicitly non-cognate pair (xyz/qmn)
    should be ranked at the top of the outlier list with a
    positive z-score."""
    regular = _regular_corpus()
    planted = CognateSet(
        cognate_id="NON_COGNATE",
        forms={"A": _form("A", "xyz"), "B": _form("B", "qmn")},
    )
    corpus = regular + [planted]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    reports = find_cognate_outliers(corpus, model)
    # Every set got scored.
    assert len(reports) == len(corpus)
    # The planted non-cognate is the top outlier and has z > 0.
    assert reports[0].cognate_id == "NON_COGNATE"
    assert reports[0].z_score > 0
    # Regular pairs have z <= the non-cognate's.
    for r in reports[1:]:
        assert r.z_score <= reports[0].z_score


def test_find_outliers_returns_all_reports_as_dataclass_instances() -> None:
    corpus = _regular_corpus()
    model = train_model(corpus)
    reports = find_cognate_outliers(corpus, model)
    assert all(isinstance(r, CognateOutlierReport) for r in reports)
    for r in reports:
        assert r.n_pairs >= 1
        # cost_per_segment can be negative: the learned model subtracts
        # a log-Z offset so observed pairs score cheaper than the
        # merkmal baseline, and that offset can push the per-segment
        # cost below zero.
        assert isinstance(r.cost_per_segment, float)
        assert isinstance(r.z_score, float)


def test_find_outliers_top_k_limits_output() -> None:
    """COMMITMENT: ``top_k`` returns only the k worst outliers."""
    corpus = _regular_corpus()
    model = train_model(corpus)
    reports = find_cognate_outliers(corpus, model, top_k=3)
    assert len(reports) == 3


def test_find_outliers_is_a_diagnostic_not_a_filter() -> None:
    """COMMITMENT: the helper does NOT mutate the input corpus
    or the model. Running it twice yields the same reports."""
    corpus = _regular_corpus()
    model = train_model(corpus)
    r1 = find_cognate_outliers(corpus, model)
    r2 = find_cognate_outliers(corpus, model)
    assert r1 == r2


def test_find_outliers_ignores_sets_with_no_trainable_pair() -> None:
    """COMMITMENT: a cognate set whose lects aren't in any
    pairwise_model is silently skipped (not an error)."""
    corpus = _regular_corpus()
    model = train_model(corpus)
    # Orphan set uses lects the model knows nothing about.
    orphan = CognateSet(
        cognate_id="ORPHAN",
        forms={"X": _form("X", "xyz"), "Y": _form("Y", "zyx")},
    )
    reports = find_cognate_outliers([orphan], model)
    assert reports == []
