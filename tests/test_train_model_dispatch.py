"""Tests for the  step-4 scaffolding of ``train_model``.

These verify that ``train_model`` dispatches between the legacy
pair-based path and the new CognateSet-based path, and that the
multi-lect path produces a MultiLectModel containing the expected
per-pair learned models. reconciliation/3 (reconciliation + context discovery lifted)
are NOT exercised here — they land in later  steps.
"""

from regulae import (
    CognateSet,
    Form,
    LearnedModel,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


# ----- dispatch ----------------------------------------------------------


def test_train_model_on_empty_corpus_returns_empty_learned_model() -> None:
    """COMMITMENT: empty input returns the legacy empty LearnedModel
    (backward compat)."""
    result = train_model([])
    assert isinstance(result, LearnedModel)


def test_train_model_on_pair_list_returns_learned_model() -> None:
    """COMMITMENT: legacy pair input still returns a LearnedModel and
    emits a DeprecationWarning telling the caller to migrate."""
    import pytest

    pairs = [
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "kata"), _form("B", "hada")),
    ]
    with pytest.warns(DeprecationWarning, match="cognate_sets_from_pairs"):
        result = train_model(pairs)
    assert isinstance(result, LearnedModel)


def test_train_model_on_cognate_sets_returns_multilect_model() -> None:
    """COMMITMENT: CognateSet input returns a MultiLectModel."""
    pairs = [
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "kata"), _form("B", "hada")),
    ]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    result = train_model(corpus)
    assert isinstance(result, MultiLectModel)


# ----- N=2 path ----------------------------------------------------------


def test_multilect_n2_has_one_pairwise_model() -> None:
    """COMMITMENT: for N=2 there is exactly one pair and one learned
    model, keyed by a frozenset of the two lect IDs."""
    pairs = [(_form("A", "pata"), _form("B", "fada"))] * 3
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert set(model.pairwise_models) == {frozenset({"A", "B"})}
    assert isinstance(model.pairwise_models[frozenset({"A", "B"})], LearnedModel)


def test_multilect_n2_learned_model_is_equivalent_to_legacy() -> None:
    """COMMITMENT: the per-pair LearnedModel inside a MultiLectModel is
    the same as what the legacy pair-based train_model would produce."""
    import pytest

    pairs = [
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "pata"), _form("B", "fada")),
    ]
    with pytest.warns(DeprecationWarning):
        legacy = train_model(pairs)
    multi = train_model(cognate_sets_from_pairs(pairs, ("src", "tgt")))
    assert isinstance(legacy, LearnedModel)
    assert isinstance(multi, MultiLectModel)
    pair_inside = multi.pairwise_models[frozenset({"src", "tgt"})]
    # Segment table counts should match (both pipelines were fed the
    # same underlying form pairs).
    assert set(pair_inside.segment_table.counts) == set(legacy.segment_table.counts)


def test_multilect_n2_records_canonical_lect_ids() -> None:
    pairs = [(_form("A", "pa"), _form("B", "fa"))]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert model.lect_ids == ("A", "B")


def test_multilect_n2_conditioned_classes_empty_until_context_discovery() -> None:
    """COMMITMENT: multi-lect context discovery is not yet wired, so
    conditioned_classes stays empty. Unconditioned classes ARE
    populated once reconciliation lands."""
    pairs = [(_form("A", "pa"), _form("B", "fa"))]
    model = train_model(cognate_sets_from_pairs(pairs, ("A", "B")))
    assert isinstance(model, MultiLectModel)
    assert model.conditioned_classes == ()


# ----- N=3 scaffolding ---------------------------------------------------


def test_multilect_n3_produces_three_pairwise_models() -> None:
    """COMMITMENT: for N=3 the scaffolding trains all 3 pair models
    (one per 2-combination of lects)."""
    a1, b1, c1 = _form("A", "pata"), _form("B", "fada"), _form("C", "pada")
    a2, b2, c2 = _form("A", "kata"), _form("B", "hada"), _form("C", "kada")
    corpus = [
        CognateSet(cognate_id="c1", forms={"A": a1, "B": b1, "C": c1}),
        CognateSet(cognate_id="c2", forms={"A": a2, "B": b2, "C": c2}),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pairs = set(model.pairwise_models)
    assert pairs == {
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"B", "C"}),
    }
    assert model.lect_ids == ("A", "B", "C")


def test_multilect_missing_pair_is_not_trained() -> None:
    """COMMITMENT: if no cognate set contains both lects in a pair,
    that pair is absent from pairwise_models."""
    corpus = [
        CognateSet(cognate_id="c1", forms={"A": _form("A", "pa"), "B": _form("B", "fa")}),
        CognateSet(cognate_id="c2", forms={"A": _form("A", "ka"), "C": _form("C", "xa")}),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert frozenset({"A", "B"}) in model.pairwise_models
    assert frozenset({"A", "C"}) in model.pairwise_models
    assert frozenset({"B", "C"}) not in model.pairwise_models


# ----- N=1 degenerate ----------------------------------------------------


def test_multilect_n1_returns_empty_pairwise_models() -> None:
    corpus = [
        CognateSet(cognate_id="c1", forms={"A": _form("A", "pata")}),
        CognateSet(cognate_id="c2", forms={"A": _form("A", "kata")}),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert model.pairwise_models == {}
    assert model.lect_ids == ("A",)
