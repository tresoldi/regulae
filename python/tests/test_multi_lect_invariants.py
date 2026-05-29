"""Invariant tests for the  multi-lect training flow.

These property-style tests exercise the whole pipeline (reconciliation
+ multi-lect context discovery) on small corpora and verify that high-level invariants
hold:

- **Determinism**: shuffling the input corpus does not change the
  output class table.
- **Lect removal stability**: removing one lect from every cognate
  set preserves the other lects' pairwise models and produces a
  strict subset of classes.
- **Sparse robustness**: very small / degenerate inputs do not
  crash multi-lect context discovery.
"""

import random

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def _make_n3_corpus() -> list[CognateSet]:
    pairs = [
        ("pata", "fada", "pada"),
        ("kata", "hada", "kada"),
        ("pona", "fona", "pona"),
        ("kono", "hono", "kono"),
        ("pila", "fila", "pila"),
        ("kina", "hina", "kina"),
        ("puto", "futo", "puto"),
        ("kuta", "huta", "kuta"),
    ]
    return [
        CognateSet(
            cognate_id=f"c{i}",
            forms={
                "A": _form("A", a),
                "B": _form("B", b),
                "C": _form("C", c),
            },
        )
        for i, (a, b, c) in enumerate(pairs)
    ]


def _canonical_class_keys(
    model: MultiLectModel,
) -> list[tuple[tuple[str, str], ...]]:
    """Canonical representation of a model's unconditioned classes:
    sorted list of sorted (lect, grapheme) tuples."""
    return sorted(
        tuple(sorted(k.segments.items())) for k in model.unconditioned_classes
    )


def _canonical_counts(
    model: MultiLectModel,
) -> dict[tuple[tuple[str, str], ...], float]:
    return {
        tuple(sorted(k.segments.items())): k.count
        for k in model.unconditioned_classes
    }


# ----- determinism under reordering --------------------------------------


def test_determinism_reorder_preserves_classes() -> None:
    """COMMITMENT: shuffling the input corpus must not change the
    unconditioned class table."""
    corpus = _make_n3_corpus()
    model_a = train_model(corpus)
    rng = random.Random(42)
    shuffled = list(corpus)
    rng.shuffle(shuffled)
    model_b = train_model(shuffled)
    assert isinstance(model_a, MultiLectModel)
    assert isinstance(model_b, MultiLectModel)
    assert _canonical_counts(model_a) == _canonical_counts(model_b)


def test_determinism_reorder_preserves_conditioned_classes() -> None:
    """COMMITMENT: multi-lect context discovery output is deterministic under input order.
    We check the set of (segments, count) pairs, since class_id is
    assigned after sort and context comparison via repr is noisy."""
    corpus = _make_n3_corpus()
    model_a = train_model(corpus)
    shuffled = list(corpus)
    random.Random(7).shuffle(shuffled)
    model_b = train_model(shuffled)
    assert isinstance(model_a, MultiLectModel)
    assert isinstance(model_b, MultiLectModel)

    def signature(m: MultiLectModel) -> set[tuple]:
        return {
            (tuple(sorted(k.segments.items())), k.count)
            for k in m.conditioned_classes
        }

    assert signature(model_a) == signature(model_b)


def test_determinism_two_runs_same_inputs() -> None:
    """COMMITMENT: training twice on the same input gives the same model."""
    corpus = _make_n3_corpus()
    m1 = train_model(corpus)
    m2 = train_model(corpus)
    assert isinstance(m1, MultiLectModel)
    assert isinstance(m2, MultiLectModel)
    assert _canonical_counts(m1) == _canonical_counts(m2)
    assert set(m1.pairwise_models) == set(m2.pairwise_models)


# ----- lect removal stability --------------------------------------------


def test_removing_lect_preserves_other_pair_coverage() -> None:
    """COMMITMENT: removing one lect from every cognate set produces
    a model whose pairwise_models is the subset of the original that
    didn't include the removed lect, and whose remaining classes
    refer only to the remaining lects."""
    corpus = _make_n3_corpus()
    full = train_model(corpus)
    assert isinstance(full, MultiLectModel)

    # Remove lect C from every set.
    pruned = [
        CognateSet(
            cognate_id=cs.cognate_id,
            forms={l: f for l, f in cs.forms.items() if l != "C"},
        )
        for cs in corpus
    ]
    reduced = train_model(pruned)
    assert isinstance(reduced, MultiLectModel)

    # Reduced pairwise models should be a subset and contain no C.
    for pair in reduced.pairwise_models:
        assert "C" not in pair
    assert frozenset({"A", "B"}) in reduced.pairwise_models

    # Reduced classes mention only A and B.
    for klass in reduced.unconditioned_classes:
        assert set(klass.segments).issubset({"A", "B"})


def test_removing_lect_does_not_empty_other_pair_models() -> None:
    """COMMITMENT: the surviving pair's learned model is still trained
    (has observed segment correspondences)."""
    corpus = _make_n3_corpus()
    pruned = [
        CognateSet(
            cognate_id=cs.cognate_id,
            forms={l: f for l, f in cs.forms.items() if l != "C"},
        )
        for cs in corpus
    ]
    reduced = train_model(pruned)
    assert isinstance(reduced, MultiLectModel)
    ab = reduced.pairwise_models[frozenset({"A", "B"})]
    assert len(ab.segment_table.counts) > 0


# ----- sparse / degenerate input robustness -------------------------------


def test_single_cognate_set_does_not_crash() -> None:
    corpus = [
        CognateSet(
            cognate_id="only",
            forms={"A": _form("A", "pata"), "B": _form("B", "fada")},
        )
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)


def test_sparse_missing_lects_does_not_crash_context_discovery() -> None:
    """COMMITMENT: multi-lect context discovery handles very sparse corpora without
    crashing, even if the resulting conditioned table is empty."""
    corpus = [
        CognateSet(
            cognate_id="c1",
            forms={"A": _form("A", "pa"), "B": _form("B", "fa")},
        ),
        CognateSet(
            cognate_id="c2",
            forms={"A": _form("A", "ka"), "C": _form("C", "xa")},
        ),
        CognateSet(
            cognate_id="c3",
            forms={"B": _form("B", "ma"), "C": _form("C", "ma")},
        ),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    # Each of the 3 pairs should have its own model.
    assert len(model.pairwise_models) == 3


def test_n1_degenerate_corpus_is_valid() -> None:
    """COMMITMENT: a corpus with only one lect produces an empty
    MultiLectModel (no pairs, no classes) without crashing."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", f"pa{i}")})
        for i in range(3)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert model.lect_ids == ("A",)
    assert model.pairwise_models == {}
    assert model.unconditioned_classes == ()
    assert model.conditioned_classes == ()


def test_empty_corpus_returns_legacy_empty_learned_model() -> None:
    """COMMITMENT: empty corpus preserves the legacy backward-compat
    contract — returns an empty LearnedModel, not a MultiLectModel."""
    from regulae import LearnedModel

    result = train_model([])
    assert isinstance(result, LearnedModel)
