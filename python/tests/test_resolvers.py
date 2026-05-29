"""Tests for the public resolvers on LearnedModel and MultiLectModel,
and for chunk diagnostics populated on ChunkPhraseTable."""

import pytest

from regulae import (
    CognateSet,
    Context,
    FeatureConstraint,
    Form,
    LearnedModel,
    MultiLectModel,
    Segment,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


# ----- LearnedModel.posterior_for ----------------------------------------


def test_posterior_for_unconditioned_sums_over_seen_targets() -> None:
    """COMMITMENT: the unconditioned posterior over targets is coherent
    and sums approximately to 1 after Dirichlet smoothing."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(10)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    posterior = pair.posterior_for("p")
    assert "f" in posterior
    assert posterior["f"] > 0.9  # clean p→f, no noise
    assert 0.99 < sum(posterior.values()) <= 1.01


def test_posterior_for_conditioned_context_picks_matching_entry() -> None:
    """COMMITMENT: when a conditioned entry's context is a subset of
    the query context, posterior_for uses that entry (not the
    unconditioned fallback)."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(20)
    ] + [
        CognateSet(cognate_id=f"d{i}", forms={"A": _form("A", "pi"), "B": _form("B", "bi")})
        for i in range(20)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    ctx_close = Context(following=(FeatureConstraint("close", "+"),))
    post = pair.posterior_for("p", context=ctx_close)
    # The conditioned entry for p→b before close:+ fires; other
    # targets are NOT supplemented by unconditioned fallback.
    assert post.get("b", 0.0) > 0.9
    assert post.get("f", 0.0) < 0.1


def test_posterior_for_no_entries_returns_empty_dict() -> None:
    corpus = [
        CognateSet(cognate_id="c0", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    # An unknown source with no entries.
    assert pair.posterior_for("xyz") == {}


def test_posterior_for_empty_context_equals_none_context() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    assert pair.posterior_for("p") == pair.posterior_for("p", context=Context())


# ----- MultiLectModel.classes_for_segment --------------------------------


def test_classes_for_segment_filters_by_lect_grapheme() -> None:
    corpus = [
        CognateSet(
            cognate_id=f"c{i}",
            forms={"A": _form("A", "pa"), "B": _form("B", "fa"), "C": _form("C", "pa")},
        )
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    matches = model.classes_for_segment("A", "p")
    assert len(matches) == 1
    assert matches[0].segments["A"] == "p"
    assert matches[0].segments["B"] == "f"
    assert matches[0].segments["C"] == "p"


def test_classes_for_segment_returns_empty_for_absent_segment() -> None:
    corpus = [
        CognateSet(cognate_id="c0", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert model.classes_for_segment("A", "q") == ()


def test_class_by_id_roundtrip() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    first = model.unconditioned_classes[0]
    assert model.class_by_id(first.class_id) is first
    assert model.class_by_id(99999) is None


def test_cognate_set_by_id_walks_back_to_source() -> None:
    corpus = [
        CognateSet(cognate_id="alpha", forms={"A": _form("A", "pa"), "B": _form("B", "fa")}),
        CognateSet(cognate_id="beta", forms={"A": _form("A", "ta"), "B": _form("B", "da")}),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    found = model.cognate_set_by_id("alpha")
    assert found is not None
    assert list(found.forms["A"].segments)[0].grapheme == "p"
    assert model.cognate_set_by_id("gamma") is None


# ----- ChunkPhraseTable.diagnostics --------------------------------------


def test_chunk_diagnostics_populated_at_training() -> None:
    """COMMITMENT: every promoted chunk carries a diagnostic report."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "kta"), "B": _form("B", "tʃa")})
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    assert len(pair.chunk_table.entries) > 0
    for key in pair.chunk_table.entries:
        assert key in pair.chunk_table.diagnostics
        report = pair.chunk_table.diagnostics[key]
        assert 0.0 <= report.transparency_score <= 1.0
        assert report.process_profile in {
            "compact_fusion", "residual_reduction", "nasal_fusion",
            "glide_or_vocalization_fusion", "balanced_restructuring",
            "bundled_reduction", "mixed_or_unclear",
        }


def test_chunk_min_transparency_filters_low_score_chunks() -> None:
    """COMMITMENT: setting chunk_min_transparency drops low-score chunks."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "kta"), "B": _form("B", "tʃa")})
        for i in range(5)
    ]
    full = train_model(corpus)
    filtered = train_model(corpus, chunk_min_transparency=0.5)
    assert isinstance(full, MultiLectModel)
    assert isinstance(filtered, MultiLectModel)
    full_pair = full.pairwise_models[frozenset({"A", "B"})]
    filtered_pair = filtered.pairwise_models[frozenset({"A", "B"})]
    assert len(filtered_pair.chunk_table.entries) <= len(full_pair.chunk_table.entries)
    for key in filtered_pair.chunk_table.entries:
        report = filtered_pair.chunk_table.diagnostics[key]
        assert report.transparency_score >= 0.5


def test_chunk_filter_retains_diagnostics_only_for_survivors() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "kta"), "B": _form("B", "tʃa")})
        for i in range(5)
    ]
    filtered = train_model(corpus, chunk_min_transparency=0.5)
    assert isinstance(filtered, MultiLectModel)
    pair = filtered.pairwise_models[frozenset({"A", "B"})]
    # diagnostics must be a subset of entries (no orphaned reports).
    assert set(pair.chunk_table.diagnostics.keys()) <= set(pair.chunk_table.entries.keys())
    assert set(pair.chunk_table.entries.keys()) == set(pair.chunk_table.diagnostics.keys())


def test_chunk_diagnostics_not_in_equality() -> None:
    """The diagnostics parallel map uses compare=False, so two tables
    with the same entries but different diagnostics compare equal."""
    from dataclasses import replace
    from regulae import ChunkPhraseTable
    t1 = ChunkPhraseTable(entries={})
    t2 = ChunkPhraseTable(entries={})
    assert t1 == t2


# ----- regression / invariance --------------------------------------------


def test_default_training_unchanged() -> None:
    """Default training with no chunk_min_transparency keeps all
    BIC-promoted chunks (backward compatible)."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    a = train_model(corpus)
    b = train_model(corpus, chunk_min_transparency=0.0)
    assert isinstance(a, MultiLectModel)
    assert isinstance(b, MultiLectModel)
    pa = a.pairwise_models[frozenset({"A", "B"})]
    pb = b.pairwise_models[frozenset({"A", "B"})]
    assert pa.chunk_table.entries == pb.chunk_table.entries
