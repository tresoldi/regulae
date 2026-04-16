"""Tests for  multi-lect context discovery: multi-lect context discovery (class-level context splits).

These use small handcrafted corpora where the expected conditioning
environment is obvious: lect A has a grapheme that takes two
different reflexes in another lect depending on its phonological
neighborhood. The test verifies that at least one conditioned class
is committed and that it captures the expected pivot lect/grapheme.
"""

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def _make_palatalization_corpus() -> list[CognateSet]:
    """A and B share most correspondences, but A's ``k`` maps to
    B's ``s`` before ``i``/``e`` (front vowels) and to ``k``
    elsewhere. Plenty of repetition to clear BIC thresholds.
    """
    pairs = [
        # k before i/e -> s (palatalization)
        ("kita", "sita"),
        ("kite", "site"),
        ("ketu", "setu"),
        ("keri", "seri"),
        ("kina", "sina"),
        ("keta", "seta"),
        ("kile", "sile"),
        ("kise", "sise"),
        # k before a/o/u -> k (preserved)
        ("kata", "kata"),
        ("koto", "koto"),
        ("kupa", "kupa"),
        ("kala", "kala"),
        ("koma", "koma"),
        ("kuma", "kuma"),
        ("kota", "kota"),
        ("kapa", "kapa"),
    ]
    return [
        CognateSet(
            cognate_id=f"c{i}",
            forms={"A": _form("A", src), "B": _form("B", tgt)},
        )
        for i, (src, tgt) in enumerate(pairs)
    ]


def test_context_discovery_runs_without_errors() -> None:
    """SMOKE: multi-lect context discovery runs on a simple corpus and returns
    a MultiLectModel with possibly non-empty conditioned_classes."""
    corpus = _make_palatalization_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    # At least the unconditioned stage populated something.
    assert len(model.unconditioned_classes) > 0


def test_context_discovery_discovers_palatalization_split() -> None:
    """COMMITMENT: on a corpus where A's ``k`` splits into B's ``s``
    before front vowels and B's ``k`` elsewhere, multi-lect context discovery should
    commit a conditioned class for (A:k, B:s) with a non-empty
    A-context that references a front/vowel feature. A palatalization
    class is the headline result of multi-lect context discovery.
    """
    corpus = _make_palatalization_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert len(model.conditioned_classes) > 0

    # There must be a conditioned class matching (A:k, B:s) with a
    # non-empty A-side following context.
    k_to_s = [
        k
        for k in model.conditioned_classes
        if k.segments.get("A") == "k"
        and k.segments.get("B") == "s"
        and k.contexts is not None
        and k.contexts.get("A") is not None
        and len(k.contexts["A"].following) > 0
    ]
    assert k_to_s, (
        "expected a conditioned (A:k, B:s) class with a non-empty "
        "following-context on A (the palatalization rule)"
    )
    # Its count should match the number of front-vowel palatalized pairs.
    assert max(k.count for k in k_to_s) >= 8


def test_context_discovery_preserves_unconditioned_classes() -> None:
    """COMMITMENT: multi-lect context discovery adds conditioned classes; it does NOT modify
    the unconditioned table (Option C: parallel tables)."""
    corpus = _make_palatalization_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert len(model.unconditioned_classes) > 0
    for klass in model.unconditioned_classes:
        assert klass.contexts is None


def test_context_discovery_conditioned_classes_carry_confidence() -> None:
    """COMMITMENT: every multi-lect context discovery conditioned class carries a
    confidence (coverage) value in [0, 1]. The palatalization
    pivot (A, k) has 16 observations and the committed front-vowel
    partition has 8 of them, so the (A:k, B:s) class should have
    confidence 0.5."""
    corpus = _make_palatalization_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert len(model.conditioned_classes) > 0
    for klass in model.conditioned_classes:
        assert 0.0 <= klass.confidence <= 1.0
    k_to_s = [
        k
        for k in model.conditioned_classes
        if k.segments.get("A") == "k" and k.segments.get("B") == "s"
    ]
    assert k_to_s
    # (A,k) had 16 observations; 8 fell in the front-vowel partition.
    # Expected coverage: 8/16 = 0.5.
    assert any(abs(k.confidence - 0.5) < 1e-9 for k in k_to_s)


def test_context_discovery_unconditioned_classes_have_default_confidence() -> None:
    """COMMITMENT: unconditioned classes leave confidence at its
    default of 1.0 — confidence is a multi-lect context discovery diagnostic only."""
    corpus = _make_palatalization_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    for klass in model.unconditioned_classes:
        assert klass.confidence == 1.0


def test_context_discovery_conditioned_classes_have_unique_class_ids() -> None:
    corpus = _make_palatalization_corpus()
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    all_ids = [k.class_id for k in model.unconditioned_classes] + [
        k.class_id for k in model.conditioned_classes
    ]
    assert len(set(all_ids)) == len(all_ids)


def test_context_discovery_min_commit_scale_zero_disables_floor() -> None:
    """COMMITMENT: multi_lect_min_commit_scale=0 disables the adaptive
    minimum commit floor and restores the looser behavior (structural
    minimum of 2)."""
    corpus = _make_palatalization_corpus()
    strict = train_model(corpus)  # defaults: scale=0.5
    loose = train_model(corpus, multi_lect_min_commit_scale=0.0)
    assert isinstance(strict, MultiLectModel)
    assert isinstance(loose, MultiLectModel)
    # Loose should have >= as many conditioned classes as strict.
    assert len(loose.conditioned_classes) >= len(strict.conditioned_classes)


def test_context_discovery_min_commit_scale_high_kills_small_commits() -> None:
    """COMMITMENT: increasing multi_lect_min_commit_scale above default
    shrinks or eliminates conditioned classes on a small corpus."""
    corpus = _make_palatalization_corpus()
    default = train_model(corpus)
    tight = train_model(corpus, multi_lect_min_commit_scale=3.0)
    assert isinstance(default, MultiLectModel)
    assert isinstance(tight, MultiLectModel)
    assert len(tight.conditioned_classes) <= len(default.conditioned_classes)


def test_context_discovery_bic_correction_can_be_disabled() -> None:
    """COMMITMENT: multi_lect_bic_correction=False reverts multi-lect context discovery to
    the original (looser) BIC penalty and may commit more classes."""
    corpus = _make_palatalization_corpus()
    corrected = train_model(corpus)  # default: True
    uncorrected = train_model(corpus, multi_lect_bic_correction=False)
    assert isinstance(corrected, MultiLectModel)
    assert isinstance(uncorrected, MultiLectModel)
    # Uncorrected should produce >= as many commits.
    assert len(uncorrected.conditioned_classes) >= len(corrected.conditioned_classes)


def test_context_discovery_empty_on_corpus_with_no_splittable_sources() -> None:
    """COMMITMENT: when no pivot has multiple competing sister tuples,
    multi-lect context discovery commits nothing (no split to find)."""
    corpus = [
        CognateSet(
            cognate_id=f"c{i}",
            forms={"A": _form("A", "pata"), "B": _form("B", "pata")},
        )
        for i in range(10)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    assert model.conditioned_classes == ()
