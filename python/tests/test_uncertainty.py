"""Tests for the uncertainty module and its integration across training."""

import pytest

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    UncertaintyEstimate,
    percentile_interval,
    train_model,
    wilson_interval,
)


# ----- Wilson math --------------------------------------------------------


def test_wilson_zero_n_returns_full_interval() -> None:
    u = wilson_interval(0, 0)
    assert u.lo == 0.0 and u.hi == 1.0
    assert u.method == "wilson"


def test_wilson_full_certainty_narrow_high() -> None:
    u = wilson_interval(100, 100)
    assert u.lo > 0.95
    assert u.hi == 1.0


def test_wilson_zero_successes_narrow_low() -> None:
    u = wilson_interval(0, 100)
    assert u.lo < 1e-10  # floating-point near-zero, not exact 0
    assert u.hi < 0.05


def test_wilson_half_half_centered() -> None:
    u = wilson_interval(50, 100)
    # Wilson interval is symmetric around 0.5 for p=0.5 regardless of n.
    assert abs((u.lo + u.hi) / 2 - 0.5) < 0.01
    # Width should be modest at n=100.
    assert u.hi - u.lo < 0.25


def test_wilson_widens_as_n_shrinks() -> None:
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)


def test_wilson_fractional_counts_ok() -> None:
    # Confidence-weighted training produces fractional counts.
    u = wilson_interval(2.5, 5.0)
    # At n=5, half-half is wide: roughly [0.17, 0.83].
    assert 0.0 < u.lo < 0.3
    assert 0.7 < u.hi < 1.0


def test_wilson_rejects_unsupported_alpha() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 10, alpha=0.07)


# ----- percentile_interval ------------------------------------------------


def test_percentile_interval_empty() -> None:
    u = percentile_interval([])
    assert u.lo == 0.0 and u.hi == 1.0


def test_percentile_interval_bracket_middle_samples() -> None:
    samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    u = percentile_interval(samples)
    # 95% CI should bracket roughly [0.1, 0.9].
    assert u.lo <= 0.2
    assert u.hi >= 0.8


# ----- integration: training populates uncertainty ------------------------


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def test_training_populates_segment_uncertainty() -> None:
    """COMMITMENT: every segment_table.counts entry has an uncertainty."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    for key in pair.segment_table.counts:
        assert key in pair.segment_table.uncertainty
        u = pair.segment_table.uncertainty[key]
        assert u.method == "wilson"
        assert 0.0 <= u.lo <= u.hi <= 1.0


def test_training_populates_displacement_uncertainty() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    for key in pair.displacement_dist.counts:
        assert key in pair.displacement_dist.uncertainty


def test_training_populates_multi_lect_class_uncertainty() -> None:
    """COMMITMENT: every multi-lect class carries a Wilson interval."""
    corpus = [
        CognateSet(
            cognate_id=f"c{i}",
            forms={"A": _form("A", "pa"), "B": _form("B", "fa"), "C": _form("C", "fa")},
        )
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    for klass in model.unconditioned_classes:
        assert klass.uncertainty is not None
        assert klass.uncertainty.method == "wilson"


def test_segment_interval_narrower_when_src_is_well_attested() -> None:
    """A source grapheme with many observations gets narrower intervals
    than one with few. ``n`` on the interval reflects ``src_totals``."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pap"), "B": _form("B", "faf")})
        for i in range(20)
    ] + [
        # 't' appears only in a couple of pairs.
        CognateSet(cognate_id=f"d{i}", forms={"A": _form("A", "ta"), "B": _form("B", "θa")})
        for i in range(3)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    p_u = next(
        u for k, u in pair.segment_table.uncertainty.items()
        if k.src == "p" and k.tgt == "f" and k.context.constraint_count() == 0
    )
    t_u = next(
        u for k, u in pair.segment_table.uncertainty.items()
        if k.src == "t" and k.tgt == "θ" and k.context.constraint_count() == 0
    )
    # p has src_totals much higher than t, so its interval is narrower
    # despite both representing a dominant correspondence.
    assert p_u.n > t_u.n
    assert (p_u.hi - p_u.lo) < (t_u.hi - t_u.lo)


# ----- bootstrap ----------------------------------------------------------


def test_bootstrap_produces_bootstrap_method_intervals() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(10)
    ]
    model = train_model(corpus, bootstrap_n=10, bootstrap_seed=1)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    # At least one entry should have been replaced to bootstrap method.
    methods = {u.method for u in pair.segment_table.uncertainty.values()}
    assert methods == {"bootstrap"}
    for klass in model.unconditioned_classes:
        assert klass.uncertainty is not None
        assert klass.uncertainty.method == "bootstrap"


def test_bootstrap_is_deterministic() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(8)
    ] + [
        CognateSet(cognate_id=f"d{i}", forms={"A": _form("A", "pa"), "B": _form("B", "ba")})
        for i in range(4)
    ]
    m1 = train_model(corpus, bootstrap_n=20, bootstrap_seed=7)
    m2 = train_model(corpus, bootstrap_n=20, bootstrap_seed=7)
    assert isinstance(m1, MultiLectModel)
    assert isinstance(m2, MultiLectModel)
    p1 = m1.pairwise_models[frozenset({"A", "B"})]
    p2 = m2.pairwise_models[frozenset({"A", "B"})]
    for k, u1 in p1.segment_table.uncertainty.items():
        u2 = p2.segment_table.uncertainty[k]
        assert u1.lo == u2.lo and u1.hi == u2.hi


def test_bootstrap_brackets_base_rate() -> None:
    """COMMITMENT: bootstrap intervals bracket the observed base rate
    (under well-mixed noise)."""
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(8)
    ] + [
        CognateSet(cognate_id=f"d{i}", forms={"A": _form("A", "pa"), "B": _form("B", "ba")})
        for i in range(3)
    ]
    model = train_model(corpus, bootstrap_n=100, bootstrap_seed=42)
    assert isinstance(model, MultiLectModel)
    pair = model.pairwise_models[frozenset({"A", "B"})]
    # base rate of p->f is 8/11 = 0.727
    pf_key = next(
        k for k in pair.segment_table.uncertainty
        if k.src == "p" and k.tgt == "f" and k.context.constraint_count() == 0
    )
    u = pair.segment_table.uncertainty[pf_key]
    base_rate = pair.segment_table.counts[pf_key] / pair.segment_table.src_totals["p"]
    assert u.lo <= base_rate <= u.hi


# ----- invariance: no regression on equality / determinism ---------------


def test_training_without_bootstrap_is_deterministic() -> None:
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    m1 = train_model(corpus)
    m2 = train_model(corpus)
    # Equality is unaffected by uncertainty field (compare=False).
    assert m1.pairwise_models == m2.pairwise_models
    assert m1.unconditioned_classes == m2.unconditioned_classes


def test_multi_lect_class_equality_ignores_uncertainty() -> None:
    from dataclasses import replace
    corpus = [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(5)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    klass = model.unconditioned_classes[0]
    other = replace(klass, uncertainty=None)
    assert klass == other
