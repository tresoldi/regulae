"""Tests for the typological prior hook."""

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    train_model,
)
from regulae.priors import combine, uniform


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def _mixed_corpus() -> list[CognateSet]:
    return [
        CognateSet(cognate_id=f"c{i}", forms={"A": _form("A", "pa"), "B": _form("B", "fa")})
        for i in range(3)
    ] + [
        CognateSet(cognate_id=f"d{i}", forms={"A": _form("A", "pe"), "B": _form("B", "be")})
        for i in range(3)
    ]


# ----- uniform prior is a no-op -------------------------------------------


def test_uniform_prior_matches_no_prior() -> None:
    """COMMITMENT: ``uniform()`` is bitwise-identical to passing no
    prior. The hook adds no implicit opinion."""
    corpus = _mixed_corpus()
    m_none = train_model(corpus)
    m_uni = train_model(corpus, typological_prior=uniform())
    assert isinstance(m_none, MultiLectModel)
    assert isinstance(m_uni, MultiLectModel)
    pn = m_none.pairwise_models[frozenset({"A", "B"})].segment_table
    pu = m_uni.pairwise_models[frozenset({"A", "B"})].segment_table
    assert pn.prior_pseudo_counts == pu.prior_pseudo_counts
    assert pn.counts == pu.counts


# ----- custom prior shifts the posterior ----------------------------------


def test_custom_prior_boosts_targeted_correspondence() -> None:
    """COMMITMENT: a prior that boosts ``p → b`` shifts the
    posterior probability of ``p → b`` upward relative to the
    no-prior baseline. Equivalent to the historical claim
    'this family favors lenition over spirantization'."""
    corpus = _mixed_corpus()

    def boost_p_to_b(s: str, t: str) -> float:
        return 3.0 if s == "p" and t == "b" else 0.0

    m_none = train_model(corpus)
    m_boost = train_model(corpus, typological_prior=boost_p_to_b)
    assert isinstance(m_none, MultiLectModel)
    assert isinstance(m_boost, MultiLectModel)

    post_none = m_none.pairwise_models[frozenset({"A", "B"})].posterior_for("p")
    post_boost = m_boost.pairwise_models[frozenset({"A", "B"})].posterior_for("p")
    assert post_boost.get("b", 0.0) > post_none.get("b", 0.0)
    assert post_boost.get("f", 0.0) < post_none.get("f", 0.0)


def test_custom_prior_suppresses_targeted_correspondence() -> None:
    """A negative prior on ``p → b`` should reduce its posterior."""
    corpus = _mixed_corpus()

    def suppress_p_to_b(s: str, t: str) -> float:
        return -3.0 if s == "p" and t == "b" else 0.0

    m_none = train_model(corpus)
    m_supp = train_model(corpus, typological_prior=suppress_p_to_b)
    assert isinstance(m_none, MultiLectModel)
    assert isinstance(m_supp, MultiLectModel)

    post_none = m_none.pairwise_models[frozenset({"A", "B"})].posterior_for("p")
    post_supp = m_supp.pairwise_models[frozenset({"A", "B"})].posterior_for("p")
    assert post_supp.get("b", 0.0) < post_none.get("b", 0.0)


# ----- composition --------------------------------------------------------


def test_combine_sums_priors() -> None:
    """``combine`` should sum the contributions of multiple priors.
    Combining a uniform prior with a custom one is equivalent to
    just the custom one."""
    corpus = _mixed_corpus()

    def boost(s: str, t: str) -> float:
        return 1.5 if s == "p" and t == "b" else 0.0

    m_solo = train_model(corpus, typological_prior=boost)
    m_combined = train_model(corpus, typological_prior=combine(uniform(), boost))
    assert isinstance(m_solo, MultiLectModel)
    assert isinstance(m_combined, MultiLectModel)
    pn = m_solo.pairwise_models[frozenset({"A", "B"})].segment_table
    pc = m_combined.pairwise_models[frozenset({"A", "B"})].segment_table
    assert pn.prior_pseudo_counts == pc.prior_pseudo_counts


def test_combine_two_custom_priors() -> None:
    """Combining two pair-specific priors should add their boosts."""
    corpus = _mixed_corpus()

    def boost_pb(s: str, t: str) -> float:
        return 1.5 if s == "p" and t == "b" else 0.0

    def boost_pf(s: str, t: str) -> float:
        return 1.5 if s == "p" and t == "f" else 0.0

    m_combined = train_model(
        corpus,
        typological_prior=combine(boost_pb, boost_pf),
    )
    assert isinstance(m_combined, MultiLectModel)
    pcs = m_combined.pairwise_models[frozenset({"A", "B"})].segment_table.prior_pseudo_counts
    p_to_b = next((c for cc, c in pcs.items() if cc.src == "p" and cc.tgt == "b"), 0.0)
    p_to_f = next((c for cc, c in pcs.items() if cc.src == "p" and cc.tgt == "f"), 0.0)
    p_to_e = next((c for cc, c in pcs.items() if cc.src == "p" and cc.tgt == "e"), 0.0)
    # Both p→b and p→f get boosted; should both exceed p→e.
    assert p_to_b > p_to_e
    assert p_to_f > p_to_e


# ----- regression: training is deterministic ------------------------------


def test_training_with_prior_is_deterministic() -> None:
    corpus = _mixed_corpus()

    def boost(s: str, t: str) -> float:
        return 2.0 if s == "p" and t == "b" else 0.0

    m1 = train_model(corpus, typological_prior=boost)
    m2 = train_model(corpus, typological_prior=boost)
    assert isinstance(m1, MultiLectModel) and isinstance(m2, MultiLectModel)
    pa1 = m1.pairwise_models[frozenset({"A", "B"})].segment_table
    pa2 = m2.pairwise_models[frozenset({"A", "B"})].segment_table
    assert pa1.prior_pseudo_counts == pa2.prior_pseudo_counts
    assert pa1.counts == pa2.counts
