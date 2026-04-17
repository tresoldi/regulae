"""Tests for the BICConfig dataclass and its effect on training."""

import pytest

from regulae import (
    BICConfig,
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


# ----- validation --------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_split_observations", 0),
        ("max_split_depth", 0),
        ("min_chunk_observations", 0),
        ("long_range_min_split_observations", 0),
        ("long_range_min_dominant_fraction", -0.1),
        ("long_range_min_dominant_fraction", 1.1),
        ("cross_dim_max_iterations", 0),
        ("cross_dim_min_rule_count", 0),
        ("cross_dim_min_rule_confidence", -0.1),
        ("cross_dim_min_rule_confidence", 1.1),
        ("multi_lect_min_commit_scale", -0.1),
    ],
)
def test_invalid_config_values_raise(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        BICConfig(**{field: value})


def test_default_config_has_historical_values() -> None:
    """COMMITMENT: BICConfig() defaults exactly match the pre-config
    historical constants."""
    c = BICConfig()
    assert c.delta_bic_threshold == -1.0
    assert c.min_split_observations == 2
    assert c.max_split_depth == 3
    assert c.min_chunk_observations == 2
    assert c.long_range_delta_bic_threshold == -5.0
    assert c.long_range_min_split_observations == 5
    assert c.long_range_min_dominant_fraction == 0.6
    assert c.cross_dim_max_iterations == 5
    assert c.cross_dim_min_rule_count == 3
    assert c.cross_dim_min_rule_confidence == 0.5
    assert c.multi_lect_bic_small_sample_correction is True
    assert c.multi_lect_min_commit_scale == 0.5


# ----- non-regression on default ------------------------------------------


def _mixed_corpus() -> list[CognateSet]:
    pairs = [
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "kta"), _form("B", "tʃa")),
        (_form("A", "kta"), _form("B", "tʃa")),
        (_form("A", "kta"), _form("B", "tʃa")),
        (_form("A", "kta"), _form("B", "tʃa")),
    ]
    return cognate_sets_from_pairs(pairs, ("A", "B"))


def test_default_config_matches_no_config() -> None:
    """COMMITMENT: passing no BICConfig and passing BICConfig() must
    produce bitwise-identical models."""
    corpus = _mixed_corpus()
    m_none = train_model(corpus)
    m_default = train_model(corpus, bic_config=BICConfig())
    assert isinstance(m_none, MultiLectModel)
    assert isinstance(m_default, MultiLectModel)
    pn = m_none.pairwise_models[frozenset({"A", "B"})]
    pd = m_default.pairwise_models[frozenset({"A", "B"})]
    assert pn.segment_table.counts == pd.segment_table.counts
    assert pn.chunk_table.entries == pd.chunk_table.entries


# ----- tightening actually drops commits ----------------------------------


def test_tight_min_chunk_observations_rejects_low_count_chunks() -> None:
    corpus = _mixed_corpus()
    loose = train_model(corpus, bic_config=BICConfig(min_chunk_observations=2))
    tight = train_model(corpus, bic_config=BICConfig(min_chunk_observations=100))
    assert isinstance(loose, MultiLectModel)
    assert isinstance(tight, MultiLectModel)
    loose_chunks = loose.pairwise_models[frozenset({"A", "B"})].chunk_table.entries
    tight_chunks = tight.pairwise_models[frozenset({"A", "B"})].chunk_table.entries
    assert len(tight_chunks) == 0
    assert len(loose_chunks) > 0


def test_tight_min_rule_count_rejects_cross_dim_rules() -> None:
    """Cross-dimensional rule with 40 observations on tonogenesis
    fixture. Raising the floor to 100 should reject it."""
    # Mini tonogenesis corpus: voiced onset → tone 4, voiceless → tone 1.
    def F(L: str, w: str, tones: str) -> Form:
        segs = []
        for i, c in enumerate(w):
            t = tones[i] if i < len(tones) and tones[i] != " " else None
            segs.append(Segment(c, tone=t))
        return Form(L, tuple(segs))

    src_forms = []
    tgt_forms = []
    voiced = ["b", "d", "g"]
    voiceless = ["p", "t", "k"]
    for v in voiced:
        for i in range(10):
            src_forms.append(F("A", f"{v}a", "  "))
            tgt_forms.append(F("B", f"{v}a", " 4"))
    for vl in voiceless:
        for i in range(10):
            src_forms.append(F("A", f"{vl}a", "  "))
            tgt_forms.append(F("B", f"{vl}a", " 1"))
    corpus = cognate_sets_from_pairs(
        list(zip(src_forms, tgt_forms)), ("A", "B")
    )
    loose = train_model(corpus, bic_config=BICConfig(cross_dim_min_rule_count=3))
    tight = train_model(corpus, bic_config=BICConfig(cross_dim_min_rule_count=100))
    assert isinstance(loose, MultiLectModel)
    assert isinstance(tight, MultiLectModel)
    loose_rules = loose.pairwise_models[frozenset({"A", "B"})].cross_dimensional_table.entries
    tight_rules = tight.pairwise_models[frozenset({"A", "B"})].cross_dimensional_table.entries
    assert len(loose_rules) > 0
    assert len(tight_rules) == 0


# ----- backward-compat kwargs ---------------------------------------------


def test_legacy_multi_lect_kwargs_override_bic_config_field() -> None:
    """The existing ``multi_lect_bic_correction`` and
    ``multi_lect_min_commit_scale`` kwargs should override the
    corresponding BICConfig fields when explicitly passed."""
    corpus = _mixed_corpus()
    # Pass a config with correction=False, but override via kwarg=True.
    m_override = train_model(
        corpus,
        bic_config=BICConfig(multi_lect_bic_small_sample_correction=False),
        multi_lect_bic_correction=True,
    )
    # Matches: config with correction=True directly.
    m_ref = train_model(
        corpus,
        bic_config=BICConfig(multi_lect_bic_small_sample_correction=True),
    )
    assert isinstance(m_override, MultiLectModel)
    assert isinstance(m_ref, MultiLectModel)
    assert m_override.unconditioned_classes == m_ref.unconditioned_classes
    assert m_override.conditioned_classes == m_ref.conditioned_classes


def test_legacy_kwargs_default_is_noop() -> None:
    """Passing no legacy kwargs (both None) leaves BICConfig unchanged."""
    corpus = _mixed_corpus()
    m_default = train_model(corpus)
    m_explicit = train_model(corpus, bic_config=BICConfig())
    assert isinstance(m_default, MultiLectModel)
    assert isinstance(m_explicit, MultiLectModel)
    assert m_default.unconditioned_classes == m_explicit.unconditioned_classes


# ----- determinism --------------------------------------------------------


def test_training_with_custom_bic_config_is_deterministic() -> None:
    corpus = _mixed_corpus()
    cfg = BICConfig(min_chunk_observations=4, delta_bic_threshold=-2.0)
    m1 = train_model(corpus, bic_config=cfg)
    m2 = train_model(corpus, bic_config=cfg)
    assert isinstance(m1, MultiLectModel) and isinstance(m2, MultiLectModel)
    p1 = m1.pairwise_models[frozenset({"A", "B"})]
    p2 = m2.pairwise_models[frozenset({"A", "B"})]
    assert p1.segment_table.counts == p2.segment_table.counts
    assert p1.chunk_table.entries == p2.chunk_table.entries
