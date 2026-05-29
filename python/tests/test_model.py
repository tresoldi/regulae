"""Tests for the learned model data types."""

from regulae.model import (
    ChunkPhraseTable,
    ConditionedCorrespondence,
    DisplacementDistribution,
    LearnedModel,
    SegmentCorrespondenceTable,
    TonalCorrespondence,
    TonalCorrespondenceTable,
)
from regulae.types import Context, FeatureDisplacement, Segment


def _cc(src: str, tgt: str) -> ConditionedCorrespondence:
    return ConditionedCorrespondence(src=src, tgt=tgt, context=Context())


# ----- SegmentCorrespondenceTable ----------------------------------------


def test_segment_correspondence_table_defaults_to_empty() -> None:
    """COMMITMENT: a default-constructed table has no counts and no prior."""
    table = SegmentCorrespondenceTable()
    assert table.counts == {}
    assert table.prior_pseudo_counts == {}
    assert table.src_totals == {}


def test_segment_correspondence_table_holds_counts() -> None:
    table = SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 5.0, _cc("p", "p"): 1.0},
        prior_pseudo_counts={_cc("p", "f"): 0.5, _cc("p", "p"): 2.0},
        src_totals={"p": 6.0},
    )
    assert table.counts[_cc("p", "f")] == 5.0
    assert table.prior_pseudo_counts[_cc("p", "p")] == 2.0
    assert table.src_totals["p"] == 6.0


# ----- DisplacementDistribution ------------------------------------------


def test_displacement_distribution_defaults_to_empty() -> None:
    dist = DisplacementDistribution()
    assert dist.counts == {}
    assert dist.total == 0.0
    assert dist.prior_pseudo_count == 1.0


def test_displacement_distribution_holds_counts_keyed_by_vector() -> None:
    d1 = (FeatureDisplacement("continuant", "-", "+"),)
    d2 = (FeatureDisplacement("voice", "-", "+"),)
    dist = DisplacementDistribution(
        counts={d1: 8.0, d2: 3.0},
        total=11.0,
        prior_pseudo_count=1.0,
    )
    assert dist.counts[d1] == 8.0
    assert dist.total == 11.0


# ----- ChunkPhraseTable --------------------------------------------------


def test_chunk_phrase_table_defaults_to_empty() -> None:
    table = ChunkPhraseTable()
    assert table.entries == {}


def test_chunk_phrase_table_holds_entries() -> None:
    kt = (Segment("k"), Segment("t"))
    tt = (Segment("t"), Segment("t"))
    table = ChunkPhraseTable(entries={(kt, tt): 0.1})
    assert table.entries[(kt, tt)] == 0.1


# ----- LearnedModel ------------------------------------------------------


def test_learned_model_holds_all_three_layers() -> None:
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    assert isinstance(model.segment_table, SegmentCorrespondenceTable)
    assert isinstance(model.displacement_dist, DisplacementDistribution)
    assert isinstance(model.chunk_table, ChunkPhraseTable)


def test_learned_model_has_default_hyperparameters() -> None:
    """COMMITMENT: default hyperparameters are documented constants."""
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    assert model.feature_system == "descriptive"
    assert model.temperature == 1.0
    assert model.concentration == 5.0
    assert model.segment_weight == 0.7
    assert model.displacement_weight == 0.3


def test_learned_model_empty_factory() -> None:
    """COMMITMENT: empty() creates a model with no learned data."""
    model = LearnedModel.empty()
    assert model.segment_table.counts == {}
    assert model.segment_table.prior_pseudo_counts == {}
    assert model.displacement_dist.counts == {}
    assert model.chunk_table.entries == {}


def test_learned_model_empty_factory_preserves_hyperparameters() -> None:
    model = LearnedModel.empty(
        feature_system="distinctive", temperature=2.0, concentration=10.0
    )
    assert model.feature_system == "distinctive"
    assert model.temperature == 2.0
    assert model.concentration == 10.0


def test_learned_model_is_frozen() -> None:
    """COMMITMENT: the model is immutable."""
    import pytest

    model = LearnedModel.empty()
    with pytest.raises(Exception):
        model.temperature = 99.0  # type: ignore[misc]


# ----- ConditionedCorrespondence () -----------------------------------


def test_conditioned_correspondence_default_is_empty_context() -> None:
    """COMMITMENT: a conditioned correspondence without an explicit
    context uses the empty context."""
    cc = ConditionedCorrespondence(src="p", tgt="f")
    assert cc.context == Context()


def test_conditioned_correspondence_is_hashable_for_dict_keys() -> None:
    """COMMITMENT: ConditionedCorrespondence can be used as a dict key."""
    key = ConditionedCorrespondence(src="p", tgt="f")
    d = {key: 1.0}
    assert d[key] == 1.0


def test_conditioned_correspondence_distinct_by_context() -> None:
    """COMMITMENT: same (src, tgt) with different contexts are distinct keys."""
    from regulae.types import FeatureConstraint

    k1 = ConditionedCorrespondence(src="p", tgt="f", context=Context())
    k2 = ConditionedCorrespondence(
        src="p",
        tgt="f",
        context=Context(following=(FeatureConstraint("vowel", "+"),)),
    )
    assert k1 != k2
    assert hash(k1) != hash(k2) or k1 == k2  # hash may collide but != still holds


# ----- TonalCorrespondence and TonalCorrespondenceTable ------------------


def test_tonal_correspondence_construction() -> None:
    t = TonalCorrespondence(src_tone="high", tgt_tone="low")
    assert t.src_tone == "high"
    assert t.tgt_tone == "low"


def test_tonal_correspondence_none_represents_untoned() -> None:
    """COMMITMENT: None tones are valid; they represent untoned segments."""
    t = TonalCorrespondence(src_tone=None, tgt_tone=None)
    assert t.src_tone is None
    assert t.tgt_tone is None


def test_tonal_correspondence_is_hashable() -> None:
    key = TonalCorrespondence(src_tone="55", tgt_tone="33")
    d = {key: 5.0}
    assert d[key] == 5.0


def test_tonal_correspondence_table_defaults_to_empty() -> None:
    table = TonalCorrespondenceTable()
    assert table.counts == {}
    assert table.prior_pseudo_counts == {}
    assert table.src_totals == {}


def test_tonal_correspondence_table_holds_counts() -> None:
    key = TonalCorrespondence(src_tone="H", tgt_tone="L")
    table = TonalCorrespondenceTable(
        counts={key: 12.0},
        prior_pseudo_counts={key: 1.0},
        src_totals={"H": 12.0},
    )
    assert table.counts[key] == 12.0


# ----- LearnedModel with tonal table -------------------------------------


def test_learned_model_has_tonal_table_by_default() -> None:
    """COMMITMENT: the learned model exposes an (empty) tonal table by default."""
    m = LearnedModel.empty()
    assert isinstance(m.tonal_table, TonalCorrespondenceTable)
    assert m.tonal_table.counts == {}


def test_learned_model_has_default_tone_weight() -> None:
    m = LearnedModel.empty()
    assert m.tone_weight == 1.0
