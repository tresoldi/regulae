"""Tests for the ``format_model`` inspection helper.

These verify content (which pieces of information appear) rather than
exact byte-for-byte output, so cosmetic tweaks don't break them.
"""

from regulae import (
    ChunkPhraseTable,
    ConditionedCorrespondence,
    Context,
    DisplacementDistribution,
    FeatureDisplacement,
    Form,
    LearnedModel,
    Segment,
    SegmentCorrespondenceTable,
    format_model,
)
from tests._helpers import train_model


def _cc(src: str, tgt: str) -> ConditionedCorrespondence:
    return ConditionedCorrespondence(src=src, tgt=tgt, context=Context())


def test_format_model_empty_shows_no_data_markers() -> None:
    """COMMITMENT: an empty model renders with explicit 'no data' markers
    rather than raising or producing a confusing output."""
    out = format_model(LearnedModel.empty())
    assert "no observations" in out.lower()
    assert "none" in out.lower()  # displacements and chunks sections


def test_format_model_shows_feature_system() -> None:
    out = format_model(LearnedModel.empty(feature_system="distinctive"))
    assert "distinctive" in out


def test_format_model_shows_top_segment_correspondences() -> None:
    table = SegmentCorrespondenceTable(
        counts={
            _cc("p", "f"): 10.0,
            _cc("t", "d"): 5.0,
            _cc("k", "h"): 3.0,
        },
        prior_pseudo_counts={},
        src_totals={"p": 10.0, "t": 5.0, "k": 3.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    out = format_model(model)
    # Each correspondence should appear.
    assert "p -> f" in out
    assert "t -> d" in out
    assert "k -> h" in out
    # Counts should appear.
    assert "10" in out
    assert "5" in out
    assert "3" in out


def test_format_model_respects_top_segments_cap() -> None:
    """COMMITMENT: top_segments caps the number of correspondences shown."""
    counts = {_cc(c, c): float(i + 1) for i, c in enumerate("abcdefghij")}
    table = SegmentCorrespondenceTable(
        counts=counts,
        prior_pseudo_counts={},
        src_totals={c: float(i + 1) for i, c in enumerate("abcdefghij")},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    out = format_model(model, top_segments=3)
    # The top 3 by count should be the last three letters (which have the
    # highest counts). The first few letters (lowest count) should NOT appear.
    assert "j -> j" in out  # highest count (10)
    assert "i -> i" in out  # second (9)
    assert "h -> h" in out  # third (8)
    # Lowest count should be omitted.
    assert "a -> a" not in out
    # "23" (total number of correspondences) should not leak — we have 10 only.
    assert "of 10" in out


def test_format_model_shows_identity_displacement_as_such() -> None:
    """COMMITMENT: empty-tuple displacements render as 'identity'."""
    dist = DisplacementDistribution(
        counts={(): 5.0},
        total=5.0,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=dist,
        chunk_table=ChunkPhraseTable(),
    )
    out = format_model(model)
    assert "identity" in out


def test_format_model_shows_displacement_features_compactly() -> None:
    """COMMITMENT: non-identity displacements render feature names and
    a ``->`` direction marker."""
    disp = (
        FeatureDisplacement("stop", "present", "absent"),
        FeatureDisplacement("fricative", "absent", "present"),
    )
    dist = DisplacementDistribution(
        counts={disp: 7.0},
        total=7.0,
    )
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=dist,
        chunk_table=ChunkPhraseTable(),
    )
    out = format_model(model)
    assert "stop" in out
    assert "fricative" in out
    assert "->" in out


def test_format_model_shows_promoted_chunks() -> None:
    kt = (Segment("k"), Segment("t"))
    tt = (Segment("t"), Segment("t"))
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(entries={(kt, tt): -1.234}),
    )
    out = format_model(model)
    assert "kt" in out
    assert "tt" in out
    assert "1.234" in out or "-1.234" in out


def test_format_model_integrates_with_trained_corpus() -> None:
    """SMOKE: format_model runs cleanly on a real trained model."""
    corpus = [
        (Form("A", tuple(Segment(c) for c in "pata")), Form("B", tuple(Segment(c) for c in "fada"))),
        (Form("A", tuple(Segment(c) for c in "pata")), Form("B", tuple(Segment(c) for c in "fada"))),
    ]
    trained = train_model(corpus)
    out = format_model(trained)
    # Header always shows.
    assert "LearnedModel" in out
    # Sections always shown.
    assert "Segment correspondences" in out
    assert "displacements" in out.lower()
    assert "chunks" in out.lower()


# ----- describe_source ---------------------------------------------------


def test_describe_source_on_untrained_grapheme_returns_empty_sections() -> None:
    """COMMITMENT: describe_source returns a structured report even for
    graphemes that have no learned data, with explicit "(none)" markers."""
    from regulae import describe_source

    model = LearnedModel.empty()
    out = describe_source(model, "p")
    assert "Source: p" in out
    assert "Unconditioned entries (0)" in out
    assert "(none)" in out


def test_describe_source_shows_unconditioned_entries() -> None:
    from regulae import describe_source

    table = SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 10.0, _cc("p", "p"): 2.0, _cc("t", "d"): 5.0},
        prior_pseudo_counts={},
        src_totals={"p": 12.0, "t": 5.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    out = describe_source(model, "p")
    # The unconditioned section should list both p→f and p→p.
    assert "→ f: 10" in out
    assert "→ p: 2" in out
    # Other sources should not appear.
    assert "→ d" not in out


def test_describe_source_groups_conditioned_entries_by_context() -> None:
    """COMMITMENT: describe_source groups context-conditioned entries
    so all targets under the same context appear together."""
    from regulae import describe_source
    from regulae.types import FeatureConstraint

    front_ctx = Context(following=(FeatureConstraint("front", "+"),))
    table = SegmentCorrespondenceTable(
        counts={
            _cc("k", "k"): 11.0,  # unconditioned
            ConditionedCorrespondence("k", "θ", front_ctx): 5.0,
            ConditionedCorrespondence("k", "k", front_ctx): 4.0,
        },
        prior_pseudo_counts={},
        src_totals={"k": 20.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    out = describe_source(model, "k")
    # Both conditioned targets should appear under the same context.
    assert "→ θ: 5" in out
    assert "→ k: 4" in out
    # And "front" should appear as the conditioning feature.
    assert "front" in out


def test_describe_source_shows_relevant_chunks() -> None:
    """COMMITMENT: describe_source shows chunks where the source
    chunk starts with the queried grapheme."""
    from regulae import describe_source

    kt = (Segment("k"), Segment("t"))
    tt = (Segment("t"), Segment("t"))
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(entries={(kt, tt): -1.5}),
    )
    out = describe_source(model, "k")
    assert "kt → tt" in out
    # But no chunks for a source grapheme not starting with k.
    out_p = describe_source(model, "p")
    assert "kt → tt" not in out_p


def test_describe_source_on_trained_model() -> None:
    """SMOKE: describe_source runs cleanly on a real trained model
    and produces a readable report."""
    from regulae import describe_source

    corpus = [
        (Form("A", tuple(Segment(c) for c in "pata")), Form("B", tuple(Segment(c) for c in "fada"))),
        (Form("A", tuple(Segment(c) for c in "pata")), Form("B", tuple(Segment(c) for c in "fada"))),
    ]
    trained = train_model(corpus)
    out = describe_source(trained, "p")
    assert "Source: p" in out
    # p→f should be recorded.
    assert "f" in out
