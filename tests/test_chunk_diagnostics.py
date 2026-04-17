"""Tests for promoted-chunk transparency diagnostics."""

from regulae import (
    ChunkPhraseTable,
    Form,
    LearnedModel,
    Segment,
    analyze_promoted_chunks,
    describe_promoted_chunk,
    format_model,
)
from tests._helpers import train_model


def _chunk(*graphemes: str) -> tuple[Segment, ...]:
    return tuple(Segment(g) for g in graphemes)


def test_chunk_transparency_prefers_compact_balanced_chunks() -> None:
    compact = (_chunk("k", "t"), _chunk("t", "ʃ"))
    opaque = (_chunk("a", "t", "e"), _chunk("a", "d"))
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries={compact: -4.9, opaque: -7.4}),
        tonal_table=model.tonal_table,
        cross_dimensional_table=model.cross_dimensional_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )
    reports = analyze_promoted_chunks(model)
    by_pair = {(r.src_chunk, r.tgt_chunk): r for r in reports}
    assert by_pair[compact].transparency_score > by_pair[opaque].transparency_score


def test_format_model_can_annotate_chunk_transparency() -> None:
    corpus = [
        (
            # recurring kt -> tʃ
            Form("A", (Segment("a"), Segment("k"), Segment("t"), Segment("a"))),
            Form("B", (Segment("a"), Segment("t"), Segment("ʃ"), Segment("a"))),
        ),
        (
            Form("A", (Segment("i"), Segment("k"), Segment("t"), Segment("i"))),
            Form("B", (Segment("i"), Segment("t"), Segment("ʃ"), Segment("i"))),
        ),
        (
            Form("A", (Segment("u"), Segment("k"), Segment("t"), Segment("u"))),
            Form("B", (Segment("u"), Segment("t"), Segment("ʃ"), Segment("u"))),
        ),
    ]
    trained = train_model(corpus)
    out = format_model(trained, annotate_chunks=True)
    assert "score=" in out


def test_describe_promoted_chunk_mentions_transparency_and_notes() -> None:
    compact = (_chunk("k", "t"), _chunk("t", "ʃ"))
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries={compact: -4.9}),
        tonal_table=model.tonal_table,
        cross_dimensional_table=model.cross_dimensional_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )
    out = describe_promoted_chunk(model, 0)
    assert "transparency score:" in out
    assert "notes:" in out
    assert "compositional sub-alignment:" in out
