"""Tests for promoted-chunk transparency diagnostics."""

from regulae import (
    ChunkPhraseTable,
    ChunkProcessFamilyReport,
    ChunkProcessSubtypeReport,
    ConditionedCorrespondence,
    Context,
    FeatureConstraint,
    Form,
    LearnedModel,
    SegmentCorrespondenceTable,
    Segment,
    analyze_promoted_chunks,
    describe_chunk_process_families,
    describe_chunk_process_subtypes,
    describe_promoted_chunk,
    format_model,
    summarize_chunk_process_families,
    summarize_chunk_process_subtypes,
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
    out = format_model(
        trained,
        annotate_chunks=True,
        summarize_chunk_processes=True,
        summarize_chunk_subtypes=True,
    )
    assert "score=" in out
    assert "profile=" in out
    assert "subtype=" in out
    assert "Chunk process families" in out
    assert "Chunk process subtypes" in out


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
    assert "process profile:" in out
    assert "process subtype:" in out
    assert "notes:" in out
    assert "compositional sub-alignment:" in out


def test_chunk_transparency_prefers_compact_fusional_reflex_over_bundle() -> None:
    compact = (_chunk("s", "k"), _chunk("ʃ",))
    bundled = (_chunk("i", "h", "t"), _chunk("aɪ", "t"))
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries={compact: -4.9, bundled: -7.4}),
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
    compact_report = by_pair[compact]
    bundled_report = by_pair[bundled]
    assert compact_report.transparency_score > bundled_report.transparency_score
    assert compact_report.process_profile == "compact_fusion"
    assert "historically opaque candidate" in bundled_report.notes
    assert bundled_report.process_profile == "bundled_reduction"


def test_chunk_transparency_distinguishes_fusional_from_residual_reduction() -> None:
    fusional = (_chunk("a", "n"), _chunk("ɑ̃",))
    residual = (_chunk("k", "w"), _chunk("k",))
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries={fusional: -4.9, residual: -4.9}),
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
    assert by_pair[fusional].process_profile == "nasal_fusion"
    assert by_pair[residual].process_profile == "residual_reduction"


def test_chunk_transparency_recognizes_glide_or_vocalization_fusion() -> None:
    glide = (_chunk("u", "l"), _chunk("j",))
    bundled = (_chunk("a", "t", "e"), _chunk("ɛ",))
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries={glide: -4.9, bundled: -7.4}),
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
    assert by_pair[glide].process_profile == "glide_or_vocalization_fusion"
    assert by_pair[bundled].process_profile == "bundled_reduction"
    assert by_pair[glide].process_subtype == "glide_formation_or_vocalization"


def test_chunk_process_family_summary_aggregates_recurrent_profiles() -> None:
    chunks = {
        (_chunk("a", "n"), _chunk("ɑ̃",)): -5.0,
        (_chunk("e", "n"), _chunk("ɛ̃",)): -4.8,
        (_chunk("k", "w"), _chunk("k",)): -4.7,
        (_chunk("a", "t", "e"), _chunk("ɛ",)): -7.4,
    }
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries=chunks),
        tonal_table=model.tonal_table,
        cross_dimensional_table=model.cross_dimensional_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )
    families = summarize_chunk_process_families(model)
    assert all(isinstance(f, ChunkProcessFamilyReport) for f in families)
    by_profile = {f.process_profile: f for f in families}
    assert by_profile["nasal_fusion"].chunk_count == 2
    assert by_profile["nasal_fusion"].weighted_support > by_profile["bundled_reduction"].weighted_support
    assert "vowel gained nasalization" in by_profile["nasal_fusion"].evidence_signatures

    subtypes = summarize_chunk_process_subtypes(model)
    assert all(isinstance(s, ChunkProcessSubtypeReport) for s in subtypes)
    by_subtype = {(s.process_profile, s.process_subtype): s for s in subtypes}
    assert by_subtype[
        ("nasal_fusion", "vowel_nasalization_with_consonant_absorption")
    ].chunk_count == 2


def test_describe_chunk_process_families_mentions_profiles_and_examples() -> None:
    chunks = {
        (_chunk("k", "t"), _chunk("t", "ʃ")): -4.9,
        (_chunk("k", "w"), _chunk("k",)): -4.8,
    }
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries=chunks),
        tonal_table=model.tonal_table,
        cross_dimensional_table=model.cross_dimensional_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )
    out = describe_chunk_process_families(model)
    assert "Chunk process families" in out
    assert "compact_fusion" in out or "residual_reduction" in out
    assert "examples:" in out

    subtype_out = describe_chunk_process_subtypes(model)
    assert "Chunk process subtypes" in subtype_out
    assert "/" in subtype_out


def test_chunk_subtypes_split_broad_profiles_into_narrower_groups() -> None:
    chunks = {
        (_chunk("c", "e"), _chunk("θ",)): -4.9,
        (_chunk("i",), _chunk("w", "a")): -4.8,
        (_chunk("b", "o"), _chunk("b",)): -4.8,
        (_chunk("e",), _chunk("j", "e")): -4.8,
    }
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries=chunks),
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
    assert by_pair[(_chunk("c", "e"), _chunk("θ",))].process_subtype == (
        "fricativizing_or_affricating_fusion"
    )
    assert by_pair[(_chunk("i",), _chunk("w", "a"))].process_subtype == (
        "insertional_glide_or_diphthongal_fusion"
    )
    assert by_pair[(_chunk("b", "o"), _chunk("b",))].process_subtype == (
        "consonant_residue_with_vowel_loss"
    )
    assert by_pair[(_chunk("e",), _chunk("j", "e"))].process_subtype == (
        "residual_expansion"
    )


def test_chunk_process_subtypes_surface_discovered_contexts() -> None:
    chunk = (_chunk("c", "e"), _chunk("θ",))
    conditioned = ConditionedCorrespondence(
        src="c",
        tgt="θ",
        context=Context(
            following=(FeatureConstraint("front", "+"),),
        ),
    )
    model = LearnedModel.empty()
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(
            counts={conditioned: 6.0},
            src_totals={"c": 6.0},
        ),
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries={chunk: -4.9}),
        tonal_table=model.tonal_table,
        cross_dimensional_table=model.cross_dimensional_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )
    subtypes = summarize_chunk_process_subtypes(model)
    by_subtype = {(s.process_profile, s.process_subtype): s for s in subtypes}
    report = by_subtype[("compact_fusion", "fricativizing_or_affricating_fusion")]
    assert any("c->θ / _[front:+]" == ctx for ctx in report.context_signatures)

    out = describe_chunk_process_subtypes(model)
    assert "contexts:" in out
    assert "c->θ / _[front:+]" in out
