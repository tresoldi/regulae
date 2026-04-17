"""Diagnostics for promoted chunk correspondences.

These helpers do not change scoring or alignment. They inspect an
trained model's promoted chunk table and estimate how historically
transparent each chunk is likely to be.

The central idea is pragmatic rather than theoretical: a chunk is more
transparent when it is short, structurally balanced, decomposes cleanly
without many gap links, and does not heavily overlap with smaller
promoted chunks that appear to explain the same material.
"""

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from functools import lru_cache

import merkmal

from regulae.model import ChunkPhraseTable, LearnedModel
from regulae.scoring import compute_displacement
from regulae.search import align_forms
from regulae.types import Alignment, Segment

CHUNK_PROCESS_PROFILES = (
    "compact_fusion",
    "residual_reduction",
    "nasal_fusion",
    "glide_or_vocalization_fusion",
    "balanced_restructuring",
    "bundled_reduction",
    "mixed_or_unclear",
)


@dataclass(frozen=True)
class ChunkTransparencyReport:
    """Interpretability report for one promoted chunk.

    ``transparency_score`` is a heuristic in ``[0, 1]`` where higher is
    more historically transparent. The score is advisory only.
    """

    src_chunk: tuple[Segment, ...]
    tgt_chunk: tuple[Segment, ...]
    promoted_cost: float
    transparency_score: float
    sub_alignment: Alignment
    asymmetry: int
    gap_ratio: float
    matched_count: int
    changed_match_count: int
    identity_match_count: int
    deletion_count: int
    insertion_count: int
    overlap_count: int
    process_profile: str
    process_subtype: str
    process_confidence: float
    process_evidence: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ChunkProcessFamilyReport:
    """Corpus-level summary of one inferred chunk process family.

    These reports aggregate promoted chunks that share the same process
    profile so downstream reconstruction logic can prefer process types
    backed by repeated evidence rather than isolated chunk proposals.
    """

    process_profile: str
    chunk_count: int
    weighted_support: float
    average_transparency: float
    average_process_confidence: float
    representative_chunks: tuple[tuple[str, str], ...]
    evidence_signatures: tuple[str, ...]


@dataclass(frozen=True)
class ChunkProcessSubtypeReport:
    """Corpus-level summary of one inferred chunk process subtype."""

    process_profile: str
    process_subtype: str
    chunk_count: int
    weighted_support: float
    average_transparency: float
    average_process_confidence: float
    representative_chunks: tuple[tuple[str, str], ...]
    evidence_signatures: tuple[str, ...]
    context_signatures: tuple[str, ...]


def analyze_promoted_chunks(
    model: LearnedModel,
) -> tuple[ChunkTransparencyReport, ...]:
    """Return transparency reports for every promoted chunk in ``model``.

    Reports are sorted by score ascending first, then by stored chunk
    cost, so the least transparent chunks surface first.
    """
    no_chunks_model = dc_replace(model, chunk_table=ChunkPhraseTable())
    reports: list[ChunkTransparencyReport] = []
    all_entries = list(model.chunk_table.entries.items())
    for (src_chunk, tgt_chunk), promoted_cost in all_entries:
        sub_alignment = align_forms(
            source=_pseudo_form("_chunk_src", src_chunk),
            target=_pseudo_form("_chunk_tgt", tgt_chunk),
            model=no_chunks_model,
            max_chunk_size=1,
        )
        asymmetry = abs(len(src_chunk) - len(tgt_chunk))
        total_links = max(len(sub_alignment.links), 1)
        matched_count = sum(
            1
            for link in sub_alignment.links
            if link.source_chunk and link.target_chunk
        )
        identity_match_count = sum(
            1
            for link in sub_alignment.links
            if link.source_chunk
            and link.target_chunk
            and link.source_chunk == link.target_chunk
        )
        changed_match_count = matched_count - identity_match_count
        deletion_count = sum(
            1
            for link in sub_alignment.links
            if link.source_chunk and not link.target_chunk
        )
        insertion_count = sum(
            1
            for link in sub_alignment.links
            if not link.source_chunk and link.target_chunk
        )
        gap_links = sum(
            1
            for link in sub_alignment.links
            if not link.source_chunk or not link.target_chunk
        )
        gap_ratio = gap_links / total_links
        overlap_count = _overlap_count(src_chunk, tgt_chunk, model)
        process_profile, process_confidence, process_evidence = _classify_chunk_process(
            sub_alignment=sub_alignment,
            feature_system=model.feature_system,
        )
        process_subtype = _classify_chunk_subtype(
            sub_alignment=sub_alignment,
            process_profile=process_profile,
            process_evidence=process_evidence,
            feature_system=model.feature_system,
        )
        score = _transparency_score(
            src_chunk=src_chunk,
            tgt_chunk=tgt_chunk,
            asymmetry=asymmetry,
            gap_ratio=gap_ratio,
            matched_count=matched_count,
            changed_match_count=changed_match_count,
            identity_match_count=identity_match_count,
            deletion_count=deletion_count,
            insertion_count=insertion_count,
            overlap_count=overlap_count,
            process_profile=process_profile,
        )
        notes = _chunk_notes(
            src_chunk=src_chunk,
            tgt_chunk=tgt_chunk,
            asymmetry=asymmetry,
            gap_ratio=gap_ratio,
            matched_count=matched_count,
            changed_match_count=changed_match_count,
            identity_match_count=identity_match_count,
            deletion_count=deletion_count,
            insertion_count=insertion_count,
            overlap_count=overlap_count,
            transparency_score=score,
            process_profile=process_profile,
        )
        reports.append(
            ChunkTransparencyReport(
                src_chunk=src_chunk,
                tgt_chunk=tgt_chunk,
                promoted_cost=promoted_cost,
                transparency_score=score,
                sub_alignment=sub_alignment,
                asymmetry=asymmetry,
                gap_ratio=gap_ratio,
                matched_count=matched_count,
                changed_match_count=changed_match_count,
                identity_match_count=identity_match_count,
                deletion_count=deletion_count,
                insertion_count=insertion_count,
                overlap_count=overlap_count,
                process_profile=process_profile,
                process_subtype=process_subtype,
                process_confidence=process_confidence,
                process_evidence=process_evidence,
                notes=notes,
            )
        )
    reports.sort(
        key=lambda r: (
            r.transparency_score,
            r.promoted_cost,
            _chunk_string(r.src_chunk),
            _chunk_string(r.tgt_chunk),
        )
    )
    return tuple(reports)


def summarize_chunk_process_families(
    model: LearnedModel,
) -> tuple[ChunkProcessFamilyReport, ...]:
    """Aggregate promoted chunks into corpus-level process families.

    ``weighted_support`` is a soft support measure: the sum over chunks of
    ``transparency_score * process_confidence``. This keeps the summary
    close to observed evidence while down-weighting opaque or weakly
    classified chunks.
    """
    reports = analyze_promoted_chunks(model)
    by_profile: dict[str, list[ChunkTransparencyReport]] = {}
    for report in reports:
        by_profile.setdefault(report.process_profile, []).append(report)

    families: list[ChunkProcessFamilyReport] = []
    for profile, members in by_profile.items():
        weighted_support = sum(
            r.transparency_score * r.process_confidence for r in members
        )
        avg_transparency = sum(r.transparency_score for r in members) / len(members)
        avg_process_confidence = sum(r.process_confidence for r in members) / len(
            members
        )
        ranked_members = sorted(
            members,
            key=lambda r: (
                -(r.transparency_score * r.process_confidence),
                -r.transparency_score,
                _chunk_string(r.src_chunk),
                _chunk_string(r.tgt_chunk),
            ),
        )
        representative_chunks = tuple(
            (
                _chunk_string(r.src_chunk),
                _chunk_string(r.tgt_chunk),
            )
            for r in ranked_members[:3]
        )
        evidence_signatures = _top_evidence_signatures(members)
        families.append(
            ChunkProcessFamilyReport(
                process_profile=profile,
                chunk_count=len(members),
                weighted_support=weighted_support,
                average_transparency=avg_transparency,
                average_process_confidence=avg_process_confidence,
                representative_chunks=representative_chunks,
                evidence_signatures=evidence_signatures,
            )
        )

    families.sort(
        key=lambda f: (
            -f.weighted_support,
            -f.chunk_count,
            -f.average_transparency,
            f.process_profile,
        )
    )
    return tuple(families)


def describe_promoted_chunk(
    model: LearnedModel,
    index: int,
) -> str:
    """Return a detailed multi-line description of one promoted chunk."""
    reports = analyze_promoted_chunks(model)
    if index < 0 or index >= len(reports):
        raise IndexError(
            f"no promoted chunk at index {index} (table has {len(reports)} entries)"
        )
    report = reports[index]
    notes = ", ".join(report.notes) if report.notes else "(none)"
    lines = [
        f"Promoted chunk #{index}",
        "=" * 50,
        f"  source: {_chunk_string(report.src_chunk)}",
        f"  target: {_chunk_string(report.tgt_chunk)}",
        f"  promoted cost: {report.promoted_cost:.3f}",
        f"  transparency score: {report.transparency_score:.2f}",
        f"  process profile: {report.process_profile}",
        f"  process subtype: {report.process_subtype}",
        f"  process confidence: {report.process_confidence:.2f}",
        f"  process evidence: {', '.join(report.process_evidence) if report.process_evidence else '(none)'}",
        f"  asymmetry: {report.asymmetry}",
        f"  gap ratio: {report.gap_ratio:.2f}",
        f"  matched links: {report.matched_count}",
        f"  changed matches: {report.changed_match_count}",
        f"  identity matches: {report.identity_match_count}",
        f"  deletions: {report.deletion_count}",
        f"  insertions: {report.insertion_count}",
        f"  overlap count: {report.overlap_count}",
        f"  notes: {notes}",
        "  compositional sub-alignment:",
    ]
    for link in report.sub_alignment.links:
        lines.append(f"    {_chunk_string(link.source_chunk)} ~ {_chunk_string(link.target_chunk)}")
    return "\n".join(lines)


def describe_chunk_process_families(model: LearnedModel) -> str:
    """Return a multi-line summary of corpus-level chunk process families."""
    families = summarize_chunk_process_families(model)
    lines = [f"Chunk process families ({len(families)}):"]
    if not families:
        lines.append("  (none)")
        return "\n".join(lines)
    for family in families:
        lines.append(
            "  "
            f"{family.process_profile}: count={family.chunk_count} "
            f"support={family.weighted_support:.2f} "
            f"avg_score={family.average_transparency:.2f} "
            f"avg_conf={family.average_process_confidence:.2f}"
        )
        if family.representative_chunks:
            reps = ", ".join(f"{src}->{tgt}" for src, tgt in family.representative_chunks)
            lines.append(f"    examples: {reps}")
        if family.evidence_signatures:
            lines.append(
                f"    evidence: {', '.join(family.evidence_signatures)}"
            )
    return "\n".join(lines)


def summarize_chunk_process_subtypes(
    model: LearnedModel,
) -> tuple[ChunkProcessSubtypeReport, ...]:
    """Aggregate promoted chunks into narrower process subtype groups."""
    reports = analyze_promoted_chunks(model)
    by_subtype: dict[tuple[str, str], list[ChunkTransparencyReport]] = {}
    for report in reports:
        key = (report.process_profile, report.process_subtype)
        by_subtype.setdefault(key, []).append(report)

    subtypes: list[ChunkProcessSubtypeReport] = []
    for (profile, subtype), members in by_subtype.items():
        weighted_support = sum(
            r.transparency_score * r.process_confidence for r in members
        )
        avg_transparency = sum(r.transparency_score for r in members) / len(members)
        avg_process_confidence = sum(r.process_confidence for r in members) / len(
            members
        )
        ranked_members = sorted(
            members,
            key=lambda r: (
                -(r.transparency_score * r.process_confidence),
                -r.transparency_score,
                _chunk_string(r.src_chunk),
                _chunk_string(r.tgt_chunk),
            ),
        )
        representative_chunks = tuple(
            (_chunk_string(r.src_chunk), _chunk_string(r.tgt_chunk))
            for r in ranked_members[:3]
        )
        evidence_signatures = _top_evidence_signatures(members)
        subtypes.append(
            ChunkProcessSubtypeReport(
                process_profile=profile,
                process_subtype=subtype,
                chunk_count=len(members),
                weighted_support=weighted_support,
                average_transparency=avg_transparency,
                average_process_confidence=avg_process_confidence,
                representative_chunks=representative_chunks,
                evidence_signatures=evidence_signatures,
                context_signatures=_top_context_signatures(model, members),
            )
        )

    subtypes.sort(
        key=lambda s: (
            -s.weighted_support,
            -s.chunk_count,
            -s.average_transparency,
            s.process_profile,
            s.process_subtype,
        )
    )
    return tuple(subtypes)


def describe_chunk_process_subtypes(model: LearnedModel) -> str:
    """Return a multi-line summary of chunk process subtypes."""
    subtypes = summarize_chunk_process_subtypes(model)
    lines = [f"Chunk process subtypes ({len(subtypes)}):"]
    if not subtypes:
        lines.append("  (none)")
        return "\n".join(lines)
    for subtype in subtypes:
        lines.append(
            "  "
            f"{subtype.process_profile}/{subtype.process_subtype}: "
            f"count={subtype.chunk_count} "
            f"support={subtype.weighted_support:.2f} "
            f"avg_score={subtype.average_transparency:.2f} "
            f"avg_conf={subtype.average_process_confidence:.2f}"
        )
        if subtype.representative_chunks:
            reps = ", ".join(
                f"{src}->{tgt}" for src, tgt in subtype.representative_chunks
            )
            lines.append(f"    examples: {reps}")
        if subtype.evidence_signatures:
            lines.append(
                f"    evidence: {', '.join(subtype.evidence_signatures)}"
            )
        if subtype.context_signatures:
            lines.append(
                f"    contexts: {', '.join(subtype.context_signatures)}"
            )
    return "\n".join(lines)


def _pseudo_form(lect_id: str, segments: tuple[Segment, ...]):
    from regulae.types import Form

    return Form(lect_id=lect_id, segments=segments)


def _chunk_string(chunk: tuple[Segment, ...]) -> str:
    if not chunk:
        return "ε"
    return "".join(seg.grapheme for seg in chunk)


def _transparency_score(
    *,
    src_chunk: tuple[Segment, ...],
    tgt_chunk: tuple[Segment, ...],
    asymmetry: int,
    gap_ratio: float,
    matched_count: int,
    changed_match_count: int,
    identity_match_count: int,
    deletion_count: int,
    insertion_count: int,
    overlap_count: int,
    process_profile: str,
) -> float:
    total_len = len(src_chunk) + len(tgt_chunk)
    score = 1.0
    score -= 0.18 * asymmetry
    score -= 0.24 * gap_ratio
    score -= 0.14 * max(total_len - 3, 0)
    score -= 0.09 * min(overlap_count, 3) / 3.0
    if total_len >= 5 and asymmetry > 0:
        score -= 0.10
    if deletion_count >= 2:
        score -= 0.10
    if insertion_count >= 2:
        score -= 0.10
    if deletion_count > 0 and insertion_count == 0 and asymmetry > 0:
        score -= 0.10
    if insertion_count > 0 and deletion_count == 0 and asymmetry > 0:
        score -= 0.10
    if (
        total_len <= 4
        and asymmetry == 1
        and matched_count == 1
        and changed_match_count == 1
        and deletion_count + insertion_count == 1
    ):
        score += 0.08
    if (
        total_len <= 4
        and asymmetry == 1
        and matched_count == 1
        and identity_match_count == 1
        and deletion_count + insertion_count == 1
    ):
        score += 0.03
    # Equal-length chunks with both deletion and insertion in the
    # atomic decomposition often reflect a compact local
    # restructuring (e.g. cluster palatalization) rather than
    # wholesale reduction.
    if asymmetry == 0 and deletion_count > 0 and insertion_count > 0:
        score += 0.12
    if matched_count >= 2 and asymmetry <= 1:
        score += 0.05
    if asymmetry == 0 and gap_ratio == 0.0 and total_len <= 4:
        score += 0.08
    elif total_len <= 4 and asymmetry <= 1:
        score += 0.05
    score += _process_score_adjustment(process_profile)
    return max(0.0, min(1.0, score))


def _chunk_notes(
    *,
    src_chunk: tuple[Segment, ...],
    tgt_chunk: tuple[Segment, ...],
    asymmetry: int,
    gap_ratio: float,
    matched_count: int,
    changed_match_count: int,
    identity_match_count: int,
    deletion_count: int,
    insertion_count: int,
    overlap_count: int,
    transparency_score: float,
    process_profile: str,
) -> tuple[str, ...]:
    notes: list[str] = []
    total_len = len(src_chunk) + len(tgt_chunk)
    specialized_short_reduction = (
        total_len <= 4
        and asymmetry == 1
        and matched_count == 1
        and deletion_count + insertion_count == 1
        and (changed_match_count == 1 or identity_match_count == 1)
    )
    if asymmetry > 0:
        notes.append("length asymmetry")
    if gap_ratio >= 0.34:
        notes.append("gap-heavy decomposition")
    if asymmetry == 0 and deletion_count > 0 and insertion_count > 0:
        notes.append("balanced local restructuring")
    if (
        total_len <= 4
        and asymmetry == 1
        and matched_count == 1
        and changed_match_count == 1
        and deletion_count + insertion_count == 1
    ):
        notes.append("compact fusional reflex")
    if process_profile == "nasal_fusion":
        notes.append("nasal fusion profile")
    if process_profile == "glide_or_vocalization_fusion":
        notes.append("glide or vocalization profile")
    if (
        total_len <= 4
        and asymmetry == 1
        and matched_count == 1
        and identity_match_count == 1
        and deletion_count + insertion_count == 1
    ):
        notes.append("residue-preserving reduction")
    if (
        deletion_count > 0
        and insertion_count == 0
        and asymmetry > 0
        and not specialized_short_reduction
    ):
        notes.append("reductive loss chunk")
    if total_len >= 5:
        notes.append("large bundled chunk")
    if overlap_count > 0:
        notes.append("overlaps with smaller promoted chunks")
    if matched_count >= 2 and asymmetry <= 1 and total_len <= 5:
        notes.append("multi-step but locally coherent")
    if not notes and transparency_score >= 0.8:
        notes.append("compact decomposition")
    if transparency_score < 0.35 or (total_len >= 5 and asymmetry > 0):
        notes.append("historically opaque candidate")
    return tuple(notes)


def _process_score_adjustment(process_profile: str) -> float:
    adjustments = {
        "compact_fusion": 0.03,
        "residual_reduction": 0.01,
        "nasal_fusion": 0.05,
        "glide_or_vocalization_fusion": 0.04,
        "balanced_restructuring": 0.02,
        "bundled_reduction": -0.04,
        "mixed_or_unclear": 0.0,
    }
    return adjustments.get(process_profile, 0.0)


def _classify_chunk_process(
    *,
    sub_alignment: Alignment,
    feature_system: str,
) -> tuple[str, float, tuple[str, ...]]:
    changed_matches: list[tuple[Segment, Segment]] = []
    identity_matches: list[tuple[Segment, Segment]] = []
    deletions: list[Segment] = []
    insertions: list[Segment] = []

    for link in sub_alignment.links:
        if link.source_chunk and link.target_chunk:
            src = link.source_chunk[0]
            tgt = link.target_chunk[0]
            if src == tgt:
                identity_matches.append((src, tgt))
            else:
                changed_matches.append((src, tgt))
        elif link.source_chunk:
            deletions.extend(link.source_chunk)
        elif link.target_chunk:
            insertions.extend(link.target_chunk)

    total_events = len(changed_matches) + len(identity_matches) + len(deletions) + len(insertions)

    if _looks_like_nasal_fusion(changed_matches, deletions, feature_system):
        return (
            "nasal_fusion",
            0.92,
            ("vowel gained nasalization", "adjacent nasal absorbed"),
        )
    if _looks_like_glide_or_vocalization_fusion(changed_matches, deletions, feature_system):
        return (
            "glide_or_vocalization_fusion",
            0.84,
            ("approximant or glide reflex", "neighboring vocalic/liquid material absorbed"),
        )
    if len(changed_matches) == 1 and len(deletions) + len(insertions) == 1 and total_events <= 2:
        return (
            "compact_fusion",
            0.82,
            _compact_fusion_evidence(changed_matches[0], deletions, insertions, feature_system),
        )
    if len(identity_matches) == 1 and len(deletions) + len(insertions) == 1 and total_events <= 2:
        return (
            "residual_reduction",
            0.88,
            ("one residue segment preserved", "adjacent material reduced"),
        )
    if len(deletions) > 0 and len(insertions) > 0 and len(changed_matches) + len(identity_matches) <= 1:
        return (
            "balanced_restructuring",
            0.68,
            ("local deletion and insertion both present",),
        )
    if total_events >= 3 and len(deletions) + len(insertions) > 0 and (
        len(changed_matches) + len(identity_matches)
    ) > 0:
        return (
            "bundled_reduction",
            0.86,
            ("multiple local steps required", "not historically atomic on present evidence"),
        )
    return ("mixed_or_unclear", 0.45, ())


def _classify_chunk_subtype(
    *,
    sub_alignment: Alignment,
    process_profile: str,
    process_evidence: tuple[str, ...],
    feature_system: str,
) -> str:
    identity_matches, changed_matches, deletions, insertions = _split_sub_alignment(
        sub_alignment
    )
    if process_profile == "compact_fusion":
        if any("fricative-like" in item for item in process_evidence):
            return "fricativizing_or_affricating_fusion"
        if any("target segment inserted" in item for item in process_evidence):
            return "insertional_glide_or_diphthongal_fusion"
        if len(changed_matches) == 1:
            src, tgt = changed_matches[0]
            if _has_feature(tgt, "palatal", feature_system):
                return "palatalizing_fusion"
            if _has_feature(src, "vowel", feature_system) and _has_feature(
                tgt, "vowel", feature_system
            ):
                return "vocalic_absorptive_fusion"
        return "absorptive_fusion"
    if process_profile == "residual_reduction":
        if insertions:
            return "residual_expansion"
        if len(identity_matches) == 1 and len(deletions) == 1:
            src, _ = identity_matches[0]
            deleted = deletions[0]
            if _has_feature(src, "consonant", feature_system) and _has_feature(
                deleted, "vowel", feature_system
            ):
                return "consonant_residue_with_vowel_loss"
            if _has_feature(src, "vowel", feature_system) and _has_feature(
                deleted, "consonant", feature_system
            ):
                return "vowel_residue_with_consonant_loss"
        return "generic_residual_reduction"
    if process_profile == "nasal_fusion":
        return "vowel_nasalization_with_consonant_absorption"
    if process_profile == "glide_or_vocalization_fusion":
        return "glide_formation_or_vocalization"
    if process_profile == "balanced_restructuring":
        return "balanced_local_restructuring"
    if process_profile == "bundled_reduction":
        return "multi_step_bundle"
    return "mixed_or_unclear"


def _compact_fusion_evidence(
    changed_match: tuple[Segment, Segment],
    deletions: list[Segment],
    insertions: list[Segment],
    feature_system: str,
) -> tuple[str, ...]:
    src, tgt = changed_match
    evidence: list[str] = []
    disp = _safe_displacement(src, tgt, feature_system)
    if any(d.feature in {"fricative", "sibilant", "affricate"} and d.to_value == "present" for d in disp):
        evidence.append("surviving segment became more fricative-like")
    if any(d.feature == "palatal" and d.to_value == "present" for d in disp):
        evidence.append("surviving segment gained palatal properties")
    if deletions:
        evidence.append("one neighboring source segment absorbed")
    if insertions:
        evidence.append("one neighboring target segment inserted")
    if not evidence:
        evidence.append("single changed reflex plus one local non-match")
    return tuple(evidence)


def _looks_like_nasal_fusion(
    changed_matches: list[tuple[Segment, Segment]],
    deletions: list[Segment],
    feature_system: str,
) -> bool:
    if len(changed_matches) != 1 or len(deletions) != 1:
        return False
    src, tgt = changed_matches[0]
    deleted = deletions[0]
    return (
        _has_feature(src, "vowel", feature_system)
        and _has_feature(tgt, "vowel", feature_system)
        and _gained_feature(src, tgt, "nasalized", feature_system)
        and _has_feature(deleted, "nasal", feature_system)
    )


def _looks_like_glide_or_vocalization_fusion(
    changed_matches: list[tuple[Segment, Segment]],
    deletions: list[Segment],
    feature_system: str,
) -> bool:
    if len(changed_matches) != 1 or len(deletions) != 1:
        return False
    src, tgt = changed_matches[0]
    deleted = deletions[0]
    return _has_feature(tgt, "approximant", feature_system) and (
        _has_feature(src, "lateral", feature_system)
        or _has_feature(src, "approximant", feature_system)
        or _has_feature(deleted, "vowel", feature_system)
        or _has_feature(deleted, "approximant", feature_system)
    )


def _gained_feature(
    src: Segment,
    tgt: Segment,
    feature: str,
    feature_system: str,
) -> bool:
    src_features = _segment_features(src.grapheme, feature_system)
    tgt_features = _segment_features(tgt.grapheme, feature_system)
    return src_features is not None and tgt_features is not None and feature not in src_features and feature in tgt_features


def _has_feature(
    segment: Segment,
    feature: str,
    feature_system: str,
) -> bool:
    features = _segment_features(segment.grapheme, feature_system)
    return features is not None and feature in features


@lru_cache(maxsize=512)
def _segment_features(
    grapheme: str,
    feature_system: str,
) -> frozenset[str] | None:
    try:
        return merkmal.get_features(grapheme, system=feature_system)
    except Exception:
        return None


def _safe_displacement(
    src: Segment,
    tgt: Segment,
    feature_system: str,
):
    try:
        return compute_displacement(src, tgt, feature_system=feature_system)
    except Exception:
        return ()


def _split_sub_alignment(
    sub_alignment: Alignment,
) -> tuple[
    list[tuple[Segment, Segment]],
    list[tuple[Segment, Segment]],
    list[Segment],
    list[Segment],
]:
    identity_matches: list[tuple[Segment, Segment]] = []
    changed_matches: list[tuple[Segment, Segment]] = []
    deletions: list[Segment] = []
    insertions: list[Segment] = []
    for link in sub_alignment.links:
        if link.source_chunk and link.target_chunk:
            src = link.source_chunk[0]
            tgt = link.target_chunk[0]
            if src == tgt:
                identity_matches.append((src, tgt))
            else:
                changed_matches.append((src, tgt))
        elif link.source_chunk:
            deletions.extend(link.source_chunk)
        elif link.target_chunk:
            insertions.extend(link.target_chunk)
    return identity_matches, changed_matches, deletions, insertions


def _top_evidence_signatures(
    reports: list[ChunkTransparencyReport],
) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for report in reports:
        for evidence in report.process_evidence:
            counts[evidence] = counts.get(evidence, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(evidence for evidence, _ in ranked[:3])


def _top_context_signatures(
    model: LearnedModel,
    reports: list[ChunkTransparencyReport],
) -> tuple[str, ...]:
    counts: dict[str, float] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for report in reports:
        for link in report.sub_alignment.links:
            if not link.source_chunk or not link.target_chunk:
                continue
            src = _chunk_string(link.source_chunk)
            tgt = _chunk_string(link.target_chunk)
            seen_pairs.add((src, tgt))
    for key, count in model.segment_table.counts.items():
        if key.context.constraint_count() == 0:
            continue
        if (key.src, key.tgt) not in seen_pairs:
            continue
        signature = f"{key.src}->{key.tgt}{_compact_context_signature(key.context)}"
        counts[signature] = counts.get(signature, 0.0) + count
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(signature for signature, _ in ranked[:3])


def _compact_context_signature(ctx) -> str:
    if ctx.constraint_count() == 0:
        return ""
    has_immediate = bool(ctx.preceding or ctx.following)
    pieces: list[str] = []
    if has_immediate:
        if ctx.preceding:
            pieces.append(
                "[" + ",".join(f"{c.feature}:{c.value}" for c in ctx.preceding) + "]"
            )
        pieces.append("_")
        if ctx.following:
            pieces.append(
                "[" + ",".join(f"{c.feature}:{c.value}" for c in ctx.following) + "]"
            )
    env = "".join(pieces)
    pos = f"@{ctx.position}" if ctx.position else ""
    lr_parts: list[str] = []
    for offset, fc in ctx.preceding_at_distance:
        lr_parts.append(f"pre@{offset}=[{fc.feature}:{fc.value}]")
    for offset, fc in ctx.following_at_distance:
        lr_parts.append(f"fol@{offset}=[{fc.feature}:{fc.value}]")
    if ctx.somewhere_preceding:
        lr_parts.append(
            "s_pre=["
            + ",".join(f"{c.feature}:{c.value}" for c in ctx.somewhere_preceding)
            + "]"
        )
    if ctx.somewhere_following:
        lr_parts.append(
            "s_fol=["
            + ",".join(f"{c.feature}:{c.value}" for c in ctx.somewhere_following)
            + "]"
        )
    if ctx.same_syllable:
        lr_parts.append(
            "same_syl=["
            + ",".join(f"{c.feature}:{c.value}" for c in ctx.same_syllable)
            + "]"
        )
    if ctx.next_syllable:
        lr_parts.append(
            "next_syl=["
            + ",".join(f"{c.feature}:{c.value}" for c in ctx.next_syllable)
            + "]"
        )
    if ctx.previous_syllable:
        lr_parts.append(
            "prev_syl=["
            + ",".join(f"{c.feature}:{c.value}" for c in ctx.previous_syllable)
            + "]"
        )
    if lr_parts:
        sep = " " if env or pos else ""
        lr_str = sep + " ".join(lr_parts)
    else:
        lr_str = ""
    return f" / {env}{pos}{lr_str}"


def _overlap_count(
    src_chunk: tuple[Segment, ...],
    tgt_chunk: tuple[Segment, ...],
    model: LearnedModel,
) -> int:
    count = 0
    for other_src, other_tgt in model.chunk_table.entries:
        if other_src == src_chunk and other_tgt == tgt_chunk:
            continue
        if len(other_src) >= len(src_chunk) and len(other_tgt) >= len(tgt_chunk):
            continue
        if _is_contiguous_subchunk(other_src, src_chunk) and _is_contiguous_subchunk(
            other_tgt, tgt_chunk
        ):
            count += 1
    return count


def _is_contiguous_subchunk(
    needle: tuple[Segment, ...],
    haystack: tuple[Segment, ...],
) -> bool:
    if not needle:
        return False
    if len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False
