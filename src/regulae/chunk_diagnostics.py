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

from regulae.model import ChunkPhraseTable, LearnedModel
from regulae.search import align_forms
from regulae.types import Alignment, Segment


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
    overlap_count: int
    notes: tuple[str, ...]


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
        gap_links = sum(
            1
            for link in sub_alignment.links
            if not link.source_chunk or not link.target_chunk
        )
        gap_ratio = gap_links / total_links
        overlap_count = _overlap_count(src_chunk, tgt_chunk, model)
        score = _transparency_score(
            src_chunk=src_chunk,
            tgt_chunk=tgt_chunk,
            asymmetry=asymmetry,
            gap_ratio=gap_ratio,
            overlap_count=overlap_count,
        )
        notes = _chunk_notes(
            src_chunk=src_chunk,
            tgt_chunk=tgt_chunk,
            asymmetry=asymmetry,
            gap_ratio=gap_ratio,
            overlap_count=overlap_count,
            transparency_score=score,
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
                overlap_count=overlap_count,
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
        f"  asymmetry: {report.asymmetry}",
        f"  gap ratio: {report.gap_ratio:.2f}",
        f"  overlap count: {report.overlap_count}",
        f"  notes: {notes}",
        "  compositional sub-alignment:",
    ]
    for link in report.sub_alignment.links:
        lines.append(f"    {_chunk_string(link.source_chunk)} ~ {_chunk_string(link.target_chunk)}")
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
    overlap_count: int,
) -> float:
    total_len = len(src_chunk) + len(tgt_chunk)
    score = 1.0
    score -= 0.18 * asymmetry
    score -= 0.24 * gap_ratio
    score -= 0.14 * max(total_len - 3, 0)
    score -= 0.09 * min(overlap_count, 3) / 3.0
    if total_len >= 5 and asymmetry > 0:
        score -= 0.10
    if asymmetry == 0 and gap_ratio == 0.0 and total_len <= 4:
        score += 0.08
    elif total_len <= 4 and asymmetry <= 1:
        score += 0.05
    return max(0.0, min(1.0, score))


def _chunk_notes(
    *,
    src_chunk: tuple[Segment, ...],
    tgt_chunk: tuple[Segment, ...],
    asymmetry: int,
    gap_ratio: float,
    overlap_count: int,
    transparency_score: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    total_len = len(src_chunk) + len(tgt_chunk)
    if asymmetry > 0:
        notes.append("length asymmetry")
    if gap_ratio >= 0.34:
        notes.append("gap-heavy decomposition")
    if total_len >= 5:
        notes.append("large bundled chunk")
    if overlap_count > 0:
        notes.append("overlaps with smaller promoted chunks")
    if not notes and transparency_score >= 0.8:
        notes.append("compact decomposition")
    if transparency_score < 0.35 or (total_len >= 5 and asymmetry > 0):
        notes.append("historically opaque candidate")
    return tuple(notes)


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
