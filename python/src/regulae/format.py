"""Human-readable formatting for alignments and learned models.

Debugging and inspection helpers. Not part of the core inference path;
these exist so a human can look at an alignment (or a trained model)
and understand it without writing ad-hoc loops every time.

``format_alignment`` renders an alignment with one link per row.
``format_link`` renders a single link on one line.
``format_model`` renders a summary of a ``LearnedModel``: top segment
correspondences, displacement vectors, and promoted chunks.
"""

from regulae.chunk_diagnostics import (
    ChunkTransparencyReport,
    analyze_promoted_chunks,
    summarize_chunk_process_families,
    summarize_chunk_process_subtypes,
)
from regulae.model import (
    ConditionedCorrespondence,
    LearnedModel,
    MultiLectCorrespondenceClass,
    MultiLectModel,
)
from regulae.scoring import score_link
from regulae.types import (
    Alignment,
    Context,
    FeatureConstraint,
    FeatureDisplacement,
    Link,
    Segment,
)

# The symbol used to render an empty chunk (0-to-N or N-to-0 link).
# Epsilon is conventional for the empty string in formal language theory.
EMPTY_CHUNK_SYMBOL: str = "ε"


def format_segments(segments: tuple[Segment, ...]) -> str:
    """Render a chunk of segments as a compact string.

    Graphemes are concatenated. Suprasegmental annotations are appended
    in brackets after their bearer. An empty chunk is rendered as
    ``EMPTY_CHUNK_SYMBOL``.
    """
    if not segments:
        return EMPTY_CHUNK_SYMBOL
    parts: list[str] = []
    for seg in segments:
        piece: str = seg.grapheme
        annotations: list[str] = []
        if seg.tone is not None:
            annotations.append(f"T={seg.tone}")
        if seg.length is not None:
            annotations.append(f"L={seg.length}")
        if seg.stress is not None:
            annotations.append(f"S={seg.stress}")
        if annotations:
            piece += "[" + ",".join(annotations) + "]"
        parts.append(piece)
    return "".join(parts)


def format_link(link: Link) -> str:
    """Single-line rendering of a link: ``source ~ target``.

    Suprasegmental annotations, if present on any segment, are shown
    in brackets. The conditioning context and feature displacement are
    not rendered here — use ``format_alignment`` for the verbose view.
    """
    src = format_segments(link.source_chunk)
    tgt = format_segments(link.target_chunk)
    return f"{src} ~ {tgt}"


def format_model(
    model: LearnedModel,
    *,
    top_segments: int = 15,
    top_displacements: int = 5,
    min_count: float = 1.0,
    annotate_chunks: bool = False,
    chunk_warning_threshold: float = 0.35,
    summarize_chunk_processes: bool = False,
    summarize_chunk_subtypes: bool = False,
) -> str:
    """Summary of a ``LearnedModel`` as a multi-line string.

    Shows:

    * **Hyperparameters** — feature system, temperature, concentration,
      segment/displacement weights.
    * **Segment correspondences** — top ``top_segments`` entries by
      observed count, filtered to those with count >= ``min_count``.
      Empty corpus → notes "no segment-level observations yet."
    * **Displacement vectors** — top ``top_displacements`` entries by
      count. A compact notation ``feat: x -> y`` is used for each
      displacement in a vector.
    * **Promoted chunks** — all entries in the phrase table with their
      stored costs.

    The output is plain text, suitable for terminals and logs. No
    guarantees about exact formatting; tests should check content, not
    byte-for-byte output.
    """
    lines: list[str] = []
    lines.append(
        f"LearnedModel (feature_system={model.feature_system!r}, "
        f"τ={model.temperature}, β={model.concentration}, "
        f"w_seg={model.segment_weight}, w_disp={model.displacement_weight}, "
        f"w_tone={model.tone_weight})"
    )

    # Segment correspondences — grouped by source for readability.
    # Within each source, the unconditioned entries come first,
    # followed by context-conditioned variants sorted by specificity.
    counts = model.segment_table.counts
    unconditioned = {k: v for k, v in counts.items() if k.context.constraint_count() == 0 and v >= min_count}
    conditioned = {k: v for k, v in counts.items() if k.context.constraint_count() > 0 and v >= min_count}
    top_unconditioned = sorted(
        unconditioned.items(),
        key=lambda kv: (-kv[1], (kv[0].src, kv[0].tgt)),
    )[:top_segments]
    lines.append(
        f"Segment correspondences — unconditioned "
        f"(top {len(top_unconditioned)} of {len(unconditioned)}):"
    )
    if not top_unconditioned:
        lines.append("  (no observations yet)")
    else:
        for key, n in top_unconditioned:
            lines.append(f"  {key.src} -> {key.tgt}: {_fmt_count(n)}")
    # Conditioned entries grouped by source, only shown when non-empty.
    if conditioned:
        lines.append("")
        lines.append(
            f"Context-conditioned splits ({len(conditioned)} entries):"
        )
        by_src: dict[str, list[tuple[ConditionedCorrespondence, float]]] = {}
        for k, v in conditioned.items():
            by_src.setdefault(k.src, []).append((k, v))
        for src in sorted(by_src):
            entries = sorted(
                by_src[src],
                key=lambda kv: (-kv[0].context.constraint_count(), -kv[1], kv[0].tgt),
            )
            for ckey, cn in entries:
                ctx_str = _compact_context(ckey.context)
                lines.append(f"  {ckey.src} -> {ckey.tgt}{ctx_str}: {_fmt_count(cn)}")

    # Tonal correspondences
    tonal_counts = model.tonal_table.counts
    lines.append(f"Tonal correspondences ({len(tonal_counts)}):")
    if not tonal_counts:
        lines.append("  (none yet)")
    else:
        top_tonal = sorted(tonal_counts.items(), key=lambda kv: -kv[1])[:top_segments]
        for tkey, tn in top_tonal:
            tsrc = tkey.src_tone if tkey.src_tone is not None else "∅"
            ttgt = tkey.tgt_tone if tkey.tgt_tone is not None else "∅"
            lines.append(f"  {tsrc} -> {ttgt}: {_fmt_count(tn)}")

    # Cross-dimensional links
    cd_entries = model.cross_dimensional_table.entries
    lines.append(f"Cross-dimensional links ({len(cd_entries)}):")
    if not cd_entries:
        lines.append("  (none)")
    else:
        # Sort by confidence desc, then by count desc, then
        # lexically for a stable tiebreaker.
        sorted_entries = sorted(
            cd_entries,
            key=lambda e: (
                -e.confidence,
                -e.count,
                e.src_feature.feature,
                e.src_position,
                e.tgt_value,
            ),
        )
        for e in sorted_entries[:top_segments]:
            # Segmental predictors render as ``voiced=+``; tonal
            # predictors as ``tone=2``. Joint rules show both
            # predictors joined by ``&``.
            src_label = _render_src_predicate(
                e.src_feature, e.src_position
            )
            if e.src_feature_2 is not None and e.src_position_2 is not None:
                src_label += " & " + _render_src_predicate(
                    e.src_feature_2, e.src_position_2
                )
            lines.append(
                f"  {src_label}"
                f" -> {e.tgt_dimension}={e.tgt_value}@"
                f"{'+' if e.tgt_position_offset >= 0 else ''}"
                f"{e.tgt_position_offset}"
                f"  count={_fmt_count(e.count)}/{_fmt_count(e.src_count)}"
                f" conf={e.confidence:.2f}"
            )
        if len(sorted_entries) > top_segments:
            lines.append(f"  ... ({len(sorted_entries) - top_segments} more)")

    # Displacement vectors
    disp_counts = model.displacement_dist.counts
    top_disp = sorted(
        disp_counts.items(),
        key=lambda kv: (-kv[1], _compact_displacement(kv[0])),
    )[:top_displacements]
    lines.append(
        f"Feature displacements (top {min(len(top_disp), top_displacements)} "
        f"of {len(disp_counts)}):"
    )
    if not top_disp:
        lines.append("  (none yet)")
    else:
        for disp, n in top_disp:
            lines.append(f"  {_compact_displacement(disp)}: {_fmt_count(n)}")

    # Chunk table
    chunks = model.chunk_table.entries
    lines.append(f"Promoted chunks ({len(chunks)}):")
    if not chunks:
        lines.append("  (none promoted)")
    else:
        chunk_reports: dict[
            tuple[tuple[Segment, ...], tuple[Segment, ...]],
            ChunkTransparencyReport,
        ] = {}
        if annotate_chunks:
            chunk_reports = {
                (report.src_chunk, report.tgt_chunk): report
                for report in analyze_promoted_chunks(model)
            }
        for (src_ch, tgt_ch), cost in sorted(chunks.items(), key=lambda kv: kv[1]):
            s = "".join(x.grapheme for x in src_ch) or EMPTY_CHUNK_SYMBOL
            t = "".join(x.grapheme for x in tgt_ch) or EMPTY_CHUNK_SYMBOL
            line = f"  ({s}, {t}): cost={cost:.3f}"
            report = chunk_reports.get((src_ch, tgt_ch))
            if report is not None:
                line += (
                    f" score={report.transparency_score:.2f}"
                    f" profile={report.process_profile}"
                    f" subtype={report.process_subtype}"
                    f" conf={report.process_confidence:.2f}"
                )
                if report.process_evidence:
                    line += f" evidence={'; '.join(report.process_evidence)}"
                if report.notes:
                    line += f" notes={'; '.join(report.notes)}"
                if report.transparency_score < chunk_warning_threshold:
                    line += " WARNING"
            lines.append(line)

    if summarize_chunk_processes:
        families = summarize_chunk_process_families(model)
        lines.append(f"Chunk process families ({len(families)}):")
        if not families:
            lines.append("  (none)")
        else:
            for family in families:
                line = (
                    f"  {family.process_profile}: count={family.chunk_count} "
                    f"support={family.weighted_support:.2f} "
                    f"avg_score={family.average_transparency:.2f} "
                    f"avg_conf={family.average_process_confidence:.2f}"
                )
                if family.representative_chunks:
                    examples = ", ".join(
                        f"{src}->{tgt}" for src, tgt in family.representative_chunks
                    )
                    line += f" examples={examples}"
                if family.evidence_signatures:
                    line += f" evidence={'; '.join(family.evidence_signatures)}"
                lines.append(line)

    if summarize_chunk_subtypes:
        subtypes = summarize_chunk_process_subtypes(model)
        lines.append(f"Chunk process subtypes ({len(subtypes)}):")
        if not subtypes:
            lines.append("  (none)")
        else:
            for subtype in subtypes:
                line = (
                    f"  {subtype.process_profile}/{subtype.process_subtype}: "
                    f"count={subtype.chunk_count} "
                    f"support={subtype.weighted_support:.2f} "
                    f"avg_score={subtype.average_transparency:.2f} "
                    f"avg_conf={subtype.average_process_confidence:.2f}"
                )
                if subtype.representative_chunks:
                    examples = ", ".join(
                        f"{src}->{tgt}" for src, tgt in subtype.representative_chunks
                    )
                    line += f" examples={examples}"
                if subtype.evidence_signatures:
                    line += f" evidence={'; '.join(subtype.evidence_signatures)}"
                if subtype.context_signatures:
                    line += f" contexts={'; '.join(subtype.context_signatures)}"
                lines.append(line)

    return "\n".join(lines)


def _fmt_count(n: float) -> str:
    """Format a count as an integer if whole, else with one decimal."""
    if n == int(n):
        return str(int(n))
    return f"{n:.1f}"


def describe_source(model: LearnedModel, source_grapheme: str) -> str:
    """Return a multi-line description of everything the model has
    learned about a particular source grapheme.

    Shows:

    * All segment-table entries (unconditioned and context-conditioned)
      where this grapheme is the source, sorted by count.
    * Any chunk-table entries where this grapheme starts the source chunk.
    * The total unconditioned mass and the sum of conditioned masses.

    Useful when you've seen a "competing correspondence" in the
    model output and want to dig into what splits were
    committed for that source.
    """
    lines: list[str] = []
    lines.append(f"Source: {source_grapheme}")

    # Segment correspondences for this source
    entries = [
        (k, v)
        for k, v in model.segment_table.counts.items()
        if k.src == source_grapheme
    ]
    unconditioned = [
        (k, v) for k, v in entries if k.context.constraint_count() == 0
    ]
    conditioned = [
        (k, v) for k, v in entries if k.context.constraint_count() > 0
    ]

    lines.append(f"  Unconditioned entries ({len(unconditioned)}):")
    if not unconditioned:
        lines.append("    (none)")
    else:
        for k, v in sorted(unconditioned, key=lambda kv: -kv[1]):
            lines.append(f"    → {k.tgt}: {_fmt_count(v)}")

    lines.append(f"  Context-conditioned entries ({len(conditioned)}):")
    if not conditioned:
        lines.append("    (none)")
    else:
        # Group by context so multiple targets under the same context
        # are shown together.
        CtxKey = tuple[
            str | None,
            tuple[tuple[str, str], ...],
            tuple[tuple[str, str], ...],
        ]
        by_context: dict[CtxKey, list[tuple[ConditionedCorrespondence, float]]] = {}
        for k, v in conditioned:
            ctx_key: CtxKey = (
                k.context.position,
                tuple(sorted((c.feature, c.value) for c in k.context.preceding)),
                tuple(sorted((c.feature, c.value) for c in k.context.following)),
            )
            by_context.setdefault(ctx_key, []).append((k, v))
        # Sort contexts by specificity descending, then by total mass
        def _context_spec(ctx_key: CtxKey) -> int:
            return (
                (1 if ctx_key[0] is not None else 0)
                + len(ctx_key[1])
                + len(ctx_key[2])
            )
        sorted_contexts = sorted(
            by_context.keys(),
            key=lambda c: (-_context_spec(c), -sum(v for _, v in by_context[c])),
        )
        for ctx_key in sorted_contexts:
            items = sorted(by_context[ctx_key], key=lambda kv: -kv[1])
            first_k = items[0][0]
            ctx_str = _compact_context(first_k.context).strip()
            if ctx_str.startswith("/"):
                ctx_str = ctx_str[1:].strip()
            lines.append(f"    [{ctx_str}]")
            for k, v in items:
                lines.append(f"      → {k.tgt}: {_fmt_count(v)}")

    # Chunks starting with this source grapheme
    relevant_chunks = [
        (src, tgt, cost)
        for (src, tgt), cost in model.chunk_table.entries.items()
        if src and src[0].grapheme == source_grapheme
    ]
    lines.append(f"  Chunks starting with {source_grapheme!r} ({len(relevant_chunks)}):")
    if not relevant_chunks:
        lines.append("    (none)")
    else:
        for src, tgt, cost in sorted(relevant_chunks, key=lambda x: x[2]):
            s = "".join(x.grapheme for x in src)
            t = "".join(x.grapheme for x in tgt) or "ε"
            lines.append(f"    {s} → {t}: cost={cost:.3f}")

    return "\n".join(lines)


def describe_cross_dimensional_rule(
    model: LearnedModel,
    index: int,
) -> str:
    """Return a human-readable description of one
    cross-dimensional rule in ``model.cross_dimensional_table``.

    Looks up the rule by its integer position in the table and
    produces a multi-line report with the rule's source
    condition, target prediction, supporting counts, Dirichlet-
    smoothed conditional probability, base rate from the tonal
    table, and per-match cost adjustment. Useful for drilling
    into a single committed rule after seeing it in
    ``format_model``'s cross-dimensional section.

    Raises :class:`IndexError` if ``index`` is out of bounds.
    """
    entries = model.cross_dimensional_table.entries
    if index < 0 or index >= len(entries):
        raise IndexError(
            f"no cross-dimensional rule at index {index} "
            f"(table has {len(entries)} entries)"
        )
    rule = entries[index]

    lines: list[str] = []
    lines.append(f"Cross-dimensional rule #{index}")
    lines.append("=" * 50)
    lines.append(
        f"  source: {rule.src_feature.feature}"
        f" (feature) at position {rule.src_position}"
    )
    offset_str = (
        f"{'+' if rule.tgt_position_offset >= 0 else ''}"
        f"{rule.tgt_position_offset}"
    )
    lines.append(
        f"  target: {rule.tgt_dimension} = {rule.tgt_value!r}"
        f" at offset {offset_str} from the link position"
    )
    lines.append(
        f"  support: {_fmt_count(rule.count)} matches out of "
        f"{_fmt_count(rule.src_count)} source observations"
    )
    lines.append(f"  confidence: {rule.confidence:.3f}")

    # Compute and show the per-match cost adjustment from the
    # scoring overlay. Import lazily to avoid the top-level
    # format→scoring import cycle that doesn't exist yet but
    # might later.
    from regulae.scoring import _precompute_rule_adjustments

    pos_adj, neg_adj = _precompute_rule_adjustments(rule, model.tonal_table)
    lines.append(f"  cost adjustment on match: {pos_adj:+.3f} (nats)")
    lines.append(f"  cost adjustment on miss:  {neg_adj:+.3f} (nats)")

    return "\n".join(lines)


def _render_src_predicate(
    feature_constraint: FeatureConstraint, position_spec: str
) -> str:
    """Render one source predicate for a cross-dimensional rule.

    Segmental features: ``voiced=+@relative_-1``.
    Tonal features: ``tone=2@relative_0``.
    """
    return f"{feature_constraint.feature}={feature_constraint.value}@{position_spec}"


def _compact_context(ctx: Context) -> str:
    """One-line notation for a context, or empty string if the context is empty.

    Immediate-neighbour contexts render as ``" / _[vowel:+]"``.
    Long-range contexts add slot tags after the
    immediate-neighbour environment, e.g. ``" / _[front:+]
    next_syl=[front:+]"``.
    """
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


def _compact_displacement(disp: tuple[FeatureDisplacement, ...]) -> str:
    """One-line notation for a displacement vector.

    Example: ``[stop: P->A, fricative: A->P]`` for a stop→fricative
    change (P = present, A = absent).
    """
    if not disp:
        return "identity"
    parts = []
    for d in disp:
        fv = "P" if d.from_value == "present" else ("A" if d.from_value == "absent" else d.from_value)
        tv = "P" if d.to_value == "present" else ("A" if d.to_value == "absent" else d.to_value)
        parts.append(f"{d.feature}: {fv}->{tv}")
    return "[" + ", ".join(parts) + "]"


def format_alignment(
    alignment: Alignment,
    *,
    feature_system: str = "descriptive",
    show_costs: bool = True,
    show_displacement: bool = False,
) -> str:
    """Multi-line rendering of an alignment.

    Includes a header with the lect identifiers and total cost, then
    one line per link. With ``show_costs`` (default), each link's score
    is shown. With ``show_displacement``, each link's feature
    displacement vector is also shown (one line per displacement).

    The output is plain text suitable for terminals and debug logs.
    """
    src = alignment.source_form
    tgt = alignment.target_form
    src_str = format_segments(src.segments)
    tgt_str = format_segments(tgt.segments)

    lines: list[str] = []
    header = f"{src.lect_id} {src_str!r} ~ {tgt.lect_id} {tgt_str!r}"
    if show_costs:
        total = sum(score_link(link, feature_system=feature_system) for link in alignment.links)
        header += f"  (cost={total:.3f})"
    lines.append(header)

    for link in alignment.links:
        line = "  " + format_link(link)
        if show_costs:
            link_cost = score_link(link, feature_system=feature_system)
            line += f"  [{link_cost:.3f}]"
        lines.append(line)
        if show_displacement and link.feature_displacement:
            for disp in link.feature_displacement:
                lines.append(
                    f"      {disp.feature}: {disp.from_value} -> {disp.to_value}"
                )

    return "\n".join(lines)


# ----- multi-lect formatting ----------------------------------------------


def _format_class_segments(
    klass: MultiLectCorrespondenceClass,
    *,
    lect_width: int | None = None,
) -> str:
    """Render a multi-lect class's segment tuple as ``lect:grapheme``
    entries joined by spaces, in lect-id order."""
    parts = []
    for lect_id, grapheme in sorted(klass.segments.items()):
        label = lect_id if lect_width is None else lect_id.ljust(lect_width)
        parts.append(f"{label}:{grapheme}")
    return " ".join(parts)


def _format_class_contexts(klass: MultiLectCorrespondenceClass) -> str:
    """Render the per-lect contexts of a conditioned class, grouping
    lects that share an identical context.

    A single multi-lect class often carries the same conditioning
    environment under several pivot lects because the class-level
    discovery loop iterates
    every pivot independently and the dedup merges the discoveries.
    Rather than show ``Haw: foll=[front] | Mao: foll=[front] | ...``
    for each of N lects, collapse lects with bitwise-identical
    contexts into a single ``{Haw, Mao, ...}: foll=[front]``
    group. Produces much shorter reports on wide (7-lect+) classes
    without losing information: the underlying ``contexts`` mapping
    on the class is unchanged, only the display is grouped.

    Returns ``""`` for an unconditioned class or for a class whose
    every lect has an empty context.
    """
    if klass.contexts is None:
        return ""

    # Canonical encoding of a context for equality bucketing. Two
    # Context objects are frozen dataclasses with equal fields
    # (preceding, following, position, morphological) so ``==``
    # already works — but dicts aren't hashable, so we group via
    # an intermediate list of (ctx, lect_set).
    groups: list[tuple[Context, list[str]]] = []
    for lect_id in sorted(klass.contexts):
        ctx = klass.contexts[lect_id]
        if ctx is None or ctx.constraint_count() == 0:
            continue
        placed = False
        for gctx, members in groups:
            if gctx == ctx:
                members.append(lect_id)
                placed = True
                break
        if not placed:
            groups.append((ctx, [lect_id]))

    if not groups:
        return ""

    pieces: list[str] = []
    for gctx, members in groups:
        body = _compact_context(gctx).lstrip(" /")
        if len(members) == 1:
            pieces.append(f"{members[0]}={body}")
        else:
            pieces.append("{" + ",".join(members) + "}=" + body)
    return " [" + " | ".join(pieces) + "]"


def format_multi_lect_model(
    model: MultiLectModel,
    *,
    top_unconditioned: int = 20,
    top_conditioned: int = 20,
) -> str:
    """Render a human-readable summary of a :class:`MultiLectModel`.

    Shows:

    * the canonical lect-id list,
    * the pairwise model coverage,
    * the top ``top_unconditioned`` classes by count,
    * the top ``top_conditioned`` conditioned classes by count.

    The output is a plain-text report suitable for experiment logs.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MultiLectModel")
    lines.append("=" * 60)
    lines.append(f"lects ({len(model.lect_ids)}): {', '.join(model.lect_ids)}")
    lines.append(f"pairwise models:     {len(model.pairwise_models)}")
    lines.append(f"unconditioned cls:   {len(model.unconditioned_classes)}")
    lines.append(f"conditioned cls:     {len(model.conditioned_classes)}")
    lines.append(f"cognate corpus size: {len(model.cognate_corpus)}")
    lines.append("")

    lines.append(f"--- Top {top_unconditioned} unconditioned classes ---")
    if not model.unconditioned_classes:
        lines.append("  (none)")
    for klass in model.unconditioned_classes[:top_unconditioned]:
        n = len(klass.segments)
        segs = _format_class_segments(klass)
        lines.append(f"  [{n}-way] count={_fmt_count(klass.count):>5}  {segs}")

    if len(model.unconditioned_classes) > top_unconditioned:
        remaining = len(model.unconditioned_classes) - top_unconditioned
        lines.append(f"  ... ({remaining} more)")

    lines.append("")
    lines.append(f"--- Top {top_conditioned} conditioned classes ---")
    if not model.conditioned_classes:
        lines.append("  (none — class-level discovery committed no splits)")
    # Sort conditioned classes by confidence (descending) then count:
    # the user wants to see "strongest signal" first, not "most
    # observations", since confidence discriminates real rules from
    # thin-data noise in a way that count alone cannot.
    conditioned_sorted = sorted(
        model.conditioned_classes,
        key=lambda k: (-k.confidence, -k.count, sorted(k.segments.items())),
    )
    for klass in conditioned_sorted[:top_conditioned]:
        segs = _format_class_segments(klass)
        ctxs = _format_class_contexts(klass)
        cov = f" cov={klass.confidence:.2f}"
        lines.append(f"  count={_fmt_count(klass.count):>5}{cov}  {segs}{ctxs}")

    if len(conditioned_sorted) > top_conditioned:
        remaining = len(conditioned_sorted) - top_conditioned
        lines.append(f"  ... ({remaining} more)")

    # Multi-lect cross-dimensional rules section.
    xd_entries = model.cross_dimensional_table.entries
    lines.append("")
    lines.append(
        f"--- Multi-lect cross-dimensional rules ({len(xd_entries)}) ---"
    )
    if not xd_entries:
        lines.append("  (none)")
    else:
        for rule in xd_entries:
            src_label = _render_src_predicate(
                rule.src_feature, rule.src_position
            )
            if rule.src_feature_2 is not None and rule.src_position_2 is not None:
                src_label += " & " + _render_src_predicate(
                    rule.src_feature_2, rule.src_position_2
                )
            lines.append(
                f"  {rule.src_lect}[{src_label}] "
                f"→ {rule.tgt_lect}[{rule.tgt_dimension}={rule.tgt_value}"
                f"@{rule.tgt_position_offset:+d}]  "
                f"count={rule.count:.0f}/{rule.src_count:.0f}  "
                f"conf={rule.confidence:.2f}"
            )

    return "\n".join(lines)


def describe_multi_lect_class(
    model: MultiLectModel,
    lect_id: str,
    grapheme: str,
) -> str:
    """Describe every multi-lect class in ``model`` that contains a
    specific ``(lect_id, grapheme)`` pair.

    Useful for drilling into one lect's behavior: "show me everything
    where Hawaiian has a `k`". Both unconditioned and conditioned
    entries are shown, separated by section.
    """
    lines: list[str] = []
    lines.append(f"Classes with {lect_id}:{grapheme}")
    lines.append("=" * 50)

    uncond_matches = [
        k for k in model.unconditioned_classes if k.segments.get(lect_id) == grapheme
    ]
    lines.append(f"Unconditioned entries ({len(uncond_matches)})")
    if not uncond_matches:
        lines.append("  (none)")
    for klass in uncond_matches:
        segs = _format_class_segments(klass)
        lines.append(f"  count={_fmt_count(klass.count):>5}  {segs}")

    lines.append("")
    cond_matches = [
        k for k in model.conditioned_classes if k.segments.get(lect_id) == grapheme
    ]
    lines.append(f"Conditioned entries ({len(cond_matches)})")
    if not cond_matches:
        lines.append("  (none)")
    cond_sorted = sorted(cond_matches, key=lambda k: (-k.confidence, -k.count))
    for klass in cond_sorted:
        segs = _format_class_segments(klass)
        ctxs = _format_class_contexts(klass)
        cov = f" cov={klass.confidence:.2f}"
        lines.append(f"  count={_fmt_count(klass.count):>5}{cov}  {segs}{ctxs}")

    return "\n".join(lines)
