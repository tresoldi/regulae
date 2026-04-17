import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import merkmal

from regulae.model import (
    ChunkPhraseTable,
    ConditionedCorrespondence,
    DisplacementDistribution,
    LearnedModel,
    SegmentCorrespondenceTable,
    TonalCorrespondenceTable,
)
from regulae.scoring import compute_displacement
from regulae.search import align_forms, alignment_cost
from regulae.uncertainty import UncertaintyEstimate, wilson_interval
from regulae.types import (
    Alignment,
    Context,
    FeatureDisplacement,
    Form,
)


def _initial_model(
    corpus: Sequence[tuple[Form, Form]],
    *,
    pair_weights: Sequence[float] | None = None,
    feature_system: str,
    temperature: float,
    concentration: float,
    segment_weight: float,
    displacement_weight: float,
) -> LearnedModel:
    """Build the starting model: no counts, merkmal-derived Dirichlet
    prior, empty displacement distribution and chunk table.

    The prior is computed for every (src_grapheme, tgt_grapheme) pair
    where the source grapheme appears in any source form and the
    target grapheme appears in any target form. Segments that appear
    only on one side are still given priors over the other side's
    inventory so the model can score any link encountered during
    alignment.

    Pairs with ``pair_weight <= 0`` are excluded from the grapheme
    inventory so that confidence=0 cognate sets do not shape the
    prior candidate space. Unknown graphemes encountered at alignment
    time still fall back to merkmal.
    """
    if pair_weights is None:
        pair_weights = [1.0] * len(corpus)
    src_graphemes: set[str] = set()
    tgt_graphemes: set[str] = set()
    for (src_form, tgt_form), pair_weight in zip(corpus, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        src_graphemes.update(seg.grapheme for seg in src_form.segments)
        tgt_graphemes.update(seg.grapheme for seg in tgt_form.segments)

    # For robustness during alignment, use the union of source and target
    # vocabularies as the candidate space for BOTH sides of the prior.
    # This way any link encountered during the search has a prior entry.
    # Sorted iteration is required for determinism: ``for x in some_set``
    # follows randomized hash order, and floating-point non-associativity
    # in the softmax/log-sum-exp below makes dict insertion order visible
    # in the final prior values. Sorting makes training bitwise
    # reproducible across runs.
    vocab = sorted(src_graphemes | tgt_graphemes)

    prior_pseudo_counts: dict[ConditionedCorrespondence, float] = {}
    log_normalizers: dict[str, float] = {}
    for s in vocab:
        logits: dict[str, float] = {}
        for t in vocab:
            try:
                d = float(merkmal.distance(s, t, system=feature_system))
            except KeyError:
                # Skip unknown graphemes in the prior; they'll fall
                # back to merkmal at scoring time if encountered.
                continue
            logits[t] = -temperature * d
        if not logits:
            continue
        # log-sum-exp for numerical stability
        max_logit = max(logits.values())
        exp_logits = {t: math.exp(lg - max_logit) for t, lg in logits.items()}
        total = sum(exp_logits.values())
        if total <= 0:
            continue
        # log Z(s) = max_logit + log(sum exp(logit - max_logit))
        log_normalizers[s] = max_logit + math.log(total)
        for t, e in exp_logits.items():
            key = ConditionedCorrespondence(src=s, tgt=t, context=Context())
            prior_pseudo_counts[key] = concentration * (e / total)

    segment_table = SegmentCorrespondenceTable(
        counts={},
        prior_pseudo_counts=prior_pseudo_counts,
        src_totals={},
        log_normalizers=log_normalizers,
    )

    from regulae.training import DEFAULT_TONE_WEIGHT

    return LearnedModel(
        segment_table=segment_table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        tonal_table=TonalCorrespondenceTable(),
        feature_system=feature_system,
        temperature=temperature,
        concentration=concentration,
        segment_weight=segment_weight,
        displacement_weight=displacement_weight,
        tone_weight=DEFAULT_TONE_WEIGHT,
    )


def _segment_em(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    initial_model: LearnedModel,
    *,
    max_chunk_size: int,
    max_iter: int,
    convergence_eps: float,
) -> LearnedModel:
    """Iterative segment-level EM.

    Each iteration:

    1. E-step: align every form pair under the current model.
    2. M-step: reset segment counts and accumulate 1-to-1 link counts
       from the alignments.

    Stops when the relative change in total corpus cost drops below
    ``convergence_eps``, or when ``max_iter`` is reached.
    """
    from regulae.training import align_corpus

    model = initial_model
    prev_cost = float("inf")

    for _ in range(max_iter):
        alignments = align_corpus(corpus, model, max_chunk_size=max_chunk_size)
        total_cost = sum(alignment_cost(a, model=model) for a in alignments)

        # Convergence check
        if prev_cost < float("inf"):
            denom = max(abs(prev_cost), 1.0)
            if abs(prev_cost - total_cost) / denom < convergence_eps:
                break
        prev_cost = total_cost

        # M-step: rebuild the segment table from scratch.
        model = _update_segment_table(model, alignments, pair_weights=pair_weights)

    return model


def _update_segment_table(
    model: LearnedModel,
    alignments: Sequence[Alignment],
    *,
    pair_weights: Sequence[float] | None = None,
) -> LearnedModel:
    """Produce a new model with the segment table updated from alignments.

    Only 1-to-1 links contribute. Chunks and gaps are ignored at this
    layer (chunks are handled by chunk promotion; gap modeling is
    deferred).

    During segment-level EM, counts are accumulated with the
    unconditioned (empty) context. Context-conditioned entries are
    created later, from these unconditioned counts.
    """
    counts: dict[ConditionedCorrespondence, float] = defaultdict(float)
    src_totals: dict[str, float] = defaultdict(float)

    empty_context = Context()
    if pair_weights is None:
        pair_weights = [1.0] * len(alignments)

    for alignment, pair_weight in zip(alignments, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        for link in alignment.links:
            if len(link.source_chunk) == 1 and len(link.target_chunk) == 1:
                s = link.source_chunk[0].grapheme
                t = link.target_chunk[0].grapheme
                key = ConditionedCorrespondence(src=s, tgt=t, context=empty_context)
                counts[key] += pair_weight
                src_totals[s] += pair_weight

    new_segment_table = SegmentCorrespondenceTable(
        counts=dict(counts),
        prior_pseudo_counts=model.segment_table.prior_pseudo_counts,
        src_totals=dict(src_totals),
        log_normalizers=model.segment_table.log_normalizers,
        uncertainty=_segment_uncertainty(counts, src_totals),
    )
    return _replace_segment_table(model, new_segment_table)


def _segment_uncertainty(
    counts: dict[ConditionedCorrespondence, float],
    src_totals: dict[str, float],
) -> dict[ConditionedCorrespondence, UncertaintyEstimate]:
    """Wilson intervals on ``P(tgt | src, context) = count / src_totals[src]``
    for every entry in ``counts``."""
    out: dict[ConditionedCorrespondence, UncertaintyEstimate] = {}
    for key, c in counts.items():
        n = src_totals.get(key.src, 0.0)
        out[key] = wilson_interval(c, n)
    return out


def _replace_segment_table(
    model: LearnedModel, segment_table: SegmentCorrespondenceTable
) -> LearnedModel:
    """Return a copy of the model with a new segment table.

    Uses ``dataclasses.replace`` so any additional fields on
    :class:`LearnedModel` (e.g. the cross-dimensional table)
    are preserved unchanged.
    """
    return replace(model, segment_table=segment_table)


def _displacement_aggregation(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
    *,
    max_chunk_size: int,
) -> LearnedModel:
    """One-pass displacement distribution update.

    Re-aligns the corpus with the post-EM model (to get the final
    alignments) and then counts every 1-to-1 link's feature
    displacement. The result is a new model with a populated
    :class:`DisplacementDistribution`.
    """
    from regulae.training import align_corpus

    alignments = align_corpus(corpus, model, max_chunk_size=max_chunk_size)

    counts: dict[tuple[FeatureDisplacement, ...], float] = defaultdict(float)
    total = 0.0

    for alignment, pair_weight in zip(alignments, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        for link in alignment.links:
            if len(link.source_chunk) != 1 or len(link.target_chunk) != 1:
                continue
            try:
                disp = compute_displacement(
                    link.source_chunk[0],
                    link.target_chunk[0],
                    feature_system=model.feature_system,
                )
            except Exception:
                continue
            counts[disp] += pair_weight
            total += pair_weight

    new_dist = DisplacementDistribution(
        counts=dict(counts),
        total=total,
        prior_pseudo_count=model.displacement_dist.prior_pseudo_count,
        uncertainty={d: wilson_interval(c, total) for d, c in counts.items()},
    )
    return LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=new_dist,
        chunk_table=model.chunk_table,
        tonal_table=model.tonal_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )
