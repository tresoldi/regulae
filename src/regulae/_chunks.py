import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from regulae.model import (
    ChunkPhraseTable,
    LearnedModel,
    SegmentCorrespondenceTable,
)
from regulae.search import align_forms
from regulae.types import (
    Alignment,
    Context,
    Form,
    Segment,
)

# Minimum number of times a chunk must appear in the corpus before it
# can be considered for promotion. Prevents BIC from promoting
# chunks that only appear once, where Laplace-smoothed MLE gives an
# over-confident probability (P = 1). Not strictly principled, but a
# pragmatic necessity for low-data settings.
MIN_CHUNK_OBSERVATIONS: int = 2


def _chunk_promotion(
    corpus: Sequence[tuple[Form, Form]],
    model: LearnedModel,
    *,
    max_chunk_size: int,
) -> LearnedModel:
    """Extract candidate chunks from corpus alignments and promote
    those that improve BIC.

    Returns a new model with the chunk table populated. Greedy
    promotion: at each step, the candidate with the most negative BIC
    delta is promoted; the loop terminates when no remaining candidate
    has a negative delta.
    """
    from regulae.training import align_corpus

    alignments = align_corpus(corpus, model, max_chunk_size=max_chunk_size)

    candidates = _extract_chunk_candidates(alignments, max_chunk_size=max_chunk_size)

    # Count of 1-to-1 observations is our proxy for corpus size N.
    n_observations = sum(
        1
        for a in alignments
        for link in a.links
        if len(link.source_chunk) == 1 and len(link.target_chunk) == 1
    )
    if n_observations <= 0:
        n_observations = 1  # avoid log(0)

    # Compute for each candidate its compositional cost and promoted
    # cost, then its BIC delta. We don't re-evaluate after each
    # promotion because our compositional cost uses only the segment
    # table, which is unchanged during this phase.
    promoted: dict[tuple[tuple[Segment, ...], tuple[Segment, ...]], float] = {}
    for (src_chunk, tgt_chunk), n_c in candidates.items():
        # Hard minimum count: below this, Laplace-smoothed MLE is too
        # optimistic (P = 1 from a single observation) and BIC cannot
        # reliably reject the spurious promotion.
        if n_c < MIN_CHUNK_OBSERVATIONS:
            continue
        comp_cost_per = _compositional_chunk_cost_raw(src_chunk, tgt_chunk, model)
        promoted_cost_per = _promoted_chunk_cost(src_chunk, tgt_chunk, candidates)
        if comp_cost_per == math.inf or promoted_cost_per == math.inf:
            continue
        reduction = n_c * (comp_cost_per - promoted_cost_per)
        # BIC parameter count: proportional to structural complexity
        # of the chunk. A longer chunk costs more parameters to
        # specify, so short frequent chunks are favored over long rare
        # ones. k = max(src_len, tgt_len) gives each length-unit one
        # parameter.
        k_params = max(len(src_chunk), len(tgt_chunk))
        delta_bic = -2.0 * reduction + k_params * math.log(n_observations)
        if delta_bic < 0.0:
            # Store the chunk cost in the SEARCH cost scale.
            # Promoted raw cost is -log P_chunk; subtract the
            # log-normalizer offset (sum over source segments) to
            # match the search's offset convention, so the chunk
            # table entry can be dropped into search scoring directly.
            log_z_sum = sum(
                model.segment_table.log_normalizers.get(s.grapheme, 0.0)
                for s in src_chunk
            )
            promoted[(src_chunk, tgt_chunk)] = promoted_cost_per - log_z_sum

    return LearnedModel(
        segment_table=model.segment_table,
        displacement_dist=model.displacement_dist,
        chunk_table=ChunkPhraseTable(entries=promoted),
        tonal_table=model.tonal_table,
        feature_system=model.feature_system,
        temperature=model.temperature,
        concentration=model.concentration,
        segment_weight=model.segment_weight,
        displacement_weight=model.displacement_weight,
        tone_weight=model.tone_weight,
    )


def _extract_chunk_candidates(
    alignments: Sequence[Alignment],
    *,
    max_chunk_size: int,
) -> dict[tuple[tuple[Segment, ...], tuple[Segment, ...]], int]:
    """Enumerate contiguous sub-alignments as candidate chunks.

    For each alignment, iterate over every contiguous subsequence of
    links [i..j] and concatenate their source and target chunks. The
    resulting (src_chunk, tgt_chunk) pair is a candidate if:

    * both sides are non-empty (no pure insertion/deletion),
    * both sides are within ``max_chunk_size``,
    * the combined length is at least 3 (excludes plain 1-to-1 links
      that belong to the segment table, not the chunk table).
    """
    counts: dict[tuple[tuple[Segment, ...], tuple[Segment, ...]], int] = defaultdict(int)
    for alignment in alignments:
        links = alignment.links
        for i in range(len(links)):
            src_chunk: tuple[Segment, ...] = ()
            tgt_chunk: tuple[Segment, ...] = ()
            for j in range(i, len(links)):
                src_chunk = src_chunk + links[j].source_chunk
                tgt_chunk = tgt_chunk + links[j].target_chunk
                if len(src_chunk) > max_chunk_size or len(tgt_chunk) > max_chunk_size:
                    break
                if len(src_chunk) == 0 or len(tgt_chunk) == 0:
                    continue
                if len(src_chunk) + len(tgt_chunk) < 3:
                    continue
                counts[(src_chunk, tgt_chunk)] += 1
    return counts


def _compositional_chunk_cost_raw(
    src_chunk: tuple[Segment, ...],
    tgt_chunk: tuple[Segment, ...],
    model: LearnedModel,
) -> float:
    """Raw compositional cost of a chunk — negative log probability of
    generating the chunk under independent segment-level draws.

    This is what BIC compares against the promoted chunk cost. The
    cost is a pure log-likelihood sum, without the ``log Z`` offset
    that the search cost uses.

    The pairing of source and target segments is found by running a
    **sub-alignment**: a mini ``align_forms`` call with
    ``max_chunk_size=1`` (so only 1-to-1, 0-to-1, and 1-to-0 links are
    considered) against a temporary model with an empty chunk table
    (to prevent recursion). This gives us the best decomposition of
    the chunk into atomic correspondences, which is what BIC needs for
    a fair comparison with the promoted (monolithic) chunk cost.

    Without the sub-alignment, naive left-to-right pairing produces
    wildly wrong costs for asymmetric chunks. For example, for
    ``(en, jen)`` the left-to-right pairing is ``e→j, n→e, gap→n``,
    which costs far more than the true best pairing ``e→je, n→n``.
    The inflated compositional cost makes BIC over-promote rule-like
    asymmetric chunks.
    """
    from regulae.scoring import _segment_posterior

    # Temporary model without chunks to prevent recursion into the
    # phrase table during the sub-alignment.
    no_chunks_model = replace(model, chunk_table=ChunkPhraseTable())
    sub_src = Form(lect_id="_sub_src", segments=src_chunk)
    sub_tgt = Form(lect_id="_sub_tgt", segments=tgt_chunk)
    sub = align_forms(sub_src, sub_tgt, model=no_chunks_model, max_chunk_size=1)

    # Compute the raw (unoffset) cost by summing over the sub-alignment's
    # chosen links.
    table = model.segment_table
    concentration = model.concentration
    # The effective vocabulary size for gap-cost purposes. Using the
    # number of known source graphemes is a reasonable proxy for
    # "how many segments could appear here."
    vocab_size = max(len(table.log_normalizers), 1)
    gap_cost_per_segment = math.log(vocab_size) if vocab_size > 1 else 1.0

    # For BIC comparison we look up the unconditioned posterior only
    # — context splits are relevant to the search-time scoring but
    # not to the BIC baseline cost.
    empty_context = Context()
    cost = 0.0
    for link in sub.links:
        sc = link.source_chunk
        tc = link.target_chunk
        if len(sc) == 1 and len(tc) == 1:
            p = _segment_posterior(
                sc[0].grapheme, tc[0].grapheme, empty_context, table, concentration
            )
            if p is None or p <= 0.0:
                return math.inf
            cost += -math.log(p)
        else:
            # Gap or multi-segment sub-link. With max_chunk_size=1 the
            # sub-alignment only emits 0-to-1 or 1-to-0 gap links, but
            # we handle the general case defensively.
            missing_segments = abs(len(sc) - len(tc)) + min(len(sc), len(tc)) * 0
            # Gap cost scales with the number of unmatched segments.
            cost += gap_cost_per_segment * max(len(sc), len(tc))

    return cost


def _promoted_chunk_cost(
    src_chunk: tuple[Segment, ...],
    tgt_chunk: tuple[Segment, ...],
    candidates: dict[tuple[tuple[Segment, ...], tuple[Segment, ...]], int],
    alpha: float = 1.0,
) -> float:
    """MLE chunk cost with Laplace smoothing: -log P(tgt | src).

    P(tgt | src) is estimated from candidate counts: for a given
    source chunk, we count how often each target chunk appears with
    it, apply add-alpha smoothing, and take the negative log.
    """
    # Counts for this source chunk across all observed target chunks.
    src_counts: dict[tuple[Segment, ...], int] = {}
    for (s, t), n in candidates.items():
        if s == src_chunk:
            src_counts[t] = n
    n_src_total = sum(src_counts.values())
    v_src = len(src_counts)
    if v_src == 0:
        return float("inf")
    n_c = src_counts.get(tgt_chunk, 0)
    prob = (n_c + alpha) / (n_src_total + alpha * v_src)
    if prob <= 0.0:
        return float("inf")
    return -math.log(prob)
