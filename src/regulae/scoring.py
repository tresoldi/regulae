"""Link scoring via merkmal feature distances.

Single-link scoring is the smallest meaningful operation in the alignment
framework: given two segment chunks, return a cost. Lower is better.

This module deliberately does very little:

* For 1-to-1 links, the cost is the merkmal distance between the source and
  target segments.
* For 0-to-N or N-to-0 links (insertion / deletion), a fixed gap cost.
* For N-to-M links (multi-segment chunks), a *compositional fallback*:
  pair the segments left-to-right and sum their distances, plus a chunk
  complexity penalty proportional to the asymmetry between chunk lengths.

The compositional fallback is the baseline for unseen chunks. Recurring
chunk correspondences are discovered by the training pipeline and promoted
to first-class entries in the learned phrase table, where they receive
holistic costs that override the compositional sum.

When a trained model is supplied, ``score_link`` uses the learned segment
table (with context-conditioned entries), the displacement distribution,
and the chunk phrase table. Without a model, it falls back to raw merkmal
distances.

**Unknown graphemes** raise :class:`UnknownGrapheme` from this module
rather than propagating merkmal's inconsistent errors. The policy is
strict by default: a grapheme not recognised by the chosen feature system
is an error that must be handled explicitly, not silently mapped to
infinity or skipped. Real data will contain unknowns (typos, unusual
phones, diacritic variation) and the caller must decide what to do about
them.
"""

import math

import merkmal

from regulae.model import (
    ConditionedCorrespondence,
    DisplacementDistribution,
    LearnedModel,
    SegmentCorrespondenceTable,
    TonalCorrespondence,
    TonalCorrespondenceTable,
)
from regulae.model import CrossDimensionalLink
from regulae.types import (
    Alignment,
    Context,
    FeatureConstraint,
    FeatureDisplacement,
    Form,
    Link,
    Segment,
)


class UnknownGrapheme(ValueError):
    """Raised when a segment's grapheme is not recognised by the
    feature system in use.

    Carries the offending grapheme and the feature system name so the
    caller can diagnose the failure. The exception message includes
    both.
    """

    def __init__(self, grapheme: str, feature_system: str) -> None:
        self.grapheme = grapheme
        self.feature_system = feature_system
        super().__init__(
            f"Unknown grapheme {grapheme!r} in feature system {feature_system!r}. "
            f"Either the grapheme is a typo, or the feature system does not "
            f"cover it. Fix the input or choose a different feature system."
        )

# Default cost for inserting or deleting a single segment.
# Calibrated empirically: this should be roughly the cost of a "moderate"
# 1-to-1 substitution, so the model doesn't prefer many gaps over a few
# substitutions or vice versa.
DEFAULT_GAP_COST: float = 0.5

# Penalty added per unit of length asymmetry in a multi-segment link.
# A chunk-of-2 ~ chunk-of-3 link gets +DEFAULT_CHUNK_PENALTY for the
# asymmetry. This will be replaced by learned chunk probabilities later.
DEFAULT_CHUNK_PENALTY: float = 0.25


def _segment_distance(
    source: Segment,
    target: Segment,
    feature_system: str,
) -> float:
    """Distance between two segments, currently using only the grapheme.

    Suprasegmental annotations (tone, length, stress) are not yet factored
    into the distance — that requires a richer scoring model. They are
    present on the Segment so the higher layers can use them.

    Raises :class:`UnknownGrapheme` if either grapheme is not recognised
    by the feature system.
    """
    try:
        return float(
            merkmal.distance(source.grapheme, target.grapheme, system=feature_system)
        )
    except KeyError as exc:
        # merkmal raises KeyError with the offending grapheme in the message.
        # We don't know which side failed, so we check both.
        if merkmal.get_features(source.grapheme, system=feature_system) is None:
            raise UnknownGrapheme(source.grapheme, feature_system) from exc
        raise UnknownGrapheme(target.grapheme, feature_system) from exc


def score_link(
    link: Link,
    feature_system: str = "descriptive",
    *,
    model: LearnedModel | None = None,
) -> float:
    """Return the cost of a single link. Lower is better.

    Without a model (``model=None``), a fixed-cost scoring applies:

    * 1-to-1: merkmal distance between the segments.
    * 0-to-N or N-to-0: ``DEFAULT_GAP_COST`` per inserted/deleted segment.
    * N-to-M: compositional sum (segments paired left-to-right, distances
      summed) plus ``DEFAULT_CHUNK_PENALTY`` per unit of length asymmetry.

    With a model, the layered learned scoring applies:

    * 1-to-1: log-linear combination of Dirichlet-posterior segment
      probability and feature-displacement-distribution probability.
      Falls back to the merkmal distance for segment pairs not in the
      prior (e.g., unseen during training).
    * Promoted chunks: direct lookup in the phrase table.
    * Unpromoted chunks: compositional fallback using the *learned*
      segment-pair costs (not the bare merkmal distances).

    The score is symmetric in the source and target chunks: swapping them
    yields the same cost. This reflects that correspondences are
    bidirectional at this stage; diachrony is a separate later inference.
    """
    src = link.source_chunk
    tgt = link.target_chunk

    # Pure insertion / deletion. Gap costs are not learned; use
    # the fixed constants regardless of whether a model is supplied.
    if not src and not tgt:
        return 0.0
    if not src:
        return DEFAULT_GAP_COST * len(tgt)
    if not tgt:
        return DEFAULT_GAP_COST * len(src)

    # Promoted chunk override: if the exact chunk pair is in the phrase
    # table, use its learned cost directly.
    if model is not None and (src, tgt) in model.chunk_table.entries:
        return model.chunk_table.entries[(src, tgt)]

    fs = model.feature_system if model is not None else feature_system

    # 1-to-1 link.
    if len(src) == 1 and len(tgt) == 1:
        if model is None:
            return _segment_distance(src[0], tgt[0], fs)
        return _segment_pair_cost_with_model(src[0], tgt[0], model, link.context)

    # N-to-M compositional fallback. Pair segments left-to-right; any
    # leftover segments on either side count as gap insertions.
    paired = min(len(src), len(tgt))
    if model is None:
        pair_cost = sum(
            _segment_distance(src[i], tgt[i], fs) for i in range(paired)
        )
    else:
        pair_cost = sum(
            _segment_pair_cost_with_model(src[i], tgt[i], model, link.context)
            for i in range(paired)
        )
    asymmetry = abs(len(src) - len(tgt))
    gap_cost = DEFAULT_GAP_COST * asymmetry
    chunk_penalty = DEFAULT_CHUNK_PENALTY * asymmetry
    return pair_cost + gap_cost + chunk_penalty


def _segment_pair_cost_with_model(
    src: Segment,
    tgt: Segment,
    model: LearnedModel,
    link_context: Context | None = None,
) -> float:
    """Cost of a 1-to-1 link under the layered learned model.

    The segment-level cost is ``-log P_post(t | s) - log Z_prior(s)``,
    where ``Z_prior(s) = sum_t' exp(-tau * d(s, t'))`` is the prior's
    log-normalizer cached in the table. This offset makes the cost
    match the bare merkmal distance at zero observations (so the
    initial model is behaviorally identical to prior-only scoring) and lets observed
    pairs get costs below the merkmal distance as the posterior
    concentrates mass.

    The segment lookup is **context-aware**: given the
    ``link_context`` (the features of the surrounding segments in the
    form), the most specific conditioned correspondence matching that
    context is used. If no conditioned match exists, the unconditioned
    (empty-context) entry is used. If that doesn't exist either, we
    fall back to bare merkmal.

    The cost is a weighted combination in log space:

        cost = w_seg * cost_seg
             + w_disp * cost_disp
             + w_tone * cost_tone

    The displacement term is zero if the displacement distribution
    hasn't been populated (e.g., during early training). The tonal
    term is zero if both segments are untoned or the tonal table is
    empty.

    Degenerate cases:

    * Segment pair not in the prior: fall back to the bare merkmal distance.
    """
    s = src.grapheme
    t = tgt.grapheme
    if link_context is None:
        link_context = Context()

    # Find the most specific matching conditioned correspondence.
    p_seg = _segment_posterior(s, t, link_context, model.segment_table, model.concentration)
    if p_seg is None:
        # Unseen in prior — fall back to the bare merkmal distance.
        return _segment_distance(src, tgt, model.feature_system)

    log_z = model.segment_table.log_normalizers.get(s, 0.0)
    cost_seg = _safe_neg_log(p_seg) - log_z

    # Displacement-level probability. An empty distribution contributes
    # nothing (segment-only scoring).
    if model.displacement_dist.counts or model.displacement_dist.total > 0.0:
        try:
            disp = compute_displacement(src, tgt, feature_system=model.feature_system)
            p_disp = _displacement_probability(disp, model.displacement_dist)
            v_eff = max(len(model.displacement_dist.counts), 1)
            cost_disp = _safe_neg_log(p_disp) - math.log(v_eff)
            cost_layered = (
                model.segment_weight * cost_seg
                + model.displacement_weight * cost_disp
            )
        except UnknownGrapheme:
            return _segment_distance(src, tgt, model.feature_system)
    else:
        cost_layered = cost_seg

    # Tonal component (additive). Zero when no tonal data to learn
    # from OR when both segments are untoned.
    tone_cost = _tonal_cost(src.tone, tgt.tone, model.tonal_table)
    return cost_layered + model.tone_weight * tone_cost


def _tonal_cost(
    src_tone: str | None,
    tgt_tone: str | None,
    table: TonalCorrespondenceTable,
) -> float:
    """Negative log probability of a tonal correspondence, offset so
    that the uninformative (empty) distribution contributes zero.

    Returns 0.0 when:

    * both tones are None (untoned segments — no tonal cost)
    * the tonal table is empty (no learned tone correspondences yet —
      we don't know anything, so we assign zero cost)
    """
    if src_tone is None and tgt_tone is None:
        return 0.0
    if not table.counts and not table.prior_pseudo_counts:
        # No tonal data at all: no cost. This preserves the
        # non-tonal training behavior.
        return 0.0
    key = TonalCorrespondence(src_tone=src_tone, tgt_tone=tgt_tone)
    alpha = table.prior_pseudo_counts.get(key, 0.0)
    n = table.counts.get(key, 0.0)
    total_for_src = table.src_totals.get(src_tone, 0.0)
    # Use a simple symmetric smoothing: the alpha of all keys is
    # the sum of the prior pseudo-count mass.
    prior_mass = sum(table.prior_pseudo_counts.values()) or 1.0
    denominator = total_for_src + prior_mass
    if denominator <= 0.0:
        return 0.0
    p = (alpha + n) / denominator
    if p <= 0.0:
        return 0.0
    # Offset by -log V (effective vocab size) so that uniform ≈ zero.
    # Use the union of counts and priors, with at least 2 so we get
    # a meaningful log offset even when only one key is observed.
    v_eff = max(
        len(set(table.counts.keys()) | set(table.prior_pseudo_counts.keys())), 2
    )
    return _safe_neg_log(p) - math.log(v_eff)


# ----- cross-dimensional overlay ------------------------------------------


_CROSS_DIM_SMOOTHING_ALPHA: float = 1.0


def apply_cross_dimensional_adjustments(
    alignment: Alignment,
    model: LearnedModel,
) -> float:
    """Return the total cost adjustment from cross-dimensional
    rule matches across the alignment.

    Cross-dimensional discovery commits :class:`CrossDimensionalLink`
    entries into ``model.cross_dimensional_table``. At scoring time the
    overlay walks the alignment, checks each 1-to-1 link against
    every entry in the table, and accumulates a Dirichlet-smoothed
    log-likelihood-ratio cost adjustment.

    Formula per rule applied to a 1-to-1 link:

    .. code-block:: text

        if source feature holds at source position:
            if target value holds at target position:
                adj = -(log P_cond - log P_base)
            else:
                adj = -(log(1 - P_cond) - log(1 - P_base))
        else:
            adj = 0

    where ``P_cond = (count + α) / (src_count + α·V)`` is the
    Dirichlet-smoothed conditional of the target value given the
    source feature at the source position, and ``P_base`` is the
    Dirichlet-smoothed unconditional probability of the target
    value from the model's ``TonalCorrespondenceTable`` (v1
    dimension is "tone"). ``V`` is the number of distinct target
    values seen in the table (clamped to ≥2).

    When the rule doesn't discriminate (``P_cond == P_base``)
    both the positive and negative adjustments are zero, so the
    empty-table inertness invariant is preserved. The same holds
    when the table is empty.

    The adjustment is additive: every committed rule contributes
    independently, and the total is the sum across rules × links.
    """
    if not model.cross_dimensional_table.entries:
        return 0.0

    # Precompute per-rule (pos_adj, neg_adj) pairs. These depend
    # only on the rule's count/src_count and the model's tonal
    # table, not on the specific link being scored, so we do them
    # once per scoring call.
    precomputed: list[tuple[float, float]] = [
        _precompute_rule_adjustments(rule, model.tonal_table)
        for rule in model.cross_dimensional_table.entries
    ]

    total = 0.0
    src_pos = 0
    tgt_pos = 0
    feature_system = model.feature_system
    src_form = alignment.source_form
    tgt_form = alignment.target_form

    for link in alignment.links:
        src_len = len(link.source_chunk)
        tgt_len = len(link.target_chunk)
        if src_len == 1 and tgt_len == 1:
            for rule, (pos_adj, neg_adj) in zip(
                model.cross_dimensional_table.entries, precomputed
            ):
                total += _apply_rule_to_link(
                    rule=rule,
                    pos_adj=pos_adj,
                    neg_adj=neg_adj,
                    src_form=src_form,
                    src_pos=src_pos,
                    tgt_form=tgt_form,
                    tgt_pos=tgt_pos,
                    feature_system=feature_system,
                )
        src_pos += src_len
        tgt_pos += tgt_len
    return total


def _precompute_rule_adjustments(
    rule: CrossDimensionalLink,
    tonal_table: TonalCorrespondenceTable,
) -> tuple[float, float]:
    """Return ``(pos_adj, neg_adj)`` for a rule against a tonal
    table.

    * ``pos_adj``: cost adjustment to apply when the source
      feature holds at the source position AND the target value
      holds at the target position. Negative when the rule's
      conditional probability exceeds the base rate.
    * ``neg_adj``: cost adjustment when the source holds but the
      target value does not. Positive when the rule confidently
      predicted a value that didn't show up (the rule gets
      charged for a bad prediction).

    Both are in nats.
    """
    # V = number of distinct target values in the tonal table.
    # Clamped to ≥2 so log(V) is well-defined and the smoothing
    # doesn't collapse to uniform-with-one-category.
    distinct_tones: set[str | None] = set()
    for key in tonal_table.counts:
        if key.tgt_tone is not None:
            distinct_tones.add(key.tgt_tone)
    v_eff = max(len(distinct_tones), 2)

    alpha = _CROSS_DIM_SMOOTHING_ALPHA

    # Dirichlet-smoothed conditional of the target value given
    # the rule's source context. Uses the rule's own count and
    # src_count fields, both populated at commit time.
    p_cond = (rule.count + alpha) / (rule.src_count + alpha * v_eff)
    p_cond = max(min(p_cond, 1.0 - 1e-12), 1e-12)

    # Dirichlet-smoothed unconditional base rate of the target
    # value across the whole tonal table. If the table is empty
    # (non-tonal corpus), fall back to a uniform prior over V.
    total_for_value = 0.0
    total_all = 0.0
    for key, count in tonal_table.counts.items():
        total_all += count
        if key.tgt_tone == rule.tgt_value:
            total_for_value += count
    if total_all <= 0.0:
        p_base = 1.0 / v_eff
    else:
        p_base = (total_for_value + alpha) / (total_all + alpha * v_eff)
    p_base = max(min(p_base, 1.0 - 1e-12), 1e-12)

    # Log-likelihood ratios: negative cost when the rule's
    # prediction is confirmed, positive cost when it's
    # contradicted. The sign flip (``-(...)``) turns an LLR into
    # a cost delta (lower is better).
    pos_adj = -(math.log(p_cond) - math.log(p_base))
    neg_adj = -(math.log(1.0 - p_cond) - math.log(1.0 - p_base))
    return pos_adj, neg_adj


def _apply_rule_to_link(
    *,
    rule: CrossDimensionalLink,
    pos_adj: float,
    neg_adj: float,
    src_form: Form,
    src_pos: int,
    tgt_form: Form,
    tgt_pos: int,
    feature_system: str,
) -> float:
    """Evaluate one rule against one 1-to-1 link position pair,
    returning the rule's cost contribution for that link.

    Returns 0.0 if the rule's source context does not hold at
    the specified source position, or if the rule's target
    context cannot be evaluated (out of bounds, missing
    annotation, unsupported dimension).
    """
    if not _source_predicate_holds(
        rule.src_feature,
        rule.src_position,
        src_form,
        src_pos,
        feature_system,
    ):
        return 0.0
    # Joint-predictor extension: when src_feature_2 is
    # set, the rule fires only when BOTH predicates hold.
    if rule.src_feature_2 is not None:
        if rule.src_position_2 is None:
            return 0.0
        if not _source_predicate_holds(
            rule.src_feature_2,
            rule.src_position_2,
            src_form,
            src_pos,
            feature_system,
        ):
            return 0.0

    # Source context holds. Now evaluate the target side.
    tgt_segment_idx = tgt_pos + rule.tgt_position_offset
    if tgt_segment_idx < 0 or tgt_segment_idx >= len(tgt_form.segments):
        return 0.0
    tgt_seg = tgt_form.segments[tgt_segment_idx]

    if rule.tgt_dimension == "tone":
        actual = tgt_seg.tone
    elif rule.tgt_dimension == "length":
        actual = tgt_seg.length
    elif rule.tgt_dimension == "stress":
        actual = tgt_seg.stress
    else:
        # Unsupported dimension (v1 only implements "tone"). Phase
        # 5 would never commit such a rule in v1, but defend
        # against it at scoring time anyway.
        return 0.0

    if actual is None:
        return 0.0

    if actual == rule.tgt_value:
        return pos_adj
    return neg_adj


def _source_predicate_holds(
    feature_constraint: FeatureConstraint,
    position_spec: str,
    src_form: Form,
    src_pos: int,
    feature_system: str,
) -> bool:
    """Return True iff ``feature_constraint`` holds at the segment
    identified by ``position_spec`` relative to ``src_pos``.

    Tonal predictors (``feature_constraint.feature == "tone"``)
    check ``Segment.tone`` directly; all other predictors query
    merkmal for the segmental feature set. Returns False on
    out-of-bounds, missing annotation, unknown grapheme, or
    mismatched value.
    """
    offset = _parse_relative_offset(position_spec)
    idx = src_pos + offset
    if idx < 0 or idx >= len(src_form.segments):
        return False
    seg = src_form.segments[idx]
    if feature_constraint.feature == "tone":
        return seg.tone == feature_constraint.value
    if not seg.grapheme:
        return False
    try:
        features = merkmal.get_features(seg.grapheme, system=feature_system)
    except KeyError:
        return False
    if features is None:
        return False
    return bool(feature_constraint.feature in features)


def _parse_relative_offset(spec: str) -> int:
    """Parse a canonical ``src_position`` spec string like
    ``"relative_-1"``, ``"relative_0"``, ``"relative_+1"`` into
    an integer offset.

    Raises :class:`ValueError` if the string is not a recognised
    spec. In v1 the only accepted forms are the three relative
    variants.
    """
    if not spec.startswith("relative_"):
        raise ValueError(f"unrecognised src_position spec: {spec!r}")
    body = spec[len("relative_"):]
    try:
        return int(body)
    except ValueError as exc:
        raise ValueError(f"unrecognised src_position spec: {spec!r}") from exc


def _segment_posterior(
    src: str,
    tgt: str,
    link_context: Context,
    table: SegmentCorrespondenceTable,
    concentration: float,
) -> float | None:
    """Dirichlet posterior P(tgt | src, link_context) from the segment table.

    Finds the most specific conditioned correspondence matching
    ``link_context`` (using :meth:`Context.is_subset_of`) and returns
    its posterior. Falls through to the unconditioned entry
    ``ConditionedCorrespondence(src, tgt, Context())`` if no
    conditioned match exists.

    Returns ``None`` if no matching entry is in the prior
    pseudo-counts, signaling that the caller should fall back to the
    merkmal distance.
    """
    best_key = _find_most_specific_match(src, tgt, link_context, table)
    if best_key is None:
        return None
    alpha = table.prior_pseudo_counts.get(best_key, 0.0)
    n = table.counts.get(best_key, 0.0)
    n_src = table.src_totals.get(src, 0.0)
    denominator = concentration + n_src
    if denominator <= 0.0:
        return None
    return (alpha + n) / denominator


def _find_most_specific_match(
    src: str,
    tgt: str,
    link_context: Context,
    table: SegmentCorrespondenceTable,
) -> ConditionedCorrespondence | None:
    """Return the most specific conditioned correspondence in the table
    whose context is a subset of ``link_context``.

    Uses an index keyed by ``(src, tgt)`` so lookups are O(number of
    contexts for that pair), which is typically 1-3 even after splits.
    The index is cached per-table by ``id()`` — since the training
    loop produces a new table instance on every update, the cache
    stays correct without explicit invalidation.
    """
    index = _get_pair_index(table)
    candidates = index.get((src, tgt))
    if not candidates:
        return None
    best: ConditionedCorrespondence | None = None
    best_specificity = -1
    for key in candidates:
        if not key.context.is_subset_of(link_context):
            continue
        specificity = key.context.constraint_count()
        if specificity > best_specificity:
            best = key
            best_specificity = specificity
    return best


# Lazy per-table index keyed by the id() of the counts and priors dicts.
# Since training produces new dict instances on every update (no
# in-place mutation of shared dicts), id() is a stable cache key for
# the lifetime of the table. The cache is bounded to prevent runaway
# memory growth.
_PAIR_INDEX_CACHE: dict[
    tuple[int, int], dict[tuple[str, str], list[ConditionedCorrespondence]]
] = {}
_MAX_CACHE_SIZE: int = 32


def _get_pair_index(
    table: SegmentCorrespondenceTable,
) -> dict[tuple[str, str], list[ConditionedCorrespondence]]:
    """Return a (src, tgt) -> list of conditioned correspondences index
    for the given table, building it lazily. O(1) cache hit via id().

    Correctness note: the training code creates a fresh dict for every
    table update, so id() is stable across calls within a single
    scoring operation. If a caller were to mutate the dicts in place
    (not supported by the framework's contract), the cache could go
    stale.
    """
    cache_key = (id(table.counts), id(table.prior_pseudo_counts))
    cached = _PAIR_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    while len(_PAIR_INDEX_CACHE) >= _MAX_CACHE_SIZE:
        oldest = next(iter(_PAIR_INDEX_CACHE))
        del _PAIR_INDEX_CACHE[oldest]

    index: dict[tuple[str, str], list[ConditionedCorrespondence]] = {}
    seen: set[ConditionedCorrespondence] = set()
    for key in table.counts:
        if key in seen:
            continue
        seen.add(key)
        index.setdefault((key.src, key.tgt), []).append(key)
    for key in table.prior_pseudo_counts:
        if key in seen:
            continue
        seen.add(key)
        index.setdefault((key.src, key.tgt), []).append(key)
    _PAIR_INDEX_CACHE[cache_key] = index
    return index


def _displacement_probability(
    disp: tuple[FeatureDisplacement, ...],
    dist: DisplacementDistribution,
) -> float:
    """Posterior probability of a displacement vector under symmetric
    Dirichlet smoothing.

        P(d) = (alpha + N(d)) / (alpha * V + N_total)

    where ``V`` is the number of observed displacement vectors in the
    distribution. For unseen displacements the probability is
    ``alpha / (alpha * V + N_total)``.
    """
    alpha = dist.prior_pseudo_count
    n_d = dist.counts.get(disp, 0.0)
    v = len(dist.counts)
    denominator = alpha * max(v, 1) + dist.total
    if denominator <= 0.0:
        return 1.0  # no data — treat as uniform, cost-free
    return (alpha + n_d) / denominator


def _safe_neg_log(p: float) -> float:
    """Negative log of a probability, with guards against zero and negatives."""
    if p <= 0.0:
        return float("inf")
    return -math.log(p)


def compute_displacement(
    source: Segment,
    target: Segment,
    feature_system: str = "descriptive",
) -> tuple[FeatureDisplacement, ...]:
    """Compute the feature displacement vector between two segments.

    Looks up each segment's feature representation via merkmal and returns
    the symmetric difference: features present in one but not the other,
    encoded as ``FeatureDisplacement(feature, from_value, to_value)``.

    For the default categorical merkmal systems (``"descriptive"``,
    ``"broad"``), features are bare strings ('voiceless', 'fricative',
    ...). A feature present in source but absent in target becomes a
    displacement ``(feature, "present", "absent")``; absent → present is
    the reverse.

    For valued feature systems (PHOIBLE, P-base), this function will need
    extension.

    Identity (source == target) returns an empty tuple.
    """
    src_features = merkmal.get_features(source.grapheme, system=feature_system)
    tgt_features = merkmal.get_features(target.grapheme, system=feature_system)

    if src_features is None:
        raise UnknownGrapheme(source.grapheme, feature_system)
    if tgt_features is None:
        raise UnknownGrapheme(target.grapheme, feature_system)

    if src_features == tgt_features:
        return ()

    displacements: list[FeatureDisplacement] = []

    # Features present in source but not target: lost.
    for feat in sorted(src_features - tgt_features):
        displacements.append(FeatureDisplacement(feat, "present", "absent"))

    # Features present in target but not source: gained.
    for feat in sorted(tgt_features - src_features):
        displacements.append(FeatureDisplacement(feat, "absent", "present"))

    return tuple(displacements)
