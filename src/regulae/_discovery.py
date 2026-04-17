import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from regulae.model import (
    ChunkPhraseTable,
    ConditionedCorrespondence,
    LearnedModel,
    SegmentCorrespondenceTable,
    TonalCorrespondence,
    TonalCorrespondenceTable,
)
from regulae.search import align_forms
from regulae.types import (
    Context,
    FeatureConstraint,
    Form,
)
from regulae.uncertainty import UncertaintyEstimate, wilson_interval


# Minimum observation count for a split branch to be considered.
# Set low (2) so that small minority targets (e.g., ``k → θ`` before
# front vowels, 3 observations against 11 ``k → k``) are still
# candidates. BIC handles the noise/signal tradeoff; the threshold
# just prevents degenerate 1-observation partitions.
MIN_SPLIT_OBSERVATIONS: int = 2

# Maximum number of context splits that can be committed on the same
# source grapheme. Caps runaway multi-level splits.
MAX_SPLIT_DEPTH: int = 3

# Safety buffer on ΔBIC: a split is committed only when its BIC
# delta is below this threshold, not just below zero. Rejects
# marginal splits that are near-zero cost reductions driven by
# noise in small samples. The experiments showed that splits with
# ΔBIC in (-1.0, 0) were typically spurious (e.g., partitions with
# the same dominant target but slightly different minority mass);
# a buffer of -1.0 rejects those while keeping clearly-positive
# splits like k → θ / _[front] (ΔBIC ≈ -5.6 on the Latin-Spanish
# corpus).
DELTA_BIC_THRESHOLD: float = -1.0


# Candidate feature dimensions tried as context split criteria.
# Each entry is ("preceding" | "following", feature_name, value).
# Position splits are handled separately.
_SPLIT_FEATURE_INVENTORY: tuple[tuple[str, str, str], ...] = (
    ("following", "vowel", "+"),
    ("following", "front", "+"),
    ("following", "back", "+"),
    ("following", "close", "+"),
    ("following", "open", "+"),
    ("preceding", "vowel", "+"),
    ("preceding", "front", "+"),
    ("preceding", "back", "+"),
    ("preceding", "voiced", "+"),
    ("preceding", "voiceless", "+"),
    ("preceding", "consonant", "+"),
)

_SPLIT_POSITIONS: tuple[str, ...] = ("initial", "medial", "final")

DEFAULT_TONE_WEIGHT: float = 1.0

# Feature inventory for long-range candidate enumeration. Deliberately
# narrower than _SPLIT_FEATURE_INVENTORY: "vowel" and "consonant" are
# omitted because they are tautological on syllable-structural slots
# (every next syllable has a vowel by definition of "syllable", and
# almost every same-syllable context contains some consonant), which
# in early testing produced spurious commits that partitioned
# observations almost trivially. The remaining six features are more
# discriminative: they distinguish natural classes that are
# individually sparse.
_LONG_RANGE_FEATURES: tuple[str, ...] = (
    "front",
    "back",
    "close",
    "open",
    "voiced",
    "voiceless",
)

# Stricter thresholds than immediate-neighbour context discovery
# because long-range enumerates ~54 candidates per split step
# (6 features × 9 slots), a much larger search space than the
# immediate-neighbour ~14 candidates. Without tightening,
# random low-count partitions cross BIC by chance. The 3× tightening
# of ΔBIC and the 5-observation floor were tuned on latin_spanish
# to remove noise commits while preserving the signal hooks that
# the umlaut and harmony synthetic fixtures are designed to recover.
_LONG_RANGE_MIN_SPLIT_OBS: int = 5
_LONG_RANGE_DELTA_BIC_THRESHOLD: float = -5.0
# Minimum fraction of YES observations that must carry the same
# target outcome. Without this, the BIC commit loop is free to
# commit grab-bag splits that lump several rare outcomes into a
# single "anomaly bucket" — the split improves BIC by virtue of
# moving noisy rare outcomes out of the baseline, but doesn't
# encode a real conditioning rule. A real long-range rule has a
# single, reproducible outcome; the filter below enforces that.
_LONG_RANGE_MIN_DOMINANT_FRACTION: float = 0.6

# Order: structural → distance-bounded → existential (Q5).
# Structural first because syllable scope is the tightest and most
# linguistically meaningful conditioning axis; distance-bounded
# second because fixed offsets give sharp predicates; existential
# last because "somewhere in the prefix" is the loosest (and most
# prone to spurious correlations on small corpora).
_LONG_RANGE_STRUCTURAL_SLOTS: tuple[str, ...] = (
    "same_syllable",
    "next_syllable",
    "previous_syllable",
)
_LONG_RANGE_DISTANCE_SLOTS: tuple[str, ...] = (
    "preceding@2",
    "preceding@3",
    "following@2",
    "following@3",
)
_LONG_RANGE_EXISTENTIAL_SLOTS: tuple[str, ...] = (
    "somewhere_preceding",
    "somewhere_following",
)


def _context_discovery(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
    *,
    max_chunk_size: int,
) -> LearnedModel:
    """Discover immediate-neighbor context splits for segment correspondences.

    Operates on the alignments produced by the post-displacement model.
    For each source grapheme with at least two targets observed, tries
    a greedy BIC-optimized split by feature or position on the
    surrounding link context. Splits that reduce BIC get committed
    as new conditioned correspondence entries in the segment table.

    The unconditioned (empty-context) entries are kept as fallbacks:
    any link whose context doesn't match a conditioned entry falls
    through to the unconditioned version.
    """
    from regulae.training import align_corpus
    from regulae._em import _replace_segment_table

    alignments = align_corpus(corpus, model, max_chunk_size=max_chunk_size)

    # Flatten alignments into a list of (src, tgt, link_context) tuples.
    #
    # Observations come from two sources:
    # 1. 1-to-1 links — added directly.
    # 2. multi-segment links — decomposed into 1-to-1 equivalents via a
    #    sub-alignment (max_chunk_size=1, no chunk table). This captures
    #    patterns that the main search expressed as chunks but are
    #    structurally 1-to-1 with gaps. Without this, context discovery
    #    is blind to cases like OE ``sk → ʃ`` (absorbed as a 2-to-1 chunk) and
    #    intervocalic fricative voicing packaged into longer chunks.
    observations: list[tuple[str, str, Context, float]] = []
    no_chunks_model = replace(model, chunk_table=ChunkPhraseTable())
    for alignment, pair_weight in zip(alignments, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        for link in alignment.links:
            sc, tc = link.source_chunk, link.target_chunk
            if len(sc) == 1 and len(tc) == 1:
                observations.append(
                    (sc[0].grapheme, tc[0].grapheme, link.context, pair_weight)
                )
                continue
            if not sc or not tc:
                continue  # pure gap — no segment-level observation
            # Multi-segment link: decompose via sub-alignment.
            sub_src = Form(lect_id="_sub", segments=sc)
            sub_tgt = Form(lect_id="_sub", segments=tc)
            sub = align_forms(
                sub_src, sub_tgt, model=no_chunks_model, max_chunk_size=1
            )
            for sub_link in sub.links:
                sub_sc, sub_tc = sub_link.source_chunk, sub_link.target_chunk
                if len(sub_sc) == 1 and len(sub_tc) == 1:
                    # Use the outer link's context (the sub-alignment has
                    # its own but it's relative to the chunk's internal
                    # position, not the full form's).
                    observations.append(
                        (sub_sc[0].grapheme, sub_tc[0].grapheme, link.context, pair_weight)
                    )

    if not observations:
        return model

    # Group by source grapheme so we can consider each source's
    # target distribution independently.
    by_source: dict[str, list[tuple[str, Context, float]]] = defaultdict(list)
    for s, t, c, w in observations:
        by_source[s].append((t, c, w))

    # The starting set of conditioned correspondences is the
    # unconditioned ones from the segment-level EM table.
    new_counts: dict[ConditionedCorrespondence, float] = dict(model.segment_table.counts)
    new_src_totals: dict[str, float] = dict(model.segment_table.src_totals)

    # Total observation count (ln N baseline for BIC).
    n_total = sum(w for _, _, _, w in observations)
    if n_total == 0:
        return model
    ln_n = math.log(n_total)

    # Process each source greedily.
    for src, obs in by_source.items():
        # Only attempt splits on sources with multiple targets and
        # enough observations to support any split at all.
        target_set = {t for t, _, _ in obs}
        if len(target_set) < 2 or sum(w for _, _, w in obs) < 4.0:
            continue
        _commit_splits_for_source(
            src=src,
            observations=obs,
            new_counts=new_counts,
            new_src_totals=new_src_totals,
            ln_n=ln_n,
            max_depth=MAX_SPLIT_DEPTH,
        )

    new_segment_table = SegmentCorrespondenceTable(
        counts=new_counts,
        prior_pseudo_counts=model.segment_table.prior_pseudo_counts,
        src_totals=new_src_totals,
        log_normalizers=model.segment_table.log_normalizers,
        uncertainty=_segment_counts_uncertainty(new_counts, new_src_totals),
    )
    return _replace_segment_table(model, new_segment_table)


def _segment_counts_uncertainty(
    counts: dict[ConditionedCorrespondence, float],
    src_totals: dict[str, float],
) -> dict[ConditionedCorrespondence, UncertaintyEstimate]:
    """Wilson interval on ``P(tgt | src, context)`` for every entry.

    Context-conditioned entries share ``src_totals[src]`` with the
    unconditioned entry, so the denominator is the total observations
    of the source grapheme across all contexts. That matches how the
    posterior is computed at scoring time.
    """
    out: dict[ConditionedCorrespondence, UncertaintyEstimate] = {}
    for key, c in counts.items():
        n = src_totals.get(key.src, 0.0)
        out[key] = wilson_interval(c, n)
    return out


def _commit_splits_for_source(
    src: str,
    observations: list[tuple[str, Context, float]],
    new_counts: dict[ConditionedCorrespondence, float],
    new_src_totals: dict[str, float],
    ln_n: float,
    max_depth: int,
) -> None:
    """Sequential greedy context splitting for a single source grapheme.

    Repeatedly finds the best BIC-improving split on the remaining
    (unsplit) observations, commits it, and continues. This captures
    orthogonal conditioning axes without inflating the entry count
    with redundant partitions: for source ``p`` with observations
    ``{p:8, b:4}``, the first iteration might commit
    ``p → p / pos=initial`` (covering 7 observations), and the second
    iteration runs on the remaining 5 and commits
    ``p → b / prec=vowel`` (covering 4 of them).

    The previous flat multi-predicate approach committed every
    BIC-passing predicate against the full data independently, which
    produced many overlapping entries that had no discriminative
    power (e.g., ``a → a / prec=voiced`` AND ``a → a / prec=voiceless``
    on a source where most preceding segments are one or the other).

    For each committed split, a second-level refinement is attempted
    on the YES partition for multi-feature contexts (e.g., first
    split by ``_[front]``, then inside that split further by
    ``_[front, open]``).
    """
    remaining = list(observations)
    committed_count = 0
    # Cap the number of top-level splits per source to prevent
    # pathological loops on messy data.
    while (
        committed_count < max_depth * 4
        and _observation_weight(remaining) >= MIN_SPLIT_OBSERVATIONS
    ):
        baseline_cost = _group_cost(remaining)
        best_predicate: tuple[str, str, str | None] | None = None
        best_partitions: tuple[list, list] | None = None
        best_delta = DELTA_BIC_THRESHOLD
        for predicate in _candidate_predicates(Context()):
            yes_obs, no_obs = _partition(remaining, predicate)
            if (
                _observation_weight(yes_obs) < MIN_SPLIT_OBSERVATIONS
                or _observation_weight(no_obs) < MIN_SPLIT_OBSERVATIONS
            ):
                continue
            split_cost = _group_cost(yes_obs) + _group_cost(no_obs)
            reduction = baseline_cost - split_cost
            delta_bic = -2.0 * reduction + ln_n
            if delta_bic < best_delta:
                best_delta = delta_bic
                best_predicate = predicate
                best_partitions = (yes_obs, no_obs)
        if best_predicate is None or best_partitions is None:
            break
        yes_obs, no_obs = best_partitions
        yes_context = _apply_predicate(Context(), best_predicate)
        _commit_group(src, yes_obs, yes_context, new_counts)
        # Attempt further refinement on the YES partition for
        # multi-feature contexts.
        _refine_split(
            src=src,
            observations=yes_obs,
            base_context=yes_context,
            depth=1,
            new_counts=new_counts,
            ln_n=ln_n,
            max_depth=max_depth,
        )
        remaining = no_obs
        committed_count += 1


def _refine_split(
    src: str,
    observations: list[tuple[str, Context, float]],
    base_context: Context,
    depth: int,
    new_counts: dict[ConditionedCorrespondence, float],
    ln_n: float,
    max_depth: int,
) -> None:
    """Recursively refine an already-committed conditioned entry.

    Tries to find a sub-split of ``observations`` (which are already
    constrained by ``base_context``) that improves BIC further. If so,
    commits the sub-split as a more specific conditioned entry and
    recurses.

    Uses greedy-best-split within the already-committed branch, since
    at this level we're looking for a single additional axis of
    conditioning. Multiple orthogonal axes at the same level would
    explode combinatorially; the framework handles most real cases
    well enough with one refinement level per branch.
    """
    if depth >= max_depth or _observation_weight(observations) < MIN_SPLIT_OBSERVATIONS:
        return
    baseline_cost = _group_cost(observations)
    best_predicate: tuple[str, str, str | None] | None = None
    best_partitions: tuple[list[tuple[str, Context, float]], list[tuple[str, Context, float]]] | None = None
    best_delta = DELTA_BIC_THRESHOLD
    for predicate in _candidate_predicates(base_context):
        yes_obs, no_obs = _partition(observations, predicate)
        if (
            _observation_weight(yes_obs) < MIN_SPLIT_OBSERVATIONS
            or _observation_weight(no_obs) < MIN_SPLIT_OBSERVATIONS
        ):
            continue
        split_cost = _group_cost(yes_obs) + _group_cost(no_obs)
        reduction = baseline_cost - split_cost
        delta_bic = -2.0 * reduction + ln_n
        if delta_bic < best_delta:
            best_delta = delta_bic
            best_predicate = predicate
            best_partitions = (yes_obs, no_obs)
    if best_predicate is None or best_partitions is None:
        return
    yes_obs, _ = best_partitions
    yes_context = _apply_predicate(base_context, best_predicate)
    _commit_group(src, yes_obs, yes_context, new_counts)
    _refine_split(
        src=src,
        observations=yes_obs,
        base_context=yes_context,
        depth=depth + 1,
        new_counts=new_counts,
        ln_n=ln_n,
        max_depth=max_depth,
    )


def _obs_weight(obs: tuple) -> float:
    """Return the weight of one observation tuple.

    Backward-compatible with older internal tests that pass
    ``(target, Context)`` pairs without an explicit weight.
    """
    if len(obs) >= 3:
        return float(obs[2])
    return 1.0


def _observation_weight(observations: list[tuple[str, Context, float]]) -> float:
    """Return the total confidence-weighted mass of the observations."""
    return sum(_obs_weight(obs) for obs in observations)


def _group_cost(observations: list[tuple[str, Context, float]]) -> float:
    """Negative log-likelihood of a group of (target, context) observations
    under a single unconditioned correspondence for them.

    For each target, estimates P(t | src) = count(t) / count(src) and
    sums -log P over all observations. This is the standard MLE fit
    for a single categorical distribution.
    """
    if not observations:
        return 0.0
    target_counts: dict[str, float] = defaultdict(float)
    for obs in observations:
        target_counts[obs[0]] += _obs_weight(obs)
    total = _observation_weight(observations)
    cost = 0.0
    for _, n in target_counts.items():
        p = n / total
        if p > 0:
            cost += -n * math.log(p)
    return cost


def _candidate_predicates(
    base_context: Context,
) -> list[tuple[str, str, str | None]]:
    """Return candidate split predicates not already in ``base_context``.

    Each predicate is a tuple:

    * ``("following", feature_name, "+")`` for a following-segment feature
    * ``("preceding", feature_name, "+")`` for a preceding-segment feature
    * ``("position", position_name, None)`` for word position
    """
    candidates: list[tuple[str, str, str | None]] = []
    existing_following = {c.feature for c in base_context.following}
    existing_preceding = {c.feature for c in base_context.preceding}
    for slot, feature, value in _SPLIT_FEATURE_INVENTORY:
        if slot == "following" and feature in existing_following:
            continue
        if slot == "preceding" and feature in existing_preceding:
            continue
        candidates.append((slot, feature, value))
    # Position splits: only if the base context has no position set.
    if base_context.position is None:
        for pos in _SPLIT_POSITIONS:
            candidates.append(("position", pos, None))
    return candidates


def _partition(
    observations: list[tuple[str, Context, float]],
    predicate: tuple[str, str, str | None],
) -> tuple[list[tuple[str, Context, float]], list[tuple[str, Context, float]]]:
    """Split the observations by whether the predicate is satisfied."""
    slot, feature, value = predicate
    yes_obs: list[tuple[str, Context, float]] = []
    no_obs: list[tuple[str, Context, float]] = []
    for obs in observations:
        t = obs[0]
        ctx = obs[1]
        weight = _obs_weight(obs)
        if _predicate_holds(ctx, slot, feature, value):
            yes_obs.append((t, ctx, weight))
        else:
            no_obs.append((t, ctx, weight))
    return yes_obs, no_obs


def _predicate_holds(
    ctx: Context,
    slot: str,
    feature: str,
    value: str | None,
) -> bool:
    if slot == "following":
        return any(c.feature == feature and c.value == value for c in ctx.following)
    if slot == "preceding":
        return any(c.feature == feature and c.value == value for c in ctx.preceding)
    if slot == "position":
        return ctx.position == feature
    # Long-range slots. Distance-bounded slots encode the offset
    # in the slot name as ``preceding@<d>`` / ``following@<d>``.
    if slot.startswith("preceding@"):
        try:
            d = int(slot.split("@", 1)[1])
        except ValueError:
            return False
        return any(
            offset == d and c.feature == feature and c.value == value
            for offset, c in ctx.preceding_at_distance
        )
    if slot.startswith("following@"):
        try:
            d = int(slot.split("@", 1)[1])
        except ValueError:
            return False
        return any(
            offset == d and c.feature == feature and c.value == value
            for offset, c in ctx.following_at_distance
        )
    if slot == "somewhere_preceding":
        return any(c.feature == feature and c.value == value for c in ctx.somewhere_preceding)
    if slot == "somewhere_following":
        return any(c.feature == feature and c.value == value for c in ctx.somewhere_following)
    if slot == "same_syllable":
        return any(c.feature == feature and c.value == value for c in ctx.same_syllable)
    if slot == "next_syllable":
        return any(c.feature == feature and c.value == value for c in ctx.next_syllable)
    if slot == "previous_syllable":
        return any(c.feature == feature and c.value == value for c in ctx.previous_syllable)
    return False


def _apply_predicate(
    base_context: Context,
    predicate: tuple[str, str, str | None],
) -> Context:
    """Return a new context with the predicate's constraint added.

    Uses ``dataclasses.replace`` so all unrelated fields (including
    long-range slots) are carried through unchanged. New
    constraints are appended to the relevant slot.
    """
    import dataclasses

    slot, feature, value = predicate
    fc = FeatureConstraint(feature, value or "+")
    if slot == "following":
        return dataclasses.replace(
            base_context, following=base_context.following + (fc,)
        )
    if slot == "preceding":
        return dataclasses.replace(
            base_context, preceding=base_context.preceding + (fc,)
        )
    if slot == "position":
        return dataclasses.replace(base_context, position=feature)
    if slot.startswith("preceding@"):
        d = int(slot.split("@", 1)[1])
        return dataclasses.replace(
            base_context,
            preceding_at_distance=base_context.preceding_at_distance + ((d, fc),),
        )
    if slot.startswith("following@"):
        d = int(slot.split("@", 1)[1])
        return dataclasses.replace(
            base_context,
            following_at_distance=base_context.following_at_distance + ((d, fc),),
        )
    if slot == "somewhere_preceding":
        return dataclasses.replace(
            base_context,
            somewhere_preceding=base_context.somewhere_preceding + (fc,),
        )
    if slot == "somewhere_following":
        return dataclasses.replace(
            base_context,
            somewhere_following=base_context.somewhere_following + (fc,),
        )
    if slot == "same_syllable":
        return dataclasses.replace(
            base_context, same_syllable=base_context.same_syllable + (fc,)
        )
    if slot == "next_syllable":
        return dataclasses.replace(
            base_context, next_syllable=base_context.next_syllable + (fc,)
        )
    if slot == "previous_syllable":
        return dataclasses.replace(
            base_context, previous_syllable=base_context.previous_syllable + (fc,)
        )
    return base_context


def _commit_group(
    src: str,
    observations: list[tuple[str, Context, float]],
    context: Context,
    new_counts: dict[ConditionedCorrespondence, float],
) -> None:
    """Write counts for each observed target in this group as new
    conditioned correspondence entries under the given context."""
    target_counts: dict[str, float] = defaultdict(float)
    for obs in observations:
        target_counts[obs[0]] += _obs_weight(obs)
    for t, n in target_counts.items():
        key = ConditionedCorrespondence(src=src, tgt=t, context=context)
        new_counts[key] = n


def _tonal_aggregation(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
    *,
    max_chunk_size: int,
) -> LearnedModel:
    """One-pass update of the tonal correspondence table.

    Re-aligns the corpus with the post-chunk-promotion model, then counts each
    1-to-1 link's tonal correspondence. A link where both segments
    are untoned contributes to the (None, None) correspondence;
    differently-toned links contribute to whichever tonal pair they
    carry.

    The resulting tonal table is inert for non-tonal corpora (if every
    link is (None, None), the table has one entry that the scoring
    function treats as zero cost).
    """
    from regulae.training import align_corpus

    alignments = align_corpus(corpus, model, max_chunk_size=max_chunk_size)
    counts: dict[TonalCorrespondence, float] = defaultdict(float)
    src_totals: dict[str | None, float] = defaultdict(float)

    any_toned = False
    for alignment, pair_weight in zip(alignments, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        for link in alignment.links:
            if len(link.source_chunk) != 1 or len(link.target_chunk) != 1:
                continue
            src_tone = link.source_chunk[0].tone
            tgt_tone = link.target_chunk[0].tone
            if src_tone is None and tgt_tone is None:
                continue  # untoned — skip entirely
            any_toned = True
            key = TonalCorrespondence(src_tone=src_tone, tgt_tone=tgt_tone)
            counts[key] += pair_weight
            src_totals[src_tone] += pair_weight

    if not any_toned:
        # Non-tonal corpus: leave the tonal table empty.
        return model

    new_tonal_table = TonalCorrespondenceTable(
        counts=dict(counts),
        prior_pseudo_counts={k: 1.0 for k in counts},  # symmetric uniform prior
        src_totals=dict(src_totals),
        uncertainty={
            k: wilson_interval(c, src_totals.get(k.src_tone, 0.0))
            for k, c in counts.items()
        },
    )
    # Use dataclasses.replace so any unrelated fields
    # (cross_dimensional_table, etc.) are preserved from
    # ``model`` without having to mention them explicitly.
    return replace(model, tonal_table=new_tonal_table)


def _long_range_candidate_predicates(
    base_context: Context,
) -> list[tuple[str, str, str | None]]:
    """Return candidate long-range split predicates.

    Only emits predicates that aren't already satisfied by
    ``base_context`` so a refinement step doesn't re-add a
    constraint that's already committed.
    """
    candidates: list[tuple[str, str, str | None]] = []
    for slot in (
        _LONG_RANGE_STRUCTURAL_SLOTS
        + _LONG_RANGE_DISTANCE_SLOTS
        + _LONG_RANGE_EXISTENTIAL_SLOTS
    ):
        for feat in _LONG_RANGE_FEATURES:
            predicate = (slot, feat, "+")
            # Skip if already in base_context.
            if _predicate_holds(base_context, slot, feat, "+"):
                continue
            candidates.append(predicate)
    return candidates


def _long_range_discovery(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
    *,
    max_chunk_size: int,
) -> LearnedModel:
    """Discover long-range context splits for segment correspondences.

    Runs after cross-dimensional discovery. Re-aligns the corpus
    under the current model and looks for splits of the same form
    as immediate-neighbour context discovery, but using
    distance-bounded, existential, and syllable-structural
    predicates.

    Only 1-to-1 link observations are used — multi-segment chunks
    don't carry a well-defined syllable scope (Q7). Splits that
    beat :data:`DELTA_BIC_THRESHOLD` are committed as conditioned
    entries in the segment table, reusing
    :class:`ConditionedCorrespondence` with long-range Context
    fields.

    Existing immediate-neighbour entries stay in place; long-range
    entries are additional, more specific
    conditionings. At scoring time, subset-matching picks the
    most specific entry that fires.
    """
    from regulae.training import align_corpus
    from regulae._em import _replace_segment_table

    alignments = align_corpus(corpus, model, max_chunk_size=max_chunk_size)

    observations: list[tuple[str, str, Context, float]] = []
    for alignment, pair_weight in zip(alignments, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        for link in alignment.links:
            sc, tc = link.source_chunk, link.target_chunk
            if len(sc) == 1 and len(tc) == 1:
                observations.append(
                    (sc[0].grapheme, tc[0].grapheme, link.context, pair_weight)
                )

    if not observations:
        return model

    by_source: dict[str, list[tuple[str, Context, float]]] = defaultdict(list)
    for s, t, c, w in observations:
        by_source[s].append((t, c, w))

    new_counts: dict[ConditionedCorrespondence, float] = dict(model.segment_table.counts)
    new_src_totals: dict[str, float] = dict(model.segment_table.src_totals)
    n_total = sum(w for _, _, _, w in observations)
    ln_n = math.log(n_total)

    for src, obs in by_source.items():
        target_set = {t for t, _, _ in obs}
        if len(target_set) < 2 or _observation_weight(obs) < 4.0:
            continue
        _commit_long_range_splits_for_source(
            src=src,
            observations=obs,
            new_counts=new_counts,
            ln_n=ln_n,
            max_depth=MAX_SPLIT_DEPTH,
        )

    new_segment_table = SegmentCorrespondenceTable(
        counts=new_counts,
        prior_pseudo_counts=model.segment_table.prior_pseudo_counts,
        src_totals=new_src_totals,
        log_normalizers=model.segment_table.log_normalizers,
        uncertainty=_segment_counts_uncertainty(new_counts, new_src_totals),
    )
    return _replace_segment_table(model, new_segment_table)


def _commit_long_range_splits_for_source(
    src: str,
    observations: list[tuple[str, Context, float]],
    new_counts: dict[ConditionedCorrespondence, float],
    ln_n: float,
    max_depth: int,
) -> None:
    """Sequential greedy long-range context splitting for one source.

    Mirrors :func:`_commit_splits_for_source` but uses
    long-range candidate predicates and searches against the full
    observation set without subtracting out immediate-neighbour commits
    (the long-range entries live in parallel — they're more
    specific overlays that fire only when their predicate holds).
    """
    remaining = list(observations)
    committed_count = 0
    while (
        committed_count < max_depth * 4
        and _observation_weight(remaining) >= _LONG_RANGE_MIN_SPLIT_OBS
    ):
        baseline_cost = _group_cost(remaining)
        best_predicate: tuple[str, str, str | None] | None = None
        best_partitions: tuple[list, list] | None = None
        best_delta = _LONG_RANGE_DELTA_BIC_THRESHOLD
        for predicate in _long_range_candidate_predicates(Context()):
            yes_obs, no_obs = _partition(remaining, predicate)
            if (
                _observation_weight(yes_obs) < _LONG_RANGE_MIN_SPLIT_OBS
                or _observation_weight(no_obs) < _LONG_RANGE_MIN_SPLIT_OBS
            ):
                continue
            # Dominance filter: the YES group must be carried by a
            # single target outcome, otherwise this is a grab-bag
            # anomaly bucket rather than a real rule.
            yes_target_counts: dict[str, float] = defaultdict(float)
            for t, _, weight in yes_obs:
                yes_target_counts[t] += weight
            yes_mode = max(yes_target_counts.values())
            yes_weight = _observation_weight(yes_obs)
            if yes_weight <= 0.0 or yes_mode / yes_weight < _LONG_RANGE_MIN_DOMINANT_FRACTION:
                continue
            split_cost = _group_cost(yes_obs) + _group_cost(no_obs)
            reduction = baseline_cost - split_cost
            delta_bic = -2.0 * reduction + ln_n
            if delta_bic < best_delta:
                best_delta = delta_bic
                best_predicate = predicate
                best_partitions = (yes_obs, no_obs)
        if best_predicate is None or best_partitions is None:
            break
        yes_obs, no_obs = best_partitions
        yes_context = _apply_predicate(Context(), best_predicate)
        _commit_group(src, yes_obs, yes_context, new_counts)
        remaining = no_obs
        committed_count += 1
