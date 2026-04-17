"""Residual-mutual-information anomaly detection.

Computes residual mutual information between feature pairs in
training-corpus alignments, conditional on what the current
model already explains. Returns a ranked list of
:class:`PatternHypothesis` objects that the cross-dimensional
discovery loop consumes as a candidate pool for BIC-driven commits.

**Pure diagnostic**: this module never mutates the model or the
corpus. The training pipeline calls it from the cross-dimensional
discovery stage; outside of that, it can be used as a standalone
diagnostic.
"""

import math
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import merkmal

from regulae.model import LearnedModel
from regulae.search import _CONTEXT_FEATURES, align_forms
from regulae.types import Form, Segment


#: Hypothesis kinds reported by :func:`find_residual_patterns`.
#: - ``cross_dimensional_tonogenesis`` — segmental feature at one
#:   position conditions a tonal value at another position. The
#:   canonical tonogenesis pattern.
#:
#: Long-range discovery uses a direct split-loop on the training
#: observations (see ``training._long_range_discovery``) rather
#: than the anomaly-detection diagnostic, so the diagnostic
#: pipeline is the canonical path only for cross-dimensional
#: patterns.
HYPOTHESIS_KINDS: tuple[str, ...] = (
    "cross_dimensional_tonogenesis",
)


@dataclass(frozen=True)
class PatternHypothesis:
    """A residual-mutual-information-based hypothesis about
    unmodelled structure in the training corpus.

    * ``kind``: one of :data:`HYPOTHESIS_KINDS`.
    * ``src_feature``: the source-side feature name
      (``"voiced"``, ``"front"``, etc.) or grapheme string that
      the hypothesis predicts from. ``"tone"`` is a special
      feature name that checks ``Segment.tone`` rather
      than a merkmal feature; in that case ``src_value`` holds
      the tone value to match (e.g. ``"2"``).
    * ``src_value``: the value ``src_feature`` must take for the
      hypothesis to fire. Defaults to ``"+"`` for segmental
      features (where presence is binary). For tonal source
      predictors (``src_feature == "tone"``) this holds the
      source tone string to match.
    * ``src_position_spec``: a string describing where on the
      source side the feature must hold, e.g. ``"relative_-1"``
      (segment immediately to the left), ``"relative_0"`` (the
      link position itself), ``"same_syllable"``, etc.
    * ``tgt_feature_or_value``: the target-side outcome. For
      cross-dimensional tonogenesis this is a tone value string
      like ``"4"``; for long-range grapheme hypotheses it's a
      feature name.
    * ``tgt_dimension``: which dimension of the target segment
      carries the outcome — ``"grapheme"``, ``"tone"``,
      ``"length"``, ``"stress"``.
    * ``tgt_position_offset``: relative position on the target
      side, or ``None`` for structural position specs.
    * ``residual_mi``: observed mutual information between the
      binary predictor (source feature present / absent at the
      specified position) and the binary outcome (target
      feature/value present / absent at the specified target
      position), in nats.
    * ``null_percentile``: the observed MI's rank in a
      permutation null distribution (shuffle the target side
      and recompute). A value ≥0.95 means "significantly above
      chance".
    * ``supporting_observations``: how many training-corpus link
      observations contributed to the MI calculation.
    """

    kind: str
    src_feature: str
    src_position_spec: str
    tgt_feature_or_value: str
    tgt_dimension: str
    tgt_position_offset: int | None
    residual_mi: float
    null_percentile: float
    supporting_observations: int
    src_value: str = "+"
    #: Joint-predictor extension: when set, the hypothesis
    #: describes a conjunctive rule "src_feature at src_position_spec
    #: AND src_feature_2 at src_position_spec_2 → target". None for
    #: single-predictor hypotheses.
    src_feature_2: str | None = None
    src_value_2: str | None = None
    src_position_spec_2: str | None = None


@dataclass(frozen=True)
class _Observation:
    """One 1-to-1 link observation harvested from a trained
    alignment. Internal bookkeeping type."""

    src_form: Form
    src_pos: int
    tgt_form: Form
    tgt_pos: int


# ----- public API --------------------------------------------------------


def find_residual_patterns(
    corpus: Sequence[tuple[Form, Form]],
    model: LearnedModel,
    *,
    min_observations: int = 5,
    null_permutations: int = 100,
    null_percentile_threshold: float = 0.95,
    feature_system: str | None = None,
    random_seed: int = 0,
    kinds: Collection[str] | None = None,
) -> list[PatternHypothesis]:
    """Rank suspicious feature-pair combinations by residual MI.

    For each candidate (``kind``, source-side predictor, target-
    side outcome), computes the mutual information between the
    predictor and the outcome across all 1-to-1 link observations
    in the corpus under the current model. Candidates with MI
    significantly above a permutation null are returned, sorted
    by MI descending.

    Arguments:

    * ``corpus``: a pair corpus. The model must be compatible
      with these pairs — normally, this is the corpus used to
      train ``model``.
    * ``model``: a trained :class:`LearnedModel`. The model's
      alignments are used to determine which link positions
      count as 1-to-1 observations.
    * ``min_observations``: hypotheses backed by fewer than this
      many observations are skipped.
    * ``null_permutations``: number of permutation shuffles used
      to estimate the null MI distribution per candidate.
    * ``null_percentile_threshold``: minimum percentile against
      the null distribution for a hypothesis to be reported.
      Default 0.95 (p ≤ 0.05 one-sided).
    * ``feature_system``: merkmal feature system. Defaults to
      ``model.feature_system``.
    * ``random_seed``: seeds the shuffle generator so results
      are deterministic across runs.
    * ``kinds``: subset of :data:`HYPOTHESIS_KINDS` to compute.
      ``None`` means all kinds.

    Returns a list of :class:`PatternHypothesis` sorted by
    ``residual_mi`` descending. Empty list if no candidates pass
    the filters.

    **Current scope**: ``cross_dimensional_tonogenesis`` is the
    only supported kind. Long-range rules are discovered through
    a direct split-loop in the training pipeline instead.
    """
    fs = feature_system or model.feature_system
    if kinds is None:
        selected_kinds = set(HYPOTHESIS_KINDS)
    else:
        unknown = set(kinds) - set(HYPOTHESIS_KINDS)
        if unknown:
            raise ValueError(f"unknown hypothesis kinds: {sorted(unknown)}")
        selected_kinds = set(kinds)

    observations = _collect_observations(corpus, model)
    if not observations:
        return []

    rng = random.Random(random_seed)
    hypotheses: list[PatternHypothesis] = []

    if "cross_dimensional_tonogenesis" in selected_kinds:
        hypotheses.extend(
            _find_cross_dimensional_tonogenesis(
                observations,
                fs,
                rng,
                min_observations=min_observations,
                null_permutations=null_permutations,
                null_percentile_threshold=null_percentile_threshold,
            )
        )

    hypotheses.sort(
        key=lambda h: (
            -h.residual_mi,
            h.kind,
            h.src_feature,
            h.src_position_spec,
            h.tgt_feature_or_value,
        )
    )
    return hypotheses


# ----- candidate enumeration: cross-dimensional tonogenesis --------------


def _find_cross_dimensional_tonogenesis(
    observations: list[_Observation],
    feature_system: str,
    rng: random.Random,
    *,
    min_observations: int,
    null_permutations: int,
    null_percentile_threshold: float,
) -> list[PatternHypothesis]:
    """Enumerate segmental→tonal candidate hypotheses and filter
    by observed MI vs. permutation null.

    Candidate shape: for each source offset in ``{-1, 0, +1}``,
    each feature in the standard inventory, each target offset
    in ``{0, +1, +2}``, and each tone value that appears in the
    target side of the corpus, build a binary predictor "source
    has this feature at this offset" and a binary outcome
    "target has this tone value at this offset", then compute
    their mutual information.
    """
    # Collect all tone values that appear on the target side.
    # Without observed tones there is nothing to predict.
    tgt_tone_values: set[str] = set()
    for obs in observations:
        for seg in obs.tgt_form.segments:
            if seg.tone is not None:
                tgt_tone_values.add(seg.tone)
    if not tgt_tone_values:
        return []
    sorted_tgt_tones = sorted(tgt_tone_values)

    # Also collect source tone values so we can enumerate
    # "source tone X predicts target tone Y" candidates alongside
    # the segmental-feature ones. Without source-tone predictors,
    # corpora where the real conditioning is partly tonal remain
    # out of reach for the commit loop.
    src_tone_values: set[str] = set()
    for obs in observations:
        for seg in obs.src_form.segments:
            if seg.tone is not None:
                src_tone_values.add(seg.tone)
    sorted_src_tones = sorted(src_tone_values)

    hypotheses: list[PatternHypothesis] = []

    # Segmental source predictors.
    for src_offset in (-1, 0, 1):
        for feature in _CONTEXT_FEATURES:
            for tgt_offset in (0, 1, 2):
                for tone_value in sorted_tgt_tones:
                    xs: list[int] = []
                    ys: list[int] = []
                    for obs in observations:
                        x = _segment_has_feature(
                            obs.src_form,
                            obs.src_pos + src_offset,
                            feature,
                            feature_system,
                        )
                        y_tone = _segment_tone(
                            obs.tgt_form, obs.tgt_pos + tgt_offset
                        )
                        if x is None or y_tone is None:
                            continue
                        xs.append(1 if x else 0)
                        ys.append(1 if y_tone == tone_value else 0)

                    if len(xs) < min_observations:
                        continue

                    observed_mi = _binary_mutual_information(xs, ys)
                    if observed_mi <= 0.0:
                        continue

                    null_scores = _permutation_null_mi(
                        xs, ys, rng=rng, permutations=null_permutations
                    )
                    percentile = _fraction_below(observed_mi, null_scores)
                    if percentile < null_percentile_threshold:
                        continue

                    hypotheses.append(
                        PatternHypothesis(
                            kind="cross_dimensional_tonogenesis",
                            src_feature=feature,
                            src_value="+",
                            src_position_spec=_format_offset(src_offset),
                            tgt_feature_or_value=tone_value,
                            tgt_dimension="tone",
                            tgt_position_offset=tgt_offset,
                            residual_mi=observed_mi,
                            null_percentile=percentile,
                            supporting_observations=len(xs),
                        )
                    )

    # Source-tone predictors. Same enumeration shape but
    # with "does src segment at offset have tone=V" as the binary
    # predictor. Skips the identity check src=tgt (zero offset +
    # same tone) because that's trivially predictive and doesn't
    # encode a real cross-dimensional rule.
    for src_offset in (-1, 0, 1):
        for src_tone_value in sorted_src_tones:
            for tgt_offset in (0, 1, 2):
                for tgt_tone_value in sorted_tgt_tones:
                    # Skip "src tone X at same position predicts
                    # tgt tone X at same position" — that's the
                    # identity test that the tonal correspondence
                    # table already captures.
                    if (
                        src_offset == 0
                        and tgt_offset == 0
                        and src_tone_value == tgt_tone_value
                    ):
                        continue
                    xs = []
                    ys = []
                    for obs in observations:
                        src_tone = _segment_tone(
                            obs.src_form, obs.src_pos + src_offset
                        )
                        tgt_tone = _segment_tone(
                            obs.tgt_form, obs.tgt_pos + tgt_offset
                        )
                        if src_tone is None or tgt_tone is None:
                            continue
                        xs.append(1 if src_tone == src_tone_value else 0)
                        ys.append(1 if tgt_tone == tgt_tone_value else 0)

                    if len(xs) < min_observations:
                        continue

                    observed_mi = _binary_mutual_information(xs, ys)
                    if observed_mi <= 0.0:
                        continue

                    null_scores = _permutation_null_mi(
                        xs, ys, rng=rng, permutations=null_permutations
                    )
                    percentile = _fraction_below(observed_mi, null_scores)
                    if percentile < null_percentile_threshold:
                        continue

                    hypotheses.append(
                        PatternHypothesis(
                            kind="cross_dimensional_tonogenesis",
                            src_feature="tone",
                            src_value=src_tone_value,
                            src_position_spec=_format_offset(src_offset),
                            tgt_feature_or_value=tgt_tone_value,
                            tgt_dimension="tone",
                            tgt_position_offset=tgt_offset,
                            residual_mi=observed_mi,
                            null_percentile=percentile,
                            supporting_observations=len(xs),
                        )
                    )

    # Joint predictors. Restricted to (segmental feature,
    # source tone value) pairs — the cross-source-dimension
    # case. Joints are only
    # admitted when they strictly improve over EITHER component
    # alone; otherwise they'd commit "passenger" predictors where
    # one component does all the predictive work and the other
    # rides along. The margin is the non-interaction guard.
    _JOINT_MARGIN = 0.05
    if sorted_src_tones:
        for seg_offset in (-1, 0, 1):
            for seg_feature in _CONTEXT_FEATURES:
                for tone_offset in (-1, 0, 1):
                    for src_tone_value in sorted_src_tones:
                        for tgt_offset in (0, 1, 2):
                            for tgt_tone_value in sorted_tgt_tones:
                                xs = []
                                xs_a: list[int] = []
                                xs_b: list[int] = []
                                ys = []
                                for obs in observations:
                                    has_feat = _segment_has_feature(
                                        obs.src_form,
                                        obs.src_pos + seg_offset,
                                        seg_feature,
                                        feature_system,
                                    )
                                    src_tone = _segment_tone(
                                        obs.src_form,
                                        obs.src_pos + tone_offset,
                                    )
                                    tgt_tone = _segment_tone(
                                        obs.tgt_form,
                                        obs.tgt_pos + tgt_offset,
                                    )
                                    if (
                                        has_feat is None
                                        or src_tone is None
                                        or tgt_tone is None
                                    ):
                                        continue
                                    a = 1 if has_feat else 0
                                    b = 1 if src_tone == src_tone_value else 0
                                    xs_a.append(a)
                                    xs_b.append(b)
                                    xs.append(1 if a and b else 0)
                                    ys.append(
                                        1 if tgt_tone == tgt_tone_value else 0
                                    )

                                if len(xs) < min_observations:
                                    continue
                                if sum(xs) < min_observations:
                                    continue

                                # Non-interaction guard: the joint
                                # must strictly improve over both
                                # component predictors. ``conf_*``
                                # is the confidence of the rule
                                # built from just that predictor.
                                conf_joint = sum(
                                    1 for x, y in zip(xs, ys) if x and y
                                ) / max(sum(xs), 1)
                                n_a = sum(xs_a)
                                n_b = sum(xs_b)
                                conf_a = (
                                    sum(
                                        1 for a, y in zip(xs_a, ys) if a and y
                                    )
                                    / n_a
                                    if n_a
                                    else 0.0
                                )
                                conf_b = (
                                    sum(
                                        1 for b, y in zip(xs_b, ys) if b and y
                                    )
                                    / n_b
                                    if n_b
                                    else 0.0
                                )
                                if (
                                    conf_joint < conf_a + _JOINT_MARGIN
                                    or conf_joint < conf_b + _JOINT_MARGIN
                                ):
                                    continue

                                observed_mi = _binary_mutual_information(xs, ys)
                                if observed_mi <= 0.0:
                                    continue

                                null_scores = _permutation_null_mi(
                                    xs,
                                    ys,
                                    rng=rng,
                                    permutations=null_permutations,
                                )
                                percentile = _fraction_below(
                                    observed_mi, null_scores
                                )
                                if percentile < null_percentile_threshold:
                                    continue

                                hypotheses.append(
                                    PatternHypothesis(
                                        kind="cross_dimensional_tonogenesis",
                                        src_feature=seg_feature,
                                        src_value="+",
                                        src_position_spec=_format_offset(
                                            seg_offset
                                        ),
                                        src_feature_2="tone",
                                        src_value_2=src_tone_value,
                                        src_position_spec_2=_format_offset(
                                            tone_offset
                                        ),
                                        tgt_feature_or_value=tgt_tone_value,
                                        tgt_dimension="tone",
                                        tgt_position_offset=tgt_offset,
                                        residual_mi=observed_mi,
                                        null_percentile=percentile,
                                        supporting_observations=len(xs),
                                    )
                                )

    return hypotheses


# ----- observation harvesting --------------------------------------------


def _collect_observations(
    corpus: Sequence[tuple[Form, Form]],
    model: LearnedModel,
) -> list[_Observation]:
    """Run each training pair through the current model's
    alignment and collect 1-to-1 link positions as observations.

    Multi-segment chunks, gaps, and anything else that isn't a
    1-to-1 link are ignored — they aren't usable for
    feature-pair MI calculation.
    """
    observations: list[_Observation] = []
    for src_form, tgt_form in corpus:
        alignment = align_forms(src_form, tgt_form, model=model)
        src_pos = 0
        tgt_pos = 0
        for link in alignment.links:
            src_len = len(link.source_chunk)
            tgt_len = len(link.target_chunk)
            if src_len == 1 and tgt_len == 1:
                observations.append(
                    _Observation(
                        src_form=src_form,
                        src_pos=src_pos,
                        tgt_form=tgt_form,
                        tgt_pos=tgt_pos,
                    )
                )
            src_pos += src_len
            tgt_pos += tgt_len
    return observations


# ----- feature accessors -------------------------------------------------


def _segment_has_feature(
    form: Form,
    pos: int,
    feature: str,
    feature_system: str,
) -> bool | None:
    """Return True if the segment at ``pos`` has ``feature``,
    False if it doesn't, or None if the position is out of
    bounds or the grapheme is unknown to the feature system.

    The None sentinel lets the MI calculation skip observations
    it can't evaluate (edge effects near word boundaries,
    unknown graphemes) without biasing the count.
    """
    if pos < 0 or pos >= len(form.segments):
        return None
    grapheme = form.segments[pos].grapheme
    if not grapheme:
        return None
    try:
        features = merkmal.get_features(grapheme, system=feature_system)
    except KeyError:
        features = None
    if features is None:
        return None
    return feature in features


def _segment_tone(form: Form, pos: int) -> str | None:
    """Return the tone annotation on the segment at ``pos``, or
    None if the position is out of bounds or the segment has
    no tone annotation."""
    if pos < 0 or pos >= len(form.segments):
        return None
    return form.segments[pos].tone


# ----- mutual information + null -----------------------------------------


def _binary_mutual_information(xs: list[int], ys: list[int]) -> float:
    """Mutual information between two parallel binary sequences,
    in nats.

    MI(X; Y) = Σ P(x, y) * log(P(x, y) / (P(x) * P(y)))

    Returns 0.0 for empty inputs or fully-independent data.
    """
    n = len(xs)
    if n == 0:
        return 0.0
    # 2×2 joint counts
    c00 = c01 = c10 = c11 = 0
    for x, y in zip(xs, ys):
        if x == 0:
            if y == 0:
                c00 += 1
            else:
                c01 += 1
        else:
            if y == 0:
                c10 += 1
            else:
                c11 += 1
    p00 = c00 / n
    p01 = c01 / n
    p10 = c10 / n
    p11 = c11 / n
    px0 = p00 + p01
    px1 = p10 + p11
    py0 = p00 + p10
    py1 = p01 + p11
    mi = 0.0
    for p_xy, p_x, p_y in (
        (p00, px0, py0),
        (p01, px0, py1),
        (p10, px1, py0),
        (p11, px1, py1),
    ):
        if p_xy > 0.0 and p_x > 0.0 and p_y > 0.0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))
    return mi


def _permutation_null_mi(
    xs: list[int],
    ys: list[int],
    *,
    rng: random.Random,
    permutations: int,
) -> list[float]:
    """Compute the permutation null distribution of MI by
    shuffling the target side and recomputing MI.

    Returns a sorted list of null MI values. Deterministic
    given the supplied ``rng``.
    """
    scores: list[float] = []
    # Work on a local copy to avoid mutating the input.
    shuffled = list(ys)
    for _ in range(permutations):
        rng.shuffle(shuffled)
        scores.append(_binary_mutual_information(xs, shuffled))
    scores.sort()
    return scores


def _fraction_below(value: float, sorted_scores: list[float]) -> float:
    """Return the fraction of ``sorted_scores`` strictly less
    than ``value``. Used to compute the percentile of an
    observed MI against the permutation null.
    """
    if not sorted_scores:
        return 0.0
    below = 0
    for s in sorted_scores:
        if s < value:
            below += 1
        else:
            break
    return below / len(sorted_scores)


def _format_offset(offset: int) -> str:
    """Canonical string representation of a position offset, for
    the ``src_position_spec`` field of :class:`PatternHypothesis`.
    """
    if offset == 0:
        return "relative_0"
    return f"relative_{offset:+d}"
