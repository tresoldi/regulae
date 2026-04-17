import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import merkmal

if TYPE_CHECKING:
    from regulae.anomaly import PatternHypothesis

from regulae._discovery import DELTA_BIC_THRESHOLD
from regulae.config import BICConfig
from regulae.model import (
    CrossDimensionalLink,
    CrossDimensionalLinkTable,
    LearnedModel,
)
from regulae.search import align_forms, alignment_cost
from regulae.uncertainty import wilson_interval
from regulae.types import (
    FeatureConstraint,
    FeatureName,
    FeatureValue,
    Form,
)


#: Maximum number of sequential-greedy iterations cross-dimensional
#: discovery will run before stopping. In each iteration it re-runs anomaly
#: detection, commits the best hypothesis, and re-evaluates the
#: residuals. In practice the loop terminates naturally after 1-2
#: iterations on typical corpora; the cap is a safety valve
#: against pathological inputs (e.g. data where every candidate
#: just barely passes BIC and the loop keeps finding new ones).
_CROSS_DIM_MAX_ITERATIONS: int = 5

#: Minimum support for a committed cross-dimensional rule.
#: Without this, the commit loop admits count=1/109-style noise
#: on corpora with many rare tonal outcomes. BIC alone is a soft
#: filter and Dirichlet-smoothed LLR on a 1/N rule still scores
#: slightly negative, so additional explicit floors are needed.
_CROSS_DIM_MIN_RULE_COUNT: int = 3
_CROSS_DIM_MIN_RULE_CONFIDENCE: float = 0.5


def _cross_dimensional_discovery(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
    *,
    random_seed: int = 0,
    bic_config: BICConfig | None = None,
) -> LearnedModel:
    """Discover cross-dimensional correspondence rules from
    residual mutual information in the training corpus.

    Runs after tonal aggregation and consumes the post-tonal
    model. Calls
    :func:`regulae.anomaly.find_residual_patterns` with
    ``kinds={"cross_dimensional_tonogenesis"}`` to obtain a
    ranked list of candidate :class:`PatternHypothesis` objects
    backed by residual-MI evidence that beats a permutation
    null. Each surviving hypothesis is tested via a BIC
    comparison against the model's log-likelihood on the
    training corpus; the single best-scoring hypothesis (lowest
    ΔBIC, must beat :data:`DELTA_BIC_THRESHOLD`) is committed
    to ``model.cross_dimensional_table``. The loop then
    re-runs anomaly detection on the updated model and tries
    to commit the next best hypothesis, up to
    :data:`_CROSS_DIM_MAX_ITERATIONS` iterations.

    Sequential-greedy with re-ranking (Q E alternative 4): the
    expensive permutation-null recomputation happens at most
    once per iteration, not once per candidate. In practice on
    a clean-signal tonogenesis fixture the loop commits one
    rule and then exits because the re-ranked hypothesis list
    is empty on iteration 2.

    The BIC likelihood delta is computed via alternative 1 from
    Q B: re-score the entire training corpus under the trial
    model (with the candidate rule added to the table) and
    subtract from the baseline cost. This is exact — no
    closed-form approximation — and matches how per-pair and
    multi-lect context discovery compute their commit deltas.
    """
    from regulae.anomaly import find_residual_patterns

    if bic_config is None:
        bic_config = BICConfig()
    max_iterations = bic_config.cross_dim_max_iterations
    min_rule_count = bic_config.cross_dim_min_rule_count
    min_rule_confidence = bic_config.cross_dim_min_rule_confidence
    delta_threshold = bic_config.delta_bic_threshold

    current_model = model
    for _iteration in range(max_iterations):
        hypotheses = find_residual_patterns(
            corpus,
            current_model,
            kinds={"cross_dimensional_tonogenesis"},
            random_seed=random_seed,
        )
        if not hypotheses:
            break

        # Baseline: total corpus cost under the current model.
        baseline_cost = _total_corpus_cost(corpus, pair_weights, current_model)
        n_obs = _count_1to1_observations(corpus, pair_weights, current_model)
        if n_obs <= 0:
            break
        ln_n = math.log(n_obs)

        # Build a set of already-committed rule signatures so the
        # loop doesn't re-commit a rule it committed in an earlier
        # iteration. find_residual_patterns doesn't know about the
        # cross_dimensional_table — it works on residual MI from
        # the tonal/segmental tables — so a committed rule's
        # hypothesis will keep re-surfacing, and we need to skip
        # it explicitly at commit time.
        committed_signatures = {
            _rule_signature(entry)
            for entry in current_model.cross_dimensional_table.entries
        }

        # Evaluate every hypothesis and track the best.
        best_link: CrossDimensionalLink | None = None
        best_delta_bic: float = delta_threshold
        for hyp in hypotheses:
            trial_link = _hypothesis_to_cross_dimensional_link(
                hyp, corpus, pair_weights, current_model
            )
            if trial_link is None:
                continue  # couldn't count supporting observations
            if _rule_signature(trial_link) in committed_signatures:
                continue  # already committed in an earlier iteration
            # Explicit support floors. BIC alone
            # admits count=1/N "anomaly" rules that aren't real
            # phonological patterns; require both a minimum absolute
            # count and a minimum confidence to commit.
            if trial_link.count < min_rule_count:
                continue
            if trial_link.confidence < min_rule_confidence:
                continue

            trial_entries = (
                current_model.cross_dimensional_table.entries + (trial_link,)
            )
            trial_model = replace(
                current_model,
                cross_dimensional_table=CrossDimensionalLinkTable(
                    entries=trial_entries
                ),
            )
            trial_cost = _total_corpus_cost(corpus, pair_weights, trial_model)
            reduction = baseline_cost - trial_cost
            # ΔBIC = -2 * (log_L_trial - log_L_baseline) + k * ln(N)
            # where k is the number of new parameters. A single-
            # predictor rule adds 1 parameter; a joint-predictor
            # rule adds 2. The extra parameter penalty
            # is what naturally keeps joint rules from winning
            # when a single-predictor rule already fits the data.
            k_params = 2 if trial_link.src_feature_2 is not None else 1
            delta_bic = -2.0 * reduction + k_params * ln_n
            if delta_bic < best_delta_bic:
                best_delta_bic = delta_bic
                best_link = trial_link

        if best_link is None:
            break

        # Commit the best hypothesis and continue.
        current_model = replace(
            current_model,
            cross_dimensional_table=CrossDimensionalLinkTable(
                entries=(
                    current_model.cross_dimensional_table.entries
                    + (best_link,)
                )
            ),
        )

    # Post-commit dual-framing dedup. Greedy
    # commit produces rules in multiple positional framings that
    # describe the same underlying pattern — e.g. the clean
    # fixture commits both ``voiced@-1 → tone=4@+0`` and
    # ``voiced@0 → tone=4@+1``, which are "the same rule shifted
    # by one link position". Collapse these to a single canonical
    # entry (lowest src_offset, i.e. leftmost framing).
    deduped = _dedup_dual_framings(
        current_model.cross_dimensional_table.entries
    )
    if len(deduped) != len(current_model.cross_dimensional_table.entries):
        current_model = replace(
            current_model,
            cross_dimensional_table=CrossDimensionalLinkTable(entries=deduped),
        )
    return current_model


def _dedup_dual_framings(
    entries: tuple["CrossDimensionalLink", ...],
) -> tuple["CrossDimensionalLink", ...]:
    """Collapse committed cross-dim rules that describe the same
    underlying pattern from different link-position perspectives.

    Two rules are duals iff they share:

    * ``src_feature`` (feature name + value)
    * ``tgt_dimension`` (e.g. ``"tone"``)
    * ``tgt_value``
    * the *delta* between target offset and source offset —
      i.e. ``tgt_position_offset - parse(src_position_spec)``
      is the same

    Under that grouping, keep the rule with the smallest (most
    negative) src_offset — the most canonical "leftmost" framing.
    Ties broken by keeping the first-committed entry.
    """
    GroupKey = tuple[
        str, str, str, str, int, str | None, str | None, int | None
    ]
    groups: dict[GroupKey, list[tuple[int, int, CrossDimensionalLink]]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        src_off = _parse_relative_position_offset(entry.src_position)
        delta = entry.tgt_position_offset - src_off
        key: GroupKey
        if entry.src_feature_2 is None or entry.src_position_2 is None:
            key = (
                entry.src_feature.feature,
                entry.src_feature.value,
                entry.tgt_dimension,
                entry.tgt_value,
                delta,
                None,
                None,
                None,
            )
        else:
            src_off_2 = _parse_relative_position_offset(entry.src_position_2)
            delta_2 = entry.tgt_position_offset - src_off_2
            key = (
                entry.src_feature.feature,
                entry.src_feature.value,
                entry.tgt_dimension,
                entry.tgt_value,
                delta,
                entry.src_feature_2.feature,
                entry.src_feature_2.value,
                delta_2,
            )
        groups[key].append((src_off, idx, entry))
    # Rebuild in original order, keeping only the chosen
    # representative per group.
    chosen_indices: set[int] = set()
    for members in groups.values():
        members.sort(key=lambda m: (m[0], m[1]))
        chosen_indices.add(members[0][1])
    return tuple(e for i, e in enumerate(entries) if i in chosen_indices)


def _total_corpus_cost(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
) -> float:
    """Return the sum of per-pair alignment costs for the whole
    corpus under ``model``.

    Used as the cross-dimensional BIC likelihood baseline. The cost
    includes the cross-dimensional overlay so that when a trial
    rule is present in the model, its effect is captured.
    """
    total = 0.0
    for (src_form, tgt_form), pair_weight in zip(corpus, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        alignment = align_forms(src_form, tgt_form, model=model)
        total += pair_weight * alignment_cost(alignment, model=model)
    return total


def _count_1to1_observations(
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
) -> float:
    """Return the total number of 1-to-1 link observations across
    the corpus under ``model``.

    Used as the ``N`` in the cross-dimensional BIC formula ``ln(N)``.
    """
    count = 0.0
    for (src_form, tgt_form), pair_weight in zip(corpus, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        alignment = align_forms(src_form, tgt_form, model=model)
        for link in alignment.links:
            if len(link.source_chunk) == 1 and len(link.target_chunk) == 1:
                count += pair_weight
    return count


def _hypothesis_to_cross_dimensional_link(
    hypothesis: "PatternHypothesis",
    corpus: Sequence[tuple[Form, Form]],
    pair_weights: Sequence[float],
    model: LearnedModel,
) -> CrossDimensionalLink | None:
    """Build a :class:`CrossDimensionalLink` from a
    :class:`PatternHypothesis` by walking the training corpus
    under the current model and counting actual rule matches.

    Returns ``None`` if no supporting observations can be found
    — in that case the hypothesis cannot be turned into a
    committed rule regardless of its residual MI.

    The function counts:

    * ``src_count`` — number of 1-to-1 link observations where
      the source feature holds at the source position.
    * ``count`` — number of those where the target dimension
      also carries the hypothesis's target value at the target
      offset.

    Both are floats for consistency with the rest of the model's
    count fields.
    """
    # Parse the src position spec and target offset from the
    # hypothesis. The hypothesis's src_position_spec is the
    # canonical form ``relative_-1`` / ``relative_0`` /
    # ``relative_+1``.
    src_offset = _parse_relative_position_offset(hypothesis.src_position_spec)
    tgt_offset = hypothesis.tgt_position_offset or 0
    feature_name = hypothesis.src_feature
    src_value = hypothesis.src_value
    tgt_value = hypothesis.tgt_feature_or_value
    tgt_dimension = hypothesis.tgt_dimension
    feature_system = model.feature_system

    # Joint-predictor support. A hypothesis with
    # ``src_feature_2`` set describes a conjunctive rule.
    is_joint = (
        hypothesis.src_feature_2 is not None
        and hypothesis.src_position_spec_2 is not None
    )
    if is_joint:
        assert hypothesis.src_position_spec_2 is not None  # narrowed by is_joint
        src_offset_2: int | None = _parse_relative_position_offset(
            hypothesis.src_position_spec_2
        )
        feature_name_2: str | None = hypothesis.src_feature_2
        src_value_2: str | None = hypothesis.src_value_2 or "+"
    else:
        src_offset_2 = None
        feature_name_2 = None
        src_value_2 = None

    def _predicate_holds(
        seg_form: Form,
        seg_idx: int,
        name: str,
        value: str,
    ) -> bool:
        if seg_idx < 0 or seg_idx >= len(seg_form.segments):
            return False
        seg = seg_form.segments[seg_idx]
        if name == "tone":
            return seg.tone == value
        if not seg.grapheme:
            return False
        try:
            features = merkmal.get_features(seg.grapheme, system=feature_system)
        except KeyError:
            return False
        return bool(features is not None and name in features)

    src_count = 0.0
    count = 0.0
    for (src_form, tgt_form), pair_weight in zip(corpus, pair_weights, strict=True):
        if pair_weight <= 0.0:
            continue
        alignment = align_forms(src_form, tgt_form, model=model)
        link_src_pos = 0
        link_tgt_pos = 0
        for link in alignment.links:
            src_len = len(link.source_chunk)
            tgt_len = len(link.target_chunk)
            if src_len == 1 and tgt_len == 1:
                holds = _predicate_holds(
                    src_form,
                    link_src_pos + src_offset,
                    feature_name,
                    src_value,
                )
                if (
                    holds
                    and is_joint
                    and src_offset_2 is not None
                    and feature_name_2 is not None
                    and src_value_2 is not None
                ):
                    holds = _predicate_holds(
                        src_form,
                        link_src_pos + src_offset_2,
                        feature_name_2,
                        src_value_2,
                    )
                if holds:
                    src_count += pair_weight
                    tgt_idx = link_tgt_pos + tgt_offset
                    if 0 <= tgt_idx < len(tgt_form.segments):
                        tgt_seg = tgt_form.segments[tgt_idx]
                        actual: str | None
                        if tgt_dimension == "tone":
                            actual = tgt_seg.tone
                        elif tgt_dimension == "length":
                            actual = tgt_seg.length
                        elif tgt_dimension == "stress":
                            actual = tgt_seg.stress
                        else:
                            actual = None
                        if actual == tgt_value:
                            count += pair_weight
            link_src_pos += src_len
            link_tgt_pos += tgt_len

    if src_count == 0:
        return None
    if count == 0:
        # No positive support for this rule. Residual MI can
        # surface "voiced → tone 1" as high-MI even when every
        # voiced observation has tone 4 instead (binary MI is
        # symmetric in the outcome), but committing a zero-count
        # rule would apply the negative-adjustment path on every
        # match and produce confusing output. Skip these and let
        # the commit loop find the positive framing
        # ("voiced → tone 4") on the same iteration.
        return None

    base_feature_2: FeatureConstraint | None = None
    base_position_2: str | None = None
    if is_joint and feature_name_2 is not None and src_value_2 is not None:
        base_feature_2 = FeatureConstraint(
            feature=FeatureName(feature_name_2), value=FeatureValue(src_value_2)
        )
        base_position_2 = hypothesis.src_position_spec_2
    return CrossDimensionalLink(
        src_feature=FeatureConstraint(
            feature=FeatureName(feature_name), value=FeatureValue(src_value)
        ),
        src_position=hypothesis.src_position_spec,
        tgt_dimension=tgt_dimension,
        tgt_value=tgt_value,
        tgt_position_offset=tgt_offset,
        count=float(count),
        src_count=float(src_count),
        confidence=count / src_count,
        src_feature_2=base_feature_2,
        src_position_2=base_position_2,
        uncertainty=wilson_interval(float(count), float(src_count)),
    )


def _parse_relative_position_offset(spec: str) -> int:
    """Local copy of regulae.scoring._parse_relative_offset to
    avoid a training-scoring import cycle."""
    if not spec.startswith("relative_"):
        raise ValueError(f"unrecognised src_position spec: {spec!r}")
    body = spec[len("relative_"):]
    return int(body)


def _rule_signature(
    rule: CrossDimensionalLink,
) -> tuple[str, str, str, str, str, int, str | None, str | None, str | None]:
    """Return a hashable signature for a cross-dimensional rule,
    used to dedup commits across cross-dimensional iterations.

    Two rules with the same signature predict the same outcome
    from the same source context; their counts and confidences
    may differ slightly (from Dirichlet smoothing or
    re-alignment drift) but committing one makes the other
    redundant.

    Joint-predictor rules include the second source feature and
    position in the signature so a joint rule isn't falsely
    deduped against a single-predictor rule with the same first
    feature.
    """
    base = (
        rule.src_feature.feature,
        rule.src_feature.value,
        rule.src_position,
        rule.tgt_dimension,
        rule.tgt_value,
        rule.tgt_position_offset,
    )
    if rule.src_feature_2 is None:
        return base + (None, None, None)
    return base + (
        rule.src_feature_2.feature,
        rule.src_feature_2.value,
        rule.src_position_2,
    )
