"""Bootstrap percentile intervals on trained-model counts.

Given a base trained model and its training corpus, this module
produces bootstrap percentile intervals on every count-bearing entry
in the base model by resampling the corpus with replacement and
retraining. The resulting intervals replace the closed-form Wilson
intervals attached at training time.

The resampler is seeded so the bootstrap is deterministic per
``bootstrap_seed``. Resample units are pair tuples (pairwise path) or
``CognateSet`` objects (multi-lect path). Per-pair ``LearnedModel``
fields on a :class:`MultiLectModel` are updated coherently — the same
resampling seed is used for the whole multi-lect run so the per-pair
bootstraps are consistent with the class-level bootstrap.

A key in the base model that is absent from a resample contributes a
count of 0 to that resample's distribution. This gives meaningful
percentile intervals even on rules that are present in the base model
but marginal in the data: they will have low lower bounds.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any

from regulae.model import (
    ChunkPhraseTable,
    CognateSet,
    ConditionedCorrespondence,
    CrossDimensionalLink,
    CrossDimensionalLinkTable,
    DisplacementDistribution,
    LearnedModel,
    MultiLectCorrespondenceClass,
    MultiLectCrossDimensionalLink,
    MultiLectCrossDimensionalLinkTable,
    MultiLectModel,
    SegmentCorrespondenceTable,
    TonalCorrespondence,
    TonalCorrespondenceTable,
)
from regulae.types import FeatureDisplacement, Form, Segment
from regulae.uncertainty import UncertaintyEstimate, bootstrap_rate_interval

if TYPE_CHECKING:
    pass


# ----- pairwise path ------------------------------------------------------


def bootstrap_pairwise_uncertainty(
    base: LearnedModel,
    *,
    corpus: Sequence[tuple[Form, Form]],
    bootstrap_n: int,
    bootstrap_seed: int,
    **train_kwargs: Any,
) -> LearnedModel:
    """Run ``bootstrap_n`` resampled trainings on a pair corpus and
    replace the base model's Wilson intervals with percentile
    intervals from the bootstrap distribution.

    ``train_kwargs`` are the training hyperparameters used for the
    base run; each bootstrap run uses the same hyperparameters. The
    resampled corpus has the same length as the base corpus.
    """
    from regulae.training import _train_pairwise_legacy

    rng = random.Random(bootstrap_seed)
    n_pairs = len(corpus)

    # Accumulators: per base-model key, a list of counts across resamples.
    segment_samples: dict[ConditionedCorrespondence, list[float]] = {
        key: [] for key in base.segment_table.counts
    }
    displacement_samples: dict[tuple[FeatureDisplacement, ...], list[float]] = {
        key: [] for key in base.displacement_dist.counts
    }
    tonal_samples: dict[TonalCorrespondence, list[float]] = {
        key: [] for key in base.tonal_table.counts
    }
    chunk_samples: dict[
        tuple[tuple[Segment, ...], tuple[Segment, ...]], list[float]
    ] = {key: [] for key in base.chunk_table.entries}
    cross_dim_samples: dict[tuple, list[float]] = {
        _cross_dim_key(r): [] for r in base.cross_dimensional_table.entries
    }

    for _ in range(bootstrap_n):
        resample = [corpus[rng.randrange(n_pairs)] for _ in range(n_pairs)]
        sample_model = _train_pairwise_legacy(resample, **train_kwargs)
        _accumulate_pair_model_samples(
            sample_model,
            segment_samples,
            displacement_samples,
            tonal_samples,
            chunk_samples,
            cross_dim_samples,
        )

    return _apply_pair_bootstrap(
        base,
        segment_samples,
        displacement_samples,
        tonal_samples,
        chunk_samples,
        cross_dim_samples,
    )


def _accumulate_pair_model_samples(
    sample_model: LearnedModel,
    segment_samples: dict[ConditionedCorrespondence, list[float]],
    displacement_samples: dict[tuple[FeatureDisplacement, ...], list[float]],
    tonal_samples: dict[TonalCorrespondence, list[float]],
    chunk_samples: dict[
        tuple[tuple[Segment, ...], tuple[Segment, ...]], list[float]
    ],
    cross_dim_samples: dict[tuple, list[float]],
) -> None:
    """Record one sample's rate for each base-model key.

    Each rate is computed *within* the sample using its own
    denominator (``src_totals`` / ``total`` / ``src_count``), so the
    sample distribution is bounded to ``[0, 1]`` for every key.
    Missing keys contribute a rate of 0.
    """
    seg_counts = sample_model.segment_table.counts
    seg_src_totals = sample_model.segment_table.src_totals
    for key, samples in segment_samples.items():
        n = seg_src_totals.get(key.src, 0.0)
        c = seg_counts.get(key, 0.0)
        samples.append(c / n if n > 0.0 else 0.0)
    disp_counts = sample_model.displacement_dist.counts
    disp_total = sample_model.displacement_dist.total
    for key, samples in displacement_samples.items():
        c = disp_counts.get(key, 0.0)
        samples.append(c / disp_total if disp_total > 0.0 else 0.0)
    tone_counts = sample_model.tonal_table.counts
    tone_src_totals = sample_model.tonal_table.src_totals
    for key, samples in tonal_samples.items():
        n = tone_src_totals.get(key.src_tone, 0.0)
        c = tone_counts.get(key, 0.0)
        samples.append(c / n if n > 0.0 else 0.0)
    chunk_counts = sample_model.chunk_table.observation_counts
    chunk_denom = _chunk_bootstrap_denominator(sample_model)
    for key, samples in chunk_samples.items():
        c = chunk_counts.get(key, 0.0)
        samples.append(c / chunk_denom if chunk_denom > 0.0 else 0.0)
    sample_cd_by_key = {
        _cross_dim_key(r): r for r in sample_model.cross_dimensional_table.entries
    }
    for key, samples in cross_dim_samples.items():
        rule = sample_cd_by_key.get(key)
        samples.append(rule.confidence if rule is not None else 0.0)


def _apply_pair_bootstrap(
    base: LearnedModel,
    segment_samples: Mapping[ConditionedCorrespondence, list[float]],
    displacement_samples: Mapping[tuple[FeatureDisplacement, ...], list[float]],
    tonal_samples: Mapping[TonalCorrespondence, list[float]],
    chunk_samples: Mapping[tuple[tuple[Segment, ...], tuple[Segment, ...]], list[float]],
    cross_dim_samples: Mapping[tuple, list[float]],
) -> LearnedModel:
    seg_unc: dict[ConditionedCorrespondence, UncertaintyEstimate] = {}
    for key, samples in segment_samples.items():
        n = base.segment_table.src_totals.get(key.src, 0.0)
        seg_unc[key] = bootstrap_rate_interval(samples, n=n)
    disp_unc: dict[tuple[FeatureDisplacement, ...], UncertaintyEstimate] = {}
    for key, samples in displacement_samples.items():
        disp_unc[key] = bootstrap_rate_interval(samples, n=base.displacement_dist.total)
    tone_unc: dict[TonalCorrespondence, UncertaintyEstimate] = {}
    for key, samples in tonal_samples.items():
        n = base.tonal_table.src_totals.get(key.src_tone, 0.0)
        tone_unc[key] = bootstrap_rate_interval(samples, n=n)
    chunk_unc: dict[
        tuple[tuple[Segment, ...], tuple[Segment, ...]], UncertaintyEstimate
    ] = {}
    chunk_denominator = _chunk_bootstrap_denominator(base)
    for key, samples in chunk_samples.items():
        chunk_unc[key] = bootstrap_rate_interval(samples, n=chunk_denominator)

    new_segment_table = dc_replace(base.segment_table, uncertainty=seg_unc)
    new_displacement = dc_replace(base.displacement_dist, uncertainty=disp_unc)
    new_tonal = dc_replace(base.tonal_table, uncertainty=tone_unc)
    new_chunks = dc_replace(base.chunk_table, uncertainty=chunk_unc)
    new_cross_dim_entries = tuple(
        dc_replace(
            r,
            uncertainty=bootstrap_rate_interval(
                cross_dim_samples.get(_cross_dim_key(r), []),
                n=r.src_count,
            ),
        )
        for r in base.cross_dimensional_table.entries
    )
    return dc_replace(
        base,
        segment_table=new_segment_table,
        displacement_dist=new_displacement,
        chunk_table=new_chunks,
        tonal_table=new_tonal,
        cross_dimensional_table=CrossDimensionalLinkTable(entries=new_cross_dim_entries),
    )


def _chunk_bootstrap_denominator(model: LearnedModel) -> float:
    """The n used in chunk uncertainty at base training is the total
    1-to-1 observation mass. We recover it from ``src_totals`` (which
    sums exactly that)."""
    return float(sum(model.segment_table.src_totals.values()))


def _cross_dim_key(rule: CrossDimensionalLink | MultiLectCrossDimensionalLink) -> tuple:
    """Return a canonical hashable identity for a cross-dim rule."""
    return (
        rule.src_feature.feature,
        rule.src_feature.value,
        rule.src_position,
        rule.tgt_dimension,
        rule.tgt_value,
        rule.tgt_position_offset,
        rule.src_feature_2.feature if rule.src_feature_2 else None,
        rule.src_feature_2.value if rule.src_feature_2 else None,
        rule.src_position_2,
    )


# ----- multi-lect path ----------------------------------------------------


def bootstrap_multi_lect_uncertainty(
    base: MultiLectModel,
    *,
    corpus: Sequence[CognateSet],
    bootstrap_n: int,
    bootstrap_seed: int,
    **train_kwargs: Any,
) -> MultiLectModel:
    """Resample the cognate-set corpus and replace intervals on the
    base :class:`MultiLectModel` with bootstrap percentile intervals.

    Updates both per-pair tables (segment, displacement, tonal,
    chunk, cross-dim) and multi-lect classes + lifted cross-dim rules.
    """
    from regulae._reconciliation import _train_multi_lect

    rng = random.Random(bootstrap_seed)
    n = len(corpus)

    # Per-pair accumulators, keyed by lect-pair frozenset.
    pair_seg: dict[frozenset[str], dict[ConditionedCorrespondence, list[float]]] = {}
    pair_disp: dict[
        frozenset[str], dict[tuple[FeatureDisplacement, ...], list[float]]
    ] = {}
    pair_tone: dict[frozenset[str], dict[TonalCorrespondence, list[float]]] = {}
    pair_chunk: dict[
        frozenset[str],
        dict[tuple[tuple[Segment, ...], tuple[Segment, ...]], list[float]],
    ] = {}
    pair_crossdim: dict[frozenset[str], dict[tuple, list[float]]] = {}
    for key, pm in base.pairwise_models.items():
        pair_seg[key] = {k: [] for k in pm.segment_table.counts}
        pair_disp[key] = {k: [] for k in pm.displacement_dist.counts}
        pair_tone[key] = {k: [] for k in pm.tonal_table.counts}
        pair_chunk[key] = {k: [] for k in pm.chunk_table.entries}
        pair_crossdim[key] = {
            _cross_dim_key(r): [] for r in pm.cross_dimensional_table.entries
        }
    uncond_samples: dict[tuple[tuple[str, str], ...], list[float]] = {
        tuple(sorted(c.segments.items())): [] for c in base.unconditioned_classes
    }
    cond_samples: dict[tuple, list[float]] = {
        _cond_class_key(c): [] for c in base.conditioned_classes
    }
    ml_cross_samples: dict[tuple, list[float]] = {
        _ml_cross_key(r): []
        for r in base.cross_dimensional_table.entries
    }

    for _ in range(bootstrap_n):
        resample = [corpus[rng.randrange(n)] for _ in range(n)]
        sample_model = _train_multi_lect(
            resample,
            **train_kwargs,
        )
        for key, pm in sample_model.pairwise_models.items():
            if key not in pair_seg:
                continue
            _accumulate_pair_model_samples(
                pm,
                pair_seg[key],
                pair_disp[key],
                pair_tone[key],
                pair_chunk[key],
                pair_crossdim[key],
            )
        # Class-level: match by sorted segment tuple. Rate comes from
        # the base-model denominator stored on the resample's own
        # class uncertainty (``n`` field). Missing class → rate 0.
        sample_uncond_by_key = {
            tuple(sorted(c.segments.items())): c
            for c in sample_model.unconditioned_classes
        }
        for k, samples in uncond_samples.items():
            c = sample_uncond_by_key.get(k)
            if c is None or c.uncertainty is None or c.uncertainty.n <= 0.0:
                samples.append(0.0)
            else:
                samples.append(c.count / c.uncertainty.n)
        sample_cond_by_key = {
            _cond_class_key(c): c for c in sample_model.conditioned_classes
        }
        for k, samples in cond_samples.items():
            c = sample_cond_by_key.get(k)
            # ``confidence`` = count / pivot_bucket_size (the discovery rate).
            samples.append(c.confidence if c is not None else 0.0)
        sample_ml_cross_by_key = {
            _ml_cross_key(r): r for r in sample_model.cross_dimensional_table.entries
        }
        for k, samples in ml_cross_samples.items():
            r = sample_ml_cross_by_key.get(k)
            samples.append(r.confidence if r is not None else 0.0)

    # Apply intervals back to the base model.
    new_pairwise: dict[frozenset[str], LearnedModel] = {}
    for key, pm in base.pairwise_models.items():
        new_pairwise[key] = _apply_pair_bootstrap(
            pm,
            pair_seg[key],
            pair_disp[key],
            pair_tone[key],
            pair_chunk[key],
            pair_crossdim[key],
        )

    new_uncond = tuple(
        dc_replace(
            c,
            uncertainty=bootstrap_rate_interval(
                uncond_samples.get(tuple(sorted(c.segments.items())), []),
                n=(c.uncertainty.n if c.uncertainty is not None else 0.0),
            ),
        )
        for c in base.unconditioned_classes
    )
    new_cond = tuple(
        dc_replace(
            c,
            uncertainty=bootstrap_rate_interval(
                cond_samples.get(_cond_class_key(c), []),
                n=(c.uncertainty.n if c.uncertainty is not None else 0.0),
            ),
        )
        for c in base.conditioned_classes
    )
    new_ml_cross_entries = tuple(
        dc_replace(
            r,
            uncertainty=bootstrap_rate_interval(
                ml_cross_samples.get(_ml_cross_key(r), []),
                n=r.src_count,
            ),
        )
        for r in base.cross_dimensional_table.entries
    )

    return dc_replace(
        base,
        pairwise_models=new_pairwise,
        unconditioned_classes=new_uncond,
        conditioned_classes=new_cond,
        cross_dimensional_table=MultiLectCrossDimensionalLinkTable(
            entries=new_ml_cross_entries,
        ),
    )


def _cond_class_key(c: MultiLectCorrespondenceClass) -> tuple:
    """Canonical hashable identity for a conditioned multi-lect class.

    Includes segments AND per-lect contexts (by constraint count and
    content) so two classes with the same segments but different
    contexts do not collide.
    """
    ctxs = c.contexts or {}
    return (
        tuple(sorted(c.segments.items())),
        tuple(
            (lect, _context_signature(ctx))
            for lect, ctx in sorted(ctxs.items())
        ),
    )


def _context_signature(ctx: Any) -> tuple:
    """Hashable signature for a Context (for bootstrap key matching)."""
    return (
        ctx.position,
        tuple((fc.feature, fc.value) for fc in ctx.preceding),
        tuple((fc.feature, fc.value) for fc in ctx.following),
        tuple((d, fc.feature, fc.value) for d, fc in ctx.preceding_at_distance),
        tuple((d, fc.feature, fc.value) for d, fc in ctx.following_at_distance),
        tuple((fc.feature, fc.value) for fc in ctx.somewhere_preceding),
        tuple((fc.feature, fc.value) for fc in ctx.somewhere_following),
        tuple((fc.feature, fc.value) for fc in ctx.same_syllable),
        tuple((fc.feature, fc.value) for fc in ctx.next_syllable),
        tuple((fc.feature, fc.value) for fc in ctx.previous_syllable),
    )


def _ml_cross_key(r: MultiLectCrossDimensionalLink) -> tuple:
    return (r.src_lect, r.tgt_lect) + _cross_dim_key(r)
