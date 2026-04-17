"""Training orchestration for the learned model.

The pairwise training pipeline runs several stages in sequence:

1. **Segment-level EM.** Iterates alignment and segment-table
   updates until the total corpus cost converges.
2. **Displacement aggregation.** One pass over the converged
   alignments to build the feature displacement distribution.
3. **Context conditioning.** A greedy BIC-gated split loop
   looks for immediate-neighbour conditioning environments that
   improve the fit; committed splits become conditioned
   correspondences.
4. **Chunk promotion.** Identifies recurring multi-segment
   correspondences and memorises those whose promotion improves
   BIC.
5. **Tonal aggregation.** Collects source-to-target tone counts
   from the converged alignments.
6. **Cross-dimensional discovery.** Searches for rules where a
   segmental feature on the source conditions a suprasegmental
   value on the target. Commits by BIC.
7. **Long-range context discovery.** A second split loop for
   conditioning environments that look beyond the immediate
   neighbours (distance-bounded, existential, and
   syllable-structural predicates).

The public entry point is :func:`train_model`, which accepts
either a pair corpus (for backward compatibility with two-lect
use) or a multi-lect ``list[CognateSet]``. A convenience
function :func:`align_corpus` applies a trained model to a
corpus.
"""

import warnings
from collections.abc import Sequence

from regulae.model import (
    CognateSet,
    LearnedModel,
    MultiLectModel,
)
from regulae.search import DEFAULT_MAX_CHUNK_SIZE, align_forms
from regulae.types import (
    Alignment,
    Form,
)

from regulae._em import (
    _displacement_aggregation,
    _initial_model,
    _replace_segment_table,
    _segment_em,
    _update_segment_table,
)
from regulae._chunks import _chunk_promotion
from regulae._discovery import (
    _context_discovery,
    _long_range_discovery,
    _tonal_aggregation,
)
from regulae._cross_dimensional import (
    _CROSS_DIM_MIN_RULE_CONFIDENCE,
    _CROSS_DIM_MIN_RULE_COUNT,
    _cross_dimensional_discovery,
    _dedup_dual_framings,
    _rule_signature,
)
from regulae._reconciliation import (
    _commit_multi_lect_long_range_splits_for_pivot,
    _lift_cross_dimensional_rules,
    _train_multi_lect,
)


# ----- defaults ------------------------------------------------------------

DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_CONCENTRATION: float = 5.0
DEFAULT_MAX_ITER: int = 30
DEFAULT_CONVERGENCE_EPS: float = 1e-4
DEFAULT_SEGMENT_WEIGHT: float = 0.7
DEFAULT_DISPLACEMENT_WEIGHT: float = 0.3
DEFAULT_TONE_WEIGHT: float = 1.0


# ----- public entry points -------------------------------------------------


def train_model(
    corpus: Sequence[tuple[Form, Form]] | Sequence[CognateSet],
    *,
    feature_system: str = "descriptive",
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    concentration: float = DEFAULT_CONCENTRATION,
    max_iter: int = DEFAULT_MAX_ITER,
    convergence_eps: float = DEFAULT_CONVERGENCE_EPS,
    segment_weight: float = DEFAULT_SEGMENT_WEIGHT,
    displacement_weight: float = DEFAULT_DISPLACEMENT_WEIGHT,
    tone_weight: float = DEFAULT_TONE_WEIGHT,
    multi_lect_bic_correction: bool = True,
    multi_lect_min_commit_scale: float = 0.5,
) -> LearnedModel | MultiLectModel:
    """Train a layered alignment model on a training corpus.

    Polymorphic over two input shapes:

    * ``list[tuple[Form, Form]]`` — legacy pairwise input. Returns a
      :class:`LearnedModel`. The pair form is deprecated; new code
      should construct :class:`CognateSet` objects.
    * ``list[CognateSet]`` — the canonical multi-lect input.
      Returns a :class:`MultiLectModel` containing per-pair learned
      models plus reconciled multi-lect correspondence classes.

    Empty input is handled as the legacy path (returns an empty
    :class:`LearnedModel`) because that's what existing call sites
    expect.

    Multi-lect-only knobs (ignored for pair input):

    * ``multi_lect_bic_correction``: add an AICc-style small-sample
      correction ``2/max(n-1, 1)`` to the class-discovery BIC
      penalty. Default True — without it, tiny pivot buckets pass
      BIC too easily.
    * ``multi_lect_min_commit_scale``: scales the adaptive floor
      for per-sister-tuple emissions in the class-discovery loop.
      The floor is ``max(2, ceil(scale * log2(n+1)))`` where
      ``n`` is the pivot's observation count. Default ``0.5``
      (gentle sub-linear scaling, tuned on multi-lect real-data
      experiments); set to ``0.0`` to disable the adaptive floor
      entirely (structural minimum of 2); set higher to tighten
      further.

    Training is deterministic given the corpus and hyperparameters.
    """
    if not corpus:
        return LearnedModel.empty(
            feature_system=feature_system,
            temperature=temperature,
            concentration=concentration,
        )

    if isinstance(corpus[0], CognateSet):
        _validate_cognate_sets(corpus)  # type: ignore[arg-type]
        return _train_multi_lect(
            corpus,  # type: ignore[arg-type]
            feature_system=feature_system,
            max_chunk_size=max_chunk_size,
            temperature=temperature,
            concentration=concentration,
            max_iter=max_iter,
            convergence_eps=convergence_eps,
            segment_weight=segment_weight,
            displacement_weight=displacement_weight,
            tone_weight=tone_weight,
            multi_lect_bic_correction=multi_lect_bic_correction,
            multi_lect_min_commit_scale=multi_lect_min_commit_scale,
        )

    warnings.warn(
        "Passing a list of (Form, Form) pairs to train_model is "
        "deprecated; use cognate_sets_from_pairs(pairs, lect_ids) to "
        "convert to a list[CognateSet] and pass that instead. The "
        "pair-input branch may be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _train_pairwise_legacy(
        corpus,  # type: ignore[arg-type]
        feature_system=feature_system,
        max_chunk_size=max_chunk_size,
        temperature=temperature,
        concentration=concentration,
        max_iter=max_iter,
        convergence_eps=convergence_eps,
        segment_weight=segment_weight,
        displacement_weight=displacement_weight,
        tone_weight=tone_weight,
    )


def _validate_cognate_sets(corpus: Sequence[CognateSet]) -> None:
    """Check cognate sets for common structural errors before training."""
    all_lects: set[str] = set()
    for cs in corpus:
        all_lects.update(cs.forms)
    n_lects = len(all_lects)

    for cs in corpus:
        if n_lects >= 2 and len(cs.forms) < 2:
            raise ValueError(
                f"CognateSet {cs.cognate_id!r} has {len(cs.forms)} form(s); "
                f"at least 2 are required for multi-lect training."
            )
        for lect_id, form in cs.forms.items():
            if not form.segments:
                raise ValueError(
                    f"CognateSet {cs.cognate_id!r}, lect {lect_id!r}: "
                    f"form has no segments."
                )
        if not 0.0 <= cs.confidence <= 1.0:
            raise ValueError(
                f"CognateSet {cs.cognate_id!r}: confidence must be in [0, 1], "
                f"got {cs.confidence!r}."
            )


def _train_pairwise_legacy(
    corpus: Sequence[tuple[Form, Form]],
    *,
    pair_weights: Sequence[float] | None = None,
    feature_system: str,
    max_chunk_size: int,
    temperature: float,
    concentration: float,
    max_iter: int,
    convergence_eps: float,
    segment_weight: float,
    displacement_weight: float,
    tone_weight: float,
) -> LearnedModel:
    """Internal: the pair-based training pipeline."""
    if pair_weights is None:
        pair_weights = [1.0] * len(corpus)
    if len(pair_weights) != len(corpus):
        raise ValueError("pair_weights must have the same length as corpus")

    initial = _initial_model(
        corpus=corpus,
        feature_system=feature_system,
        temperature=temperature,
        concentration=concentration,
        segment_weight=segment_weight,
        displacement_weight=displacement_weight,
    )

    after_em = _segment_em(
        corpus=corpus,
        pair_weights=pair_weights,
        initial_model=initial,
        max_chunk_size=max_chunk_size,
        max_iter=max_iter,
        convergence_eps=convergence_eps,
    )

    after_displacement = _displacement_aggregation(
        corpus=corpus,
        pair_weights=pair_weights,
        model=after_em,
        max_chunk_size=max_chunk_size,
    )

    after_context = _context_discovery(
        corpus=corpus,
        pair_weights=pair_weights,
        model=after_displacement,
        max_chunk_size=max_chunk_size,
    )

    after_chunks = _chunk_promotion(
        corpus=corpus,
        pair_weights=pair_weights,
        model=after_context,
        max_chunk_size=max_chunk_size,
    )

    after_tonal = _tonal_aggregation(
        corpus=corpus,
        pair_weights=pair_weights,
        model=after_chunks,
        max_chunk_size=max_chunk_size,
    )

    after_cross_dim = _cross_dimensional_discovery(
        corpus=corpus,
        pair_weights=pair_weights,
        model=after_tonal,
    )

    after_long_range = _long_range_discovery(
        corpus=corpus,
        pair_weights=pair_weights,
        model=after_cross_dim,
        max_chunk_size=max_chunk_size,
    )

    return after_long_range


def cognate_sets_from_pairs(
    pairs: Sequence[tuple[Form, Form]],
    lect_ids: tuple[str, str] = ("src", "tgt"),
    *,
    cognate_id_prefix: str = "pair",
) -> list[CognateSet]:
    """Convert a list of pairwise form tuples to a list of CognateSets.

    Helper for code that operates on pair lists. Each
    ``(src_form, tgt_form)`` becomes a CognateSet with a synthetic
    cognate ID and the given pair of lect IDs.

    The lect IDs supplied here are the identifiers used in the
    resulting ``CognateSet.forms`` mapping and in the
    ``Form.lect_id`` of each form (both are rewritten to match).
    """
    src_lect, tgt_lect = lect_ids
    out: list[CognateSet] = []
    for i, (src, tgt) in enumerate(pairs):
        src_form = Form(
            lect_id=src_lect, segments=src.segments, syllable_breaks=src.syllable_breaks
        )
        tgt_form = Form(
            lect_id=tgt_lect, segments=tgt.segments, syllable_breaks=tgt.syllable_breaks
        )
        out.append(
            CognateSet(
                cognate_id=f"{cognate_id_prefix}.{i:05d}",
                forms={src_lect: src_form, tgt_lect: tgt_form},
            )
        )
    return out


def align_corpus(
    corpus: Sequence[tuple[Form, Form]],
    model: LearnedModel,
    *,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Alignment]:
    """Align every pair in a corpus under a trained model.

    Convenience wrapper around :func:`align_forms` with the model
    parameter fixed. Returns one :class:`Alignment` per input pair, in
    the same order as the corpus.
    """
    return [
        align_forms(src, tgt, max_chunk_size=max_chunk_size, model=model)
        for src, tgt in corpus
    ]
