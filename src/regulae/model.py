"""Data types for the learned model.

This module contains only *data types* — frozen dataclasses that hold
the learned parameters. No training logic, no scoring logic. Training
lives in ``training.py``; scoring lives in ``scoring.py``. Keeping data
types separate from logic prevents circular imports and makes the
model's structure easy to inspect.

The layered model composes several contributions on top of the
merkmal feature-geometry prior:

1. :class:`SegmentCorrespondenceTable` — Dirichlet-smoothed segment-pair
   probabilities estimated from 1-to-1 link counts in the training
   corpus, with context-conditioned entries where the data supports a
   split.

2. :class:`DisplacementDistribution` — feature-displacement probabilities
   for generalising across natural classes.

3. :class:`ChunkPhraseTable` — memorised multi-segment correspondences,
   each with a direct cost.

4. :class:`TonalCorrespondenceTable` — source → target tone mappings
   with a Dirichlet prior.

5. :class:`CrossDimensionalLinkTable` — rules where a segmental feature
   on the source predicts a suprasegmental value on the target.

:class:`LearnedModel` composes them with the hyperparameters used
during training.
"""

from dataclasses import dataclass, field
from typing import Mapping

from regulae.types import (
    Context,
    FeatureConstraint,
    FeatureDisplacement,
    Form,
    Segment,
)
from regulae.uncertainty import UncertaintyEstimate


@dataclass(frozen=True)
class ConditionedCorrespondence:
    """A key identifying one segment correspondence together with its
    conditioning context.

    Used as the key type in :class:`SegmentCorrespondenceTable`.
    Context-free correspondences use ``Context()`` (the empty
    context). Context-conditioned correspondences use a non-empty
    :class:`Context` with any combination of position,
    preceding/following feature constraints, long-range constraints,
    or morphological information.
    """

    src: str
    tgt: str
    context: Context = field(default_factory=Context)


@dataclass(frozen=True)
class TonalCorrespondence:
    """A key identifying one tone-to-tone correspondence.

    Tones are stored as strings on :class:`Segment`; a ``None`` tone
    represents an untoned segment. The tonal table is populated
    during training and used by the scoring function as an additive
    term parallel to the segmental cost. Current tonal correspondences
    are context-free; context-conditioning for tones may be added in
    a future version.
    """

    src_tone: str | None
    tgt_tone: str | None


@dataclass(frozen=True)
class TonalCorrespondenceTable:
    """Learned tone-to-tone correspondence counts with a Dirichlet prior.

    Parallels :class:`SegmentCorrespondenceTable` but on the tonal
    dimension. An empty table (no observed tonal data) contributes
    zero cost at scoring time, so the table is inert for non-tonal
    corpora.
    """

    counts: dict[TonalCorrespondence, float] = field(default_factory=dict)
    prior_pseudo_counts: dict[TonalCorrespondence, float] = field(default_factory=dict)
    src_totals: dict[str | None, float] = field(default_factory=dict)
    #: Parallel map from each ``counts`` key to a Wilson (or bootstrap)
    #: interval on the rate ``count / src_totals[src_tone]``. Populated
    #: at training time; empty on an empty table. Excluded from
    #: equality and hashing so two otherwise-identical tables compare
    #: equal regardless of uncertainty population.
    uncertainty: dict[TonalCorrespondence, UncertaintyEstimate] = field(
        default_factory=dict, compare=False, hash=False
    )


@dataclass(frozen=True)
class CrossDimensionalLink:
    """A learned correspondence that crosses phonological dimensions.

    The framework uses these as a first-class type to express
    rules like "source voiced initial consonant ↔ target
    low-register tone on the following vowel" (tonogenesis)
    without burying the cross-dimensional structure inside
    :class:`Context`.

    The canonical tonogenesis example:

    .. code-block:: python

        CrossDimensionalLink(
            src_feature=FeatureConstraint("voiced", "+"),
            src_position="relative_-1",
            tgt_dimension="tone",
            tgt_value="4",
            tgt_position_offset=0,
            count=18.0,
            src_count=20.0,
            confidence=0.9,
        )

    Interpretation: at every 1-to-1 link where the **source**
    segment one position to the left has the ``voiced:+``
    feature, the **target** segment at the link position itself
    is expected to carry tone ``"4"``.

    The link contributes an additive cost adjustment at scoring
    time equal to the log-likelihood ratio of the target value
    under the rule vs. under the unconditional tonal base rate:

    .. code-block:: text

        adjustment = -(log P_cond - log P_base)

    where ``P_cond = (count + α) / (src_count + α·V)`` is the
    Dirichlet-smoothed conditional of ``tgt_value`` given the
    source feature at the source position, and ``P_base`` is the
    unconditional probability of ``tgt_value`` from the model's
    existing :class:`TonalCorrespondenceTable`. When the rule
    doesn't discriminate (``P_cond == P_base``) the adjustment
    is zero, so the empty-table inertness invariant is
    preserved.

    Fields:

    * ``src_feature``: the :class:`FeatureConstraint` that gates
      the rule on the source side.
    * ``src_position``: locator for where on the source form the
      source feature must hold. The supported values are
      ``"relative_-1"`` (segment immediately to the left of the
      link), ``"relative_0"`` (the link position itself), and
      ``"relative_+1"`` (segment immediately to the right).
      Structural position specs are deliberately out of scope.
    * ``tgt_dimension``: which suprasegmental dimension of the
      target segment carries the gated value. Currently only
      ``"tone"`` is supported; ``"length"`` and ``"stress"``
      are possible extensions.
    * ``tgt_value``: the specific value expected on
      ``tgt_dimension``.
    * ``tgt_position_offset``: offset from the link's own
      alignment position at which ``tgt_value`` must be
      observed.
    * ``count``: number of corpus observations where the rule's
      predictor and outcome both held.
    * ``src_count``: number of corpus observations where the
      rule's predictor held (regardless of whether the outcome
      held). Needed to compute the Dirichlet-smoothed
      conditional ``P_cond`` at scoring time.
    * ``confidence``: ``count / src_count``, cached for display
      and ranking. Parallel to the multi-lect class coverage
      field.
    """

    src_feature: FeatureConstraint
    src_position: str
    tgt_dimension: str
    tgt_value: str
    tgt_position_offset: int
    count: float = 0.0
    src_count: float = 0.0
    confidence: float = 1.0
    #: Optional second predictor for joint rules. When set, the
    #: rule fires only when BOTH ``src_feature`` at
    #: ``src_position`` AND ``src_feature_2`` at
    #: ``src_position_2`` hold. When None (the default), the
    #: rule is a single-predictor rule.
    src_feature_2: FeatureConstraint | None = None
    src_position_2: str | None = None
    #: Interval on ``confidence`` (= ``count / src_count``). Populated
    #: at rule-commit time. Excluded from equality and hashing so two
    #: otherwise-identical rules compare equal regardless of
    #: uncertainty population.
    uncertainty: UncertaintyEstimate | None = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True)
class CrossDimensionalLinkTable:
    """Table of committed :class:`CrossDimensionalLink` rules.

    An empty table contributes zero cost at scoring time, so any
    model without committed cross-dimensional rules remains
    inert with respect to the cross-dimensional overlay.

    Stored as a frozen tuple so the table is hashable and safe
    to pass around.
    """

    entries: tuple[CrossDimensionalLink, ...] = ()


@dataclass(frozen=True)
class MultiLectCrossDimensionalLink:
    """A cross-dimensional rule surfaced at the multi-lect level.

    The multi-lect analogue of :class:`CrossDimensionalLink`:
    the same rule shape, with explicit ``src_lect`` and
    ``tgt_lect`` labels naming which lect carries the source
    predicate and which lect carries the target value. This is
    the minimum information a downstream consumer needs to
    interpret a tonogenesis rule without drilling into
    :attr:`MultiLectModel.pairwise_models` and reconstructing
    the pair identifiers by hand.

    The rule itself is the same: a segmental feature at one
    position on ``src_lect`` predicts a suprasegmental value at
    another position on ``tgt_lect``. Both ``src_lect`` and
    ``tgt_lect`` are ``lect_id`` strings from the parent
    :class:`MultiLectModel`.

    The multi-lect table is built by projecting per-pair
    cross-dimensional commits — no independent discovery runs
    at the multi-lect level in the current implementation.
    """

    src_lect: str
    tgt_lect: str
    src_feature: FeatureConstraint
    src_position: str
    tgt_dimension: str
    tgt_value: str
    tgt_position_offset: int
    count: float = 0.0
    src_count: float = 0.0
    confidence: float = 1.0
    #: Optional joint-predictor extension, mirroring
    #: :class:`CrossDimensionalLink`.
    src_feature_2: FeatureConstraint | None = None
    src_position_2: str | None = None
    #: Interval on ``confidence``. Propagated from the underlying
    #: per-pair :class:`CrossDimensionalLink`.
    uncertainty: UncertaintyEstimate | None = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True)
class MultiLectCrossDimensionalLinkTable:
    """Table of committed :class:`MultiLectCrossDimensionalLink` rules.

    Populated in training after the multi-lect class-discovery
    stage by projecting per-pair cross-dimensional commits.
    Empty by default.
    """

    entries: tuple[MultiLectCrossDimensionalLink, ...] = ()


@dataclass(frozen=True)
class SegmentCorrespondenceTable:
    """Observed segment-pair counts plus Dirichlet prior pseudo-counts.

    The posterior predictive distribution of ``tgt_grapheme`` given
    ``src_grapheme`` under this table is

        P(t | s) = (alpha(t | s) + N(t | s)) / (beta + N(s))

    where:

    * ``counts[(s, t)]`` = ``N(t | s)``, observed co-occurrence count
    * ``prior_pseudo_counts[(s, t)]`` = ``alpha(t | s)``, the Dirichlet
      prior derived from merkmal distances via softmax
    * ``src_totals[s]`` = ``N(s)``, total observation mass per source
    * ``beta`` (the Dirichlet concentration) lives on the parent
      :class:`LearnedModel`, not on this table, because the concentration
      is a model-wide hyperparameter.

    ``log_normalizers[s]`` caches ``log(Z(s))``, the log of the prior
    partition function for source ``s``:

        Z(s) = sum_t exp(-tau * d(s, t))

    This offset is subtracted from ``-log P_post`` at scoring time, so
    that at zero observations the adjusted cost equals the merkmal
    distance ``d(s, t)`` exactly. This keeps the initial model
    behaviourally identical to the prior-only alignment and lets
    observed pairs get cost below the merkmal distance as evidence
    accumulates.

    The table is produced by the training loop and is intended to be
    immutable after construction. The dict fields are not hash-safe;
    do not attempt to put a table in a set.
    """

    counts: dict[ConditionedCorrespondence, float] = field(default_factory=dict)
    prior_pseudo_counts: dict[ConditionedCorrespondence, float] = field(default_factory=dict)
    src_totals: dict[str, float] = field(default_factory=dict)
    log_normalizers: dict[str, float] = field(default_factory=dict)
    #: Parallel map from each ``counts`` key to an interval on the rate
    #: ``count / src_totals[key.src]`` (the conditional probability
    #: ``P(tgt | src, context)``). Populated at training time; empty on
    #: a freshly constructed table. Excluded from equality and hashing.
    uncertainty: dict[ConditionedCorrespondence, UncertaintyEstimate] = field(
        default_factory=dict, compare=False, hash=False
    )


@dataclass(frozen=True)
class DisplacementDistribution:
    """Categorical distribution over feature displacement vectors.

    Keys are canonical tuples of :class:`FeatureDisplacement` (sorted
    to make them order-insensitive keys). Values are observed counts.

    The posterior predictive under a symmetric Dirichlet prior with
    pseudo-count ``prior_pseudo_count`` per entry is

        P(d) = (alpha + N(d)) / (alpha * V + N_total)

    where V is the number of distinct displacements seen. Unseen
    displacements (not in ``counts``) get the base probability
    ``alpha / (alpha * V + N_total)`` — this is implemented in
    ``scoring.py`` rather than here.

    An empty distribution (``counts == {}`` and ``total == 0.0``)
    represents the state before displacement aggregation has run.
    The scoring function treats an empty distribution as contributing
    nothing to the final cost, so early-stage alignments use
    segment-level scoring only.
    """

    counts: dict[tuple[FeatureDisplacement, ...], float] = field(default_factory=dict)
    total: float = 0.0
    prior_pseudo_count: float = 1.0
    #: Parallel map from each ``counts`` key to an interval on the rate
    #: ``count / total``. Populated after displacement aggregation;
    #: empty on an empty distribution.
    uncertainty: dict[tuple[FeatureDisplacement, ...], UncertaintyEstimate] = field(
        default_factory=dict, compare=False, hash=False
    )


@dataclass(frozen=True)
class ChunkPhraseTable:
    """Table of promoted multi-segment correspondences.

    Each entry maps a (source chunk, target chunk) pair to a direct
    cost. Only chunks that passed the BIC promotion test appear
    here. Entries override the compositional fallback for their
    specific chunk pair when scoring.
    """

    entries: dict[tuple[tuple[Segment, ...], tuple[Segment, ...]], float] = field(
        default_factory=dict
    )
    #: Parallel map from each ``entries`` key to the number of corpus
    #: observations of that chunk at promotion time. The chunk table
    #: itself stores the promoted cost (on the search scale), not the
    #: count; this map recovers the raw count for downstream consumers
    #: and for the bootstrap uncertainty path.
    observation_counts: dict[
        tuple[tuple[Segment, ...], tuple[Segment, ...]], float
    ] = field(default_factory=dict)
    #: Parallel map from each ``entries`` key to an interval on the
    #: rate ``chunk_observations / total_1to1_observations`` used as
    #: the BIC denominator during promotion. Populated at
    #: chunk-promotion time; empty on an empty table.
    uncertainty: dict[
        tuple[tuple[Segment, ...], tuple[Segment, ...]], UncertaintyEstimate
    ] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class LearnedModel:
    """A trained learned model: the layered tables plus hyperparameters.

    Constructed by ``training.train_model`` from a corpus of form
    pairs. Passed to ``score_link`` and ``align_forms`` as an optional
    parameter; when ``model=None`` the framework falls back to
    prior-only scoring (merkmal distances without learned counts).

    The four hyperparameters:

    * ``temperature`` (``tau``): controls how sharply the merkmal
      distance is converted to a prior probability. Higher tau means
      the prior concentrates more on near matches.
    * ``concentration`` (``beta``): Dirichlet concentration parameter.
      Controls how quickly observed data overwhelms the prior. Higher
      beta means the prior dominates longer.
    * ``segment_weight`` and ``displacement_weight``: log-linear
      combination weights for the segment and displacement layers.
      Must sum to 1.0 (not enforced at construction time).
    """

    segment_table: SegmentCorrespondenceTable
    displacement_dist: DisplacementDistribution
    chunk_table: ChunkPhraseTable
    tonal_table: TonalCorrespondenceTable = field(default_factory=TonalCorrespondenceTable)
    feature_system: str = "descriptive"
    temperature: float = 1.0
    concentration: float = 5.0
    segment_weight: float = 0.7
    displacement_weight: float = 0.3
    tone_weight: float = 1.0
    #: Cross-dimensional link table. Empty by default; populated
    #: by the cross-dimensional discovery stage with committed
    #: rules.
    cross_dimensional_table: CrossDimensionalLinkTable = field(
        default_factory=CrossDimensionalLinkTable
    )

    @classmethod
    def empty(
        cls,
        feature_system: str = "descriptive",
        temperature: float = 1.0,
        concentration: float = 5.0,
    ) -> "LearnedModel":
        """Create an empty model (no learned data).

        An empty model has no prior pseudo-counts, no observed counts,
        no displacement data, and no promoted chunks. Scoring under an
        empty model falls back to merkmal distances for everything
        — the prior-only fallback used before any training data is
        available.

        This is a convenience for tests and for the initial state of
        training before the prior is computed from the corpus.
        """
        return cls(
            segment_table=SegmentCorrespondenceTable(),
            displacement_dist=DisplacementDistribution(),
            chunk_table=ChunkPhraseTable(),
            tonal_table=TonalCorrespondenceTable(),
            cross_dimensional_table=CrossDimensionalLinkTable(),
            feature_system=feature_system,
            temperature=temperature,
            concentration=concentration,
        )


# ----- multi-lect types ---------------------------------------------------


@dataclass(frozen=True)
class CognateSet:
    """A set of cognate forms across N lects.

    For N=1 this is a single-lect entry; N=2 is the pairwise
    case; N≥3 is the multi-lect case. Training takes a
    ``list[CognateSet]`` regardless of N.

    Cognate membership is explicit user input. The framework does
    not attempt to infer cognacy from glosses and does not drop
    pairs that "look" non-cognate during training — every input
    is trained on. If the user passes a non-cognate pair (the
    classic example being Latin ``kaput`` "head" vs. Spanish
    ``kabeθa`` "head", which are NOT reflexes of the same
    proto-word), the framework will learn its segment
    correspondences alongside the real ones. This is a deliberate
    choice: cognate detection is a separate problem and belongs
    to a separate tool.

    For post-hoc inspection of likely-non-cognate pairs in an
    already-trained model, see
    :func:`regulae.find_cognate_outliers`, which ranks the input
    cognate sets by alignment-cost z-score so a human reviewer
    can decide which sets to drop and retrain on.

    Fields:

    * ``cognate_id``: user-supplied identifier for the cognate set.
    * ``forms``: mapping ``lect_id -> Form``. A lect absent from the
      mapping is absent from this cognate set.
    * ``alignments``: optional per-lect pre-aligned columns. If given,
      each lect's value is a tuple of the same length, with ``None``
      marking gaps. Reserved as a warm-start hint for pairwise
      training; not currently consumed.
    * ``morpheme_boundaries``: reserved for future morph-aware
      alignment. Captured by loaders when present but not currently
      consumed.
    * ``confidence``: membership confidence in [0, 1]. Training uses
      this as an evidence weight: lower-confidence cognate sets still
      align and remain in the retained corpus for auditability, but
      contribute proportionally less mass to pairwise correspondence
      counts, chunk promotion, tonal/cross-dimensional discovery, and
      multi-lect class aggregation. ``0.0`` means "inspect but do not
      learn from this set"; ``1.0`` is the default full contribution.
    """

    cognate_id: str
    forms: Mapping[str, Form]
    alignments: Mapping[str, tuple[Segment | None, ...]] | None = None
    morpheme_boundaries: Mapping[str, tuple[int, ...]] | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class MultiLectCorrespondenceClass:
    """One multi-lect correspondence class.

    A class binds one grapheme per participating lect, optionally with
    per-lect contexts. Classes with ``contexts is None`` are
    unconditioned; classes with a non-None ``contexts`` are context-
    conditioned — the multi-lect analogue of
    :class:`ConditionedCorrespondence` with one context per
    participating lect.

    Fields:

    * ``class_id``: unique integer within a :class:`MultiLectModel`.
    * ``segments``: mapping ``lect_id -> grapheme``. Only lects
      participating in this class are present.
    * ``contexts``: optional mapping ``lect_id -> Context`` giving the
      per-lect conditioning environment. When present, lects without
      a conditioning constraint have the empty ``Context()``.
    * ``count``: observation count for this class across the corpus.
    * ``supporting_cognates``: cognate IDs from which this class was
      observed, for provenance and debugging.
    * ``confidence``: diagnostic strength indicator in ``[0, 1]``,
      set by the class-discovery loop for conditioned classes and
      left at 1.0 for unconditioned classes. Defined as the
      pivot-bucket coverage: ``count / pivot_bucket_size`` where
      ``pivot_bucket_size`` is the number of observations of the
      pivot lect+grapheme that the discovery loop saw. A value of
      1.0 means "every observation of the pivot is in this class"
      (strong rule); 0.1 means "only 10% of the pivot's
      observations landed here" (weak minority). For dedup-merged
      conditioned classes (one class discovered by multiple pivots)
      the maximum coverage across contributing pivots is kept,
      since that is the pivot that saw the clearest signal. The
      field is advisory — nothing in the training or scoring
      pipelines uses it — it exists so the formatter and downstream
      consumers can sort or filter classes by signal strength on
      thin corpora.
    """

    class_id: int
    segments: Mapping[str, str]
    contexts: Mapping[str, Context] | None = None
    count: float = 0.0
    supporting_cognates: tuple[str, ...] = ()
    confidence: float = 1.0
    #: Interval on the rate ``count / pivot_bucket_size`` (for
    #: conditioned classes) or ``count / total_observations_of_lect_combo``
    #: (for unconditioned classes). Populated at class-discovery time.
    uncertainty: UncertaintyEstimate | None = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True)
class MultiLectModel:
    """A trained multi-lect alignment model.

    Contains the per-pair learned models, the reconciled
    multi-lect correspondence classes (both unconditioned and
    conditioned tables), and a reference to the corpus used for
    training.

    Fields:

    * ``pairwise_models``: mapping from 2-element frozensets of lect
      IDs to the :class:`LearnedModel` trained on that pair. A pair
      missing from the mapping means there was no shared cognate data
      for it in the corpus.
    * ``unconditioned_classes``: tuple of
      :class:`MultiLectCorrespondenceClass` entries with
      ``contexts is None``.
    * ``conditioned_classes``: tuple of class entries with non-None
      ``contexts``, produced by the multi-lect class-discovery
      loop.
    * ``cognate_corpus``: the input corpus retained for provenance.
    * ``lect_ids``: canonical ordered list of all lect IDs observed
      in the corpus.
    * ``cross_dimensional_table``: aggregated table of
      :class:`MultiLectCrossDimensionalLink` entries lifted
      from per-pair commits. Empty by default. Consumers can
      iterate ``cross_dimensional_table.entries`` to see every
      tonogenesis-style rule committed in any pairwise model,
      each annotated with explicit ``src_lect`` and
      ``tgt_lect`` labels.
    """

    pairwise_models: Mapping[frozenset[str], LearnedModel]
    unconditioned_classes: tuple[MultiLectCorrespondenceClass, ...]
    conditioned_classes: tuple[MultiLectCorrespondenceClass, ...]
    cognate_corpus: tuple[CognateSet, ...]
    lect_ids: tuple[str, ...]
    cross_dimensional_table: MultiLectCrossDimensionalLinkTable = field(
        default_factory=MultiLectCrossDimensionalLinkTable
    )

    @classmethod
    def empty(cls) -> "MultiLectModel":
        """Create an empty multi-lect model with no pairs and no classes."""
        return cls(
            pairwise_models={},
            unconditioned_classes=(),
            conditioned_classes=(),
            cognate_corpus=(),
            lect_ids=(),
        )
