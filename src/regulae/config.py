"""Tunable configuration for BIC-gated discovery stages.

Most of the thresholds in the training pipeline are BIC
parameters: delta thresholds, minimum-observation floors, support
fractions, iteration caps. Before this module existed, every
threshold was a module-level constant — one central location per
file, but not adjustable at train time.

:class:`BICConfig` collects every such knob into a single frozen
dataclass. ``train_model(... bic_config=BICConfig(...))`` lets
callers tighten or loosen discovery without monkey-patching
internals.

The default ``BICConfig()`` preserves the behavior the training
pipeline had before this kwarg existed, so passing no config is
bitwise-equivalent to the previous defaults.

Organization:

* **Segment-level context discovery** — ``delta_bic_threshold``,
  ``min_split_observations``, ``max_split_depth``.
* **Chunk promotion** — ``min_chunk_observations``.
* **Long-range context discovery** — ``long_range_delta_bic_threshold``,
  ``long_range_min_split_observations``,
  ``long_range_min_dominant_fraction``.
* **Cross-dimensional discovery** — ``cross_dim_max_iterations``,
  ``cross_dim_min_rule_count``, ``cross_dim_min_rule_confidence``.
* **Multi-lect reconciliation** — ``multi_lect_bic_small_sample_correction``,
  ``multi_lect_min_commit_scale``.

Every field has a one-line rationale in the docstring so future
contributors don't have to re-derive them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BICConfig:
    """Tunable thresholds for the BIC-gated discovery stages.

    The defaults match the historical constants used in
    ``_discovery``, ``_chunks``, ``_cross_dimensional``, and
    ``_reconciliation``. A default-constructed ``BICConfig`` is
    bitwise-equivalent to the pre-config behavior, so training
    is backward compatible.

    Tighten to reject more commits on noisy data; loosen to
    commit more rules on sparse data. Individual fields are
    documented below.
    """

    #: Safety buffer on ΔBIC for immediate-neighbour context splits.
    #: A split is committed only when its BIC delta is below this
    #: threshold (i.e. more negative than this). -1.0 rejects the
    #: near-zero spurious splits observed on Latin-Spanish during
    #: calibration.
    delta_bic_threshold: float = -1.0

    #: Minimum observation count (weighted) for a split branch.
    #: Prevents degenerate 1-observation partitions.
    min_split_observations: int = 2

    #: Maximum number of context splits committed on the same
    #: source grapheme. Caps pathological multi-level splits.
    max_split_depth: int = 3

    #: Minimum chunk observations for promotion consideration.
    #: Below this, Laplace-smoothed MLE gives ``P = 1`` from a
    #: single observation and BIC cannot reject the spurious
    #: promotion.
    min_chunk_observations: int = 2

    #: ΔBIC threshold for long-range context splits. Stricter than
    #: the immediate-neighbour threshold because the long-range
    #: candidate space is ~4× larger.
    long_range_delta_bic_threshold: float = -5.0

    #: Minimum weighted observation count per branch for long-range
    #: splits. Stricter than the immediate-neighbour floor.
    long_range_min_split_observations: int = 5

    #: Minimum fraction of YES observations that must carry the
    #: same target outcome for a long-range split to commit.
    #: Rejects grab-bag splits that lump heterogeneous rare
    #: outcomes.
    long_range_min_dominant_fraction: float = 0.6

    #: Maximum sequential-greedy iterations in cross-dimensional
    #: discovery. Safety valve for pathological inputs.
    cross_dim_max_iterations: int = 5

    #: Minimum support count for a committed cross-dimensional
    #: rule. BIC alone admits ``count=1/N`` noise on corpora with
    #: many rare tonal outcomes.
    cross_dim_min_rule_count: int = 3

    #: Minimum ``confidence`` (``count / src_count``) for a
    #: committed cross-dimensional rule.
    cross_dim_min_rule_confidence: float = 0.5

    #: AICc-style small-sample correction on the multi-lect
    #: class-discovery BIC penalty. Without it, tiny pivot
    #: buckets pass BIC too easily.
    multi_lect_bic_small_sample_correction: bool = True

    #: Scales the adaptive floor for per-sister-tuple emissions
    #: in the multi-lect class-discovery loop. Floor is
    #: ``max(2, ceil(scale * log2(n+1)))``; ``0.0`` disables the
    #: adaptive floor entirely.
    multi_lect_min_commit_scale: float = 0.5

    def __post_init__(self) -> None:
        """Validate field bounds; fail fast on nonsense values."""
        if self.min_split_observations < 1:
            raise ValueError(
                f"min_split_observations must be >= 1, got "
                f"{self.min_split_observations}"
            )
        if self.max_split_depth < 1:
            raise ValueError(
                f"max_split_depth must be >= 1, got {self.max_split_depth}"
            )
        if self.min_chunk_observations < 1:
            raise ValueError(
                f"min_chunk_observations must be >= 1, got "
                f"{self.min_chunk_observations}"
            )
        if self.long_range_min_split_observations < 1:
            raise ValueError(
                f"long_range_min_split_observations must be >= 1, got "
                f"{self.long_range_min_split_observations}"
            )
        if not 0.0 <= self.long_range_min_dominant_fraction <= 1.0:
            raise ValueError(
                f"long_range_min_dominant_fraction must be in [0, 1], got "
                f"{self.long_range_min_dominant_fraction}"
            )
        if self.cross_dim_max_iterations < 1:
            raise ValueError(
                f"cross_dim_max_iterations must be >= 1, got "
                f"{self.cross_dim_max_iterations}"
            )
        if self.cross_dim_min_rule_count < 1:
            raise ValueError(
                f"cross_dim_min_rule_count must be >= 1, got "
                f"{self.cross_dim_min_rule_count}"
            )
        if not 0.0 <= self.cross_dim_min_rule_confidence <= 1.0:
            raise ValueError(
                f"cross_dim_min_rule_confidence must be in [0, 1], got "
                f"{self.cross_dim_min_rule_confidence}"
            )
        if self.multi_lect_min_commit_scale < 0.0:
            raise ValueError(
                f"multi_lect_min_commit_scale must be >= 0, got "
                f"{self.multi_lect_min_commit_scale}"
            )


# Module-level singleton for backward-compat paths that want "the
# default" without constructing a fresh instance each call.
DEFAULT_BIC_CONFIG: BICConfig = BICConfig()
