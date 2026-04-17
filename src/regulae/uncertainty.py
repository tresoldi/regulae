"""Uncertainty quantification for count-bearing entries.

Every count in a trained model is an estimate from finite observations.
Downstream consumers (``historia`` especially) need an interval on each
count to weight it against others and to propagate uncertainty into
posteriors over histories.

This module provides:

* :class:`UncertaintyEstimate` — a lightweight dataclass carrying a
  two-sided interval on a rate (or count-derived rate), along with the
  method that produced it so consumers can tell what it means.
* :func:`wilson_interval` — closed-form Wilson score interval on a
  binomial rate. Default method; used for every count at training
  time with no measurable overhead.
* :func:`percentile_interval` — distribution-free percentile interval,
  used by the bootstrap loop to summarize per-entry count samples.

Wilson is the default because:

* It is calibrated well for small n (unlike the normal approximation).
* It has no pathological behavior at rate = 0 or rate = 1 (unlike the
  normal).
* It is closed-form in elementary functions (unlike Jeffreys, which
  requires the inverse incomplete beta).
* Its calibration matches Jeffreys at n ≥ 5 and is marginally better
  for very small n.

The ``UncertaintyEstimate.method`` string is kept so a future switch to
Jeffreys (e.g. once scipy becomes an optional dependency) can coexist
with Wilson in the same model without ambiguity.

The interval is always reported on a **rate** in ``[0, 1]`` — the
conditional probability the count represents. For each count-bearing
object the denominator is documented at the point of use; see
``docs/training_pipeline.md`` for the mapping.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

# Two-sided z-score for a 95% Wilson interval. Hard-coded so there's no
# scipy dependency; change ``DEFAULT_Z`` if a different confidence level
# becomes a configuration knob later.
DEFAULT_ALPHA: float = 0.05
_Z_BY_ALPHA: dict[float, float] = {
    0.10: 1.6448536269514722,
    0.05: 1.959963984540054,
    0.01: 2.5758293035489004,
}


@dataclass(frozen=True)
class UncertaintyEstimate:
    """A two-sided interval on a rate, plus provenance.

    ``lo`` and ``hi`` bracket the rate (a conditional probability in
    ``[0, 1]``) estimated by the count. The interval's width reflects
    how much data backs that rate: a count of 3 out of 4 produces a
    much wider interval than 300 out of 400, even though both have
    point estimate 0.75.

    ``method`` records how the interval was produced:

    * ``"wilson"`` — Wilson score interval, closed form, computed
      at training time by default.
    * ``"bootstrap"`` — percentile interval across ``N`` resampled
      trainings, computed when ``train_model`` is called with
      ``bootstrap_n > 0``.

    ``n`` is the denominator used (e.g. ``src_totals[src]`` for a
    segment-level correspondence, ``pivot_bucket_size`` for a
    multi-lect class). ``alpha`` is the significance level the interval
    was constructed for (0.05 = 95% CI).

    ``UncertaintyEstimate`` fields are declared ``compare=False`` when
    attached to frozen dataclasses elsewhere, so a model's equality and
    hash are unaffected by the presence or shape of uncertainty.
    """

    lo: float
    hi: float
    method: str
    n: float = field(default=0.0)
    alpha: float = field(default=DEFAULT_ALPHA)


def wilson_interval(k: float, n: float, alpha: float = DEFAULT_ALPHA) -> UncertaintyEstimate:
    """Wilson score interval on the rate ``k / n``.

    Handles ``n <= 0`` by returning ``[0, 1]``; handles ``k <= 0`` and
    ``k >= n`` with the usual Wilson boundary behavior (a one-sided
    interval that touches 0 or 1 rather than going negative or above
    1). Accepts fractional ``k`` and ``n`` because confidence-weighted
    training produces non-integer counts.

    The formula:

        z = Phi^-1(1 - alpha/2)
        p_hat = k / n
        center = (p_hat + z^2 / (2n)) / (1 + z^2/n)
        half_width = (z / (1 + z^2/n)) * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4 n^2))
        lo, hi = center - half_width, center + half_width
    """
    if alpha not in _Z_BY_ALPHA:
        raise ValueError(
            f"alpha {alpha!r} not supported; expected one of {sorted(_Z_BY_ALPHA)}."
        )
    if n <= 0.0:
        return UncertaintyEstimate(lo=0.0, hi=1.0, method="wilson", n=0.0, alpha=alpha)
    z = _Z_BY_ALPHA[alpha]
    p_hat = max(0.0, min(1.0, k / n))
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    inner = p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)
    half = (z / denom) * math.sqrt(max(inner, 0.0))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return UncertaintyEstimate(lo=lo, hi=hi, method="wilson", n=n, alpha=alpha)


def percentile_interval(
    samples: Sequence[float],
    *,
    n: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
) -> UncertaintyEstimate:
    """Percentile interval over a distribution of samples.

    Used by the bootstrap loop: each sample is the rate (or count) from
    one resampled training. The returned ``lo``/``hi`` are the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the sample
    distribution. Empty ``samples`` returns ``[0, 1]``.

    ``n`` records the denominator associated with the point estimate,
    not the sample size; for a bootstrap of 100 resamples of a training
    where a rate has denominator 40, store ``n=40`` here (the sample
    size goes implicitly into the width of the distribution).
    """
    if not samples:
        return UncertaintyEstimate(lo=0.0, hi=1.0, method="bootstrap", n=n, alpha=alpha)
    ordered = sorted(samples)
    lo = _quantile(ordered, alpha / 2.0)
    hi = _quantile(ordered, 1.0 - alpha / 2.0)
    return UncertaintyEstimate(lo=lo, hi=hi, method="bootstrap", n=n, alpha=alpha)


def _quantile(ordered: list[float], q: float) -> float:
    """Linear-interpolated quantile on a pre-sorted list. q in [0, 1]."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    pos = q * (len(ordered) - 1)
    lo_idx = int(math.floor(pos))
    hi_idx = int(math.ceil(pos))
    if lo_idx == hi_idx:
        return ordered[lo_idx]
    frac = pos - lo_idx
    return ordered[lo_idx] * (1.0 - frac) + ordered[hi_idx] * frac


# ----- bootstrap ---------------------------------------------------------


def bootstrap_rate_interval(
    samples: Sequence[float],
    n: float,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> UncertaintyEstimate:
    """Bootstrap percentile interval on a rate.

    ``samples`` is the distribution of rates (in ``[0, 1]``) observed
    across resampled trainings — each sample is computed *within* its
    own resample's denominator, not relative to the base ``n``. ``n``
    is the base denominator, stored for reference only.

    The returned ``lo``/``hi`` are the ``alpha/2`` and ``1-alpha/2``
    quantiles of the rate distribution, each clamped to ``[0, 1]``.

    An empty ``samples`` returns ``[0, 1]`` (no information).
    """
    if not samples:
        return UncertaintyEstimate(lo=0.0, hi=1.0, method="bootstrap", n=n, alpha=alpha)
    ordered = sorted(max(0.0, min(1.0, s)) for s in samples)
    lo = _quantile(ordered, alpha / 2.0)
    hi = _quantile(ordered, 1.0 - alpha / 2.0)
    return UncertaintyEstimate(lo=lo, hi=hi, method="bootstrap", n=n, alpha=alpha)
