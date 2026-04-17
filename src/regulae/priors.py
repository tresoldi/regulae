"""Typological priors on segment-pair correspondences.

A :class:`TypologicalPrior` is any callable
``(src_grapheme, tgt_grapheme) -> float`` returning a log-prior
adjustment in nats. The adjustment is added to the merkmal
feature-distance logit before the softmax that produces the
Dirichlet pseudo-counts in :func:`regulae._em._initial_model`:

.. code-block:: text

    logit(s, t) = -temperature * d(s, t) + prior(s, t)

Positive returns boost the prior probability of ``t`` given ``s``;
negative returns suppress it; zero leaves the merkmal-distance
prior unchanged.

This module ships with **no opinionated priors**. The framework
deliberately does not encode "lenition is more likely than
fortition" or any other typological asymmetry as a default —
the project's stance is that those asymmetries belong with the
user, not the framework. :func:`uniform` returns a no-op prior
that preserves the current merkmal-only behavior.

Users who want to inject typological knowledge construct their
own callable. Example shapes:

.. code-block:: python

    # Boost a specific correspondence pair (e.g., known
    # palatalization in this family).
    def boost_k_to_tch(s, t):
        if s == "k" and t == "tʃ":
            return 1.0
        return 0.0

    # Use a typology table (Mapping[(s, t), float]).
    table = {("p", "f"): 0.5, ("t", "θ"): 0.5, ("k", "x"): 0.5}
    def from_table(s, t):
        return table.get((s, t), 0.0)

    # Compose a uniform prior with a sparse table.
    prior = combine(uniform(), from_table)

The prior is consulted only at the prior-construction step. It does
not influence alignment scoring directly (segment scoring uses the
trained Dirichlet posterior, into which the prior has already been
folded as pseudo-count mass).
"""

from collections.abc import Callable
from typing import TypeAlias

#: A typological-prior callable. Takes ``(src_grapheme, tgt_grapheme)``
#: and returns a log-prior adjustment in nats. Implementations must
#: be pure functions: identical inputs must produce identical outputs
#: across calls so training stays deterministic.
TypologicalPrior: TypeAlias = Callable[[str, str], float]


def uniform() -> TypologicalPrior:
    """Return the no-op prior: zero adjustment for every pair.

    Preserves the merkmal-only behavior. This is the default for
    :func:`regulae.train_model` and exists so that the prior hook
    can be passed through training pipelines without conditional
    branches.
    """
    def _uniform(_src: str, _tgt: str) -> float:
        return 0.0

    return _uniform


def combine(*priors: TypologicalPrior) -> TypologicalPrior:
    """Sum multiple priors. Returns a new callable that adds each
    contributing prior's adjustment.

    Useful for composing a sparse user-supplied table with a
    structural prior, or for stacking family-level and
    pair-specific priors.
    """
    def _combined(src: str, tgt: str) -> float:
        return sum(p(src, tgt) for p in priors)

    return _combined
