"""Shared test helpers for the regulae test suite.

``train_model`` here is a thin wrapper around ``regulae.train_model``
that routes legacy pair-list input through the CognateSet API so
legacy tests don't trigger the pair-input DeprecationWarning. The
legacy tests still exercise the underlying pair-training pipeline —
they're kept as regression coverage until the pair branch is removed.
"""

from regulae import cognate_sets_from_pairs
from regulae.training import train_model as _public_train_model


def train_model(corpus, **kwargs):
    """Drop-in wrapper: accepts a pair list or CognateSet list.

    For pair input this converts to CognateSets with the synthetic
    ``("src", "tgt")`` lect IDs, runs the real ``train_model``, and
    unwraps the single pair's :class:`LearnedModel`. That keeps the
    legacy tests' ``trained.segment_table`` access patterns working.
    """
    if corpus and isinstance(corpus[0], tuple):
        multi = _public_train_model(
            cognate_sets_from_pairs(corpus, ("src", "tgt")),
            **kwargs,
        )
        return multi.pairwise_models[frozenset({"src", "tgt"})]
    return _public_train_model(corpus, **kwargs)
