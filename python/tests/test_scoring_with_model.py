"""Tests for model-aware link scoring.

The key commitments:

* **Prior-only compatibility**: ``score_link(link, model=None)`` returns the
  same value as the M2 ``score_link(link)``.
* **Prior-only model**: an ``empty()`` model scores segment pairs
  via merkmal fallback (since the prior pseudo-counts are empty).
* **Learned segment table**: adding counts for a specific segment pair
  makes it cheaper than an untrained pair with the same merkmal
  distance.
* **Displacement layer**: adding displacement counts makes pairs with
  the corresponding displacement cheaper.
* **Chunk table override**: a chunk pair in the phrase table uses the
  stored cost directly, ignoring compositional scoring.
"""

import math

import pytest

from regulae import (
    ConditionedCorrespondence,
    Context,
    Link,
    Segment,
    compute_displacement,
    score_link,
)
from regulae.model import (
    ChunkPhraseTable,
    DisplacementDistribution,
    LearnedModel,
    SegmentCorrespondenceTable,
)


def _cc(src: str, tgt: str, context: Context | None = None) -> ConditionedCorrespondence:
    """Shorthand for building a ConditionedCorrespondence key."""
    return ConditionedCorrespondence(src=src, tgt=tgt, context=context or Context())


# ----- M2 compatibility ---------------------------------------------------


def test_score_link_with_model_none_matches_m2_behavior() -> None:
    """COMMITMENT: ``model=None`` reproduces the M2 scoring exactly."""
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    without = score_link(link)
    with_none = score_link(link, model=None)
    assert without == with_none


def test_score_link_with_model_none_for_chunks() -> None:
    link = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t")),
    )
    without = score_link(link)
    with_none = score_link(link, model=None)
    assert without == with_none


def test_empty_model_falls_back_to_merkmal_for_1_to_1() -> None:
    """COMMITMENT: an empty model (no prior, no data) produces the same
    cost as merkmal distance, because the fallback kicks in for any
    segment pair not in the (empty) prior."""
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    empty = LearnedModel.empty()
    assert score_link(link, model=empty) == score_link(link)


# ----- Learned segment table lowers cost for seen pairs ------------------


def _table_with_strong_pf_correspondence() -> SegmentCorrespondenceTable:
    """A segment table where ``p`` strongly prefers ``f``: heavy observed
    counts for (p, f) and a uniform prior over two candidates (p, f) and
    (p, p)."""
    return SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 50.0, _cc("p", "p"): 0.0},
        prior_pseudo_counts={_cc("p", "f"): 1.0, _cc("p", "p"): 1.0},
        src_totals={"p": 50.0},
    )


def test_strong_learned_pf_is_cheaper_than_unlearned() -> None:
    """COMMITMENT: a segment pair with many observations scores lower
    than the same pair under an empty (prior-only) model."""
    model = LearnedModel(
        segment_table=_table_with_strong_pf_correspondence(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        feature_system="descriptive",
        concentration=5.0,
    )
    link_pf = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    score_learned = score_link(link_pf, model=model)
    score_empty = score_link(link_pf, model=None)
    assert score_learned < score_empty


def test_learned_rare_is_costlier_than_prior_fallback() -> None:
    """COMMITMENT: a segment pair with low learned probability is more
    expensive than the merkmal-only fallback of the same pair, because
    the learned model has seen many other things instead."""
    # In this table, p has been observed often as f and not often as p.
    # P(p | p) is very small.
    model = LearnedModel(
        segment_table=_table_with_strong_pf_correspondence(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        feature_system="descriptive",
        concentration=5.0,
    )
    link_pp = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("p"),))
    # The learned model makes this identity expensive because P(p | p) is small.
    # Empty-model score for identity would be 0.
    # We check that the learned cost is positive — identity is no longer free.
    score_learned = score_link(link_pp, model=model)
    assert score_learned > 0.0


# ----- Displacement layer adds generalization ----------------------------


def test_displacement_layer_lowers_cost_when_active() -> None:
    """COMMITMENT: adding displacement counts for a feature change makes
    segment pairs carrying that displacement cheaper.

    This test constructs two scenarios differing only in whether the
    displacement distribution has been populated, and verifies the
    cost drops with displacement data.
    """
    # Compute the actual displacement for (p, f) so we can populate the
    # distribution with the right key.
    pf_disp = compute_displacement(Segment("p"), Segment("f"))

    table = SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 1.0},
        prior_pseudo_counts={_cc("p", "f"): 1.0, _cc("p", "p"): 1.0},
        src_totals={"p": 1.0},
    )

    # Case A: empty displacement distribution — segment-only scoring.
    model_a = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )

    # Case B: displacement distribution has strong support for the
    # (p, f) displacement.
    model_b = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(
            counts={pf_disp: 100.0},
            total=100.0,
        ),
        chunk_table=ChunkPhraseTable(),
    )

    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    cost_a = score_link(link, model=model_a)
    cost_b = score_link(link, model=model_b)
    assert cost_b < cost_a


# ----- Chunk table override ----------------------------------------------


def test_chunk_table_entry_overrides_compositional_scoring() -> None:
    """COMMITMENT: when a chunk is in the phrase table, its stored cost
    is used directly, regardless of compositional cost."""
    kt = (Segment("k"), Segment("t"))
    tt = (Segment("t"), Segment("t"))
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(entries={(kt, tt): 0.01}),
    )
    link = Link(source_chunk=kt, target_chunk=tt)
    assert score_link(link, model=model) == pytest.approx(0.01)


def test_chunk_table_does_not_affect_other_chunks() -> None:
    """COMMITMENT: chunk table entries are keyed exactly; similar but
    distinct chunks are NOT matched."""
    kt = (Segment("k"), Segment("t"))
    tt = (Segment("t"), Segment("t"))
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(entries={(kt, tt): 0.01}),
    )
    # A different chunk not in the table.
    pt = (Segment("p"), Segment("t"))
    link = Link(source_chunk=pt, target_chunk=tt)
    # This chunk is NOT in the table, so it falls through to compositional.
    # The cost should be the compositional cost, which is not 0.01.
    assert score_link(link, model=model) != pytest.approx(0.01)


# ----- Unseen-in-prior fallback ------------------------------------------


def test_unseen_segment_pair_falls_back_to_merkmal() -> None:
    """COMMITMENT: a 1-to-1 link with a segment pair not in the prior
    pseudo-counts falls back to the merkmal distance.

    This is the graceful degradation path: a trained model aligned
    against a form containing out-of-corpus segments should not crash;
    it should produce a reasonable score."""
    # Model knows only about (p, f).
    model = LearnedModel(
        segment_table=SegmentCorrespondenceTable(
            counts={_cc("p", "f"): 1.0},
            prior_pseudo_counts={_cc("p", "f"): 1.0},
            src_totals={"p": 1.0},
        ),
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    # Score a pair (k, x) which is not in the prior.
    link = Link(source_chunk=(Segment("k"),), target_chunk=(Segment("x"),))
    expected = score_link(link, model=None)
    assert score_link(link, model=model) == expected


# ----- Symmetry is preserved with a model --------------------------------


def test_learned_scoring_preserves_link_symmetry_for_symmetric_model() -> None:
    """COMMITMENT: if a learned segment table is symmetric (each seen
    pair has its reverse with equal count), the layered cost is also
    symmetric under chunk swap.

    Note that a general learned model may not be symmetric — if only
    (p, f) is observed but (f, p) is not, the learned cost will be
    asymmetric, which reflects real data. This is acceptable because
    correspondences are symmetric at the SYSTEM level (the corpus
    implies both directions equally), but the learned model is conditioned
    on source. The test below uses a manually symmetrized table.
    """
    table = SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 5.0, _cc("f", "p"): 5.0},
        prior_pseudo_counts={
            _cc("p", "f"): 1.0,
            _cc("p", "p"): 1.0,
            _cc("f", "p"): 1.0,
            _cc("f", "f"): 1.0,
        },
        src_totals={"p": 5.0, "f": 5.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
    )
    fwd = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    bwd = Link(source_chunk=(Segment("f"),), target_chunk=(Segment("p"),))
    assert score_link(fwd, model=model) == pytest.approx(score_link(bwd, model=model))


# ----- Posterior math sanity ----------------------------------------------


def test_dirichlet_posterior_with_zero_counts_equals_prior() -> None:
    """COMMITMENT: when there are no observations, the Dirichlet posterior
    reduces to the prior probability.

    Given prior pseudo-counts alpha(p, f) = 3 and alpha(p, p) = 1, and
    zero data, the posterior P(f | p) = 3 / (concentration) = 3/5 if
    beta = 5.
    """
    table = SegmentCorrespondenceTable(
        counts={},
        prior_pseudo_counts={_cc("p", "f"): 3.0, _cc("p", "p"): 1.0},
        src_totals={},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        concentration=5.0,
    )
    # cost_seg = -log P(f | p) = -log(3 / 5) = -log(0.6)
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    cost = score_link(link, model=model)
    # Empty displacement → pure segment-only cost (weight 1.0 effectively)
    # Actually wait — with segment_weight=0.7 and empty displacement,
    # the helper returns cost_seg directly (not weighted). Let me verify.
    expected = -math.log(3.0 / 5.0)
    assert cost == pytest.approx(expected)


def test_dirichlet_posterior_updates_with_counts() -> None:
    """COMMITMENT: observed counts shift the posterior away from the
    prior. With prior (3, 1) and data (0, 10), the posterior for (p, p)
    should be higher than for (p, f)."""
    table = SegmentCorrespondenceTable(
        counts={_cc("p", "p"): 10.0},
        prior_pseudo_counts={_cc("p", "f"): 3.0, _cc("p", "p"): 1.0},
        src_totals={"p": 10.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        concentration=4.0,
    )
    link_pp = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("p"),))
    link_pf = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    cost_pp = score_link(link_pp, model=model)
    cost_pf = score_link(link_pf, model=model)
    # (p, p) is now the stronger correspondence; lower cost means preferred.
    assert cost_pp < cost_pf


# ----- Context-aware lookup fallback hierarchy -----------------------------


def test_conditioned_entry_beats_unconditioned_when_context_matches() -> None:
    """COMMITMENT: when a conditioned entry matches the link's context,
    it is used instead of the unconditioned fallback, producing a
    different cost."""
    from regulae import FeatureConstraint

    front_ctx = Context(following=(FeatureConstraint("front", "+"),))
    table = SegmentCorrespondenceTable(
        counts={
            _cc("k", "s"): 1.0,
            _cc("k", "k"): 10.0,
            _cc("k", "s", front_ctx): 8.0,
            _cc("k", "k", front_ctx): 2.0,
        },
        prior_pseudo_counts={
            _cc("k", "s"): 1.0,
            _cc("k", "k"): 1.0,
            _cc("k", "s", front_ctx): 1.0,
            _cc("k", "k", front_ctx): 1.0,
        },
        src_totals={"k": 11.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        concentration=2.0,
    )
    link_uncond = Link(
        source_chunk=(Segment("k"),),
        target_chunk=(Segment("s"),),
        context=Context(),
    )
    link_cond = Link(
        source_chunk=(Segment("k"),),
        target_chunk=(Segment("s"),),
        context=front_ctx,
    )
    cost_uncond = score_link(link_uncond, model=model)
    cost_cond = score_link(link_cond, model=model)
    assert cost_cond < cost_uncond


def test_unconditioned_fallback_used_when_no_context_match() -> None:
    """COMMITMENT: a conditioned entry that doesn't match the link's
    context is ignored; the unconditioned entry is used."""
    from regulae import FeatureConstraint
    from regulae.scoring import _segment_posterior

    front_ctx = Context(following=(FeatureConstraint("front", "+"),))
    back_ctx = Context(following=(FeatureConstraint("back", "+"),))
    table = SegmentCorrespondenceTable(
        counts={
            _cc("k", "s"): 5.0,
            _cc("k", "s", front_ctx): 8.0,
        },
        prior_pseudo_counts={
            _cc("k", "s"): 1.0,
            _cc("k", "s", front_ctx): 1.0,
        },
        src_totals={"k": 13.0},
    )
    p_back = _segment_posterior("k", "s", back_ctx, table, 2.0)
    p_empty = _segment_posterior("k", "s", Context(), table, 2.0)
    assert p_back == p_empty


def test_segment_posterior_returns_none_for_unseen_pair() -> None:
    """COMMITMENT: a segment pair not in the prior returns None,
    signaling merkmal fallback."""
    from regulae.scoring import _segment_posterior

    table = SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 5.0},
        prior_pseudo_counts={_cc("p", "f"): 1.0},
        src_totals={"p": 5.0},
    )
    result = _segment_posterior("x", "y", Context(), table, 2.0)
    assert result is None


def test_gap_link_cost_with_model_equals_without() -> None:
    """COMMITMENT: gap links use fixed costs regardless of the model."""
    table = SegmentCorrespondenceTable(
        counts={_cc("p", "f"): 10.0},
        prior_pseudo_counts={_cc("p", "f"): 1.0},
        src_totals={"p": 10.0},
    )
    model = LearnedModel(
        segment_table=table,
        displacement_dist=DisplacementDistribution(),
        chunk_table=ChunkPhraseTable(),
        concentration=5.0,
    )
    gap_link = Link(source_chunk=(Segment("p"),), target_chunk=())
    cost_with = score_link(gap_link, model=model)
    cost_without = score_link(gap_link, model=None)
    assert cost_with == cost_without
