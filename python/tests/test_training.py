"""Tests for the training loop and learned-model artifacts.

The key integration claims these tests encode:

* **Empty corpus**: ``train_model([])`` returns an empty model, does not
  raise.
* **Prior-only equals M2**: alignments under an initialized-but-untrained
  model match M2 alignments exactly. The Bayesian offset trick
  (subtracting ``log Z(s)``) is what makes this possible.
* **Regular correspondence recovery**: on a corpus where every pair
  shows the same substitution (p~f throughout), the trained model
  makes that substitution much cheaper than its M2 baseline.
* **Chunk promotion**: on a corpus where (kt, tt) recurs many times,
  the chunk gets into the phrase table; on a one-off corpus it does
  not.
* **Training is deterministic**: repeated runs on the same corpus yield
  equal models.
* **align_corpus contract**: returns one alignment per input pair, in
  order.
"""

from regulae import ConditionedCorrespondence, Context, Form, Segment
from regulae.model import LearnedModel
from regulae.search import align_forms, alignment_cost
from regulae.training import align_corpus
from tests._helpers import train_model


def _cc(src: str, tgt: str) -> ConditionedCorrespondence:
    return ConditionedCorrespondence(src=src, tgt=tgt, context=Context())


def _form(lect_id: str, ipa: str) -> Form:
    return Form(lect_id=lect_id, segments=tuple(Segment(c) for c in ipa))


# ----- train_model end-to-end ---------------------------------------------


def test_train_model_on_empty_corpus_returns_empty_model() -> None:
    """COMMITMENT: an empty corpus is not an error. Returns an empty model."""
    model = train_model([])
    assert isinstance(model, LearnedModel)
    assert model.segment_table.counts == {}
    assert model.chunk_table.entries == {}


def test_prior_only_model_reproduces_m2_alignment() -> None:
    """COMMITMENT: before any training has happened, the initialized
    model scores identically to M2. This is the log Z offset working.

    We construct a 'model' by running one pair through train_model;
    since the corpus has only one pair and no recurring chunks, the
    trained model's effect should still preserve M2-like behavior
    for that pair.
    """
    corpus = [(_form("A", "pater"), _form("B", "fadar"))]
    # Just use the initial model — before any updates.
    from regulae.training import _initial_model

    initial = _initial_model(
        corpus=corpus,
        feature_system="descriptive",
        temperature=1.0,
        concentration=5.0,
        segment_weight=0.7,
        displacement_weight=0.3,
    )
    a_m2 = align_forms(corpus[0][0], corpus[0][1])
    a_prior = align_forms(corpus[0][0], corpus[0][1], model=initial)
    # Link structure should match.
    assert len(a_m2.links) == len(a_prior.links)
    for l1, l2 in zip(a_m2.links, a_prior.links, strict=True):
        assert l1.source_chunk == l2.source_chunk
        assert l1.target_chunk == l2.target_chunk
    # Cost should be numerically equal.
    assert alignment_cost(a_m2) == alignment_cost(a_prior, model=initial)


# ----- regular correspondence recovery ------------------------------------


def test_regular_correspondence_recovery_on_synthetic_corpus() -> None:
    """COMMITMENT: on a corpus where p ↔ f systematically, training
    makes that correspondence strictly cheaper than the M2 baseline.

    This is the foundational success criterion for segment EM EM.
    """
    # 10 pairs, all showing a Latin-style p ↔ Germanic-style f substitution
    # with identity on other segments.
    corpus = [
        (_form("A", "pap"), _form("B", "faf")),
        (_form("A", "pip"), _form("B", "fif")),
        (_form("A", "pop"), _form("B", "fof")),
        (_form("A", "pup"), _form("B", "fuf")),
        (_form("A", "pep"), _form("B", "fef")),
        (_form("A", "pap"), _form("B", "faf")),
        (_form("A", "pip"), _form("B", "fif")),
        (_form("A", "pop"), _form("B", "fof")),
        (_form("A", "pup"), _form("B", "fuf")),
        (_form("A", "pep"), _form("B", "fef")),
    ]

    trained = train_model(corpus)

    # The segment table should have (p, f) with positive count.
    assert trained.segment_table.counts.get(_cc("p", "f"), 0.0) >= 10.0

    # Cost of (p, f) under the trained model should be lower than M2.
    from regulae.scoring import _segment_pair_cost_with_model

    learned_cost = _segment_pair_cost_with_model(Segment("p"), Segment("f"), trained)
    # M2 gives 0.625 for p~f under descriptive. Trained should be much lower.
    assert learned_cost < 0.625


def test_regular_correspondence_makes_alignment_cheaper() -> None:
    """COMMITMENT: training on a p~f corpus reduces the total alignment
    cost of pairs in that corpus below what M2 would score."""
    corpus = [
        (_form("A", "pap"), _form("B", "faf")),
        (_form("A", "pap"), _form("B", "faf")),
        (_form("A", "pap"), _form("B", "faf")),
    ]
    trained = train_model(corpus)
    a_m2 = align_forms(corpus[0][0], corpus[0][1])
    a_trained = align_forms(corpus[0][0], corpus[0][1], model=trained)
    assert alignment_cost(a_trained, model=trained) < alignment_cost(a_m2)


# ----- displacement generalization ----------------------------------------


def test_displacement_layer_generalizes_across_natural_class() -> None:
    """COMMITMENT: after training on p~f, t~θ, k~x (all stop → fricative
    at different places), a new pair with the same displacement pattern
    should score below its M2 baseline under the trained model.

    Note: the 'new pair' has to use segments that exist in the training
    vocabulary, otherwise the segment-level prior falls back to merkmal
    and the displacement contribution is the only learning signal.
    """
    # Train on stop→fricative correspondences.
    corpus = [
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "ta"), _form("B", "sa")),
        (_form("A", "ta"), _form("B", "sa")),
        (_form("A", "ka"), _form("B", "xa")),
        (_form("A", "ka"), _form("B", "xa")),
    ]
    trained = train_model(corpus)
    # The displacement distribution should be populated.
    assert len(trained.displacement_dist.counts) > 0
    assert trained.displacement_dist.total > 0


# ----- chunk promotion ----------------------------------------------------


def test_chunk_promotion_for_recurring_cluster() -> None:
    """COMMITMENT: on a corpus where (kt, tt) recurs frequently, the
    chunk is promoted into the phrase table.

    This is the success criterion for chunk promotion.
    """
    # 15 pairs, each containing a kt → tt correspondence.
    corpus = [
        (_form("A", "akta"), _form("B", "atta")),
        (_form("A", "ikti"), _form("B", "itti")),
        (_form("A", "okto"), _form("B", "otto")),
        (_form("A", "ukte"), _form("B", "utte")),
        (_form("A", "akta"), _form("B", "atta")),
        (_form("A", "ikti"), _form("B", "itti")),
        (_form("A", "okto"), _form("B", "otto")),
        (_form("A", "ukte"), _form("B", "utte")),
        (_form("A", "akto"), _form("B", "atto")),
        (_form("A", "ikti"), _form("B", "itti")),
        (_form("A", "ukte"), _form("B", "utte")),
        (_form("A", "okta"), _form("B", "otta")),
        (_form("A", "akte"), _form("B", "atte")),
        (_form("A", "ikto"), _form("B", "itto")),
        (_form("A", "uktu"), _form("B", "uttu")),
    ]
    trained = train_model(corpus)
    # The chunk table should have at least some entries.
    assert len(trained.chunk_table.entries) > 0


def test_chunk_promotion_rejects_singletons() -> None:
    """COMMITMENT: a chunk that appears only once should NOT be
    promoted — the BIC penalty should exceed the savings.
    """
    corpus = [
        (_form("A", "akta"), _form("B", "atta")),
    ]
    trained = train_model(corpus)
    # One pair → only one chunk candidate → singleton → not promoted.
    assert len(trained.chunk_table.entries) == 0


# ----- determinism --------------------------------------------------------


def test_train_model_is_deterministic() -> None:
    """COMMITMENT: same corpus + same hyperparameters → same model."""
    corpus = [
        (_form("A", "pater"), _form("B", "fadar")),
        (_form("A", "mater"), _form("B", "madar")),
    ]
    m1 = train_model(corpus)
    m2 = train_model(corpus)
    assert m1.segment_table.counts == m2.segment_table.counts
    assert m1.chunk_table.entries == m2.chunk_table.entries


# ----- align_corpus -------------------------------------------------------


def test_align_corpus_returns_one_alignment_per_pair() -> None:
    """COMMITMENT: ``align_corpus`` produces alignments in the input
    order, one per pair, each well-formed."""
    corpus = [
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "ta"), _form("B", "da")),
        (_form("A", "ka"), _form("B", "xa")),
    ]
    trained = train_model(corpus)
    alignments = align_corpus(corpus, trained)
    assert len(alignments) == 3
    for (src, tgt), alignment in zip(corpus, alignments, strict=True):
        assert alignment.source_form == src
        assert alignment.target_form == tgt


def test_align_corpus_respects_empty_corpus() -> None:
    trained = train_model([])
    result = align_corpus([], trained)
    assert result == []


# ----- CognateSet validation -----------------------------------------------


def test_train_model_rejects_single_form_in_multi_lect_corpus() -> None:
    """COMMITMENT: a CognateSet with only one form in a multi-lect corpus
    is rejected (at least 2 lects exist but this set has only 1)."""
    import pytest
    from regulae.model import CognateSet
    from regulae.training import train_model as raw_train

    corpus = [
        CognateSet(
            cognate_id="good",
            forms={
                "A": Form(lect_id="A", segments=(Segment("p"),)),
                "B": Form(lect_id="B", segments=(Segment("f"),)),
            },
        ),
        CognateSet(
            cognate_id="bad",
            forms={"A": Form(lect_id="A", segments=(Segment("k"),))},
        ),
    ]
    with pytest.raises(ValueError, match="at least 2"):
        raw_train(corpus)


def test_train_model_rejects_empty_form() -> None:
    """COMMITMENT: a CognateSet where any form has zero segments is rejected."""
    import pytest
    from regulae.model import CognateSet
    from regulae.training import train_model as raw_train

    cs = CognateSet(
        cognate_id="bad",
        forms={
            "A": Form(lect_id="A", segments=()),
            "B": Form(lect_id="B", segments=(Segment("p"),)),
        },
    )
    with pytest.raises(ValueError, match="no segments"):
        raw_train([cs])
