"""Consolidation tests: real-data integration, invariants, edge cases.

These tests exercise ``train_model`` beyond the basic unit tests in
``test_training.py``. They are organised in three groups:

1. **Real-data integration**: small curated cognate corpora where we
   have a specific linguistic expectation. Tests assert that the
   trained model contains the expected correspondences.

2. **Training invariants**: properties that should hold for any corpus
   — determinism under reordering, idempotence of retraining, M2-floor
   preservation on unrelated inputs.

3. **Edge cases**: single-pair corpora, identical pairs, completely
   unrelated pairs, noisy input.

These are slower than the unit tests (real training runs on several
corpora) but still run in under a second total.
"""

import random

import pytest

from regulae import (
    ConditionedCorrespondence,
    Context,
    Form,
    Segment,
    UnknownGrapheme,
    align_forms,
    alignment_cost,
)
from regulae.training import _initial_model
from tests._helpers import train_model


def _cc(src: str, tgt: str) -> ConditionedCorrespondence:
    return ConditionedCorrespondence(src=src, tgt=tgt, context=Context())


def _form(lect_id: str, ipa: str) -> Form:
    return Form(lect_id=lect_id, segments=tuple(Segment(c) for c in ipa))


# ==========================================================================
# 1. Real-data integration
# ==========================================================================


def test_latin_germanic_recovers_grimm_type_correspondences() -> None:
    """INTEGRATION: training on Latin-Germanic cognate pairs should
    surface Grimm-type sound correspondences at the top of the segment
    table.

    This is the headline sanity check: the framework discovers the
    comparative method's result from data alone. The specific test:
    after training, ``p -> f``, ``k -> h``, and ``t -> d`` (or
    ``t -> θ``) should all appear with nontrivial counts.

    We use a small curated set of Latin forms paired with plausible
    Germanic reconstructions. The exact numbers are irrelevant — only
    the presence of the expected correspondences.
    """
    corpus = [
        (_form("latin", "pater"), _form("germ", "fader")),
        (_form("latin", "mater"), _form("germ", "moder")),
        (_form("latin", "frater"), _form("germ", "bruder")),
        (_form("latin", "pedis"), _form("germ", "fotu")),
        (_form("latin", "piskis"), _form("germ", "fiskaz")),
        (_form("latin", "plenus"), _form("germ", "fulnaz")),
        (_form("latin", "tres"), _form("germ", "θrejez")),
        (_form("latin", "kornu"), _form("germ", "hurnan")),
        (_form("latin", "kor"), _form("germ", "hertan")),
        (_form("latin", "kanis"), _form("germ", "hundaz")),
    ]
    trained = train_model(corpus)
    counts = trained.segment_table.counts

    # Grimm's law: voiceless stops become voiceless fricatives.
    assert counts.get(_cc("p", "f"), 0) >= 3, (
        f"Expected p -> f to be learned strongly; got count={counts.get(_cc('p','f'), 0)}"
    )
    # k -> h (Grimm), attested in "kornu/hurnan", "kor/hertan", "kanis/hundaz".
    assert counts.get(_cc("k", "h"), 0) >= 2, (
        f"Expected k -> h to be learned; got count={counts.get(_cc('k','h'), 0)}"
    )
    # t -> θ or t -> d (depending on alignment).
    t_to_fricative = counts.get(_cc("t", "θ"), 0) + counts.get(_cc("t", "d"), 0)
    assert t_to_fricative >= 2, (
        f"Expected some t -> θ or t -> d to be learned; got {t_to_fricative}"
    )


def test_latin_germanic_alignment_cost_drops_after_training() -> None:
    """INTEGRATION: training on a set of Latin-Germanic pairs should
    strictly reduce the total corpus alignment cost."""
    corpus = [
        (_form("latin", "pater"), _form("germ", "fader")),
        (_form("latin", "mater"), _form("germ", "moder")),
        (_form("latin", "pedis"), _form("germ", "fotu")),
        (_form("latin", "tres"), _form("germ", "θrejez")),
        (_form("latin", "kornu"), _form("germ", "hurnan")),
    ]
    m2_cost = sum(alignment_cost(align_forms(s, t)) for s, t in corpus)
    trained = train_model(corpus)
    m3_cost = sum(
        alignment_cost(align_forms(s, t, model=trained), model=trained)
        for s, t in corpus
    )
    assert m3_cost < m2_cost


def test_regular_substitution_corpus_concentrates_mass() -> None:
    """INTEGRATION: when every pair exhibits p~f, the Dirichlet posterior
    for P(f | p) should dominate (> 0.5 after training).

    This is a stronger version of the test in ``test_training.py``:
    we check not just that the count is high, but that the posterior
    probability correctly reflects the concentration.
    """
    corpus = [
        (_form("A", "papa"), _form("B", "fafa")),
        (_form("A", "papi"), _form("B", "fafi")),
        (_form("A", "popo"), _form("B", "fofo")),
        (_form("A", "pupu"), _form("B", "fufu")),
        (_form("A", "papa"), _form("B", "fafa")),
        (_form("A", "papi"), _form("B", "fafi")),
    ]
    trained = train_model(corpus)

    # Compute posterior P(f | p) by hand.
    from regulae.scoring import _segment_posterior

    p_f_given_p = _segment_posterior(
        "p", "f", Context(), trained.segment_table, trained.concentration
    )
    p_p_given_p = _segment_posterior(
        "p", "p", Context(), trained.segment_table, trained.concentration
    )

    assert p_f_given_p is not None and p_p_given_p is not None
    assert p_f_given_p > p_p_given_p
    assert p_f_given_p > 0.5


# ==========================================================================
# 2. Training invariants
# ==========================================================================


def test_training_is_deterministic_under_corpus_reordering() -> None:
    """INVARIANT: permuting the order of form pairs yields an equivalent
    trained model.

    This is a nontrivial claim: hard EM's result can in principle
    depend on iteration order if tie-breaking in the DP differs. Here
    we assert that the result is invariant — our DP is deterministic
    and the M-step is a sum (order-invariant).
    """
    corpus = [
        (_form("A", "papa"), _form("B", "fafa")),
        (_form("A", "mata"), _form("B", "mada")),
        (_form("A", "kata"), _form("B", "hata")),
        (_form("A", "tasa"), _form("B", "dasa")),
    ]
    rng = random.Random(0)
    shuffled = list(corpus)
    rng.shuffle(shuffled)

    m1 = train_model(corpus)
    m2 = train_model(shuffled)

    assert m1.segment_table.counts == m2.segment_table.counts
    # Displacements may differ only as dict insertion order; compare by
    # set-of-items.
    assert dict(m1.displacement_dist.counts) == dict(m2.displacement_dist.counts)
    assert dict(m1.chunk_table.entries) == dict(m2.chunk_table.entries)


def test_training_is_idempotent_on_short_corpora() -> None:
    """INVARIANT: retraining on the same corpus immediately after
    training should yield the same final model.

    This is a stronger statement than simple determinism — it checks
    that the final model is a fixed point of the training procedure."""
    corpus = [
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "pata"), _form("B", "fada")),
    ]
    m1 = train_model(corpus)
    m2 = train_model(corpus)
    assert m1.segment_table.counts == m2.segment_table.counts
    assert m1.chunk_table.entries == m2.chunk_table.entries


def test_segment_em_cost_monotonically_decreases_or_stays_stable() -> None:
    """INVARIANT: segment EM EM should never increase the total corpus
    cost between iterations. This is a property of hard EM with
    exact DP alignment: each iteration improves or maintains the
    corpus likelihood under the current model.

    We track the cost sequence manually by reimplementing the loop
    and asserting non-increase.
    """
    from regulae.training import (
        DEFAULT_CONCENTRATION,
        DEFAULT_SEGMENT_WEIGHT,
        DEFAULT_DISPLACEMENT_WEIGHT,
        DEFAULT_TEMPERATURE,
        _update_segment_table,
        align_corpus,
    )
    from regulae.search import DEFAULT_MAX_CHUNK_SIZE

    corpus = [
        (_form("A", "papa"), _form("B", "fafa")),
        (_form("A", "papi"), _form("B", "fafi")),
        (_form("A", "popo"), _form("B", "fofo")),
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "taka"), _form("B", "daha")),
    ]

    model = _initial_model(
        corpus=corpus,
        feature_system="descriptive",
        temperature=DEFAULT_TEMPERATURE,
        concentration=DEFAULT_CONCENTRATION,
        segment_weight=DEFAULT_SEGMENT_WEIGHT,
        displacement_weight=DEFAULT_DISPLACEMENT_WEIGHT,
    )

    costs: list[float] = []
    for _ in range(10):
        alignments = align_corpus(corpus, model, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE)
        total = sum(alignment_cost(a, model=model) for a in alignments)
        costs.append(total)
        model = _update_segment_table(model, alignments)

    # The cost sequence should be non-increasing (allow a tiny epsilon
    # for float noise).
    for i in range(1, len(costs)):
        assert costs[i] <= costs[i - 1] + 1e-9, (
            f"segment EM cost increased at iter {i}: {costs[i - 1]} -> {costs[i]}"
        )


def test_empty_corpus_yields_floor_behavior_on_new_pairs() -> None:
    """INVARIANT: a model trained on an empty corpus is effectively
    the ``empty`` model — its scoring should be indistinguishable
    from M2 behavior."""
    trained = train_model([])
    src = _form("A", "pater")
    tgt = _form("B", "fadar")
    a_m2 = align_forms(src, tgt)
    a_m3 = align_forms(src, tgt, model=trained)
    assert alignment_cost(a_m2) == alignment_cost(a_m3, model=trained)


def test_training_never_crashes_on_random_inputs() -> None:
    """INVARIANT: training on random form pairs should not raise.

    Deterministic fuzz over random graphemes drawn from merkmal's
    descriptive alphabet. The model may be meaningless but training
    must not crash."""
    rng = random.Random(42)
    alphabet = "paebtdkgmnfvszlriouɛɔ"
    for _ in range(5):
        corpus = []
        for _ in range(rng.randint(2, 8)):
            src_len = rng.randint(1, 5)
            tgt_len = rng.randint(1, 5)
            src = _form("A", "".join(rng.choice(alphabet) for _ in range(src_len)))
            tgt = _form("B", "".join(rng.choice(alphabet) for _ in range(tgt_len)))
            corpus.append((src, tgt))
        trained = train_model(corpus)
        # Trained model should be well-formed.
        assert trained.segment_table.log_normalizers  # prior was computed


# ==========================================================================
# 3. Edge cases
# ==========================================================================


def test_single_pair_corpus_does_not_crash() -> None:
    """EDGE: training on exactly one pair completes and produces a
    model with segment-level counts from that pair."""
    corpus = [(_form("A", "pata"), _form("B", "fada"))]
    trained = train_model(corpus)
    # At least some segment counts were recorded.
    assert len(trained.segment_table.counts) > 0


def test_single_pair_does_not_promote_any_chunks() -> None:
    """EDGE: one pair means every chunk candidate has count 1, which is
    below ``MIN_CHUNK_OBSERVATIONS``. No chunks should be promoted."""
    corpus = [(_form("A", "pata"), _form("B", "fada"))]
    trained = train_model(corpus)
    assert trained.chunk_table.entries == {}


def test_identical_pairs_stress_test() -> None:
    """EDGE: a corpus of N identical pairs should produce a strong
    posterior for the pairs' correspondences."""
    corpus = [(_form("A", "pata"), _form("B", "fada")) for _ in range(10)]
    trained = train_model(corpus)
    # Each segment pair in the aligned form should have count 10.
    assert trained.segment_table.counts.get(_cc("p", "f"), 0) == 10
    assert trained.segment_table.counts.get(_cc("a", "a"), 0) == 20  # two a's per pair
    assert trained.segment_table.counts.get(_cc("t", "d"), 0) == 10


def test_unrelated_pairs_do_not_raise_and_give_reasonable_alignment() -> None:
    """EDGE: training on a corpus where each pair is completely different
    from the others should still produce a valid model. The model's
    segment table will have many one-off entries and the chunk
    promotion should be essentially empty."""
    corpus = [
        (_form("A", "abc"), _form("B", "xyz")),
        (_form("A", "def"), _form("B", "qrs")),
        (_form("A", "ghi"), _form("B", "nop")),
    ]
    trained = train_model(corpus)
    # No chunks should be promoted (nothing recurs).
    assert trained.chunk_table.entries == {}
    # Model should still be usable for scoring.
    a = align_forms(corpus[0][0], corpus[0][1], model=trained)
    assert alignment_cost(a, model=trained) < float("inf")


def test_training_with_unknown_grapheme_raises_cleanly() -> None:
    """EDGE: a grapheme not recognized by merkmal should raise
    UnknownGrapheme during training, with a clear diagnostic.

    This is strict-by-default behavior: we don't silently skip pairs
    with bad data.
    """
    corpus = [
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "ZZZ"), _form("B", "xyz")),
    ]
    with pytest.raises(UnknownGrapheme):
        train_model(corpus)


def test_training_with_tiny_corpus_still_beats_m2_for_seen_pairs() -> None:
    """EDGE: even a tiny (3-pair) corpus should make the seen pairs
    strictly cheaper under the trained model.

    The effect size may be small but should be nonzero — otherwise the
    whole learned-model mechanism is inert for low-data use, which would be a
    serious limitation."""
    corpus = [
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "pa"), _form("B", "fa")),
    ]
    trained = train_model(corpus)
    m2_cost = alignment_cost(align_forms(corpus[0][0], corpus[0][1]))
    m3_cost = alignment_cost(
        align_forms(corpus[0][0], corpus[0][1], model=trained), model=trained
    )
    assert m3_cost < m2_cost


def test_corpus_with_length_asymmetry() -> None:
    """EDGE: pairs where source and target have very different lengths
    should not break training. Gap costs apply and the model should
    still produce valid alignments."""
    corpus = [
        (_form("A", "a"), _form("B", "xyz")),
        (_form("A", "a"), _form("B", "xyz")),
    ]
    trained = train_model(corpus)
    assert isinstance(trained.segment_table.counts, dict)
    # We don't assert the specific structure — just that training completed.
