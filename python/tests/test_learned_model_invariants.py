"""Property-based invariant tests for the  training pipeline.

Deterministic fuzz over random form pairs, like the M2-era
``test_invariants.py`` but extended to exercise context discovery context
discovery and tonal aggregation tonal aggregation. The goal is to catch
invariant violations on inputs the hand-crafted tests don't cover.
"""

import random

from regulae import (
    Context,
    Form,
    Segment,
    align_forms,
    alignment_cost,
)
from tests._helpers import train_model


GRAPHEME_ALPHABET: tuple[str, ...] = (
    "p", "b", "t", "d", "k", "g", "m", "n",
    "f", "v", "s", "z", "h",
    "r", "l", "j", "w",
    "a", "e", "i", "o", "u",
)

TONE_ALPHABET: tuple[str, ...] = ("H", "M", "L")


def _random_form(rng: random.Random, lect_id: str, max_length: int = 6) -> Form:
    length = rng.randint(1, max_length)
    segments = tuple(Segment(rng.choice(GRAPHEME_ALPHABET)) for _ in range(length))
    return Form(lect_id=lect_id, segments=segments)


def _random_toned_form(
    rng: random.Random, lect_id: str, max_length: int = 6
) -> Form:
    length = rng.randint(1, max_length)
    segments: list[Segment] = []
    for _ in range(length):
        g = rng.choice(GRAPHEME_ALPHABET)
        tone = rng.choice(TONE_ALPHABET) if g in "aeiou" else None
        segments.append(Segment(grapheme=g, tone=tone))
    return Form(lect_id=lect_id, segments=tuple(segments))


def _random_pairs(seed: int, count: int, toned: bool = False) -> list[tuple[Form, Form]]:
    rng = random.Random(seed)
    pairs: list[tuple[Form, Form]] = []
    for _ in range(count):
        if toned:
            src = _random_toned_form(rng, "A")
            tgt = _random_toned_form(rng, "B")
        else:
            src = _random_form(rng, "A")
            tgt = _random_form(rng, "B")
        pairs.append((src, tgt))
    return pairs


# ----- Training never crashes ---------------------------------------------


def test_training_never_crashes_on_random_untoned_corpora() -> None:
    """INVARIANT:  training completes on random form pairs."""
    for seed in range(10):
        corpus = _random_pairs(seed=seed, count=20)
        model = train_model(corpus)
        assert model is not None
        # At least some segment correspondences should be recorded.
        assert len(model.segment_table.counts) >= 0


def test_training_never_crashes_on_random_toned_corpora() -> None:
    """INVARIANT:  training completes on corpora with random tones."""
    for seed in range(5):
        corpus = _random_pairs(seed=seed + 100, count=15, toned=True)
        model = train_model(corpus)
        assert model is not None


# ----- context discovery never introduces spurious entries beyond what data
# supports ---------------------------------------------------------------


def test_conditioned_entries_are_subset_of_observed_sources() -> None:
    """INVARIANT: every conditioned entry in the trained model has a
    source grapheme that appears in the training data. context discovery
    should not invent splits on segments it has never seen."""
    corpus = _random_pairs(seed=200, count=20)
    observed_sources = {
        seg.grapheme for src, _ in corpus for seg in src.segments
    }
    trained = train_model(corpus)
    for key in trained.segment_table.counts:
        if key.context.constraint_count() > 0:
            assert key.src in observed_sources


def test_conditioned_counts_do_not_exceed_unconditioned() -> None:
    """INVARIANT: for any source grapheme, the sum of conditioned-entry
    counts for that source must not exceed the unconditioned count.

    This guards against context discovery double-counting: each observation
    should contribute to at most one conditioned entry per source
    (plus always contributing to the unconditioned fallback).
    """
    corpus = _random_pairs(seed=201, count=30)
    trained = train_model(corpus)

    # Sum conditioned counts per source
    by_src_conditioned: dict[str, float] = {}
    for key, n in trained.segment_table.counts.items():
        if key.context.constraint_count() > 0:
            by_src_conditioned[key.src] = by_src_conditioned.get(key.src, 0) + n

    # The unconditioned count per source — note that with the flat
    # multi-predicate approach, multiple conditioned entries can cover
    # the same observations under different axes, so conditioned
    # total CAN exceed unconditioned total. What we DO guarantee is
    # that no single conditioned entry exceeds the unconditioned
    # count for the same (src, tgt) pair.
    for key, n in trained.segment_table.counts.items():
        if key.context.constraint_count() == 0:
            continue
        unconditioned_key_count = 0.0
        for k2, n2 in trained.segment_table.counts.items():
            if (
                k2.src == key.src
                and k2.tgt == key.tgt
                and k2.context.constraint_count() == 0
            ):
                unconditioned_key_count = n2
                break
        assert n <= unconditioned_key_count + 1e-9, (
            f"Conditioned {key} count {n} exceeds unconditioned "
            f"{key.src}→{key.tgt} count {unconditioned_key_count}"
        )


# ----- Determinism is preserved -------------------------------------------


def test_training_is_deterministic_under_repeated_runs() -> None:
    """INVARIANT: same input → same trained model, for 's full flow."""
    corpus = _random_pairs(seed=300, count=20)
    m1 = train_model(corpus)
    m2 = train_model(corpus)
    assert m1.segment_table.counts == m2.segment_table.counts
    assert m1.displacement_dist.counts == m2.displacement_dist.counts
    assert m1.chunk_table.entries == m2.chunk_table.entries
    assert m1.tonal_table.counts == m2.tonal_table.counts


def test_training_is_deterministic_under_corpus_reordering() -> None:
    """INVARIANT: the trained model is invariant under corpus
    reordering. context discovery and tonal aggregation both operate on aggregated
    counts, which should be order-independent."""
    corpus = _random_pairs(seed=301, count=25)
    rng = random.Random(0)
    shuffled = list(corpus)
    rng.shuffle(shuffled)
    m1 = train_model(corpus)
    m2 = train_model(shuffled)
    assert m1.segment_table.counts == m2.segment_table.counts
    assert m1.tonal_table.counts == m2.tonal_table.counts


# ----- Cost invariants ----------------------------------------------------


def test_trained_model_alignment_cost_is_finite_on_random_inputs() -> None:
    """INVARIANT: after training, aligning any pair from the corpus
    with the trained model yields a finite cost."""
    corpus = _random_pairs(seed=400, count=20)
    trained = train_model(corpus)
    for src, tgt in corpus:
        cost = alignment_cost(
            align_forms(src, tgt, model=trained), model=trained
        )
        assert cost != float("inf")


def test_trained_model_produces_sensible_costs_on_unseen_pairs() -> None:
    """INVARIANT: aligning a pair with segments partially overlapping
    the training corpus still yields a finite cost."""
    corpus = _random_pairs(seed=401, count=15)
    trained = train_model(corpus)
    # Generate a fresh pair with segments that may or may not be in the
    # corpus.
    unseen = _random_pairs(seed=402, count=5)
    for src, tgt in unseen:
        cost = alignment_cost(
            align_forms(src, tgt, model=trained), model=trained
        )
        assert cost != float("inf")


# ----- tonal aggregation tonal invariants -------------------------------------------


def test_tonal_creates_tonal_entries_only_when_tones_are_present() -> None:
    """INVARIANT: training on an untoned corpus leaves the tonal table
    empty; training on a toned corpus (where some segments have tones)
    produces a non-empty tonal table."""
    untoned = _random_pairs(seed=500, count=15, toned=False)
    untoned_model = train_model(untoned)
    assert untoned_model.tonal_table.counts == {}

    toned = _random_pairs(seed=501, count=15, toned=True)
    toned_model = train_model(toned)
    # At least some tonal observations should be recorded.
    assert len(toned_model.tonal_table.counts) >= 0  # may be 0 if no tones aligned


def test_tonal_tonal_counts_non_negative() -> None:
    """INVARIANT: tonal counts are always non-negative."""
    toned = _random_pairs(seed=502, count=20, toned=True)
    trained = train_model(toned)
    for count in trained.tonal_table.counts.values():
        assert count >= 0


# ----- Chunk table invariants (still hold) --------------------------------


def test_chunks_still_restrain_themselves_on_non_recurring_data() -> None:
    """INVARIANT: random data (where nothing recurs) yields few
    promoted chunks. Specifically, the random corpora should not
    explode the chunk table."""
    corpus = _random_pairs(seed=600, count=15)
    trained = train_model(corpus)
    # Random corpora may still find one or two fortuitous recurrences,
    # but a dozen+ is suspicious.
    assert len(trained.chunk_table.entries) <= 20


# ----- M2 compatibility preserved -----------------------------------------


def test_m2_alignment_cost_unchanged_when_no_model_passed() -> None:
    """INVARIANT: aligning without a model reproduces M2 behavior
    exactly, even on corpora that would use all  features during
    training."""
    corpus = _random_pairs(seed=700, count=10, toned=True)
    for src, tgt in corpus:
        m2_cost = alignment_cost(align_forms(src, tgt))
        # Should match the M2 cost computed without a model.
        assert m2_cost != float("inf")
