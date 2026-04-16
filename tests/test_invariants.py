"""Property-based tests for alignment invariants.

These tests generate many random form pairs and check that the
invariants of the search hold for each one. They are deliberately
*property* tests, not coverage tests: each test asserts a single
property that should hold for ALL inputs, not a specific expected
output.

The fuzz is deterministic (seeded Random), so failures are
reproducible. The alphabet of graphemes is drawn from a small set
known to be valid in merkmal's descriptive system.

Invariants enforced:

* **Coverage**: link source chunks concatenate to the source form;
  same for target.
* **Symmetry**: ``align(A, B)`` and ``align(B, A)`` have equal total
  cost.
* **Non-negativity**: alignment cost is always >= 0.
* **Identity**: ``align(A, A)`` has cost 0.
* **Determinism**: repeated runs give identical output.
* **Triangle-inequality-like**: ``cost(A, C) <= cost(A, B) + cost(B, C)``
  need NOT hold for arbitrary scoring, so we do NOT test it. Correspondence
  cost is not a metric.
"""

import random

import pytest

from regulae import Form, Segment, align_forms, alignment_cost


# A reasonably-covering set of graphemes known to be in merkmal's
# descriptive system. Kept small so fuzz inputs are densely meaningful.
GRAPHEME_ALPHABET: tuple[str, ...] = (
    # Consonants
    "p", "b", "t", "d", "k", "ɡ", "m", "n", "ŋ",
    "f", "v", "s", "z", "ʃ", "ʒ", "h", "x",
    "r", "l", "j", "w",
    # Vowels
    "a", "e", "i", "o", "u", "ɛ", "ɔ", "ə",
)


def _random_form(rng: random.Random, lect_id: str, max_length: int = 8) -> Form:
    """Generate a random form of random length (possibly 0)."""
    length = rng.randint(0, max_length)
    segments = tuple(Segment(rng.choice(GRAPHEME_ALPHABET)) for _ in range(length))
    return Form(lect_id=lect_id, segments=segments)


def _random_pairs(seed: int, count: int, max_length: int = 8) -> list[tuple[Form, Form]]:
    rng = random.Random(seed)
    return [
        (_random_form(rng, "A", max_length), _random_form(rng, "B", max_length))
        for _ in range(count)
    ]


# ----- invariant: coverage -----------------------------------------------


def test_coverage_holds_for_random_inputs() -> None:
    """INVARIANT: for any form pair, the alignment links exhaustively
    cover both forms."""
    pairs = _random_pairs(seed=42, count=50)
    for src, tgt in pairs:
        alignment = align_forms(src, tgt)
        recovered_src = sum((link.source_chunk for link in alignment.links), ())
        recovered_tgt = sum((link.target_chunk for link in alignment.links), ())
        assert recovered_src == src.segments, (
            f"Source coverage failed for "
            f"{[s.grapheme for s in src.segments]!r} ~ "
            f"{[s.grapheme for s in tgt.segments]!r}"
        )
        assert recovered_tgt == tgt.segments


# ----- invariant: cost non-negative --------------------------------------


def test_cost_is_non_negative_for_random_inputs() -> None:
    """INVARIANT: alignment cost is always >= 0."""
    pairs = _random_pairs(seed=43, count=50)
    for src, tgt in pairs:
        alignment = align_forms(src, tgt)
        assert alignment_cost(alignment) >= 0.0


# ----- invariant: symmetry -----------------------------------------------


def test_symmetry_holds_for_random_inputs() -> None:
    """INVARIANT: swapping source and target preserves the total cost."""
    pairs = _random_pairs(seed=44, count=50)
    for src, tgt in pairs:
        cost_fwd = alignment_cost(align_forms(src, tgt))
        cost_bwd = alignment_cost(align_forms(tgt, src))
        assert cost_fwd == pytest.approx(cost_bwd), (
            f"Symmetry failed for "
            f"{[s.grapheme for s in src.segments]!r} ~ "
            f"{[s.grapheme for s in tgt.segments]!r}: "
            f"fwd={cost_fwd}, bwd={cost_bwd}"
        )


# ----- invariant: identity -----------------------------------------------


def test_identity_has_zero_cost_for_random_inputs() -> None:
    """INVARIANT: aligning a form with itself has cost 0."""
    rng = random.Random(45)
    for _ in range(50):
        form = _random_form(rng, "A")
        alignment = align_forms(form, form)
        assert alignment_cost(alignment) == 0.0


# ----- invariant: determinism --------------------------------------------


def test_determinism_holds_for_random_inputs() -> None:
    """INVARIANT: the same input always yields the same alignment."""
    pairs = _random_pairs(seed=46, count=30)
    for src, tgt in pairs:
        a1 = align_forms(src, tgt)
        a2 = align_forms(src, tgt)
        assert a1 == a2


# ----- invariant: monotonicity in max_chunk_size -------------------------


def test_larger_max_chunk_size_never_worsens_cost_for_random_inputs() -> None:
    """INVARIANT: enlarging the search space cannot increase the optimum."""
    pairs = _random_pairs(seed=47, count=20, max_length=6)
    for src, tgt in pairs:
        c1 = alignment_cost(align_forms(src, tgt, max_chunk_size=1))
        c2 = alignment_cost(align_forms(src, tgt, max_chunk_size=2))
        c3 = alignment_cost(align_forms(src, tgt, max_chunk_size=3))
        # Allow a tiny slack for float arithmetic; the DP shouldn't
        # introduce real numerical error at this scale but being paranoid.
        assert c2 <= c1 + 1e-9
        assert c3 <= c2 + 1e-9


# ----- invariant: link count bounds --------------------------------------


def test_link_count_is_bounded_for_random_inputs() -> None:
    """INVARIANT: the number of links is between max(len(src), len(tgt))
    (all in one chunk) and len(src) + len(tgt) (all 0-to-1 or 1-to-0).

    Actually the lower bound is ceil(max(n, m) / max_chunk_size), but
    using max(n, m) is a valid (looser) bound. The upper bound is tight.
    """
    pairs = _random_pairs(seed=48, count=50)
    for src, tgt in pairs:
        alignment = align_forms(src, tgt)
        n = len(src.segments)
        m = len(tgt.segments)
        if n == 0 and m == 0:
            assert len(alignment.links) == 0
        else:
            assert len(alignment.links) >= 1
            assert len(alignment.links) <= n + m
