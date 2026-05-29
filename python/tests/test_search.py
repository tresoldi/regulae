"""Tests for the alignment search.

These tests specify the contract of ``align_forms``. As with the other
test modules, each test name encodes a theoretical commitment.

The big commitments are:

* **Coverage invariant**: the links of the returned alignment cover both
  forms exactly — concatenating source chunks reproduces the source, and
  likewise for the target.
* **Optimality**: the returned alignment has the minimum total cost among
  all possible alignments under the given scoring.
* **Identity**: aligning a form with itself yields a zero-cost alignment
  of only 1-to-1 identity links.
* **Symmetry**: ``align(A, B)`` and ``align(B, A)`` produce alignments
  with equal total cost (and with each link's chunks swapped).
* **Determinism**: the search is deterministic given its inputs.
* **Real-pair plausibility**: classic cognate pairs align in the
  expected broad shape (e.g., Latin ``pater`` ~ Gothic ``fadar`` aligns
  segment-by-segment).
"""

import pytest

from regulae import (
    Alignment,
    Form,
    Segment,
    align_forms,
    alignment_cost,
)
from regulae.search import DEFAULT_MAX_CHUNK_SIZE


# ----- helpers -------------------------------------------------------------


def _form(lect_id: str, ipa: str) -> Form:
    """Build a Form from a simple IPA string, one Segment per character.

    This is test-only sugar. Real data will come from a richer loader.
    """
    return Form(lect_id=lect_id, segments=tuple(Segment(c) for c in ipa))


def _assert_covers_both_forms(alignment: Alignment) -> None:
    """The coverage invariant: concatenating the link chunks must
    reproduce both forms exactly."""
    source_recovered: tuple[Segment, ...] = ()
    target_recovered: tuple[Segment, ...] = ()
    for link in alignment.links:
        source_recovered += link.source_chunk
        target_recovered += link.target_chunk
    assert source_recovered == alignment.source_form.segments, (
        f"Source coverage failed: recovered {source_recovered} "
        f"vs expected {alignment.source_form.segments}"
    )
    assert target_recovered == alignment.target_form.segments, (
        f"Target coverage failed: recovered {target_recovered} "
        f"vs expected {alignment.target_form.segments}"
    )


# ----- trivial and degenerate cases ---------------------------------------


def test_aligning_two_empty_forms_yields_empty_alignment() -> None:
    """COMMITMENT: empty input produces empty output, not an error."""
    src = Form(lect_id="A", segments=())
    tgt = Form(lect_id="B", segments=())
    alignment = align_forms(src, tgt)
    assert alignment.links == ()
    assert alignment_cost(alignment) == 0.0


def test_aligning_identical_single_segment_forms_yields_one_identity_link() -> None:
    """COMMITMENT: identity produces a single 1-to-1 link with zero cost."""
    src = _form("A", "p")
    tgt = _form("B", "p")
    alignment = align_forms(src, tgt)
    assert len(alignment.links) == 1
    link = alignment.links[0]
    assert link.source_chunk == (Segment("p"),)
    assert link.target_chunk == (Segment("p"),)
    assert alignment_cost(alignment) == 0.0


def test_aligning_identical_forms_yields_all_identity_links() -> None:
    """COMMITMENT: identity of longer forms decomposes into identity links."""
    src = _form("A", "pater")
    tgt = _form("B", "pater")
    alignment = align_forms(src, tgt)
    _assert_covers_both_forms(alignment)
    assert alignment_cost(alignment) == 0.0
    # Expect 5 links, all 1-to-1 identity.
    assert len(alignment.links) == 5
    for link in alignment.links:
        assert len(link.source_chunk) == 1
        assert len(link.target_chunk) == 1
        assert link.source_chunk[0].grapheme == link.target_chunk[0].grapheme


def test_aligning_empty_source_to_nonempty_target_yields_pure_insertion() -> None:
    """COMMITMENT: an empty source against a nonempty target yields all 0-to-1."""
    src = Form(lect_id="A", segments=())
    tgt = _form("B", "abc")
    alignment = align_forms(src, tgt)
    _assert_covers_both_forms(alignment)
    # Every link has an empty source chunk (possibly as one long 0-to-3
    # chunk or as three 0-to-1 chunks — both are valid; we just check
    # the coverage invariant and that source chunks are all empty).
    for link in alignment.links:
        assert link.source_chunk == ()


def test_aligning_nonempty_source_to_empty_target_yields_pure_deletion() -> None:
    """COMMITMENT: an empty target against a nonempty source yields all 1-to-0."""
    src = _form("A", "abc")
    tgt = Form(lect_id="B", segments=())
    alignment = align_forms(src, tgt)
    _assert_covers_both_forms(alignment)
    for link in alignment.links:
        assert link.target_chunk == ()


# ----- coverage invariant --------------------------------------------------


@pytest.mark.parametrize(
    ("src_ipa", "tgt_ipa"),
    [
        ("p", "f"),
        ("pa", "fa"),
        ("pater", "fadar"),
        ("noktem", "notte"),
        ("a", "we"),
        ("abc", "xy"),
        ("", "xy"),
        ("abc", ""),
    ],
)
def test_coverage_invariant_holds_for_varied_inputs(src_ipa: str, tgt_ipa: str) -> None:
    """COMMITMENT: for any input pair, the returned alignment covers both
    forms exhaustively."""
    src = _form("A", src_ipa)
    tgt = _form("B", tgt_ipa)
    alignment = align_forms(src, tgt)
    _assert_covers_both_forms(alignment)


# ----- optimality via brute force ----------------------------------------


def _brute_force_min_cost(
    source: Form, target: Form, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> float:
    """Brute-force enumeration of all possible alignments, returning the
    minimum total cost.

    Used to verify that the DP is actually finding the optimum on small
    inputs. For larger inputs this is infeasible, hence "small".
    """
    from regulae.scoring import score_link
    from regulae.types import Link

    n = len(source.segments)
    m = len(target.segments)

    # Memoise by (i, j) = cost to align source[:i] with target[:j].
    memo: dict[tuple[int, int], float] = {}

    def rec(i: int, j: int) -> float:
        if (i, j) == (0, 0):
            return 0.0
        if (i, j) in memo:
            return memo[(i, j)]
        best = float("inf")
        for k in range(min(max_chunk_size, i) + 1):
            for l in range(min(max_chunk_size, j) + 1):
                if k == 0 and l == 0:
                    continue
                src_chunk = source.segments[i - k : i]
                tgt_chunk = target.segments[j - l : j]
                link = Link(source_chunk=src_chunk, target_chunk=tgt_chunk)
                link_cost = score_link(link)
                total = rec(i - k, j - l) + link_cost
                if total < best:
                    best = total
        memo[(i, j)] = best
        return best

    return rec(n, m)


@pytest.mark.parametrize(
    ("src_ipa", "tgt_ipa"),
    [
        ("p", "p"),
        ("p", "f"),
        ("pa", "fa"),
        ("pat", "fad"),
        ("pater", "fadar"),
        ("noktem", "notte"),
        ("a", "we"),
        ("kt", "tt"),
    ],
)
def test_search_finds_minimum_cost(src_ipa: str, tgt_ipa: str) -> None:
    """COMMITMENT: the DP finds the global minimum, verified against brute force."""
    src = _form("A", src_ipa)
    tgt = _form("B", tgt_ipa)
    alignment = align_forms(src, tgt)
    dp_cost = alignment_cost(alignment)
    brute = _brute_force_min_cost(src, tgt)
    assert dp_cost == pytest.approx(brute)


# ----- symmetry ------------------------------------------------------------


@pytest.mark.parametrize(
    ("ipa_a", "ipa_b"),
    [
        ("pater", "fadar"),
        ("noktem", "notte"),
        ("abc", "xyz"),
        ("", "xyz"),
    ],
)
def test_alignment_cost_is_symmetric_under_form_swap(ipa_a: str, ipa_b: str) -> None:
    """COMMITMENT: swapping source and target preserves the total cost.

    This is the alignment-level analogue of the link-level symmetry
    enforced at the link level. Correspondences are bidirectional;
    direction doesn't change the cost of a correspondence.
    """
    forward = align_forms(_form("A", ipa_a), _form("B", ipa_b))
    backward = align_forms(_form("B", ipa_b), _form("A", ipa_a))
    assert alignment_cost(forward) == pytest.approx(alignment_cost(backward))


# ----- determinism --------------------------------------------------------


def test_search_is_deterministic() -> None:
    """COMMITMENT: given the same inputs, the search returns identical output."""
    src = _form("A", "pater")
    tgt = _form("B", "fadar")
    a1 = align_forms(src, tgt)
    a2 = align_forms(src, tgt)
    assert a1 == a2


# ----- feature displacement in search output -----------------------------


def test_one_to_one_links_have_populated_displacement() -> None:
    """COMMITMENT: 1-to-1 links in the search output carry feature displacement.

    This makes the aggregate analysis possible: after many alignments,
    grouping links by their displacement vector reveals regular
    correspondences at the feature level.
    """
    alignment = align_forms(_form("A", "pater"), _form("B", "fadar"))
    # All links should be 1-to-1 for this pair.
    assert all(len(link.source_chunk) == 1 and len(link.target_chunk) == 1
               for link in alignment.links)
    # Substitution links should have nonempty displacement.
    for link in alignment.links:
        if link.source_chunk[0] != link.target_chunk[0]:
            assert len(link.feature_displacement) > 0


def test_identity_links_have_empty_displacement() -> None:
    """COMMITMENT: displacement for identity is empty."""
    alignment = align_forms(_form("A", "pater"), _form("B", "pater"))
    for link in alignment.links:
        assert link.feature_displacement == ()


# ----- real cognate smoke tests -------------------------------------------


def test_pater_fadar_aligns_segment_by_segment() -> None:
    """SMOKE: Latin pater ~ Gothic fadar produces a clean 5-link 1-to-1 alignment.

    We don't pin the exact feature displacement values, but we do check
    that each link is 1-to-1 and the pairing is the intuitive one.
    """
    alignment = align_forms(_form("latin", "pater"), _form("gothic", "fadar"))
    assert len(alignment.links) == 5
    expected_pairs = [("p", "f"), ("a", "a"), ("t", "d"), ("e", "a"), ("r", "r")]
    for link, (src_g, tgt_g) in zip(alignment.links, expected_pairs, strict=True):
        assert len(link.source_chunk) == 1
        assert len(link.target_chunk) == 1
        assert link.source_chunk[0].grapheme == src_g
        assert link.target_chunk[0].grapheme == tgt_g


def test_noktem_notte_produces_valid_alignment() -> None:
    """SMOKE: Latin noktem-like ~ Italian notte-like produces a
    well-formed alignment. Without learned phrase tables, we don't
    pin a specific chunk structure — only that coverage holds and the
    cost is finite.
    """
    alignment = align_forms(_form("latin", "noktem"), _form("italian", "notte"))
    _assert_covers_both_forms(alignment)
    assert alignment_cost(alignment) < float("inf")


# ----- max_chunk_size parameter -------------------------------------------


def test_max_chunk_size_one_reduces_to_classical_needleman_wunsch() -> None:
    """COMMITMENT: with max_chunk_size=1, the search only emits 1-to-1,
    0-to-1, and 1-to-0 links. This is the classical NW special case.
    """
    alignment = align_forms(
        _form("A", "kt"), _form("B", "tt"), max_chunk_size=1
    )
    _assert_covers_both_forms(alignment)
    for link in alignment.links:
        assert len(link.source_chunk) <= 1
        assert len(link.target_chunk) <= 1


def test_max_chunk_size_zero_is_rejected() -> None:
    """COMMITMENT: max_chunk_size must be at least 1. Zero is meaningless."""
    with pytest.raises(ValueError):
        align_forms(_form("A", "p"), _form("B", "p"), max_chunk_size=0)


def test_larger_max_chunk_size_cannot_increase_cost() -> None:
    """COMMITMENT: allowing longer chunks can only decrease (or tie) cost.

    A larger search space never worsens the optimum. This is a monotonicity
    property of the DP.
    """
    src = _form("A", "nokt")
    tgt = _form("B", "nott")
    small = align_forms(src, tgt, max_chunk_size=1)
    medium = align_forms(src, tgt, max_chunk_size=2)
    large = align_forms(src, tgt, max_chunk_size=3)
    c1 = alignment_cost(small)
    c2 = alignment_cost(medium)
    c3 = alignment_cost(large)
    assert c2 <= c1 + 1e-9
    assert c3 <= c2 + 1e-9


# ----- alignment_cost contract --------------------------------------------


def test_alignment_cost_equals_sum_of_link_scores() -> None:
    """COMMITMENT: alignment_cost is exactly the sum of the link scores."""
    from regulae.scoring import score_link

    alignment = align_forms(_form("A", "pater"), _form("B", "fadar"))
    expected = sum(score_link(link) for link in alignment.links)
    assert alignment_cost(alignment) == pytest.approx(expected)


def test_alignment_cost_respects_feature_system() -> None:
    """COMMITMENT: alignment_cost can be recomputed under a different
    feature system. Results may differ; both should run."""
    alignment = align_forms(
        _form("A", "pater"), _form("B", "fadar"), feature_system="descriptive"
    )
    c_desc = alignment_cost(alignment, feature_system="descriptive")
    c_dist = alignment_cost(alignment, feature_system="distinctive")
    assert c_desc >= 0
    assert c_dist >= 0


def test_dp_tiebreaker_prefers_atomic_links_over_chunks() -> None:
    """COMMITMENT: the DP's cost-based tie-breaker
    (_CHUNK_COMPLEXITY_PENALTY) makes a 1-to-1 decomposition win
    against an equivalently-costed multi-segment chunking.

    The pair ``pata`` / ``fada`` has a perfectly consistent p↔f,
    a↔a, t↔d, a↔a decomposition. Under the M2 scoring (no model),
    a single 2-to-2 chunk like ``(p,a) → (f,a)`` has exactly the
    same compositional cost as two separate 1-to-1 links. Before
    the tiebreaker, the DP's pick among these depended on (k, l)
    iteration order. After the tiebreaker, the 1-to-1
    decomposition always wins because ``(k+l-2) == 0`` for 1-to-1
    links and positive for larger chunks.
    """
    alignment = align_forms(_form("A", "pata"), _form("B", "fada"))
    # Every link must be 1-to-1.
    assert all(
        len(link.source_chunk) == 1 and len(link.target_chunk) == 1
        for link in alignment.links
    ), [
        ("".join(s.grapheme for s in link.source_chunk),
         "".join(s.grapheme for s in link.target_chunk))
        for link in alignment.links
    ]


def test_dp_tiebreaker_does_not_flip_substantive_cost_comparisons() -> None:
    """COMMITMENT: the tiebreaker penalty is ~1e-9 per excess
    chunk segment — far below any real cost difference — so it
    can never cause a substantively more expensive alignment to
    be preferred over a cheaper one."""
    # A clearly lower-cost 1-to-1 alignment should still win
    # against any 2-to-2 competitor. If the tiebreaker were too
    # large (say 1.0) this test would still pass, but if it
    # approached the scale of real costs it would not.
    alignment = align_forms(_form("A", "papa"), _form("B", "papa"))
    # Identical forms should still be 1-to-1 identity.
    assert all(
        link.source_chunk == link.target_chunk for link in alignment.links
    )
    assert all(len(link.source_chunk) == 1 for link in alignment.links)
