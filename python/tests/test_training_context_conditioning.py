"""Tests for context discovery and tonal aggregation.

These tests check that context discovery and tonal aggregation produce
expected results on small hand-crafted corpora where the ground truth
is known. They are integration tests — they run the full ``train_model``
flow and inspect the resulting ``LearnedModel`` fields.
"""

import pytest

from regulae import (
    ConditionedCorrespondence,
    Context,
    FeatureConstraint,
    Form,
    Segment,
    TonalCorrespondence,
    align_forms,
    alignment_cost,
)
from tests._helpers import train_model


def _form(lect_id: str, ipa: str) -> Form:
    return Form(lect_id=lect_id, segments=tuple(Segment(c) for c in ipa))


def _toned_form(lect_id: str, pairs: list[tuple[str, str | None]]) -> Form:
    """Build a form from a list of (grapheme, tone) pairs."""
    return Form(
        lect_id=lect_id,
        segments=tuple(Segment(grapheme=g, tone=t) for g, t in pairs),
    )


# ----- context discovery: context discovery --------------------------------------


def test_context_discovery_recovers_intervocalic_voicing_split() -> None:
    """COMMITMENT: on a corpus where /p/ becomes /b/ intervocalically
    but /p/ elsewhere, context discovery should commit a conditioned split:

        p → b / [vowel:+]_[vowel:+]   (for the intervocalic cases)
        p → p                         (unconditioned fallback)

    The synthetic corpus is designed so the split is visible in
    1-to-1 alignments at context discovery time.
    """
    corpus = [
        (_form("A", "pap"), _form("B", "pap")),   # p word-initial, p word-final
        (_form("A", "pap"), _form("B", "pap")),
        (_form("A", "pap"), _form("B", "pap")),
        (_form("A", "pop"), _form("B", "pop")),
        (_form("A", "pop"), _form("B", "pop")),
        (_form("A", "pop"), _form("B", "pop")),
        (_form("A", "apa"), _form("B", "aba")),   # INTERVOCALIC p → b
        (_form("A", "apa"), _form("B", "aba")),
        (_form("A", "apa"), _form("B", "aba")),
        (_form("A", "opa"), _form("B", "oba")),
        (_form("A", "opa"), _form("B", "oba")),
        (_form("A", "upa"), _form("B", "uba")),
    ]
    trained = train_model(corpus)

    # The table should contain at least one conditioned entry for p.
    p_entries = [
        key
        for key in trained.segment_table.counts
        if key.src == "p" and key.context.constraint_count() > 0
    ]
    assert len(p_entries) > 0, (
        "Expected at least one conditioned p entry after context discovery"
    )

    # There should be a conditioned entry for p → b (the intervocalic
    # lenition target).
    p_to_b_conditioned = [
        key for key in p_entries if key.tgt == "b"
    ]
    assert len(p_to_b_conditioned) > 0, (
        "Expected a conditioned p → b entry (intervocalic lenition)"
    )


def test_context_discovery_does_not_split_clean_merger_data() -> None:
    """COMMITMENT: a corpus with a clean unconditional merger (p → f
    in every context) should NOT produce any conditioned entries.

    This is the Polynesian-style case. If context discovery were too eager,
    it would invent spurious splits on data with no real conditioning.
    """
    corpus = [
        (_form("A", "pap"), _form("B", "faf")),
        (_form("A", "pap"), _form("B", "faf")),
        (_form("A", "pop"), _form("B", "fof")),
        (_form("A", "pop"), _form("B", "fof")),
        (_form("A", "pep"), _form("B", "fef")),
        (_form("A", "pep"), _form("B", "fef")),
        (_form("A", "pip"), _form("B", "fif")),
        (_form("A", "pip"), _form("B", "fif")),
        (_form("A", "apa"), _form("B", "afa")),
        (_form("A", "opo"), _form("B", "ofo")),
        (_form("A", "epe"), _form("B", "efe")),
    ]
    trained = train_model(corpus)

    # There should be NO conditioned p entries — all p → f observations
    # are consistent and no split improves BIC.
    p_conditioned = [
        key
        for key in trained.segment_table.counts
        if key.src == "p" and key.context.constraint_count() > 0
    ]
    assert p_conditioned == [], (
        f"Expected no conditioned p entries, got {p_conditioned}"
    )


def test_context_discovery_respects_min_split_observations() -> None:
    """COMMITMENT: a corpus with a very small minority target
    (below MIN_SPLIT_OBSERVATIONS) should not produce a conditioned
    split for that minority.

    Here only 1 occurrence of p → f appears; the rest are p → p.
    MIN_SPLIT_OBSERVATIONS defaults to 2, so a 1-observation minority
    can't form a valid split partition.
    """
    corpus = [
        (_form("A", "pa"), _form("B", "pa")),
        (_form("A", "pa"), _form("B", "pa")),
        (_form("A", "pa"), _form("B", "pa")),
        (_form("A", "pa"), _form("B", "pa")),
        (_form("A", "pa"), _form("B", "pa")),
        (_form("A", "pi"), _form("B", "fi")),  # single oddball
    ]
    trained = train_model(corpus)

    p_conditioned = [
        key
        for key in trained.segment_table.counts
        if key.src == "p" and key.context.constraint_count() > 0
    ]
    # The single p → f occurrence isn't enough to justify a split.
    # Specifically, no conditioned p → f should exist.
    conditioned_pf = [k for k in p_conditioned if k.tgt == "f"]
    assert conditioned_pf == []


# ----- tonal aggregation: tonal aggregation ----------------------------------------


def test_phase_4_learns_tone_correspondence_from_toned_data() -> None:
    """COMMITMENT: a corpus where every source tone is H and every
    target tone is L should yield a tonal correspondence H → L."""
    corpus = [
        (
            _toned_form("A", [("p", None), ("a", "H")]),
            _toned_form("B", [("p", None), ("a", "L")]),
        ),
        (
            _toned_form("A", [("t", None), ("a", "H")]),
            _toned_form("B", [("t", None), ("a", "L")]),
        ),
        (
            _toned_form("A", [("k", None), ("a", "H")]),
            _toned_form("B", [("k", None), ("a", "L")]),
        ),
        (
            _toned_form("A", [("m", None), ("a", "H")]),
            _toned_form("B", [("m", None), ("a", "L")]),
        ),
        (
            _toned_form("A", [("n", None), ("a", "H")]),
            _toned_form("B", [("n", None), ("a", "L")]),
        ),
    ]
    trained = train_model(corpus)

    # The tonal table should contain (H, L) with count 5.
    key = TonalCorrespondence(src_tone="H", tgt_tone="L")
    assert trained.tonal_table.counts.get(key, 0) == 5


def test_phase_4_leaves_table_empty_for_non_tonal_corpus() -> None:
    """COMMITMENT: a corpus with no tone annotations leaves the tonal
    table empty (inert on non-tonal data)."""
    corpus = [
        (_form("A", "pat"), _form("B", "pat")),
        (_form("A", "pit"), _form("B", "pit")),
    ]
    trained = train_model(corpus)
    assert trained.tonal_table.counts == {}


def test_phase_4_does_not_store_none_to_none_pair() -> None:
    """COMMITMENT: ``(None, None)`` links (both segments untoned) do
    not contribute to the tonal table."""
    # Mixed: some links with tones, some without.
    corpus = [
        (
            _toned_form("A", [("p", None), ("a", "H"), ("t", None)]),
            _toned_form("B", [("p", None), ("a", "M"), ("t", None)]),
        ),
        (
            _toned_form("A", [("k", None), ("i", "H"), ("n", None)]),
            _toned_form("B", [("k", None), ("i", "M"), ("n", None)]),
        ),
        (
            _toned_form("A", [("m", None), ("u", "H"), ("r", None)]),
            _toned_form("B", [("m", None), ("u", "M"), ("r", None)]),
        ),
    ]
    trained = train_model(corpus)
    # Only (H, M) should be recorded, not (None, None) for the toneless
    # consonant pairings.
    none_none = TonalCorrespondence(src_tone=None, tgt_tone=None)
    assert none_none not in trained.tonal_table.counts


def test_tonal_scoring_zero_cost_for_untoned_segments() -> None:
    """COMMITMENT: untoned segments score identically to the baseline — the tonal
    cost is zero when both tones are None, regardless of the tonal
    table."""
    from regulae.scoring import _tonal_cost
    from regulae.model import TonalCorrespondenceTable

    table = TonalCorrespondenceTable(
        counts={TonalCorrespondence("H", "L"): 5.0},
        prior_pseudo_counts={TonalCorrespondence("H", "L"): 1.0},
        src_totals={"H": 5.0},
    )
    assert _tonal_cost(None, None, table) == 0.0


def test_tonal_scoring_nonzero_cost_for_different_tones() -> None:
    """COMMITMENT: differently-toned segments incur a positive cost
    proportional to how unlikely the pairing is under the tonal
    table."""
    from regulae.scoring import _tonal_cost
    from regulae.model import TonalCorrespondenceTable

    table = TonalCorrespondenceTable(
        counts={TonalCorrespondence("H", "H"): 100.0},  # H→H very common
        prior_pseudo_counts={
            TonalCorrespondence("H", "H"): 1.0,
            TonalCorrespondence("H", "L"): 1.0,
        },
        src_totals={"H": 100.0},
    )
    # H→L is much less likely than H→H
    cost_hl = _tonal_cost("H", "L", table)
    cost_hh = _tonal_cost("H", "H", table)
    # Both should be finite
    assert cost_hl != float("inf")
    assert cost_hh != float("inf")
    # H→L (the rare one) should cost more than H→H (the common one)
    assert cost_hl > cost_hh
