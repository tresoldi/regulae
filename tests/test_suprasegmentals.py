"""Suprasegmental flow-through tests for the alignment pipeline.

Prior-only scoring does not use suprasegmental information, but the
data model carries it and downstream layers will need it. These tests
pin the contract that suprasegmental annotations travel through the
pipeline unchanged. A future refactor that silently drops them would
break these tests.
"""

from regulae import (
    Form,
    Segment,
    align_forms,
    format_alignment,
)


def test_toned_segments_survive_search_unchanged() -> None:
    """A segment entering the search with a tone annotation must exit
    with the same tone annotation in the corresponding link."""
    src = Form(
        "hmong_a",
        (
            Segment("p"),
            Segment("a", tone="55"),
        ),
    )
    tgt = Form(
        "hmong_b",
        (
            Segment("p"),
            Segment("a", tone="33"),
        ),
    )
    alignment = align_forms(src, tgt)
    # The alignment should include the a~a link with the tone annotations intact.
    tone_link = [
        link for link in alignment.links
        if link.source_chunk and link.source_chunk[0].grapheme == "a"
    ][0]
    assert tone_link.source_chunk[0].tone == "55"
    assert tone_link.target_chunk[0].tone == "33"


def test_stress_and_length_survive_search() -> None:
    src = Form(
        "A",
        (Segment("a", length="long", stress="primary"), Segment("b")),
    )
    tgt = Form(
        "B",
        (Segment("a", length="short"), Segment("b")),
    )
    alignment = align_forms(src, tgt)
    a_link = alignment.links[0]
    assert a_link.source_chunk[0].length == "long"
    assert a_link.source_chunk[0].stress == "primary"
    assert a_link.target_chunk[0].length == "short"
    assert a_link.target_chunk[0].stress is None


def test_toned_segments_render_in_formatter() -> None:
    """Suprasegmentals should show up in format_alignment output."""
    src = Form("A", (Segment("a", tone="high"),))
    tgt = Form("B", (Segment("a", tone="low"),))
    alignment = align_forms(src, tgt)
    out = format_alignment(alignment)
    assert "T=high" in out
    assert "T=low" in out


def test_tone_difference_does_not_affect_alignment_cost() -> None:
    """ROADMAP: scoring ignores suprasegmentals. Two forms differing
    only in tone have zero alignment cost. This will change when
    suprasegmentals enter the scoring model.
    """
    src = Form("A", (Segment("a", tone="high"),))
    tgt = Form("B", (Segment("a", tone="low"),))
    alignment = align_forms(src, tgt)
    from regulae import alignment_cost
    assert alignment_cost(alignment) == 0.0
