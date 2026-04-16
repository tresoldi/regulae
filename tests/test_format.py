"""Tests for the human-readable formatter.

These tests check that the formatter produces the expected shape of
output. They deliberately do NOT over-pin the exact characters (so
cosmetic tweaks don't break the suite), but they do pin the content:
which pieces of information must appear.
"""

from regulae import (
    Alignment,
    Form,
    Link,
    Segment,
    align_forms,
    format_alignment,
    format_link,
    format_segments,
)
from regulae.format import EMPTY_CHUNK_SYMBOL


# ----- format_segments ----------------------------------------------------


def test_format_segments_empty_uses_epsilon_marker() -> None:
    assert format_segments(()) == EMPTY_CHUNK_SYMBOL


def test_format_segments_concatenates_graphemes() -> None:
    segs = (Segment("p"), Segment("a"), Segment("t"))
    assert format_segments(segs) == "pat"


def test_format_segments_shows_tone_annotation() -> None:
    segs = (Segment("a", tone="high"),)
    out = format_segments(segs)
    assert "a" in out
    assert "T=high" in out


def test_format_segments_shows_length_and_stress() -> None:
    segs = (Segment("a", length="long", stress="primary"),)
    out = format_segments(segs)
    assert "L=long" in out
    assert "S=primary" in out


def test_format_segments_combines_multiple_annotations() -> None:
    segs = (Segment("a", tone="high", length="long"),)
    out = format_segments(segs)
    assert "T=high" in out
    assert "L=long" in out


# ----- format_link --------------------------------------------------------


def test_format_link_shows_both_chunks_with_separator() -> None:
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    out = format_link(link)
    assert "p" in out
    assert "f" in out
    assert "~" in out


def test_format_link_empty_source_uses_epsilon() -> None:
    link = Link(source_chunk=(), target_chunk=(Segment("e"),))
    out = format_link(link)
    assert EMPTY_CHUNK_SYMBOL in out
    assert "e" in out


def test_format_link_empty_target_uses_epsilon() -> None:
    link = Link(source_chunk=(Segment("e"),), target_chunk=())
    out = format_link(link)
    assert EMPTY_CHUNK_SYMBOL in out


def test_format_link_handles_multi_segment_chunks() -> None:
    link = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t")),
    )
    out = format_link(link)
    assert "kt" in out
    assert "tt" in out


# ----- format_alignment ---------------------------------------------------


def test_format_alignment_has_header_with_lect_ids() -> None:
    alignment = align_forms(
        Form("latin", (Segment("p"), Segment("a"))),
        Form("gothic", (Segment("f"), Segment("a"))),
    )
    out = format_alignment(alignment)
    assert "latin" in out
    assert "gothic" in out


def test_format_alignment_shows_total_cost_by_default() -> None:
    alignment = align_forms(
        Form("A", (Segment("p"),)),
        Form("B", (Segment("f"),)),
    )
    out = format_alignment(alignment)
    assert "cost" in out.lower()


def test_format_alignment_hides_cost_when_requested() -> None:
    alignment = align_forms(
        Form("A", (Segment("p"),)),
        Form("B", (Segment("f"),)),
    )
    out = format_alignment(alignment, show_costs=False)
    assert "cost" not in out.lower()


def test_format_alignment_shows_displacement_when_requested() -> None:
    alignment = align_forms(
        Form("A", (Segment("p"),)),
        Form("B", (Segment("f"),)),
    )
    out = format_alignment(alignment, show_displacement=True)
    # At least one feature name should appear in the output.
    # p and f differ on manner features; the exact names come from merkmal.
    assert "->" in out  # displacement arrow is part of the rendering


def test_format_alignment_has_one_line_per_link() -> None:
    alignment = align_forms(
        Form("A", (Segment("p"), Segment("a"), Segment("t"))),
        Form("B", (Segment("f"), Segment("a"), Segment("d"))),
    )
    out = format_alignment(alignment, show_costs=False)
    # header + 3 links = 4 lines
    assert len(out.split("\n")) == 1 + len(alignment.links)


def test_format_alignment_with_empty_alignment() -> None:
    """Empty alignment (both forms empty) should format without crashing."""
    alignment = Alignment(
        source_form=Form("A", ()),
        target_form=Form("B", ()),
        links=(),
    )
    out = format_alignment(alignment)
    assert "A" in out and "B" in out
