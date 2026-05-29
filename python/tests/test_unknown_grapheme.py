"""Tests for unknown grapheme handling.

Policy: unknown graphemes raise ``UnknownGrapheme`` from the regulae
package, carrying the offending grapheme and the feature system name.
This is strict by default — silent fallback is not the right behavior
for historical linguistics data, where typos and diacritic variants
are common and need to be diagnosed, not absorbed.
"""

import pytest

from regulae import (
    Form,
    Link,
    Segment,
    UnknownGrapheme,
    align_forms,
    compute_displacement,
    score_link,
)


# ----- score_link on unknown graphemes -----------------------------------


def test_score_link_raises_on_unknown_source() -> None:
    link = Link(source_chunk=(Segment("ZZZ"),), target_chunk=(Segment("p"),))
    with pytest.raises(UnknownGrapheme) as exc_info:
        score_link(link)
    assert exc_info.value.grapheme == "ZZZ"
    assert "ZZZ" in str(exc_info.value)


def test_score_link_raises_on_unknown_target() -> None:
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("ZZZ"),))
    with pytest.raises(UnknownGrapheme) as exc_info:
        score_link(link)
    assert exc_info.value.grapheme == "ZZZ"


def test_score_link_mentions_feature_system_in_error() -> None:
    """The error carries the feature system name so the caller knows
    which system failed to recognise the grapheme."""
    link = Link(source_chunk=(Segment("ZZZ"),), target_chunk=(Segment("p"),))
    with pytest.raises(UnknownGrapheme) as exc_info:
        score_link(link, feature_system="distinctive")
    assert exc_info.value.feature_system == "distinctive"


def test_score_link_is_fine_with_valid_graphemes() -> None:
    """Regression: the unknown-grapheme check must not break the happy path."""
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    cost = score_link(link)
    assert cost > 0.0


# ----- compute_displacement on unknown graphemes -------------------------


def test_compute_displacement_raises_on_unknown_source() -> None:
    with pytest.raises(UnknownGrapheme) as exc_info:
        compute_displacement(Segment("ZZZ"), Segment("p"))
    assert exc_info.value.grapheme == "ZZZ"


def test_compute_displacement_raises_on_unknown_target() -> None:
    with pytest.raises(UnknownGrapheme) as exc_info:
        compute_displacement(Segment("p"), Segment("ZZZ"))
    assert exc_info.value.grapheme == "ZZZ"


# ----- align_forms on unknown graphemes ----------------------------------


def test_align_forms_raises_cleanly_on_unknown_grapheme() -> None:
    """The search should propagate UnknownGrapheme so the caller can
    diagnose which form or which position is the problem."""
    src = Form("A", (Segment("p"), Segment("ZZZ"), Segment("t")))
    tgt = Form("B", (Segment("p"), Segment("a"), Segment("t")))
    with pytest.raises(UnknownGrapheme) as exc_info:
        align_forms(src, tgt)
    assert exc_info.value.grapheme == "ZZZ"


def test_unknown_grapheme_is_a_value_error() -> None:
    """COMMITMENT: UnknownGrapheme inherits from ValueError so callers
    can catch it generically if they prefer."""
    assert issubclass(UnknownGrapheme, ValueError)
