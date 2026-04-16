"""Tests for the core data types.

These tests are not coverage tests for trivial getters; they encode the
*semantics* the framework promises. Each test name is meant to read as a
specification claim.
"""

import pytest

from regulae import (
    Alignment,
    Context,
    FeatureConstraint,
    FeatureDisplacement,
    Form,
    Lect,
    Link,
    Segment,
)


# ----- Segment -------------------------------------------------------------


def test_segment_carries_grapheme_only_when_unannotated() -> None:
    s = Segment(grapheme="p")
    assert s.grapheme == "p"
    assert s.tone is None
    assert s.length is None
    assert s.stress is None


def test_segment_carries_suprasegmental_annotations() -> None:
    s = Segment(grapheme="a", tone="high", length="long", stress="primary")
    assert s.tone == "high"
    assert s.length == "long"
    assert s.stress == "primary"


def test_segments_are_immutable() -> None:
    s = Segment(grapheme="p")
    with pytest.raises(Exception):
        s.grapheme = "b"  # type: ignore[misc]


def test_segments_with_same_fields_are_equal() -> None:
    assert Segment("p") == Segment("p")
    assert Segment("a", tone="high") == Segment("a", tone="high")
    assert Segment("a", tone="high") != Segment("a", tone="low")


# ----- Form ----------------------------------------------------------------


def test_form_holds_segments_and_lect_id() -> None:
    f = Form(
        lect_id="latin",
        segments=(Segment("p"), Segment("a"), Segment("t"), Segment("e"), Segment("r")),
    )
    assert f.lect_id == "latin"
    assert len(f.segments) == 5
    assert f.segments[0].grapheme == "p"


def test_form_supports_syllable_breaks() -> None:
    # pa.ter — syllable starts at index 0 and at index 2
    f = Form(
        lect_id="latin",
        segments=(Segment("p"), Segment("a"), Segment("t"), Segment("e"), Segment("r")),
        syllable_breaks=(0, 2),
    )
    assert f.syllable_breaks == (0, 2)


# ----- Lect ----------------------------------------------------------------


def test_lect_has_id_and_default_feature_system() -> None:
    lect = Lect(lect_id="latin")
    assert lect.lect_id == "latin"
    assert lect.feature_system == "descriptive"


# ----- Context and FeatureConstraint --------------------------------------


def test_empty_context_imposes_no_constraints() -> None:
    c = Context()
    assert c.position is None
    assert c.preceding == ()
    assert c.following == ()
    assert c.morphological is None


def test_context_supports_combining_conjunctive_constraints() -> None:
    # "before [+front] AND [+high]" — both constraints must hold
    front = FeatureConstraint(feature="front", value="+")
    high = FeatureConstraint(feature="high", value="+")
    c = Context(following=(front, high))
    assert len(c.following) == 2
    assert front in c.following
    assert high in c.following


def test_context_position_can_be_specified() -> None:
    c = Context(position="initial")
    assert c.position == "initial"


# ----- FeatureDisplacement -------------------------------------------------


def test_feature_displacement_records_named_change() -> None:
    d = FeatureDisplacement(feature="continuant", from_value="-", to_value="+")
    assert d.feature == "continuant"
    assert d.from_value == "-"
    assert d.to_value == "+"


def test_feature_displacements_are_hashable() -> None:
    # Needed because Link stores them in a tuple but we may want to
    # compare/dedupe them across links.
    d1 = FeatureDisplacement("continuant", "-", "+")
    d2 = FeatureDisplacement("continuant", "-", "+")
    assert hash(d1) == hash(d2)
    assert {d1, d2} == {d1}


# ----- Link ----------------------------------------------------------------


def test_one_to_one_link_has_single_segment_chunks() -> None:
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    assert len(link.source_chunk) == 1
    assert len(link.target_chunk) == 1


def test_link_default_context_is_empty() -> None:
    link = Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),))
    assert link.context == Context()


def test_zero_to_one_link_represents_insertion() -> None:
    link = Link(source_chunk=(), target_chunk=(Segment("e"),))
    assert link.source_chunk == ()
    assert len(link.target_chunk) == 1


def test_one_to_zero_link_represents_deletion() -> None:
    link = Link(source_chunk=(Segment("u"),), target_chunk=())
    assert len(link.source_chunk) == 1
    assert link.target_chunk == ()


def test_many_to_many_link_holds_multi_segment_chunks() -> None:
    # Latin /kt/ ~ Italian /tt/
    link = Link(
        source_chunk=(Segment("k"), Segment("t")),
        target_chunk=(Segment("t"), Segment("t")),
    )
    assert len(link.source_chunk) == 2
    assert len(link.target_chunk) == 2


# ----- Alignment -----------------------------------------------------------


def test_alignment_holds_forms_and_links() -> None:
    src = Form("latin", (Segment("p"), Segment("a"), Segment("t")))
    tgt = Form("germanic", (Segment("f"), Segment("a"), Segment("d")))
    links = (
        Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),)),
        Link(source_chunk=(Segment("a"),), target_chunk=(Segment("a"),)),
        Link(source_chunk=(Segment("t"),), target_chunk=(Segment("d"),)),
    )
    a = Alignment(source_form=src, target_form=tgt, links=links)
    assert a.source_form is src
    assert a.target_form is tgt
    assert len(a.links) == 3


# ----- Hashability ---------------------------------------------------------
#
# All core types are frozen dataclasses, so they should hash. This matters
# because higher layers will store them in sets and use them as dict keys
# (e.g., for memoising scores or grouping links by displacement type).


def test_all_core_types_are_hashable() -> None:
    """COMMITMENT: every core data type can be used as a set/dict key."""
    items = [
        Segment("p"),
        Segment("a", tone="high", length="long"),
        Form("latin", (Segment("p"), Segment("a"))),
        Form("latin", (Segment("p"),), syllable_breaks=(0,)),
        Lect("latin"),
        Lect("greek", feature_system="distinctive"),
        FeatureConstraint("voice", "+"),
        Context(),
        Context(position="initial", preceding=(FeatureConstraint("voice", "-"),)),
        FeatureDisplacement("continuant", "-", "+"),
        Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),)),
        Alignment(
            source_form=Form("latin", (Segment("p"),)),
            target_form=Form("germanic", (Segment("f"),)),
            links=(Link(source_chunk=(Segment("p"),), target_chunk=(Segment("f"),)),),
        ),
    ]
    # Each item should be hashable; a frozenset of all of them constructible.
    s = frozenset(items)
    assert len(s) == len(items)


# ----- Suprasegmental data flow -------------------------------------------
#
# The data model promises that suprasegmentals (tone, length, stress)
# travel through the types unchanged. Even though prior-only scoring
# ignores them, downstream layers will need them, and the tests pin the
# contract.


def test_suprasegmentals_survive_form_construction() -> None:
    """COMMITMENT: suprasegmental annotations are preserved verbatim through Form."""
    seg = Segment("a", tone="55", length="long", stress="primary")
    form = Form("hmong", (Segment("p"), seg))
    assert form.segments[1].tone == "55"
    assert form.segments[1].length == "long"
    assert form.segments[1].stress == "primary"


def test_suprasegmentals_distinguish_segments_for_equality() -> None:
    """COMMITMENT: tone-bearing minimal pairs are distinct objects.

    Required for cross-dimensional correspondences (tonogenesis): the
    framework must be able to tell that 'a with high tone' is different
    from 'a with low tone' even though the grapheme is the same.
    """
    a_high = Segment("a", tone="high")
    a_low = Segment("a", tone="low")
    a_plain = Segment("a")
    assert a_high != a_low
    assert a_high != a_plain
    assert a_low != a_plain


def test_form_with_no_syllable_breaks_has_empty_tuple() -> None:
    """COMMITMENT: syllable_breaks defaults to empty (not None) so the
    field is uniformly indexable."""
    form = Form("latin", (Segment("p"), Segment("a")))
    assert form.syllable_breaks == ()


def test_form_with_multiple_syllable_breaks() -> None:
    """COMMITMENT: multi-syllable forms carry per-syllable-start indices."""
    # ka.pu.tem (3 syllables)
    form = Form(
        "latin",
        (
            Segment("k"),
            Segment("a"),
            Segment("p"),
            Segment("u"),
            Segment("t"),
            Segment("e"),
            Segment("m"),
        ),
        syllable_breaks=(0, 2, 4),
    )
    assert form.syllable_breaks == (0, 2, 4)
    assert len(form.syllable_breaks) == 3


# ----- Lect with custom feature system ------------------------------------


def test_lect_with_custom_feature_system() -> None:
    """COMMITMENT: a lect can be configured to use any merkmal system."""
    lect = Lect(lect_id="sanskrit", feature_system="distinctive")
    assert lect.feature_system == "distinctive"


# ----- Context with combined preceding and following constraints ---------


def test_context_can_combine_preceding_and_following_constraints() -> None:
    """COMMITMENT: a single context can constrain BOTH neighbours.

    Required for correspondences like 'k ~ tS / [-cons]_[+front]'
    (intervocalic palatalisation conditioned by following vowel).
    """
    voc = FeatureConstraint("vowel", "+")
    front = FeatureConstraint("front", "+")
    ctx = Context(preceding=(voc,), following=(front,))
    assert ctx.preceding == (voc,)
    assert ctx.following == (front,)
    # And both at once must be retained.
    assert len(ctx.preceding) == 1
    assert len(ctx.following) == 1


# ----- Context subset matching and specificity () ---------------------


def test_empty_context_is_subset_of_everything() -> None:
    """COMMITMENT: the unconditioned context matches every link."""
    empty = Context()
    specific = Context(
        position="initial",
        following=(FeatureConstraint("vowel", "+"),),
    )
    assert empty.is_subset_of(specific)
    assert empty.is_subset_of(empty)


def test_context_subset_respects_position() -> None:
    initial = Context(position="initial")
    medial = Context(position="medial")
    assert not initial.is_subset_of(medial)
    assert not medial.is_subset_of(initial)
    assert initial.is_subset_of(Context(position="initial"))


def test_context_subset_respects_following_features() -> None:
    """COMMITMENT: every feature constraint in self must be in other."""
    voc = FeatureConstraint("vowel", "+")
    front = FeatureConstraint("front", "+")
    c1 = Context(following=(voc,))
    c2 = Context(following=(voc, front))
    assert c1.is_subset_of(c2)       # just `vowel` is less specific than `vowel + front`
    assert not c2.is_subset_of(c1)   # but `vowel + front` is not a subset of `vowel`


def test_context_constraint_count_counts_all_slots() -> None:
    c = Context(
        position="medial",
        preceding=(FeatureConstraint("vowel", "+"),),
        following=(
            FeatureConstraint("vowel", "+"),
            FeatureConstraint("front", "+"),
        ),
    )
    assert c.constraint_count() == 4  # position + 1 preceding + 2 following


def test_empty_context_has_zero_constraint_count() -> None:
    assert Context().constraint_count() == 0


# ----- Context long-range fields () ---------------------------------


def _front() -> FeatureConstraint:
    return FeatureConstraint("front", "+")


def _voc() -> FeatureConstraint:
    return FeatureConstraint("vowel", "+")


def test_context_preceding_at_distance_defaults_empty() -> None:
    c = Context()
    assert c.preceding_at_distance == ()
    assert c.following_at_distance == ()
    assert c.somewhere_preceding == ()
    assert c.somewhere_following == ()
    assert c.same_syllable == ()
    assert c.next_syllable == ()
    assert c.previous_syllable == ()


def test_context_long_range_fields_frozen() -> None:
    import dataclasses

    c = Context(same_syllable=(_front(),))
    try:
        c.same_syllable = ()  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Context should be frozen")


def test_is_subset_preceding_at_distance() -> None:
    c1 = Context(preceding_at_distance=((2, _front()),))
    c2 = Context(preceding_at_distance=((2, _front()), (3, _voc())))
    assert c1.is_subset_of(c2)
    assert not c2.is_subset_of(c1)
    # Different offset is not a match.
    c3 = Context(preceding_at_distance=((3, _front()),))
    assert not c1.is_subset_of(c3)


def test_is_subset_following_at_distance() -> None:
    c1 = Context(following_at_distance=((2, _front()),))
    c2 = Context(following_at_distance=((2, _front()),))
    assert c1.is_subset_of(c2)


def test_is_subset_somewhere_preceding() -> None:
    c1 = Context(somewhere_preceding=(_voc(),))
    c2 = Context(somewhere_preceding=(_voc(), _front()))
    assert c1.is_subset_of(c2)
    assert not c2.is_subset_of(c1)


def test_is_subset_somewhere_following() -> None:
    c1 = Context(somewhere_following=(_front(),))
    c2 = Context(somewhere_following=(_front(),))
    assert c1.is_subset_of(c2)


def test_is_subset_same_syllable() -> None:
    c1 = Context(same_syllable=(_voc(),))
    c2 = Context(same_syllable=(_voc(),))
    assert c1.is_subset_of(c2)
    assert not Context(same_syllable=(_front(),)).is_subset_of(c2)


def test_is_subset_next_syllable() -> None:
    c1 = Context(next_syllable=(_front(),))
    c2 = Context(next_syllable=(_front(), _voc()))
    assert c1.is_subset_of(c2)
    assert not c2.is_subset_of(c1)


def test_is_subset_previous_syllable() -> None:
    c1 = Context(previous_syllable=(_voc(),))
    assert c1.is_subset_of(Context(previous_syllable=(_voc(),)))
    assert not c1.is_subset_of(Context(previous_syllable=()))


def test_empty_context_subset_of_long_range_context() -> None:
    """Backward compatibility: empty context still matches anything,
    including contexts that carry long-range constraints."""
    specific = Context(
        next_syllable=(_front(),),
        somewhere_preceding=(_voc(),),
        preceding_at_distance=((2, _front()),),
    )
    assert Context().is_subset_of(specific)


def test_constraint_count_includes_long_range_slots() -> None:
    c = Context(
        position="medial",
        preceding=(_voc(),),
        preceding_at_distance=((2, _front()),),
        following_at_distance=((2, _voc()), (3, _front())),
        somewhere_preceding=(_voc(),),
        somewhere_following=(_front(),),
        same_syllable=(_voc(),),
        next_syllable=(_front(),),
        previous_syllable=(_voc(),),
    )
    # position(1) + preceding(1) + pre@d(1) + fol@d(2) + s_pre(1)
    # + s_fol(1) + same(1) + next(1) + prev(1) = 10
    assert c.constraint_count() == 10


def test_long_range_independent_of_immediate_neighbours() -> None:
    """Constraints on different slots don't interfere with each other."""
    table = Context(preceding=(_voc(),))
    link = Context(
        preceding=(_voc(),),
        next_syllable=(_front(),),
    )
    assert table.is_subset_of(link)
    # And a long-range-only table entry matches a long-range-carrying link.
    table2 = Context(next_syllable=(_front(),))
    assert table2.is_subset_of(link)
