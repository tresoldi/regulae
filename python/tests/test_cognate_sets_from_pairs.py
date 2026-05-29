"""Tests for the ``cognate_sets_from_pairs`` backward-compat helper."""

from regulae import CognateSet, Form, Segment, cognate_sets_from_pairs


def _form(word: str) -> Form:
    return Form(lect_id="x", segments=tuple(Segment(c) for c in word))


def test_pairs_to_cognate_sets_basic() -> None:
    """COMMITMENT: each pair becomes one CognateSet with the two forms
    keyed by the given lect IDs."""
    pairs = [(_form("pata"), _form("fada")), (_form("kat"), _form("gat"))]
    sets = cognate_sets_from_pairs(pairs, ("latin", "spanish"))
    assert len(sets) == 2
    assert all(isinstance(s, CognateSet) for s in sets)
    assert set(sets[0].forms) == {"latin", "spanish"}
    assert sets[0].forms["latin"].segments[0].grapheme == "p"
    assert sets[0].forms["spanish"].segments[0].grapheme == "f"


def test_pairs_to_cognate_sets_rewrites_lect_ids_on_forms() -> None:
    """COMMITMENT: the Form.lect_id is overwritten with the requested
    lect_ids so downstream code doesn't see a stale label."""
    pairs = [(_form("pa"), _form("fa"))]
    sets = cognate_sets_from_pairs(pairs, ("A", "B"))
    assert sets[0].forms["A"].lect_id == "A"
    assert sets[0].forms["B"].lect_id == "B"


def test_pairs_to_cognate_sets_generates_unique_cognate_ids() -> None:
    pairs = [(_form("a"), _form("a")) for _ in range(5)]
    sets = cognate_sets_from_pairs(pairs)
    ids = [s.cognate_id for s in sets]
    assert len(set(ids)) == 5


def test_pairs_to_cognate_sets_defaults_to_src_tgt_lect_ids() -> None:
    pairs = [(_form("a"), _form("b"))]
    sets = cognate_sets_from_pairs(pairs)
    assert set(sets[0].forms) == {"src", "tgt"}


def test_pairs_to_cognate_sets_empty_input() -> None:
    assert cognate_sets_from_pairs([]) == []


def test_pairs_to_cognate_sets_preserves_syllable_breaks() -> None:
    src = Form(lect_id="x", segments=tuple(Segment(c) for c in "pata"), syllable_breaks=(2,))
    tgt = Form(lect_id="y", segments=tuple(Segment(c) for c in "fada"), syllable_breaks=(2,))
    sets = cognate_sets_from_pairs([(src, tgt)], ("A", "B"))
    assert sets[0].forms["A"].syllable_breaks == (2,)
    assert sets[0].forms["B"].syllable_breaks == (2,)
