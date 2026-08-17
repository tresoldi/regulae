"""Tests for  reconciliation: reconciliation of pairwise alignments into
multi-lect correspondence class observations.

These tests use small synthetic corpora where the expected classes
are obvious by inspection.
"""

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


def _classes_by_key(model: MultiLectModel) -> dict[tuple[tuple[str, str], ...], float]:
    """Return a dict keyed by sorted (lect, grapheme) tuples to count."""
    return {
        tuple(sorted(k.segments.items())): k.count
        for k in model.unconditioned_classes
    }


# ----- N=2 sanity --------------------------------------------------------


def test_reconciliation_n2_matches_pairwise_counts() -> None:
    """COMMITMENT: for N=2 with a simple 1-to-1 corpus, reconciled
    classes match the raw pairwise correspondence counts."""
    pairs = [
        (_form("A", "pata"), _form("B", "fada")),
        (_form("A", "pata"), _form("B", "fada")),
    ]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    by_key = _classes_by_key(model)
    # p↔f, a↔a, t↔d observed twice each; one a at the end
    p_f = (("A", "p"), ("B", "f"))
    a_a = (("A", "a"), ("B", "a"))
    t_d = (("A", "t"), ("B", "d"))
    assert by_key.get(p_f, 0) == 2
    assert by_key.get(t_d, 0) == 2
    assert by_key.get(a_a, 0) == 4  # two positions × two pairs


def test_reconciliation_classes_are_sorted_by_count_desc() -> None:
    """COMMITMENT: open question 4 — classes are ordered by count
    descending, with lexicographic tiebreak."""
    pairs = [
        (_form("A", "aaab"), _form("B", "aaac")),
    ]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    counts = [k.count for k in model.unconditioned_classes]
    assert counts == sorted(counts, reverse=True)


def test_reconciliation_supporting_cognates_tracked() -> None:
    pairs = [
        (_form("A", "pa"), _form("B", "fa")),
        (_form("A", "pa"), _form("B", "fa")),
    ]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    # Every class in this micro-corpus should list both cognate IDs.
    for klass in model.unconditioned_classes:
        assert set(klass.supporting_cognates) == {"pair.00000", "pair.00001"}


def test_supporting_cognates_are_distinct_sets() -> None:
    """The number `count` cannot give: how much of the lexicon a class rests on.

    Each word realises p~f twice, so the class is worth four aligned positions
    and rests on two cognate sets. M6's adjudicators read `count` as sets and
    accepted rows a single word supported; this is the number they wanted.
    """
    pairs = [
        (_form("A", "papa"), _form("B", "fafa")),
        (_form("A", "pipi"), _form("B", "fifi")),
    ]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    matched = [k for k in model.unconditioned_classes
               if k.segments == {"A": "p", "B": "f"}]
    assert len(matched) == 1
    assert matched[0].count == 4
    assert matched[0].supporting_cognates == ("pair.00000", "pair.00001")
    for klass in model.unconditioned_classes:
        support = klass.supporting_cognates
        assert len(set(support)) == len(support)


# ----- N=3: merger disambiguation ---------------------------------------


def test_reconciliation_n3_produces_three_lect_classes() -> None:
    """COMMITMENT: with three lects and consistent pairwise alignments,
    reconciliation yields three-lect classes."""
    corpus = [
        CognateSet(
            cognate_id="c1",
            forms={
                "A": _form("A", "pa"),
                "B": _form("B", "fa"),
                "C": _form("C", "pa"),
            },
        ),
        CognateSet(
            cognate_id="c2",
            forms={
                "A": _form("A", "pa"),
                "B": _form("B", "fa"),
                "C": _form("C", "pa"),
            },
        ),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    classes = model.unconditioned_classes
    assert len(classes) >= 1
    # There should be a class with all three lects participating.
    three_lect = [k for k in classes if len(k.segments) == 3]
    assert len(three_lect) >= 1
    # Exact p↔f↔p + a↔a↔a classes should be present.
    by_key = _classes_by_key(model)
    assert by_key.get((("A", "p"), ("B", "f"), ("C", "p")), 0) == 2
    assert by_key.get((("A", "a"), ("B", "a"), ("C", "a")), 0) == 2


def test_reconciliation_merger_disambiguation_visible() -> None:
    """COMMITMENT: when lect A has merged two proto-segments but lect
    B preserves the distinction, the A↔B correspondence should yield
    two distinct classes — one where A's merged segment aligns to
    B's preserved reflex 1, and one where it aligns to reflex 2.

    Here lect A has ``k`` corresponding to B's ``t`` in one cognate
    and to B's ``k`` in another. The two classes must be distinct
    in the output.
    """
    corpus = [
        CognateSet(
            cognate_id="c1",
            forms={"A": _form("A", "ka"), "B": _form("B", "ta")},
        ),
        CognateSet(
            cognate_id="c2",
            forms={"A": _form("A", "ka"), "B": _form("B", "ka")},
        ),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    by_key = _classes_by_key(model)
    k_t = (("A", "k"), ("B", "t"))
    k_k = (("A", "k"), ("B", "k"))
    assert by_key.get(k_t, 0) >= 1
    assert by_key.get(k_k, 0) >= 1


# ----- sparse data -------------------------------------------------------


def test_reconciliation_recovers_data_from_unequal_length_chunks() -> None:
    """COMMITMENT: when  promotes an unequal-length chunk (e.g.
    2-to-1 like OE ``sk → ʃ``), reconciliation reconciliation decomposes
    the chunk via a sub-alignment and recovers position-level
    observations, rather than dropping the whole alignment as
    inconsistent.

    We build a corpus where A ↔ B is biased toward a 2-to-1 chunk
    (``sk → ʃ``) that  is likely to promote, then check that
    the reconciliation still surfaces segment-level observations
    for the chunk's internal positions.
    """
    pairs = [
        (_form("A", "ska"), _form("B", "ʃa")),
        (_form("A", "ski"), _form("B", "ʃi")),
        (_form("A", "sku"), _form("B", "ʃu")),
        (_form("A", "sko"), _form("B", "ʃo")),
        (_form("A", "ske"), _form("B", "ʃe")),
        (_form("A", "aska"), _form("B", "aʃa")),
    ]
    corpus = cognate_sets_from_pairs(pairs, ("A", "B"))
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)

    by_key = _classes_by_key(model)
    # After sub-alignment decomposition, we should see sk → ʃ expressed
    # at the segment level in at least one of these forms. Either as
    # (A:s ↔ B:ʃ) with A:k landing in a gap, or as (A:s ↔ B:ʃ) and
    # (A:k ↔ B:ʃ), depending on how the sub-alignment pairs the
    # chunk's internals. Either way there must be at least one class
    # with A:s -> B:ʃ — which previously was silently dropped.
    assert any(
        segs == (("A", "s"), ("B", "ʃ"))
        for segs in by_key
    ), (
        "expected a (A:s, B:ʃ) class from sub-alignment decomposition "
        f"of the 2-to-1 chunk, got: {sorted(by_key.keys())}"
    )


def test_reconciliation_sparse_missing_lects_only_affects_that_set() -> None:
    """COMMITMENT: a cognate set missing one of several lects
    contributes classes only for the lects that are present."""
    corpus = [
        CognateSet(
            cognate_id="c1",
            forms={
                "A": _form("A", "pa"),
                "B": _form("B", "fa"),
                "C": _form("C", "pa"),
            },
        ),
        CognateSet(
            cognate_id="c2",
            forms={
                "A": _form("A", "pa"),
                "B": _form("B", "fa"),
            },
        ),
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    classes = model.unconditioned_classes
    # Some three-lect and some two-lect classes should exist.
    assert any(len(k.segments) == 3 for k in classes)
    assert any(len(k.segments) == 2 for k in classes)
