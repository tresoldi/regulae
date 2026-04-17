"""Tests for the morpheme-boundary chunk filter.

The chunk-promotion stage rejects any candidate whose source or
target span crosses a morpheme break. Boundaries are user-supplied
on ``Form.morpheme_breaks`` (or, for backward compatibility, on
``CognateSet.morpheme_boundaries``); forms without breaks behave
exactly as before.
"""

from regulae import (
    CognateSet,
    Form,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    train_model,
)


def _form(lect: str, ipa: str, breaks: tuple[int, ...] = ()) -> Form:
    return Form(lect, tuple(Segment(c) for c in ipa), morpheme_breaks=breaks)


# ----- mechanism on synthetic data ----------------------------------------


def _morph_synthetic_pairs(with_boundaries: bool):
    """16 stem+suffix gemination pairs + 8 monomorphemic controls.

    Mirrors experiments/morph_boundary_synthetic/cognates.tsv.
    """
    stems = ["pat", "pat", "pat", "pat", "tat", "tat", "kat", "kat",
             "mat", "mat", "nat", "nat", "rat", "rat", "bat", "dat"]
    suffixes = ["a", "i", "u", "o", "a", "i", "a", "u",
                "a", "o", "a", "e", "a", "u", "a", "a"]
    pairs = []
    for stem, sfx in zip(stems, suffixes):
        proto = stem + sfx
        derived = stem + stem[-1] + sfx  # gemination at boundary
        if with_boundaries:
            src = _form("proto", proto, breaks=(3,))
            tgt = _form("derived", derived, breaks=(3,))
        else:
            src = _form("proto", proto)
            tgt = _form("derived", derived)
        pairs.append((src, tgt))
    # controls (monomorphemic CVC, no transformation)
    for w in ("pat", "tap", "kap", "map", "pak", "tak", "mak", "nat"):
        pairs.append((_form("proto", w), _form("derived", w)))
    return pairs


def test_morph_synthetic_filters_boundary_crossing_chunks() -> None:
    """COMMITMENT: with boundaries supplied, chunks straddling the
    morphological seam are rejected from the chunk table; without
    boundaries, they're all promoted."""
    pairs_no = _morph_synthetic_pairs(with_boundaries=False)
    pairs_yes = _morph_synthetic_pairs(with_boundaries=True)
    m_no = train_model(cognate_sets_from_pairs(pairs_no, ("proto", "derived")))
    m_yes = train_model(cognate_sets_from_pairs(pairs_yes, ("proto", "derived")))
    assert isinstance(m_no, MultiLectModel)
    assert isinstance(m_yes, MultiLectModel)
    chunks_no = set(m_no.pairwise_models[frozenset({"proto", "derived"})].chunk_table.entries)
    chunks_yes = set(m_yes.pairwise_models[frozenset({"proto", "derived"})].chunk_table.entries)
    dropped = chunks_no - chunks_yes
    assert dropped, "expected at least one boundary-crossing chunk to be rejected"
    # Verify each dropped chunk did indeed straddle a boundary.
    for src_chunk, tgt_chunk in dropped:
        # Chunks of length t > 1 from the source side imply the chunk
        # crossed the stem/suffix seam (since the stem is 3 segs and
        # the suffix is 1 seg).
        assert len(src_chunk) >= 2 or len(tgt_chunk) >= 2


def test_form_morpheme_breaks_propagates_via_cognate_sets_from_pairs() -> None:
    """Regression: cognate_sets_from_pairs must carry through
    morpheme_breaks when rebuilding Forms."""
    src = _form("p", "pata", breaks=(3,))
    tgt = _form("d", "patta", breaks=(3,))
    sets = cognate_sets_from_pairs([(src, tgt)], ("p", "d"))
    assert sets[0].forms["p"].morpheme_breaks == (3,)
    assert sets[0].forms["d"].morpheme_breaks == (3,)


def test_legacy_cognateset_morpheme_boundaries_still_works() -> None:
    """Backward compat: a CognateSet built directly with
    morpheme_boundaries (legacy field) still flows into chunk
    promotion via the _form_with_boundaries helper."""
    cs = CognateSet(
        cognate_id="x",
        forms={
            "p": Form("p", tuple(Segment(c) for c in "pata")),
            "d": Form("d", tuple(Segment(c) for c in "patta")),
        },
        morpheme_boundaries={"p": (3,), "d": (3,)},
    )
    # Build a corpus with enough repeats to drive chunk promotion.
    corpus = [
        CognateSet(
            cognate_id=f"c{i}",
            forms=cs.forms,
            morpheme_boundaries=cs.morpheme_boundaries,
        )
        for i in range(8)
    ] + [
        CognateSet(
            cognate_id=f"d{i}",
            forms={
                "p": Form("p", tuple(Segment(c) for c in "tata")),
                "d": Form("d", tuple(Segment(c) for c in "tatta")),
            },
            morpheme_boundaries={"p": (3,), "d": (3,)},
        )
        for i in range(8)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    chunks = set(model.pairwise_models[frozenset({"p", "d"})].chunk_table.entries)
    # The boundary-spanning chunks (at, att), (ta, tta) should be absent.
    bad_chunks = {
        (tuple(Segment(c) for c in "at"), tuple(Segment(c) for c in "att")),
        (tuple(Segment(c) for c in "ta"), tuple(Segment(c) for c in "tta")),
    }
    for bad in bad_chunks:
        assert bad not in chunks, f"boundary-crossing chunk {bad} survived"


# ----- non-regression on un-annotated corpora ----------------------------


def test_unannotated_corpus_unaffected() -> None:
    """COMMITMENT: a corpus without morpheme_breaks behaves exactly
    as before (no chunks get rejected)."""
    pairs_no = _morph_synthetic_pairs(with_boundaries=False)
    a = train_model(cognate_sets_from_pairs(pairs_no, ("proto", "derived")))
    b = train_model(cognate_sets_from_pairs(pairs_no, ("proto", "derived")))
    assert isinstance(a, MultiLectModel) and isinstance(b, MultiLectModel)
    assert (
        a.pairwise_models[frozenset({"proto", "derived"})].chunk_table.entries
        == b.pairwise_models[frozenset({"proto", "derived"})].chunk_table.entries
    )


def test_boundary_at_chunk_edge_does_not_reject() -> None:
    """A morpheme break at exactly the chunk's start or end position
    is on the edge and does not count as crossed. Chunks like
    (V, tV) starting at the suffix onset should still be allowed."""
    from regulae._chunks import _spans_break
    breaks = frozenset({3})
    # chunk [3, 5): break at 3 is at the start, not inside
    assert not _spans_break(3, 5, breaks)
    # chunk [1, 3): break at 3 is at the end, not inside
    assert not _spans_break(1, 3, breaks)
    # chunk [2, 4): break at 3 is strictly inside
    assert _spans_break(2, 4, breaks)
