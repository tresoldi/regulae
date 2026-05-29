"""Tests for ``regulae.loaders`` — generic cognate-set loaders."""

from pathlib import Path

import pytest

from regulae import CognateSet, load_arcaverborum, load_cognates_from_tsv, load_gled


def _write_tsv(tmp_path: Path, name: str, rows: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


# ----- load_cognates_from_tsv --------------------------------------------


def test_load_tsv_minimal_roundtrip(tmp_path: Path) -> None:
    """COMMITMENT: a minimal TSV with cognate_id/lect_id/segments
    yields one CognateSet per cognate ID, one form per lect."""
    path = _write_tsv(
        tmp_path,
        "mini.tsv",
        [
            "cognate_id\tlect_id\tsegments",
            "hand.001\tlatin\tm a n u s",
            "hand.001\tspanish\tm a n o",
            "foot.002\tlatin\tp e d",
            "foot.002\tspanish\tp j e",
        ],
    )
    sets = load_cognates_from_tsv(path)
    assert len(sets) == 2
    assert [s.cognate_id for s in sets] == ["hand.001", "foot.002"]
    hand = sets[0]
    assert set(hand.forms) == {"latin", "spanish"}
    assert hand.forms["latin"].segments[0].grapheme == "m"
    assert len(hand.forms["spanish"].segments) == 4


def test_load_tsv_deterministic_ordering_by_first_occurrence(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "ord.tsv",
        [
            "cognate_id\tlect_id\tsegments",
            "b\tx\tp a",
            "a\tx\tt a",
            "b\ty\tb a",
            "a\ty\td a",
        ],
    )
    sets = load_cognates_from_tsv(path)
    assert [s.cognate_id for s in sets] == ["b", "a"]


def test_load_tsv_missing_required_column_raises(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "bad.tsv",
        ["cognate_id\tlect_id", "x\ta"],
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_cognates_from_tsv(path)


def test_load_tsv_duplicate_lect_for_cognate_raises(tmp_path: Path) -> None:
    """COMMITMENT: one cognate cannot have two rows for the same lect."""
    path = _write_tsv(
        tmp_path,
        "dup.tsv",
        [
            "cognate_id\tlect_id\tsegments",
            "x\tlatin\tp a",
            "x\tlatin\tt a",
        ],
    )
    with pytest.raises(ValueError, match="duplicate row"):
        load_cognates_from_tsv(path)


def test_load_tsv_skips_empty_segments(tmp_path: Path) -> None:
    """COMMITMENT: rows with empty segment strings are silently
    skipped rather than producing an empty form."""
    path = _write_tsv(
        tmp_path,
        "empty.tsv",
        [
            "cognate_id\tlect_id\tsegments",
            "x\ta\tp a",
            "x\tb\t",
        ],
    )
    sets = load_cognates_from_tsv(path)
    assert len(sets) == 1
    assert set(sets[0].forms) == {"a"}


def test_load_tsv_custom_column_names(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "custom.tsv",
        [
            "COGSET\tDOCULECT\tIPA",
            "c1\tLatin\tp a",
            "c1\tSpanish\tp a",
        ],
    )
    sets = load_cognates_from_tsv(
        path,
        cognate_id_col="COGSET",
        lect_id_col="DOCULECT",
        segments_col="IPA",
    )
    assert len(sets) == 1
    assert set(sets[0].forms) == {"Latin", "Spanish"}


def test_load_tsv_with_alignment_column(tmp_path: Path) -> None:
    """COMMITMENT: when alignment_col is given, dashes become None
    and the alignments field is populated."""
    path = _write_tsv(
        tmp_path,
        "aligned.tsv",
        [
            "cognate_id\tlect_id\tsegments\talignment",
            "hand\tlatin\tm a n u s\tm a n u s",
            "hand\tspanish\tm a n o\tm a n o -",
        ],
    )
    sets = load_cognates_from_tsv(path, alignment_col="alignment")
    assert len(sets) == 1
    cs = sets[0]
    assert cs.alignments is not None
    assert len(cs.alignments["latin"]) == 5
    assert len(cs.alignments["spanish"]) == 5
    assert cs.alignments["spanish"][-1] is None


def test_load_tsv_alignment_length_mismatch_raises(tmp_path: Path) -> None:
    """COMMITMENT: alignment columns within one cognate must have
    the same length across lects (column-aligned data)."""
    path = _write_tsv(
        tmp_path,
        "mismatch.tsv",
        [
            "cognate_id\tlect_id\tsegments\talignment",
            "x\ta\tp a\tp a",
            "x\tb\tt\tt a n",
        ],
    )
    with pytest.raises(ValueError, match="alignment length mismatch"):
        load_cognates_from_tsv(path, alignment_col="alignment")


def test_load_tsv_with_confidence_column(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "conf.tsv",
        [
            "cognate_id\tlect_id\tsegments\tconfidence",
            "c1\ta\tp a\t0.9",
            "c1\tb\tf a\t0.9",
            "c2\ta\tt a\t1.0",
            "c2\tb\td a\t1.0",
        ],
    )
    sets = load_cognates_from_tsv(path, confidence_col="confidence")
    by_id = {s.cognate_id: s for s in sets}
    assert by_id["c1"].confidence == 0.9
    assert by_id["c2"].confidence == 1.0


def test_load_tsv_confidence_takes_minimum_on_disagreement(tmp_path: Path) -> None:
    """COMMITMENT: when rows of one cognate disagree on confidence,
    the minimum is kept (pessimistic)."""
    path = _write_tsv(
        tmp_path,
        "conf.tsv",
        [
            "cognate_id\tlect_id\tsegments\tconfidence",
            "c1\ta\tp a\t0.9",
            "c1\tb\tf a\t0.4",
        ],
    )
    sets = load_cognates_from_tsv(path, confidence_col="confidence")
    assert sets[0].confidence == 0.4


def test_load_tsv_empty_file_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    assert load_cognates_from_tsv(path) == []


GLED_HEADER = (
    "ID\tDOCULECT\tLANGUAGE_NAME\tGLOTTOCODE\tGLOTTOLOG_NAME\tFAMILY\t"
    "CONCEPT\tCONCEPTICON_ID\tASJP_FORM\tFORM\tIPA\tALIGNMENT\tCOGSET\tCOGSET_INT"
)


def _gled_row(
    row_id: str,
    doculect: str,
    family: str,
    ipa: str,
    alignment: str,
    cogset: str,
) -> str:
    return (
        f"{row_id}\t{doculect}\tLang\tglot1234\tGlot\t{family}\t"
        f"concept\t1\tasjp\tform\t{ipa}\t{alignment}\t{cogset}\t1"
    )


def _write_gled(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "gled.tsv"
    p.write_text("\n".join([GLED_HEADER] + rows) + "\n", encoding="utf-8")
    return p


# ----- load_gled ---------------------------------------------------------


def test_load_gled_basic_romance(tmp_path: Path) -> None:
    """COMMITMENT: GLED rows grouped by COGSET yield one CognateSet each."""
    path = _write_gled(
        tmp_path,
        [
            _gled_row("r1", "LATIN", "Indo-European", "m a n u s", "m a n u s", "hand.0001"),
            _gled_row("r2", "SPANISH", "Indo-European", "m a n o", "m a n o -", "hand.0001"),
            _gled_row("r3", "ITALIAN", "Indo-European", "m a n o", "m a n o -", "hand.0001"),
            _gled_row("r4", "LATIN", "Indo-European", "p e d", "p e d", "foot.0001"),
            _gled_row("r5", "SPANISH", "Indo-European", "p j e", "p j e", "foot.0001"),
        ],
    )
    sets = load_gled(path)
    by_id = {s.cognate_id: s for s in sets}
    assert set(by_id) == {"hand.0001", "foot.0001"}
    assert set(by_id["hand.0001"].forms) == {"LATIN", "SPANISH", "ITALIAN"}
    assert by_id["hand.0001"].alignments is not None
    assert by_id["hand.0001"].alignments["SPANISH"][-1] is None


def test_load_gled_filters_by_family(tmp_path: Path) -> None:
    path = _write_gled(
        tmp_path,
        [
            _gled_row("r1", "LATIN", "Indo-European", "p a", "p a", "c1"),
            _gled_row("r2", "SPANISH", "Indo-European", "p a", "p a", "c1"),
            _gled_row("r3", "YORUBA", "Niger-Congo", "p a", "p a", "c2"),
            _gled_row("r4", "HAUSA", "Afro-Asiatic", "p a", "p a", "c2"),
        ],
    )
    sets = load_gled(path, family="Indo-European")
    assert len(sets) == 1
    assert sets[0].cognate_id == "c1"


def test_load_gled_filters_by_doculect(tmp_path: Path) -> None:
    path = _write_gled(
        tmp_path,
        [
            _gled_row("r1", "LATIN", "IE", "p a", "p a", "c1"),
            _gled_row("r2", "SPANISH", "IE", "p a", "p a", "c1"),
            _gled_row("r3", "FRENCH", "IE", "p", "p", "c1"),
            _gled_row("r4", "ITALIAN", "IE", "p a", "p a", "c1"),
        ],
    )
    sets = load_gled(path, doculects={"LATIN", "SPANISH"})
    assert len(sets) == 1
    assert set(sets[0].forms) == {"LATIN", "SPANISH"}


def test_load_gled_drops_cognate_below_min_lects(tmp_path: Path) -> None:
    """COMMITMENT: cognates with fewer than min_lects are dropped."""
    path = _write_gled(
        tmp_path,
        [
            _gled_row("r1", "LATIN", "IE", "p a", "p a", "c1"),
            _gled_row("r2", "SPANISH", "IE", "p a", "p a", "c1"),
            _gled_row("r3", "LATIN", "IE", "k a", "k a", "c2"),
        ],
    )
    sets = load_gled(path, min_lects=2)
    assert [s.cognate_id for s in sets] == ["c1"]


def test_load_gled_alignment_length_mismatch_drops_alignment_only(
    tmp_path: Path,
) -> None:
    """COMMITMENT: malformed alignment data is silently discarded but
    the cognate itself is kept — we have enough good data in real
    GLED to throw away a few inconsistent alignment columns."""
    path = _write_gled(
        tmp_path,
        [
            _gled_row("r1", "LATIN", "IE", "p a", "p a", "c1"),
            _gled_row("r2", "SPANISH", "IE", "p a t", "p a t o", "c1"),
        ],
    )
    sets = load_gled(path)
    assert len(sets) == 1
    assert sets[0].alignments is None
    assert set(sets[0].forms) == {"LATIN", "SPANISH"}


def test_load_gled_duplicate_lect_keeps_first(tmp_path: Path) -> None:
    """COMMITMENT: when a (lect, cognate) has multiple reflexes, the
    first one wins rather than raising."""
    path = _write_gled(
        tmp_path,
        [
            _gled_row("r1", "LATIN", "IE", "p a", "p a", "c1"),
            _gled_row("r2", "LATIN", "IE", "p o", "p o", "c1"),
            _gled_row("r3", "SPANISH", "IE", "p a", "p a", "c1"),
        ],
    )
    sets = load_gled(path)
    assert len(sets) == 1
    assert sets[0].forms["LATIN"].segments[1].grapheme == "a"


def test_load_gled_missing_required_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.tsv"
    p.write_text("ID\tDOCULECT\n" "r1\tLATIN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        load_gled(p)


def test_load_tsv_returns_cognate_set_instances(tmp_path: Path) -> None:
    path = _write_tsv(
        tmp_path,
        "x.tsv",
        ["cognate_id\tlect_id\tsegments", "c\tl\tp a"],
    )
    sets = load_cognates_from_tsv(path)
    assert all(isinstance(s, CognateSet) for s in sets)


# ----- arcaverborum loader -----------------------------------------------


ARCA_HEADER = (
    "ID,Dataset,Language_ID,Parameter_ID,Form,Segments,Cognacy,Alignment,Family"
)


def _arca_row(
    row_id: str,
    dataset: str,
    lang: str,
    param: str,
    form: str,
    segs: str,
    cognacy: str,
    alignment: str = "",
    family: str = "Romance",
) -> str:
    return (
        f"{row_id},{dataset},{lang},{param},{form},{segs},"
        f"{cognacy},{alignment},{family}"
    )


def _write_arca(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "forms.csv"
    p.write_text("\n".join([ARCA_HEADER] + rows) + "\n", encoding="utf-8")
    return p


def test_load_arcaverborum_basic(tmp_path: Path) -> None:
    """COMMITMENT: rows with the same Cognacy first-ID are grouped into
    one CognateSet keyed by that ID, with Language_ID becoming the lect."""
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "romance", "Latin", "hand", "manus", "m a n u s", "romance_hand-1"),
            _arca_row("r2", "romance", "Spanish", "hand", "mano", "m a n o", "romance_hand-1"),
            _arca_row("r3", "romance", "Italian", "hand", "mano", "m a n o", "romance_hand-1"),
            _arca_row("r4", "romance", "Latin", "foot", "pes", "p e s", "romance_foot-1"),
            _arca_row("r5", "romance", "Spanish", "foot", "pje", "p j e", "romance_foot-1"),
        ],
    )
    sets = load_arcaverborum(path)
    by_id = {s.cognate_id: s for s in sets}
    assert set(by_id) == {"romance_hand-1", "romance_foot-1"}
    assert set(by_id["romance_hand-1"].forms) == {"Latin", "Spanish", "Italian"}


def test_load_arcaverborum_parses_morpheme_boundaries(tmp_path: Path) -> None:
    """COMMITMENT: ``+`` tokens in Segments are recorded as
    morpheme_boundaries and stripped from the segment sequence."""
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "d", "A", "p", "abc", "ɐ ŋ + dz eː", "d_c1"),
            _arca_row("r2", "d", "B", "p", "abc", "ɐ ŋ + dz eː", "d_c1"),
        ],
    )
    sets = load_arcaverborum(path)
    assert len(sets) == 1
    cs = sets[0]
    # Segments contains 4 phonemes (the '+' is stripped).
    assert len(cs.forms["A"].segments) == 4
    assert cs.forms["A"].segments[0].grapheme == "ɐ"
    # Morpheme boundary is at position 2 (after 'ɐ ŋ').
    assert cs.morpheme_boundaries is not None
    assert cs.morpheme_boundaries["A"] == (2,)


def test_load_arcaverborum_uses_first_cognacy_id(tmp_path: Path) -> None:
    """COMMITMENT: arcaverborum's Cognacy can be a semicolon-separated
    list; the loader uses the FIRST ID as the grouping key."""
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "d", "A", "p", "pa", "p a", "d_c1;d_c2"),
            _arca_row("r2", "d", "B", "p", "fa", "f a", "d_c1;d_c2"),
        ],
    )
    sets = load_arcaverborum(path)
    assert len(sets) == 1
    assert sets[0].cognate_id == "d_c1"


def test_load_arcaverborum_filters_by_dataset(tmp_path: Path) -> None:
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "d1", "A", "p", "pa", "p a", "d1_c1"),
            _arca_row("r2", "d1", "B", "p", "fa", "f a", "d1_c1"),
            _arca_row("r3", "d2", "X", "p", "pa", "p a", "d2_c1"),
            _arca_row("r4", "d2", "Y", "p", "ba", "b a", "d2_c1"),
        ],
    )
    sets = load_arcaverborum(path, dataset="d1")
    assert len(sets) == 1
    assert sets[0].cognate_id == "d1_c1"


def test_load_arcaverborum_filters_by_language_ids(tmp_path: Path) -> None:
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "d", "A", "p", "pa", "p a", "d_c1"),
            _arca_row("r2", "d", "B", "p", "fa", "f a", "d_c1"),
            _arca_row("r3", "d", "C", "p", "ga", "g a", "d_c1"),
        ],
    )
    sets = load_arcaverborum(path, language_ids={"A", "B"})
    assert len(sets) == 1
    assert set(sets[0].forms) == {"A", "B"}


def test_load_arcaverborum_skips_na_cognacy(tmp_path: Path) -> None:
    """COMMITMENT: rows with ``<NA>`` or empty Cognacy are skipped."""
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "d", "A", "p", "pa", "p a", "<NA>"),
            _arca_row("r2", "d", "B", "p", "fa", "f a", ""),
            _arca_row("r3", "d", "C", "p", "ga", "g a", "d_c1"),
            _arca_row("r4", "d", "D", "p", "ka", "k a", "d_c1"),
        ],
    )
    sets = load_arcaverborum(path)
    assert len(sets) == 1
    assert set(sets[0].forms) == {"C", "D"}


def test_load_arcaverborum_parses_alignment(tmp_path: Path) -> None:
    path = _write_arca(
        tmp_path,
        [
            _arca_row("r1", "d", "A", "p", "pata", "p a t a", "d_c1", "p a t a"),
            _arca_row("r2", "d", "B", "p", "fada", "f a d a", "d_c1", "f a d a"),
        ],
    )
    sets = load_arcaverborum(path)
    assert len(sets) == 1
    assert sets[0].alignments is not None
    assert len(sets[0].alignments["A"]) == 4


def test_load_arcaverborum_missing_required_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("ID,Dataset\nr1,d\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_arcaverborum(p)
