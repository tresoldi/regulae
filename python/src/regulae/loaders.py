"""Loaders for external cognate-set data formats.

Three loaders are provided: a generic TSV loader plus format-specific
loaders for GLED and arcaverborum (Lexibank-derived) data.

Each loader returns ``list[CognateSet]`` so downstream training code
is format-agnostic.
"""

import csv
from collections.abc import Collection
from pathlib import Path

from regulae.model import CognateSet
from regulae.types import CognateId, Form, Grapheme, LectId, Segment


def _parse_segments(raw: str) -> tuple[Segment, ...]:
    """Split a space-separated segment string into ``Segment`` objects.

    Empty graphemes (from double spaces or leading/trailing whitespace)
    are dropped. Dashes ``-`` are treated as gap markers and are NOT
    returned as segments — a form is the sequence of non-gap segments.
    """
    return tuple(Segment(Grapheme(tok)) for tok in raw.split() if tok and tok != "-")


def _parse_alignment(raw: str) -> tuple[Segment | None, ...]:
    """Parse a space-separated alignment column.

    Dashes become ``None`` (gap); other tokens become ``Segment``.
    """
    out: list[Segment | None] = []
    for tok in raw.split():
        if not tok:
            continue
        if tok == "-":
            out.append(None)
        else:
            out.append(Segment(Grapheme(tok)))
    return tuple(out)


def load_cognates_from_tsv(
    path: Path | str,
    *,
    cognate_id_col: str = "cognate_id",
    lect_id_col: str = "lect_id",
    segments_col: str = "segments",
    alignment_col: str | None = None,
    confidence_col: str | None = None,
) -> list[CognateSet]:
    """Load cognate sets from a TSV file.

    Expected layout: one row per (lect, cognate) pair. Required columns
    identify the cognate ID, the lect ID, and the segments. Optional
    columns add pre-alignment hints and a per-row confidence score.

    Rows sharing a cognate ID are grouped into one :class:`CognateSet`.
    Cognate sets are returned in the order their IDs first appear in
    the file (deterministic on any ordered input).

    Arguments:

    * ``path``: file path.
    * ``cognate_id_col``, ``lect_id_col``, ``segments_col``: required
      column names.
    * ``alignment_col``: if given, parsed as space-separated alignment
      with ``-`` for gaps. All rows for one cognate must produce
      alignment tuples of the same length; otherwise a ``ValueError``
      is raised.
    * ``confidence_col``: if given, parsed as a float. When different
      rows of the same cognate disagree, the minimum is kept (pessimistic).

    Raises:

    * ``ValueError`` if required columns are missing, if a cognate
      appears twice for the same lect, or if alignment lengths
      mismatch within a cognate.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            return []
        required = {cognate_id_col, lect_id_col, segments_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"TSV {path} missing required columns: {sorted(missing)}"
            )

        order: list[str] = []
        forms_by_id: dict[str, dict[str, Form]] = {}
        aligns_by_id: dict[str, dict[str, tuple[Segment | None, ...]]] = {}
        conf_by_id: dict[str, float] = {}

        for row in reader:
            cog_id = row[cognate_id_col].strip()
            lect_id = row[lect_id_col].strip()
            if not cog_id or not lect_id:
                continue
            segs = _parse_segments(row[segments_col])
            if not segs:
                continue
            if cog_id not in forms_by_id:
                order.append(cog_id)
                forms_by_id[cog_id] = {}
                aligns_by_id[cog_id] = {}
            if lect_id in forms_by_id[cog_id]:
                raise ValueError(
                    f"TSV {path}: cognate {cog_id!r} has duplicate row for "
                    f"lect {lect_id!r}"
                )
            forms_by_id[cog_id][lect_id] = Form(lect_id=LectId(lect_id), segments=segs)

            if alignment_col is not None:
                raw_align = row.get(alignment_col, "") or ""
                aligned = _parse_alignment(raw_align)
                if aligned:
                    existing = aligns_by_id[cog_id]
                    if existing and len(next(iter(existing.values()))) != len(aligned):
                        raise ValueError(
                            f"TSV {path}: cognate {cog_id!r} has alignment "
                            f"length mismatch across lects"
                        )
                    aligns_by_id[cog_id][lect_id] = aligned

            if confidence_col is not None:
                raw_conf = row.get(confidence_col, "") or ""
                if raw_conf.strip():
                    try:
                        c = float(raw_conf)
                    except ValueError as exc:
                        raise ValueError(
                            f"TSV {path}: cognate {cog_id!r} has non-numeric "
                            f"{confidence_col}={raw_conf!r}"
                        ) from exc
                    conf_by_id[cog_id] = min(conf_by_id.get(cog_id, c), c)

    out: list[CognateSet] = []
    for cog_id in order:
        forms = forms_by_id[cog_id]
        aligns = aligns_by_id[cog_id] or None
        confidence = conf_by_id.get(cog_id, 1.0)
        out.append(
            CognateSet(
                cognate_id=CognateId(cog_id),
                forms=forms,
                alignments=aligns,
                confidence=confidence,
            )
        )
    return out


# ----- GLED loader -------------------------------------------------------


def load_gled(
    path: Path | str,
    *,
    family: str | None = None,
    doculects: Collection[str] | None = None,
    min_lects: int = 2,
) -> list[CognateSet]:
    """Load GLED-format data from a TSV file.

    GLED (https://github.com/tresoldi/gled) ships one TSV file per
    release with the columns:

        ID DOCULECT LANGUAGE_NAME GLOTTOCODE GLOTTOLOG_NAME FAMILY
        CONCEPT CONCEPTICON_ID ASJP_FORM FORM IPA ALIGNMENT
        COGSET COGSET_INT

    This loader groups rows by ``COGSET`` (the cognate class
    identifier, which already embeds the CONCEPT), filters by family
    and/or doculect, and returns one :class:`CognateSet` per cognate.

    Arguments:

    * ``path``: path to the GLED TSV.
    * ``family``: if given, only rows whose FAMILY matches are kept.
    * ``doculects``: if given, only rows whose DOCULECT is in the
      collection are kept.
    * ``min_lects``: drop cognate sets with fewer than this many
      distinct lects after filtering. Defaults to 2 (below 2 there
      is no pair to train on).

    The IPA column is used for ``Form.segments`` and the ALIGNMENT
    column populates ``CognateSet.alignments`` (dashes become gap
    ``None``). When alignment lengths disagree across lects within a
    cognate (data error), that cognate is silently dropped rather
    than raising — GLED has enough data that a few bad rows
    shouldn't kill a whole experiment.
    """
    path = Path(path)
    order: list[str] = []
    forms_by_id: dict[str, dict[str, Form]] = {}
    aligns_by_id: dict[str, dict[str, tuple[Segment | None, ...]]] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"DOCULECT", "FAMILY", "IPA", "COGSET"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(
                f"GLED TSV {path}: expected columns {sorted(required)}, "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            if family is not None and row["FAMILY"] != family:
                continue
            lect_id = row["DOCULECT"]
            if doculects is not None and lect_id not in doculects:
                continue
            cog_id = row["COGSET"].strip()
            if not cog_id:
                continue
            segs = _parse_segments(row["IPA"])
            if not segs:
                continue
            if cog_id not in forms_by_id:
                order.append(cog_id)
                forms_by_id[cog_id] = {}
                aligns_by_id[cog_id] = {}
            if lect_id in forms_by_id[cog_id]:
                # Duplicate (lect, cognate) rows do exist in GLED when
                # a language has multiple reflexes of the same proto-
                # form. Keep the first one; drop subsequent ones.
                continue
            forms_by_id[cog_id][lect_id] = Form(lect_id=LectId(lect_id), segments=segs)
            raw_align = row.get("ALIGNMENT", "") or ""
            if raw_align:
                aligns_by_id[cog_id][lect_id] = _parse_alignment(raw_align)

    out: list[CognateSet] = []
    for cog_id in order:
        forms = forms_by_id[cog_id]
        if len(forms) < min_lects:
            continue
        aligns = aligns_by_id[cog_id] or None
        # Reject cognates with inconsistent alignment lengths.
        if aligns:
            lens = {len(v) for v in aligns.values()}
            if len(lens) > 1:
                aligns = None  # silently discard malformed alignment hint
        out.append(CognateSet(cognate_id=CognateId(cog_id), forms=forms, alignments=aligns))
    return out


# ----- arcaverborum loader -----------------------------------------------


def _parse_arcaverborum_segments(
    raw: str,
) -> tuple[tuple[Segment, ...], tuple[int, ...]]:
    """Parse an arcaverborum Segments cell.

    The format is space-separated graphemes with ``+`` tokens marking
    morpheme boundaries. Example: ``ɐ ŋ + dz eː`` has a boundary
    after position 2.

    Returns a tuple ``(segments, boundaries)`` where ``boundaries`` is
    the tuple of positions (in the filtered segment sequence) where
    morpheme boundaries occur. The boundaries are reserved for future
    morph-aware alignment; stored on the CognateSet but not yet
    consumed.
    """
    segments: list[Segment] = []
    boundaries: list[int] = []
    for tok in raw.split():
        if not tok:
            continue
        if tok == "+":
            if segments:
                boundaries.append(len(segments))
            continue
        if tok == "-":
            continue
        segments.append(Segment(Grapheme(tok)))
    return tuple(segments), tuple(boundaries)


def load_arcaverborum(
    path: Path | str,
    *,
    dataset: str | None = None,
    language_ids: Collection[str] | None = None,
    family: str | None = None,
    min_lects: int = 2,
) -> list[CognateSet]:
    """Load arcaverborum-format (merged Lexibank CLDF) data.

    arcaverborum (https://github.com/tresoldi/arcaverborum) ships
    merged ``forms.csv`` files with the schema documented in
    ``docs/MERGER_SPECIFICATION.md``. The columns this loader reads:

    * ``Language_ID`` → ``lect_id``
    * ``Segments`` → ``Form.segments`` (space-separated graphemes;
      ``+`` tokens mark morpheme boundaries and are recorded on
      ``CognateSet.morpheme_boundaries`` but stripped from the
      segment sequence)
    * ``Cognacy`` → ``cognate_id`` (arcaverborum stores this as a
      semicolon-separated list of cognate-set IDs; the FIRST ID is
      used as the grouping key and the rest are ignored)
    * ``Alignment`` → optional pre-alignment hint (``-`` tokens
      become gap ``None``)
    * ``Family`` → optional filter

    Arguments:

    * ``path``: path to the merged forms CSV (or TSV).
    * ``dataset``: if given, only rows with matching ``Dataset`` are
      kept.
    * ``language_ids``: if given, only rows whose ``Language_ID`` is
      in this collection are kept.
    * ``family``: if given and the file includes a ``Family`` column,
      only rows with matching family are kept.
    * ``min_lects``: drop cognate sets with fewer than this many
      distinct lects after filtering. Defaults to 2.

    Cognates with inconsistent alignment lengths are kept but their
    alignment hint is discarded. Duplicate (lect, cognate) rows keep
    the first one.
    """
    path = Path(path)
    # Auto-detect TSV vs CSV by extension.
    delim = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","

    order: list[str] = []
    forms_by_id: dict[str, dict[str, Form]] = {}
    aligns_by_id: dict[str, dict[str, tuple[Segment | None, ...]]] = {}
    boundaries_by_id: dict[str, dict[str, tuple[int, ...]]] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            return []
        required = {"Language_ID", "Segments", "Cognacy"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"arcaverborum file {path}: missing required columns {sorted(missing)}"
            )

        for row in reader:
            if dataset is not None and row.get("Dataset") != dataset:
                continue
            if family is not None and row.get("Family") != family:
                continue
            lect_id = row["Language_ID"]
            if not lect_id:
                continue
            if language_ids is not None and lect_id not in language_ids:
                continue
            raw_cog = row["Cognacy"]
            if not raw_cog or raw_cog in ("<NA>", "NA"):
                continue
            # First cognate ID is the grouping key.
            cog_id = raw_cog.split(";")[0].strip()
            if not cog_id:
                continue
            segs, morph_bounds = _parse_arcaverborum_segments(row["Segments"])
            if not segs:
                continue
            if cog_id not in forms_by_id:
                order.append(cog_id)
                forms_by_id[cog_id] = {}
                aligns_by_id[cog_id] = {}
                boundaries_by_id[cog_id] = {}
            if lect_id in forms_by_id[cog_id]:
                continue  # keep first reflex per (lect, cognate)
            forms_by_id[cog_id][lect_id] = Form(lect_id=LectId(lect_id), segments=segs)
            if morph_bounds:
                boundaries_by_id[cog_id][lect_id] = morph_bounds
            raw_align = row.get("Alignment") or ""
            if raw_align and raw_align not in ("<NA>", "NA"):
                aligns_by_id[cog_id][lect_id] = _parse_alignment(raw_align)

    out: list[CognateSet] = []
    for cog_id in order:
        forms = forms_by_id[cog_id]
        if len(forms) < min_lects:
            continue
        aligns = aligns_by_id[cog_id] or None
        if aligns:
            lens = {len(v) for v in aligns.values()}
            if len(lens) > 1:
                aligns = None
        bounds = boundaries_by_id[cog_id] or None
        out.append(
            CognateSet(
                cognate_id=CognateId(cog_id),
                forms=forms,
                alignments=aligns,
                morpheme_boundaries=bounds,
            )
        )
    return out
