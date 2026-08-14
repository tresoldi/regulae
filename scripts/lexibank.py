#!/usr/bin/env python3
"""Turns a CLDF Wordlist with expert cognate judgements into a regulae corpus.

Lexibank datasets are the field's shared reference data: expert-coded cognate
sets over CLTS-segmented forms, across a hundred-odd families. They are what
regulae's discovery stages have to be right about, and a synthetic fixture
cannot stand in for them -- a corpus of 5 CV syllables exercises no merger, no
gap, no unequal syllable count, and no rare-but-real correspondence.

Nothing is vendored. The datasets carry their own licences and citations, so
this reads a local checkout and writes corpora the caller can throw away.

Usage:
    scripts/lexibank.py <dataset> --clone ~/lexibank_clone -o out.tsv
    scripts/lexibank.py <dataset> --lects a,b        # restrict to two lects
    scripts/lexibank.py --list --clone ~/lexibank_clone
"""

import argparse
import collections
import csv
import pathlib
import sys

# Cognate judgements a dataset marks as doubtful are excluded: regulae never
# decides cognacy itself, so a doubtful set would enter training at full weight
# and its segment pairings would be learned as though they were secure.
DOUBTFUL = {"true", "1", "yes"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dataset_dir(clone, name):
    return pathlib.Path(clone).expanduser() / name / "cldf"


def usable(clone):
    """Datasets carrying both cognate judgements and segmented forms."""
    root = pathlib.Path(clone).expanduser()
    out = []
    for entry in sorted(root.iterdir()):
        cldf = entry / "cldf"
        if (cldf / "cognates.csv").is_file() and (cldf / "forms.csv").is_file():
            out.append(entry.name)
    return out


def convert(clone, name, lects=None, min_lects=2, max_lects=None, drop_doubt=True):
    """Returns (rows, stats). Rows are (cognate_id, lect_id, segments)."""
    cldf = dataset_dir(clone, name)
    forms = {row["ID"]: row for row in read_csv(cldf / "forms.csv")}
    cognates = read_csv(cldf / "cognates.csv")

    stats = collections.Counter()
    stats["forms"] = len(forms)
    stats["judgements"] = len(cognates)

    # A cognate set is scoped to its concept. Two forms glossed differently are
    # not evidence of one correspondence, whatever their identifiers do.
    sets = collections.defaultdict(list)
    for judgement in cognates:
        if drop_doubt and judgement.get("Doubt", "").strip().lower() in DOUBTFUL:
            stats["doubtful"] += 1
            continue
        form = forms.get(judgement["Form_ID"])
        if form is None:
            stats["dangling"] += 1
            continue
        segments = (form.get("Segments") or "").strip()
        if not segments:
            stats["unsegmented"] += 1
            continue
        key = (form["Parameter_ID"], judgement["Cognateset_ID"])
        sets[key].append((form["Language_ID"], segments))

    counts = collections.Counter(
        lect for members in sets.values() for lect, _ in members
    )
    chosen = set(lects) if lects else set(counts)
    if not lects and max_lects:
        chosen = {lect for lect, _ in counts.most_common(max_lects)}
    stats["lects_available"] = len(counts)
    stats["lects_used"] = len(chosen)

    rows = []
    for (concept, cogid), members in sorted(sets.items()):
        # One form per lect per set. A lect with two forms in one cognate set is
        # a doublet, and picking one arbitrarily would invent a correspondence;
        # the first in file order is at least deterministic, and the count is
        # reported so a caller can see how often it happened.
        seen = {}
        for lect, segments in members:
            if lect not in chosen:
                continue
            if lect in seen:
                stats["doublets"] += 1
                continue
            seen[lect] = segments
        if len(seen) < min_lects:
            stats["below_min_lects"] += 1
            continue
        cognate_id = f"{concept}-{cogid}"
        for lect in sorted(seen):
            rows.append((cognate_id, lect, seen[lect]))
        stats["sets"] += 1

    stats["rows"] = len(rows)
    return rows, stats


def clean_segments(segments, drop_markers=True):
    """CLDF markup is not a transcription of a sound.

    `+` and `_` are boundary marks, `<?>` is CLTS's mark for a grapheme it could
    not convert, and `<<...>>` is source material left unparsed. regulae refuses
    all of them at the feature system, correctly, so a corpus built for training
    has to say what it means to do with them. Dropping the boundary marks keeps
    the form; a form containing an unconvertible grapheme is dropped whole,
    because the gap is in the middle of the word and the rest is not a word.
    """
    tokens = segments.split()
    if any(t.startswith("<") for t in tokens):
        return None
    if drop_markers:
        tokens = [t for t in tokens if t not in ("+", "_", "#")]
    return " ".join(tokens) if tokens else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--clone", default="~/lexibank_clone")
    parser.add_argument("--out", "-o")
    parser.add_argument("--lects", help="comma-separated lect ids")
    parser.add_argument("--max-lects", type=int)
    parser.add_argument("--min-lects", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in usable(args.clone):
            print(name)
        return 0
    if not args.dataset:
        parser.error("a dataset name is required unless --list is given")

    lects = args.lects.split(",") if args.lects else None
    rows, stats = convert(args.clone, args.dataset, lects=lects,
                          min_lects=args.min_lects, max_lects=args.max_lects)

    kept = []
    for cognate_id, lect, segments in rows:
        cleaned = clean_segments(segments)
        if cleaned is None:
            stats["markup"] += 1
            continue
        kept.append((cognate_id, lect, cleaned))

    handle = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        handle.write("cognate_id\tlect_id\tsegments\n")
        for row in kept:
            handle.write("\t".join(row) + "\n")
    finally:
        if args.out:
            handle.close()

    for key in sorted(stats):
        print(f"{key}\t{stats[key]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
