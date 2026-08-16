#!/usr/bin/env python3
"""Generate corpora for the linguistic research questions, not engine tests."""

import argparse
import csv
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = REPO / "testdata" / "linguistic"


def render(rows, source_group="synthetic"):
    lines = ["cognate_id\tlect_id\tsegments\tetymon_group\tsource_group"]
    for row in rows:
        if len(row) == 3:
            row = (*row, row[0], source_group)
        lines.append("\t".join(row))
    return "\n".join(lines) + "\n"


def conditioning_rows(lects, vowels):
    rows = []
    for index in range(32):
        vowel, fires = vowels[index % len(vowels)]
        for lect in lects:
            outcome = "f" if lect.startswith("innovator") and fires else "p"
            rows.append((f"w{index:02d}", lect, f"k a {outcome} {vowel}"))
    return rows


def repeated_etymon_rows():
    rows = []
    endings = ("a", "e", "i", "o", "u", "a n", "a s", "a m")
    for fires, etymon, trigger in ((True, "fire", "i"), (False, "control", "a")):
        for index, ending in enumerate(endings):
            cognate_id = f"{etymon}_{index}"
            rows.append((cognate_id, "ancestor", f"k a p {trigger} {ending}", etymon,
                         "synthetic-repeated-etymon"))
            outcome = "f" if fires else "p"
            rows.append((cognate_id, "innovator", f"k a {outcome} {trigger} {ending}", etymon,
                         "synthetic-repeated-etymon"))
    return rows


def confounded_rows():
    rows = []
    onsets = ("k", "t", "m", "s")
    for index in range(32):
        vowel, fires = (("i", True), ("a", False))[index % 2]
        onset = onsets[(index // 2) % len(onsets)]
        for lect in ("ancestor", "innovator"):
            outcome = "f" if lect == "innovator" and fires else "p"
            rows.append((f"w{index:02d}", lect, f"{onset} a {outcome} {vowel}"))
    return rows


def write_synthetic():
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    corpora = {
        "taxon_sampling_2lect.tsv": conditioning_rows(
            ("ancestor", "innovator"), (("i", True), ("a", False))
        ),
        "taxon_sampling_3lect.tsv": conditioning_rows(
            ("ancestor", "conservative", "innovator"),
            (("i", True), ("a", False)),
        ),
        "taxon_sampling_4lect.tsv": conditioning_rows(
            ("ancestor", "conservative", "innovator1", "innovator2"),
            (("i", True), ("a", False)),
        ),
        "repeated_etymon.tsv": repeated_etymon_rows(),
        "confounded_conditioning.tsv": confounded_rows(),
        "deconfounded_conditioning.tsv": conditioning_rows(
            ("ancestor", "innovator"),
            (
                ("i", True),
                ("e", True),
                ("y", True),
                ("ø", True),
                ("u", False),
                ("o", False),
                ("ɯ", False),
                ("ɤ", False),
            ),
        ),
    }
    for name, rows in corpora.items():
        (SYNTHETIC_DIR / name).write_text(render(rows), encoding="utf-8")


def kessler_rows(dataset, first, second):
    path = dataset.expanduser() / "cldf" / "forms.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        forms = list(csv.DictReader(handle))
    by_concept = {
        (row["Parameter_ID"], row["Language_ID"]): row["Segments"]
        for row in forms
    }
    concepts = sorted(
        concept
        for concept, lect in by_concept
        if lect == first and (concept, second) in by_concept
    )
    rows = []
    for concept in concepts:
        rows.append((concept, first, by_concept[(concept, first)]))
        rows.append((concept, second, by_concept[(concept, second)]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kessler", type=pathlib.Path)
    parser.add_argument("--pair", default="Hawaiian,Navajo")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    if args.kessler:
        if not args.out:
            parser.error("--out is required with --kessler")
        first, second = args.pair.split(",", 1)
        args.out.write_text(
            render(kessler_rows(args.kessler, first, second),
                   source_group="kesslersignificance"), encoding="utf-8"
        )
        return 0

    write_synthetic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
