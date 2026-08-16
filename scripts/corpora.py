#!/usr/bin/env python3
"""Generates web/corpora.js: every corpus the page can load, keyed by path.

The example list and the guide both need corpus text, and embedding it twice
would ship several corpora in two places that could drift. This is the single
source; both reference entries by path.

Corpora are grouped for the example list. "Start here" is a ladder: each entry
shows something the previous one cannot. "Not readable in this format yet"
carries the grapheme that blocks it, so the list says why rather than just
failing when picked.

Usage:
    scripts/corpora.py            # rewrite web/corpora.js
    scripts/corpora.py --check    # exit non-zero if it would change
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = REPO / "web" / "corpora.js"

# The ladder, in order. Each earns its place by showing a capability the
# previous entries cannot.
LADDER = [
    ("experiments/finnish_estonian/cognates.tsv",
     "Finnish and Estonian",
     "Two closely related lects. A first look at what a correspondence class is."),
    ("experiments/latin_spanish/cognates.tsv",
     "Latin and Spanish",
     "97 cognates, 22 conditioned classes. The environments are discovered, not declared."),
    ("experiments/contaminated_cognates_synthetic/cognates.tsv",
     "Contaminated cognates",
     "Confidence weighting and outlier ranking on deliberately noisy data."),
    ("testdata/corpora/real_romance_4lect.tsv",
     "Romance, four lects",
     "Latin, Spanish, French and Italian reconciled into classes binding all four."),
    # The last three are what the tool looks like when it is right to find
    # nothing, and they belong on the ladder for the same reason the contrast
    # environments belong in a fixture. A reader who has only seen it succeed
    # cannot tell success from output.
    ("testdata/restraint/chance.tsv",
     "Two unrelated lects",
     "76 classes and 47 environments, from wordlists with no history between them. "
     "Run the shuffled baseline and the corpus is indistinguishable from its own noise."),
    ("testdata/soundlaws/final_devoicing.tsv",
     "German final devoicing",
     "A neutralisation: /t/ and /d/ merge word-finally, so the four environments "
     "reported for it are correlates. All four fall below the baseline."),
    ("testdata/diagnostics/drift.tsv",
     "One language, two transcriptions",
     "The same forty words segmented two ways. Reads as deaffrication, loss of "
     "aspiration and loss of length -- none of which happened, and nothing catches it."),
]

DESCRIPTIONS = {
    "arabic_hebrew": "Semitic, with conditioned classes.",
    "georgian_svan": "Kartvelian. The richest conditioning in the set.",
    "latin_french": "Romance, a second daughter for comparison.",
    "latin_italian": "Romance, a third daughter.",
    "oe_english": "Old English against the modern language.",
    "swahili_zulu": "Bantu.",
    "turkish_azerbaijani": "Turkic, closely related.",
    "ppn_hawaiian": "Proto-Polynesian and Hawaiian.",
    "mandarin_historical": "Middle Chinese and Mandarin, tone held in separate columns.",
    "harmony_synthetic": "Vowel harmony, synthetic.",
    "umlaut_synthetic": "Umlaut, synthetic.",
    "length_conditioned_synthetic": "Length conditioning, synthetic.",
}


def corpus_format(path):
    if path.suffix == ".csv":
        return "arcaverborum"
    if "experiments/" in path.as_posix():
        return "wide"
    return "tsv"


def probe(cli, path):
    """Runs a corpus to find out whether the page can load it, and if not,
    which grapheme stops it."""
    result = subprocess.run(
        [str(cli), "train", "--format", corpus_format(path), str(REPO / path)],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode == 0:
        first = result.stdout.splitlines()[0] if result.stdout else ""
        return {"readable": True, "lects": first.split("\t")[1].split(" ") if "\t" in first else []}
    # Both refusals the CLI can report name the offending token: a sound the
    # feature system does not cover, and CLDF/CLTS markup that never
    # transcribed one.
    match = (re.search(r'unknown grapheme "([^"]*)"', result.stderr) or
             re.search(r'"([^"]*)" is CLDF/CLTS markup', result.stderr))
    return {"readable": False, "blockedBy": match.group(1) if match else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cli", default=str(REPO / "build" / "c" / "regulae"))
    args = parser.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"corpora: CLI not found at {cli}; build it first", file=sys.stderr)
        return 2

    paths = [pathlib.Path(entry[0]) for entry in LADDER]
    for path in sorted((REPO / "experiments").glob("*/cognates.tsv")):
        relative = path.relative_to(REPO)
        if relative not in paths:
            paths.append(relative)

    ladder_paths = {entry[0]: (entry[1], entry[2]) for entry in LADDER}
    corpora = {}
    entries = []
    for path in paths:
        key = path.as_posix()
        probed = probe(cli, path)
        corpora[key] = (REPO / path).read_text(encoding="utf-8")

        if key in ladder_paths:
            label, description = ladder_paths[key]
            group = "Start here"
        else:
            name = path.parent.name
            label = name.replace("_", " ")
            description = DESCRIPTIONS.get(name, "")
            group = "More corpora" if probed["readable"] else "Not readable in this format yet"

        entry = {
            "path": key,
            "label": label,
            "description": description,
            "group": group,
            "format": corpus_format(path),
            "readable": probed["readable"],
        }
        if not probed["readable"] and probed["blockedBy"]:
            entry["blockedBy"] = probed["blockedBy"]
        entries.append(entry)

    # Ladder first, then readable, then blocked; stable within each group.
    order = {"Start here": 0, "More corpora": 1, "Not readable in this format yet": 2}
    entries.sort(key=lambda e: order[e["group"]])

    text = (
        "// Generated by scripts/corpora.js. Do not edit by hand.\n"
        "//\n"
        "// Every corpus the page can load, keyed by repository path, plus the\n"
        "// grouped list the example picker shows. The guide references these by\n"
        "// path rather than embedding its own copies.\n"
        "const CORPORA = " + json.dumps(corpora, indent=2, ensure_ascii=False) + ";\n\n"
        "const CORPUS_LIST = " + json.dumps(entries, indent=2, ensure_ascii=False) + ";\n"
    ).replace("scripts/corpora.js", "scripts/corpora.py")

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("corpora: web/corpora.js is out of date; run scripts/corpora.py",
                  file=sys.stderr)
            return 1
        print("corpora: up to date")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    readable = sum(1 for e in entries if e["readable"])
    print(f"corpora: wrote {OUTPUT.relative_to(REPO)} "
          f"({len(entries)} corpora, {readable} readable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
