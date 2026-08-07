#!/usr/bin/env python3
"""Generates docs/capabilities.md: what regulae can do, and which corpora it
can currently read.

The table is keyed on capability rather than on corpus. Keying it on corpus
would report that regulae cannot do tone, which is false: tone is supported and
tested, and the tone corpora fail because no loader carries a tone column. A
corpus that fails to load says something about the input path, not about the
engine, and the tables are arranged to say which.

Capability rows are declared here, because "is this supported, and where is the
evidence" is a judgement that cannot be measured. Corpus rows are measured by
running the CLI, including the grapheme that blocks a failing one.

Usage:
    scripts/capabilities.py            # rewrite docs/capabilities.md
    scripts/capabilities.py --check    # exit non-zero if it would change
"""

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = REPO / "docs" / "capabilities.md"

# Declared, not measured. "engine" is whether the training pipeline implements
# it; "input path" is which loaders can express it; "evidence" is what proves
# the first column.
CAPABILITIES = [
    (
        "segment correspondences",
        "supported",
        "every loader",
        "`testdata/parity/*` (13 corpora, byte-identical to the Go reference)",
    ),
    (
        "conditioned environments",
        "supported",
        "every loader",
        "`testdata/parity/conditioned_multilect.tsv`",
    ),
    (
        "long-range conditioning",
        "supported",
        "every loader",
        "`testdata/parity/long_range.tsv`",
    ),
    (
        "multi-lect reconciliation",
        "supported",
        "every loader",
        "`testdata/parity/real_romance_4lect.tsv` (4 lects)",
    ),
    (
        "confidence weighting and outliers",
        "supported",
        "TSV and wide (`confidence` column)",
        "`testdata/parity/real_contaminated.tsv`",
    ),
    (
        "morpheme boundaries",
        "supported",
        "arcaverborum, and wide via `<lect>_breaks`",
        "`testdata/parity/morph_boundary.csv`",
    ),
    (
        "cross-dimensional rules",
        "supported",
        "every loader",
        "`tests/c/test_pairwise_model.c`",
    ),
    (
        "tone",
        "supported",
        "**none** — no loader carries tone",
        "`tests/c/test_pairwise_model.c`",
    ),
    (
        "stress conditioning",
        "supported",
        "**none** — no loader carries stress",
        "`src/model.c` stress split candidates",
    ),
    (
        "bootstrap uncertainty",
        "**not ported**",
        "n/a",
        "`bootstrap.go` (`bootstrap_n` is accepted and ignored)",
    ),
    (
        "chunk transparency screening",
        "**not ported**",
        "n/a",
        "`chunk_diagnostics.go` (`chunk_min_transparency > 0` is refused)",
    ),
    (
        "anomaly detection",
        "**not ported**",
        "n/a",
        "`anomaly.go`",
    ),
]

# Maps a blocking grapheme to why it blocks and which capability row it points
# at. Anything unrecognised is reported as-is rather than guessed at.
BLOCKERS = [
    (re.compile(r"^[0-9]$"), "tone digit", "tone"),
    (re.compile(r"^\+$"), "in-word morpheme boundary", "morpheme boundaries"),
    (re.compile(r"^-$"), "syllable separator", "stress conditioning"),
]


def classify(grapheme):
    for pattern, cause, capability in BLOCKERS:
        if pattern.match(grapheme):
            return cause, capability
    return "not covered by the feature system", "upstream (merkmal)"


def run(cli, path):
    """Trains one corpus, returning a measured row."""
    result = subprocess.run(
        [str(cli), "train", "--format", "wide", str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 0:
        lines = result.stdout.splitlines()
        lects = lines[0].split("\t")[1] if lines else ""
        return {
            "status": "reads",
            "lects": lects.replace(" ", ", "),
            "classes": sum(1 for line in lines if line.startswith("UNCOND")),
            "conditioned": sum(1 for line in lines if line.startswith("COND")),
        }
    match = re.search(r'unknown grapheme "([^"]*)"', result.stderr)
    if not match:
        return {"status": "fails", "detail": result.stderr.strip().splitlines()[0]}
    grapheme, = match.groups()
    cause, capability = classify(grapheme)
    return {
        "status": "cannot read",
        "grapheme": grapheme,
        "cause": cause,
        "capability": capability,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the file is out of date")
    parser.add_argument("--cli", default=str(REPO / "build" / "c" / "regulae"))
    args = parser.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"capabilities: CLI not found at {cli}; build it first", file=sys.stderr)
        return 2

    corpora = sorted((REPO / "experiments").glob("*/cognates.tsv"))
    measured = [(path.parent.name, run(cli, path)) for path in corpora]

    reads = [row for row in measured if row[1]["status"] == "reads"]
    blocked = [row for row in measured if row[1]["status"] != "reads"]

    out = []
    out.append("# Capabilities and corpus coverage")
    out.append("")
    out.append("<!-- Generated by scripts/capabilities.py. Do not edit by hand. -->")
    out.append("")
    out.append("What regulae implements, and which of the corpora in `experiments/` it can")
    out.append("currently read. These are different questions: a corpus that fails to load says")
    out.append("something about the input path, not about the engine. Tone is the clearest case")
    out.append("— it is supported and tested, and every tone corpus below is unreadable, because")
    out.append("no loader carries a tone column.")
    out.append("")
    out.append("## Capabilities")
    out.append("")
    out.append("| capability | engine | input path | evidence |")
    out.append("| --- | --- | --- | --- |")
    for name, engine, path, evidence in CAPABILITIES:
        out.append(f"| {name} | {engine} | {path} | {evidence} |")
    out.append("")
    out.append(f"## Corpora regulae reads ({len(reads)} of {len(measured)})")
    out.append("")
    out.append("Trained through the wide loader with default options.")
    out.append("")
    out.append("| corpus | lects | classes | conditioned |")
    out.append("| --- | --- | --- | --- |")
    for name, row in reads:
        out.append(f"| {name} | {row['lects']} | {row['classes']} | {row['conditioned']} |")
    out.append("")
    out.append(f"## Corpora regulae cannot read yet ({len(blocked)} of {len(measured)})")
    out.append("")
    out.append("Each names the grapheme that blocks it and the capability row it points at.")
    out.append("None of these is an engine limitation.")
    out.append("")
    out.append("| corpus | blocked by | cause | see capability |")
    out.append("| --- | --- | --- | --- |")
    for name, row in blocked:
        if "grapheme" in row:
            out.append(f"| {name} | `{row['grapheme']}` | {row['cause']} | {row['capability']} |")
        else:
            out.append(f"| {name} | — | {row['detail']} | — |")
    out.append("")

    text = "\n".join(out)
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("capabilities: docs/capabilities.md is out of date; "
                  "run scripts/capabilities.py", file=sys.stderr)
            return 1
        print("capabilities: up to date")
        return 0

    OUTPUT.write_text(text, encoding="utf-8")
    print(f"capabilities: wrote {OUTPUT.relative_to(REPO)} "
          f"({len(reads)} readable, {len(blocked)} blocked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
