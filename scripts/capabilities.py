#!/usr/bin/env python3
"""Generates docs/capabilities.md: what regulae can do, and which corpora it
can currently read.

The table is keyed on capability rather than on corpus. Keying it on corpus
would once have reported that regulae cannot do tone, which was never true: the
tonal corpora failed because they wrote tone as an ASCII digit, not because the
engine lacked tone. A corpus that fails to load says something about the input
path, not about the engine, and the tables are arranged to say which.

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
        "`testdata/corpora/*` (13 corpora)",
    ),
    (
        "conditioned environments",
        "supported",
        "every loader",
        "`testdata/corpora/conditioned_multilect.tsv`",
    ),
    (
        "conditioned-split scorers",
        "corrected BIC default; exact NML and Dirichlet marginal experimental",
        "C options, JSON options, or `--scorer`",
        "`tests/c/test_split_score.c`, `docs/m3_evaluation.md`, ADR 0001",
    ),
    (
        "long-range conditioning",
        "supported",
        "every loader",
        "`testdata/corpora/long_range.tsv`",
    ),
    (
        "multi-lect reconciliation",
        "supported",
        "every loader",
        "`testdata/corpora/real_romance_4lect.tsv` (4 lects)",
    ),
    (
        "confidence weighting and outliers",
        "supported",
        "TSV and wide (`confidence` column)",
        "`testdata/corpora/real_contaminated.tsv`",
    ),
    (
        "morpheme boundaries",
        "supported",
        "arcaverborum, and wide via `<lect>_breaks`",
        "`testdata/corpora/morph_boundary.csv`",
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
        "every loader, from Chao superscripts in the transcription or a tone column",
        "`tests/c/test_pairwise_model.c`, `tests/c/test_merkmal_bridge.c`",
    ),
    (
        "stress conditioning",
        "supported",
        "every loader, from IPA marks or a `_stress` column",
        "`testdata/soundlaws/verner.tsv`, `tests/c/test_sound_laws.c`",
    ),
    (
        "manner conditioning",
        "supported",
        "every loader",
        "`testdata/soundlaws/graded_5_distance_two.tsv`",
    ),
    (
        "observation-group metadata",
        "supported",
        "TSV and wide (`etymon_group`, `source_group`)",
        "`tests/c/test_evaluation_m2.c`",
    ),
    (
        "group-held-out predictive evidence",
        "supported; opt-in because it retrains per fold and orientation",
        "`predictive_folds`, JSON options, or `--predictive-folds`",
        "`tests/c/test_evaluation_m4.c`, `docs/m4_evaluation.md`",
    ),
    (
        "bootstrap uncertainty",
        "supported",
        "`bootstrap_n`; cognate, etymon, or source groups",
        "`tests/c/test_uncertainty.c`: resamples named groups",
    ),
    (
        "chunk transparency screening",
        "supported",
        "`chunk_min_transparency`; every chunk row carries its score",
        "`testdata/soundlaws/metathesis_adjacent.tsv`, `tests/c/test_sound_laws.c`",
    ),
    (
        "anomaly detection",
        "**not ported**",
        "n/a",
        "never ported",
    ),
    (
        "opaque conditioning (the daughter lost the trigger)",
        "supported",
        "every loader",
        "`testdata/soundlaws/graded_9_lost_trigger.tsv`, `opaque_umlaut.tsv`",
    ),
    (
        "chain shifts kept apart from mergers",
        "supported",
        "every loader",
        "`testdata/soundlaws/great_vowel_shift.tsv`",
    ),
    (
        "compensatory lengthening",
        "supported",
        "every loader",
        "`testdata/soundlaws/compensatory_lengthening.tsv`",
    ),
    (
        "a verdict on whether supplied pairings carry structure beyond a shuffle",
        "supported",
        "`permutation_count`, or `--permutations`",
        "`testdata/restraint/chance.tsv`",
    ),
    (
        "restraint on a change with no environment",
        "supported",
        "every loader",
        "`testdata/restraint/diffusion.tsv`, `stratum.tsv`",
    ),
    (
        "ranking cognate sets worst-first, for triage",
        "supported",
        "every loader",
        "`testdata/diagnostics/contaminated.tsv`, `partial.tsv`",
    ),
    (
        "disjunctive trigger sets (RUKI's shape)",
        "supported, as a decision list",
        "every loader",
        "`testdata/soundlaws/graded_7_disjunction.tsv`: one rule per trigger, all with one outcome",
    ),
    (
        "syllable shape and weight",
        "supported",
        "every loader",
        "`testdata/soundlaws/graded_8_weight.tsv`: `prev-syl[syllable_weight:heavy]`, Sievers' shape in one rule",
    ),
    (
        "several environments for one correspondence",
        "supported",
        "every loader",
        "`tests/c/test_sound_laws.c`: rows merge on their observations, not on the tuple",
    ),
    (
        "partial cognacy, per morpheme",
        "**not expressible**",
        "n/a",
        "`testdata/diagnostics/partial.tsv`: visible in the ranking, not usable in training",
    ),
    (
        "transcription drift between sources",
        "reported, not refused",
        "`regulae check`, or `rg_find_transcription_drift`",
        "`testdata/diagnostics/drift.tsv`: 5 of 5 found, and no corpus in this repository reports a false one",
    ),
    (
        "telling borrowing from inheritance",
        "**out of scope**",
        "n/a",
        "`testdata/restraint/contact.tsv`: the correspondences are real either way; the ranking shows the split",
    ),
]

# Maps a blocking grapheme to why it blocks and which capability row it points
# at. Anything unrecognised is reported as-is rather than guessed at.
BLOCKERS = [
    (re.compile(r"^[0-9]$"), "ASCII digit, not Chao tone notation", "tone"),
    (re.compile(r"^\+$"), "in-word morpheme boundary", "morpheme boundaries"),
    (re.compile(r"^[_#]$"), "CLDF boundary marker", "morpheme boundaries"),
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
    # The CLI names the offending grapheme in both refusals it can report: a
    # sound the feature system does not cover, and CLDF/CLTS markup that never
    # transcribed one.
    match = re.search(r'unknown grapheme "([^"]*)"', result.stderr)
    markup = re.search(r'"([^"]*)" is CLDF/CLTS markup', result.stderr)
    if not match and not markup:
        return {"status": "fails", "detail": result.stderr.strip().splitlines()[0]}
    grapheme, = (match or markup).groups()
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
    out.append("something about the input path, not about the engine. Tone made the case: the")
    out.append("tonal corpora were unreadable for as long as they wrote tone as an ASCII digit,")
    out.append("which no transcription standard defines, and every one of them reads now that")
    out.append("they carry Chao superscripts on the word or a `<lect>_tone` column. Nothing in")
    out.append("the engine changed to allow it.")
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
