#!/usr/bin/env python3
"""Measures report quality across every corpus in the tree.

`model_hashes.py` answers "did the model move". This answers "is what it
published readable", which a passing test suite does not check and which four
numbers make comparable across a change:

  identity      published conditioned rules whose correspondence is X ~ X.
                Either the retention side of a real split, with the change
                hidden in contrast_class_id, or noise. Neither belongs in a
                headline.
  fragmentation observation mass in classes binding every lect, over total.
                The rest sits in partial classes that mostly restate a
                correspondence already published at higher arity.
  density       cross-dimensional rows whose environment is segmental, over
                all of them. Low means the table is mostly tone conditioning
                tone, which is a tonal correspondence rather than the
                segmental-conditions-suprasegmental rule the stage exists for.
  conjuncts     mean predicates per committed environment. Rises when the
                greedy search welds on correlates it never checks again.

Two of these read close to perfect on the tree as it stands, and that is a
statement about the tree rather than about the engine. Fragmentation is 1.00
wherever every cognate set covers every lect, so only `partial_coverage`
exercises it; density has a denominator only on the five corpora carrying
suprasegmentals. Both collapse on real wordlists -- a five-lect Polynesian
sample from Lexibank puts 24% of its mass in full-arity rows, and three Sinitic
dialects publish 156 cross-dimensional rows over 21 decisions. A number that
cannot move is not yet a gate, and the fixtures that make these two move are
worth more than tightening the two that already do.

None of the four counts a decision. A committed split publishes its rule and
then one row per outcome of its complement, so a table of 156 rows can be 21
findings, and a reader who counts rows is counting the wrong thing. Whether
that is the right shape to publish is a question for the report rather than
for the search, and it is not measured here.

Usage:
    scripts/quality.py             # print the table
    scripts/quality.py --check     # exit non-zero if any number regressed
    scripts/quality.py --write     # record a new baseline, deliberately
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPO / "testdata" / "quality_baseline.txt"

GROUPS = [
    ("soundlaws", "testdata/soundlaws/*.tsv"),
    ("restraint", "testdata/restraint/*.tsv"),
    ("diagnostics", "testdata/diagnostics/*.tsv"),
    ("linguistic", "testdata/linguistic/*.tsv"),
    ("corpora", "testdata/corpora/*.tsv"),
]

# Corpora that are not a corpus: the loader's error cases.
SKIP = {"missing_lect_column.tsv"}


def train(cli, path):
    result = subprocess.run(
        [str(cli), "train", "--json", str(path)],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def conjunct_count(segment):
    total = 0
    for value in segment.get("context", {}).values():
        total += len(value) if isinstance(value, list) else 1
    return total


def measure(model):
    """The four numbers, as (numerator, denominator) pairs so a run over many
    corpora can pool them instead of averaging averages."""
    conditioned = model["classes"]["conditioned"]
    unconditioned = model["classes"]["unconditioned"]
    cross = model.get("cross_dimensional", [])

    identity = sum(
        1 for row in conditioned
        if len({s["grapheme"] for s in row["segments"]}) == 1
    )

    # Against the corpus's lect count, not against the widest class observed.
    # Taking the widest observed makes every two-lect corpus read 1.00 by
    # construction and hides the thing being measured: a corpus where nothing
    # binds all the lects would score perfectly.
    lects = int(round(model["fit"]["lect_count"]))
    full_mass = sum(c["count"] for c in unconditioned if len(c["segments"]) == lects)
    total_mass = sum(c["count"] for c in unconditioned)

    segmental = sum(
        1 for row in cross
        if "tone" not in json.dumps(row.get("environment", {}), sort_keys=True)
    )

    conjuncts = [sum(conjunct_count(s) for s in row["segments"]) for row in conditioned]

    return {
        "identity": (identity, len(conditioned)),
        "fragmentation": (full_mass, total_mass),
        "density": (segmental, len(cross)),
        "conjuncts": (sum(conjuncts), len(conjuncts)),
    }


def ratio(pair):
    return pair[0] / pair[1] if pair[1] else 0.0


def cell(pair, width=7):
    """A ratio over an empty denominator is not zero, and printing it as zero
    is how a metric that measures nothing reads as a perfect score."""
    return f"{'':>{width - 3}}n/a" if not pair[1] else f"{ratio(pair):>{width}.2f}"


def render(rows, totals):
    out = [f"{'corpus':<50}{'cond':>6}{'ident':>7}{'frag':>7}{'xdim':>6}{'dens':>7}{'conj':>7}"]
    for name, counts in rows:
        out.append(
            f"{name:<50}{counts['identity'][1]:>6}"
            f"{cell(counts['identity'])}"
            f"{cell(counts['fragmentation'])}"
            f"{counts['density'][1]:>6}"
            f"{cell(counts['density'])}"
            f"{cell(counts['conjuncts'])}"
        )
    out.append("")
    for key in ("identity", "fragmentation", "density", "conjuncts"):
        num, den = totals[key]
        value = "n/a" if not den else f"{ratio(totals[key]):.4f}"
        out.append(f"TOTAL {key:<16}{num:>8.1f} / {den:<8.1f} = {value}")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--cli", default=str(REPO / "build" / "c" / "regulae"))
    args = parser.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"quality: CLI not found at {cli}; build it first", file=sys.stderr)
        return 2

    rows = []
    totals = {k: [0.0, 0.0] for k in ("identity", "fragmentation", "density", "conjuncts")}
    for _, pattern in GROUPS:
        for path in sorted(REPO.glob(pattern)):
            if path.name in SKIP:
                continue
            model = train(cli, path)
            if model is None:
                print(f"quality: {path.relative_to(REPO)} did not train", file=sys.stderr)
                return 2
            counts = measure(model)
            rows.append((path.relative_to(REPO).as_posix(), counts))
            for key, pair in counts.items():
                totals[key][0] += pair[0]
                totals[key][1] += pair[1]

    text = render(rows, {k: tuple(v) for k, v in totals.items()})

    if args.write:
        BASELINE.write_text(text, encoding="utf-8")
        print(f"quality: wrote {BASELINE.relative_to(REPO)}")
        return 0

    if args.check:
        if not BASELINE.exists():
            print("quality: no baseline; run scripts/quality.py --write", file=sys.stderr)
            return 1
        if BASELINE.read_text(encoding="utf-8") != text:
            print("quality: report quality moved; review the diff and rerun with --write "
                  "in the same commit", file=sys.stderr)
            subprocess.run(["diff", "-u", str(BASELINE), "-"], input=text, text=True)
            return 1
        print("quality: unchanged")
        return 0

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
