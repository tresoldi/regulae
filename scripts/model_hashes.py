#!/usr/bin/env python3
"""Records the model every corpus trains to, as one hash per corpus.

This exists so that "this was a refactor" is a checkable claim rather than an
assurance. `ctest` passing is compatible with a great many changed models: the
suite asserts that particular rules are found, not that the whole published
model is the one it was before. `wasm_smoke` compares the native build against
the WebAssembly build, which is a determinism check across two builds of one
version of the code, not a check across two versions.

So: train every corpus in the tree through `train --json` and hash the result.
A change to the engine that is meant to be invisible must leave every hash
alone. A change that is meant to be visible updates the baseline in the same
commit, which is what makes it reviewable -- the diff says which corpora moved.

The JSON is used rather than the human or summary output because it is the
complete published model, and because it carries `regulae_version` rather than
`RG_ABI_VERSION`: an ABI bump is not a model change and must not read as one.
`--permutations` is deliberately not passed. The shuffled baseline costs one
training run per permutation, and what it adds -- `standing` on each rule -- is
a function of the model already hashed here.

    scripts/model_hashes.py            check against the committed baseline
    scripts/model_hashes.py --write    record a new baseline
    scripts/model_hashes.py --jobs N   parallelism (default: CPU count)

Exit status is 0 when every corpus matches, 1 when any differs, 2 when the
setup is wrong. Decide by that, never by grepping the output.
"""

import argparse
import concurrent.futures
import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPO / "testdata" / "model_hashes.txt"

# The per-corpus training runs are independent, but a corpus that hangs would
# otherwise hang the whole harness. Every real corpus in the tree trains in
# well under this.
TIMEOUT_SECONDS = 900


def corpus_format(path):
    """The same rule scripts/corpora.py applies, so the two agree about what a
    given file is. Kept in step by hand; they are four lines each."""
    if path.suffix == ".csv":
        return "arcaverborum"
    if "experiments/" in path.as_posix():
        return "wide"
    return "tsv"


def corpora():
    """Every corpus the tree can train, in a stable order.

    testdata/soundlaws is the set built so that exactly one answer is right,
    testdata/restraint the set where the right answer is nothing,
    testdata/diagnostics the set where the corpus rather than the method is
    what is wrong, testdata/corpora is what the tests and CLI smoke checks
    read, and experiments/ is the real data. A corpus that does not load is
    still recorded: that it refuses, and goes on refusing, is also behaviour.
    """
    paths = []
    paths += sorted((REPO / "testdata" / "soundlaws").glob("*.tsv"))
    paths += sorted((REPO / "testdata" / "restraint").glob("*.tsv"))
    paths += sorted((REPO / "testdata" / "diagnostics").glob("*.tsv"))
    paths += sorted((REPO / "testdata" / "corpora").glob("*.tsv"))
    paths += sorted((REPO / "testdata" / "corpora").glob("*.csv"))
    paths += sorted((REPO / "experiments").glob("*/cognates.tsv"))
    return [p.relative_to(REPO) for p in paths]


def train(cli, path):
    """Returns (relative path, exit status, hash of stdout).

    stderr is deliberately not hashed. A refusal prints the offending file's
    path, which differs between checkouts, and a hash that depends on where the
    repository sits would fail for everyone but its author. The exit status
    carries whether the corpus loaded; stdout carries the model.
    """
    result = subprocess.run(
        [str(cli), "train", "--json", "--format", corpus_format(path), str(REPO / path)],
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
    )
    digest = hashlib.sha256(result.stdout).hexdigest()
    return (path.as_posix(), result.returncode, digest)


def measure(cli, jobs):
    paths = corpora()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(train, cli, path) for path in paths]
        for future in futures:
            rows.append(future.result())
    rows.sort(key=lambda row: row[0])
    return rows


def render(rows):
    lines = [
        "# The model every corpus trains to, as `train --json` hashed.",
        "#",
        "# Regenerate with scripts/model_hashes.py --write, and only in a commit",
        "# that means to change what regulae learns. A refactor leaves this file",
        "# untouched; that it is untouched is the evidence that it was one.",
        "#",
        "# status  sha256-of-stdout  corpus",
    ]
    for path, status, digest in rows:
        lines.append(f"{status:>6}  {digest}  {path}")
    return "\n".join(lines) + "\n"


def parse(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        status, digest, path = line.split()
        rows.append((path, int(status), digest))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="record a new baseline instead of checking one")
    parser.add_argument("--cli", default=str(REPO / "build" / "c" / "regulae"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    cli = pathlib.Path(args.cli)
    if not cli.exists():
        print(f"model_hashes: CLI not found at {cli}; build it first", file=sys.stderr)
        return 2

    rows = measure(cli, args.jobs)
    if not rows:
        print("model_hashes: no corpora found", file=sys.stderr)
        return 2

    if args.write:
        BASELINE.write_text(render(rows))
        loaded = sum(1 for _, status, _ in rows if status == 0)
        print(f"model_hashes: wrote {len(rows)} corpora ({loaded} trained) to "
              f"{BASELINE.relative_to(REPO)}")
        return 0

    if not BASELINE.exists():
        print(f"model_hashes: no baseline at {BASELINE.relative_to(REPO)}; "
              f"run with --write", file=sys.stderr)
        return 2

    want = {path: (status, digest) for path, status, digest in parse(BASELINE.read_text())}
    have = {path: (status, digest) for path, status, digest in rows}

    changed = []
    for path in sorted(set(want) | set(have)):
        if path not in want:
            changed.append(f"  new     {path}")
        elif path not in have:
            changed.append(f"  removed {path}")
        elif want[path] != have[path]:
            was_status, was_digest = want[path]
            now_status, now_digest = have[path]
            if was_status != now_status:
                changed.append(f"  {path}: exit {was_status} -> {now_status}")
            else:
                changed.append(f"  {path}: {was_digest[:12]} -> {now_digest[:12]}")

    if changed:
        print(f"model_hashes: {len(changed)} corpora differ from the baseline:",
              file=sys.stderr)
        for line in changed:
            print(line, file=sys.stderr)
        print("\nIf this change was meant, rerun with --write and commit the "
              "baseline alongside it.", file=sys.stderr)
        return 1

    print(f"model_hashes: {len(rows)} corpora match the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
