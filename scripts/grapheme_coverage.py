#!/usr/bin/env python3
"""Measure what the feature system cannot read, across the Lexibank clone.

`docs/grapheme_coverage.md` was written by hand and went stale the moment
merkmal gained a diacritic: it listed a dataset as blocked on a grapheme that
had started resolving. A standing measurement needs a generator, so this is it.

What it measures is the corpus regulae would actually train on -- every usable
dataset converted through `scripts/lexibank.py`, then read by `regulae check`.
A dataset is blocked when one token in one form cannot be read, because that is
what refusing a form costs: the whole training run.

Nothing is vendored, so the numbers depend on which revision of the clone
produced them, and the report says so.

    scripts/grapheme_coverage.py --clone ~/lexibank_clone
    scripts/grapheme_coverage.py --check      # report drift, change nothing
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lexibank  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "grapheme_coverage.md"

# Tokens that are not a transcription of a sound and never will be. Each is a
# convention with a meaning, and the meaning is why guessing is refused rather
# than a limitation to be worked around. Anything not named here and not caught
# by the shape rules below is a gap in the feature system.
MIS_ENCODINGS = {
    "∼": "U+223C TILDE OPERATOR, a mathematical symbol for a tilde",
    "→": "an editorial arrow",
    "←": "an editorial arrow",
    "*": "a reconstruction mark left in the segment column",
    "ε": "Greek epsilon for IPA ɛ",
}
ORTHOGRAPHY = {"qu"}


def category(token):
    """Who owns the fix, decided by the token's shape rather than by hand."""
    if "/" in token:
        return "ambiguity or markup"
    if token.startswith("<"):
        return "ambiguity or markup"
    # A mis-encoded letter carrying a diacritic is still mis-encoded: `εː` is
    # Greek epsilon with a length mark, not a sound the feature system lacks.
    if any(token.startswith(mark) for mark in MIS_ENCODINGS):
        return "mis-encoding or editorial"
    if token in ORTHOGRAPHY:
        return "orthography, not IPA"
    if token.isascii() and token.isalpha() and token.isupper() and len(token) <= 2:
        return "cover symbol"
    return "IPA the feature system does not cover"


CATEGORY_ORDER = [
    "IPA the feature system does not cover",
    "mis-encoding or editorial",
    "cover symbol",
    "ambiguity or markup",
    "orthography, not IPA",
]


def check_corpus(cli, path):
    """Returns (unreadable_forms, {token: (status, count)})."""
    completed = subprocess.run([str(cli), "check", str(path)],
                               capture_output=True, text=True)
    unreadable = 0
    refused = {}
    for line in completed.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "unreadable" and len(parts) > 1:
            unreadable = int(parts[1])
        elif parts[0] == "GRAPHEME" and len(parts) > 3:
            refused[parts[1]] = (parts[2], int(parts[3]))
    return unreadable, refused


def sweep(clone, cli):
    datasets = lexibank.usable(clone)
    tokens = 0
    types = set()
    per_token = collections.defaultdict(lambda: {"status": "", "tokens": 0, "datasets": set()})
    blocked = {}
    empty = []
    dropped_forms = 0

    with tempfile.TemporaryDirectory() as raw:
        workdir = pathlib.Path(raw)
        for name in datasets:
            try:
                rows, _ = lexibank.convert(clone, name)
            except Exception:
                empty.append((name, "did not convert"))
                continue
            kept = []
            for cognate_id, lect, segments in rows:
                cleaned = lexibank.clean_segments(segments)
                if cleaned is None:
                    dropped_forms += 1
                    continue
                kept.append((cognate_id, lect, cleaned))
                for token in cleaned.split():
                    tokens += 1
                    types.add(token)
            if not kept:
                empty.append((name, "no cognate-linked segmented forms"))
                continue
            path = workdir / f"{name}.tsv"
            with path.open("w", encoding="utf-8") as handle:
                handle.write("cognate_id\tlect_id\tsegments\n")
                for cognate_id, lect, segments in kept:
                    handle.write(f"{cognate_id}\t{lect}\t{segments}\n")
            unreadable, refused = check_corpus(cli, path)
            if unreadable:
                blocked[name] = (unreadable, len(kept), sorted(refused))
            for token, (status, count) in refused.items():
                entry = per_token[token]
                entry["status"] = status
                entry["tokens"] += count
                entry["datasets"].add(name)

    return {
        "datasets": datasets, "tokens": tokens, "types": types,
        "per_token": per_token, "blocked": blocked, "empty": empty,
        "dropped_forms": dropped_forms,
    }


def render(result, clone):
    per_token = result["per_token"]
    refused_tokens = sum(entry["tokens"] for entry in per_token.values())
    resolving = result["tokens"] - refused_tokens
    share = 100.0 * resolving / result["tokens"] if result["tokens"] else 0.0
    by_category = collections.defaultdict(list)
    for token, entry in per_token.items():
        by_category[category(token)].append((token, entry))

    lines = [
        "# Grapheme coverage against Lexibank",
        "",
        "<!-- Generated by scripts/grapheme_coverage.py. Not checked by CI: it needs a",
        "     local Lexibank clone, and the numbers depend on which revision of it you",
        "     have. Regenerate deliberately and review the diff as data. -->",
        "",
        "What the feature system cannot read, measured over every usable dataset in a",
        "local Lexibank clone: one carrying both expert cognate judgements and",
        "CLTS-segmented forms. Generated by `scripts/grapheme_coverage.py`, which",
        "converts each dataset through `scripts/lexibank.py` and reads the result with",
        "`regulae check`. Nothing is vendored, so the numbers depend on which revision",
        "of the clone produced them; regenerate after any merkmal upgrade.",
        "",
        f"**{result['tokens']:,} segment tokens, {len(result['types']):,} distinct types. "
        f"{share:.2f}% of tokens resolve.**",
        "",
        f"{len(result['datasets'])} datasets are usable. {len(result['blocked'])} of them "
        "carry at least one token that cannot be read.",
        "",
        "A form regulae cannot read refuses the whole training run, so "
        "`scripts/lexibank.py` drops those forms and counts them, the way it already "
        "drops forms carrying markup: the word has a hole in it and the rest of it is "
        "not a word. What the table below reports is therefore a *cost* and not a "
        "verdict — two tokens cost `grollemundbantu` two forms of 35,918 rather than "
        "all of them. Pass `--keep-unreadable` to see the refusals instead of paying "
        "for them.",
        "",
        "## What each dataset costs",
        "",
        "| dataset | forms dropped | of | refused |",
        "| --- | ---: | ---: | --- |",
    ]
    for name in sorted(result["blocked"], key=lambda n: (-result["blocked"][n][0], n)):
        unreadable, forms, offenders = result["blocked"][name]
        shown = ", ".join(f"`{token}`" for token in offenders[:4])
        if len(offenders) > 4:
            shown += f" … {len(offenders) - 4} more"
        lines.append(f"| `{name}` | {unreadable} | {forms} | {shown} |")

    lines.extend([
        "",
        "## What is refused, and who owns each fix",
        "",
        "The shape of a token decides the category, so a new gap lands in one of these "
        "rather than going unnoticed. Only the first is the feature system's.",
        "",
    ])
    for name in CATEGORY_ORDER:
        entries = by_category.get(name, [])
        if not entries:
            lines.extend([f"### {name} — none", ""])
            continue
        total = sum(entry["tokens"] for _, entry in entries)
        lines.extend([
            f"### {name} — {len(entries)} types, {total:,} tokens",
            "",
            "| token | status | tokens | datasets |",
            "| --- | --- | ---: | --- |",
        ])
        ordered = sorted(entries, key=lambda item: (-item[1]["tokens"], item[0]))
        for token, entry in ordered[:20]:
            where = ", ".join(sorted(entry["datasets"])[:3])
            if len(entry["datasets"]) > 3:
                where += " …"
            lines.append(f"| `{token}` | {entry['status']} | {entry['tokens']} | {where} |")
        if len(ordered) > 20:
            lines.append(f"| … | | | {len(ordered) - 20} more types |")
        lines.append("")

    lines.extend([
        "## Datasets that produced no corpus",
        "",
        "Not a coverage problem. A dataset whose cognate judgements do not meet its "
        "segmented forms is invisible to every method in this repository, and saying so "
        "is more useful than leaving it out of the count.",
        "",
        "| dataset | why |",
        "| --- | --- |",
    ])
    for name, why in result["empty"]:
        lines.append(f"| `{name}` | {why} |")

    lines.extend([
        "",
        f"{result['dropped_forms']:,} forms were dropped before the feature system saw "
        "them, for carrying CLDF or CLTS markup rather than a transcription. That is a "
        "hole in the word, and the rest of it is not a word.",
        "",
        "## Reproduce",
        "",
        "```sh",
        f"scripts/grapheme_coverage.py --clone {clone}",
        "```",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", default="~/lexibank_clone")
    parser.add_argument("--cli", type=pathlib.Path, default=ROOT / "build" / "c" / "regulae")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.cli.exists():
        parser.error(f"CLI not found: {args.cli}")
    text = render(sweep(args.clone, args.cli), args.clone)
    if args.check:
        if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != text:
            print("grapheme coverage: the committed report is not what this clone "
                  "produces; regenerate it deliberately", file=sys.stderr)
            return 1
        print("grapheme coverage: current")
        return 0
    REPORT.write_text(text, encoding="utf-8")
    print(f"grapheme coverage: wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
