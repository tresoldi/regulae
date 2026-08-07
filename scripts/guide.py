#!/usr/bin/env python3
"""Generates web/guide-content.js from docs/GUIDE.md.

The guide is written once, in Markdown, and the in-page walkthrough is derived
from it. yvyra keeps the same prose in a Markdown file and a JavaScript file by
hand; they will drift, and the JavaScript copy is the one nobody proofreads.

Each `##` heading starts a step. A step may name a corpus with

    <!-- example: experiments/latin_spanish/cognates.tsv -->

which the page loads into the editor when the step is opened. The loader format
is inferred from where the file lives.

Only a small Markdown subset is supported, and anything outside it is an error
rather than being passed through: a guide that renders wrong in the browser but
right on GitHub is worse than one that refuses to build.

Usage:
    scripts/guide.py            # rewrite web/guide-content.js
    scripts/guide.py --check    # exit non-zero if it would change
"""

import argparse
import html
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "docs" / "GUIDE.md"
OUTPUT = REPO / "web" / "guide-content.js"

EXAMPLE = re.compile(r"^<!--\s*example:\s*(\S+)\s*-->$")
INLINE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<link>\[[^\]]+\]\([^)]+\))"
    r"|(?P<bold>\*\*[^*]+\*\*)"
    r"|(?P<italic>\*[^*]+\*)"
)


class GuideError(Exception):
    pass


def corpus_format(path):
    """Which loader reads this file, inferred from where it lives."""
    if path.suffix == ".csv":
        return "arcaverborum"
    if "experiments/" in path.as_posix():
        return "wide"
    return "tsv"


def render_inline(text, line_number):
    """Renders the supported inline constructs, escaping everything else."""
    out = []
    position = 0
    for match in INLINE.finditer(text):
        out.append(html.escape(text[position:match.start()]))
        if match.group("code"):
            out.append("<code>" + html.escape(match.group("code")[1:-1]) + "</code>")
        elif match.group("link"):
            label, _, target = match.group("link")[1:].partition("](")
            out.append(
                '<a href="' + html.escape(target[:-1]) + '">' + html.escape(label) + "</a>"
            )
        elif match.group("bold"):
            out.append("<strong>" + html.escape(match.group("bold")[2:-2]) + "</strong>")
        else:
            out.append("<em>" + html.escape(match.group("italic")[1:-1]) + "</em>")
        position = match.end()
    out.append(html.escape(text[position:]))
    rendered = "".join(out)
    if "*" in rendered or "_" in text and "__" in text:
        raise GuideError(f"line {line_number}: unpaired emphasis marker")
    return rendered


def render(lines, start_line):
    """Renders one step's body. Deliberately narrow: headings, paragraphs,
    bullet lists and fenced code, nothing else."""
    out = []
    index = 0
    while index < len(lines):
        line = lines[index]
        number = start_line + index

        if not line.strip():
            index += 1
            continue

        if line.startswith("```"):
            end = index + 1
            while end < len(lines) and not lines[end].startswith("```"):
                end += 1
            if end >= len(lines):
                raise GuideError(f"line {number}: unterminated code fence")
            body = "\n".join(lines[index + 1:end])
            out.append("<pre><code>" + html.escape(body) + "</code></pre>")
            index = end + 1
            continue

        if line.startswith("### "):
            out.append("<h3>" + render_inline(line[4:].strip(), number) + "</h3>")
            index += 1
            continue

        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(
                    "<li>" + render_inline(lines[index][2:].strip(), start_line + index) + "</li>"
                )
                index += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue

        if line.startswith(("#", ">", "|", "    ")) or re.match(r"^\d+\.", line):
            raise GuideError(
                f"line {number}: unsupported construct {line.strip()[:40]!r}. "
                "The guide renders through a deliberately small Markdown subset; "
                "use paragraphs, ### headings, bullet lists or fenced code."
            )

        paragraph = []
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(
            ("```", "### ", "- ", "<!--")
        ):
            paragraph.append(lines[index].strip())
            index += 1
        out.append("<p>" + render_inline(" ".join(paragraph), number) + "</p>")

    return "".join(out)


def parse(text):
    """Splits the guide into steps at `##` headings."""
    lines = text.split("\n")
    steps = []
    current = None
    body = []
    body_start = 0

    for number, line in enumerate(lines, start=1):
        if line.startswith("## "):
            if current is not None:
                current["body"] = body
                current["body_start"] = body_start
                steps.append(current)
            current = {"title": line[3:].strip(), "example": None}
            body = []
            body_start = number + 1
            continue
        if current is None:
            continue
        match = EXAMPLE.match(line.strip())
        if match:
            if current["example"] is not None:
                raise GuideError(f"line {number}: step has more than one example")
            current["example"] = match.group(1)
            continue
        body.append(line)

    if current is not None:
        current["body"] = body
        current["body_start"] = body_start
        steps.append(current)
    if not steps:
        raise GuideError("no ## headings found; the guide has no steps")
    return steps


def build():
    if not SOURCE.exists():
        raise GuideError(f"{SOURCE.relative_to(REPO)} does not exist")

    steps = []
    for step in parse(SOURCE.read_text(encoding="utf-8")):
        entry = {
            "title": step["title"],
            "content": render(step["body"], step["body_start"]),
        }
        if step["example"]:
            path = REPO / step["example"]
            if not path.exists():
                raise GuideError(
                    f"step {step['title']!r} names a corpus that does not exist: "
                    f"{step['example']}"
                )
            # The corpus text itself lives in web/corpora.js; embedding it here
            # too would ship several corpora twice, in two places that could
            # drift apart.
            entry["example"] = {
                "path": step["example"],
                "format": corpus_format(path),
            }
        steps.append(entry)

    header = (
        "// Generated by scripts/guide.py from docs/GUIDE.md. Do not edit by hand.\n"
        "//\n"
        "// Each step carries its prose and, where it has one, a reference to the\n"
        "// corpus the page loads into the editor. The corpus text lives in\n"
        "// corpora.js so it is not shipped twice.\n"
    )
    return header + "const GUIDE_STEPS = " + json.dumps(steps, indent=2, ensure_ascii=False) + ";\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the generated file is out of date")
    args = parser.parse_args()

    try:
        text = build()
    except GuideError as error:
        print(f"guide: {error}", file=sys.stderr)
        return 2

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("guide: web/guide-content.js is out of date; run scripts/guide.py",
                  file=sys.stderr)
            return 1
        print("guide: up to date")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    step_count = text.count('"title":')
    print(f"guide: wrote {OUTPUT.relative_to(REPO)} ({step_count} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
