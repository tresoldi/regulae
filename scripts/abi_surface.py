#!/usr/bin/env python3
"""Every field of every public struct must be used by code that does the work.

This project has repeatedly grown a public field that nothing but a test ever
touches. It is declared in the header, copied by the copy helper, compared by
the comparison helper, serialised by the JSON writer, rendered by the
formatter -- and never read or written by anything that analyses a corpus. Six
such fields were removed in one session, `rg_segment.length` was another, and
`rg_train_options.feature_system` was a seventh, found by this script's first
run: the feature system is chosen with `rg_context_use_system`, so setting the
option did nothing and said nothing.

The tests cannot catch this on their own. A test that sets a field and reads it
back passes whether or not the rest of the library has ever heard of it, and
the round-trip tests pass because the plumbing is uniform: it handles every
field, including the ones that mean nothing.

That is why "is it mentioned in the library" is the wrong question and this
script does not ask it. It asks whether the field is mentioned *outside the
uniform plumbing* -- outside the copy, free, compare, serialise, parse, format
and defaults code, all of which handle every field alike and so are evidence of
nothing. A field named only there is decoration.

Limits, stated rather than papered over:

  - It matches member syntax (`.field`, `->field`) textually, so a field whose
    name is shared with a field of another struct is credited by any use of
    either. `count` and `source` are effectively unchecked.
  - It proves a field is referenced, not that the reference is meaningful.

It catches the failure this codebase actually has, which is a field that no
working code names at all.

Run with --check to fail on any finding; run bare for the report.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / "include" / "regulae.h"
IMPLEMENTATION_DIRS = ("src", "cmd")

# Files that handle every field uniformly. A field appearing only in these is
# carried, not used.
PLUMBING = {
    "memory.c",   # copy and free
    "json.c",     # serialise and parse
    "format.c",   # render
    "options.c",  # defaults
}

# struct.field -> why the plumbing is the right and only home for it.
EXPECTED_PLUMBING_ONLY = {
    f"rg_format_model_options.{field}": "the formatter's own options; format.c is where it belongs"
    for field in ("top_segments", "top_displacements", "min_count", "top_chunks", "top_classes")
}

# A public option the library declares, accepts, and then refuses. Each entry
# would be a promise in the header the code does not keep, recorded so that
# adding one is deliberate rather than an oversight.
#
# Empty, and the check keeps it empty in both directions: it fails on an
# unrecorded refusal, and it fails on a recorded one that no longer exists.
# `chunk_min_transparency` was the only entry, and implementing it is what
# emptied the list -- the check said so rather than the line being quietly
# deleted.
EXPECTED_REFUSED_OPTIONS = {}

STRUCT_START = re.compile(r"^typedef struct (rg_\w+) \{")
STRUCT_END = re.compile(r"^\} (rg_\w+);")
FIELD = re.compile(r"^\s*(?:const\s+)?[\w ]+?[\w*\s]*?(\w+)\s*(?:\[[^\]]*\])?\s*;\s*(?:/\*.*)?$")


def public_struct_fields():
    """struct name -> field names, in declaration order."""
    structs = {}
    current = None
    depth = 0
    for raw in HEADER.read_text().splitlines():
        start = STRUCT_START.match(raw)
        if start is not None:
            current, depth = start.group(1), 0
            structs[current] = []
            continue
        if current is None:
            continue
        if STRUCT_END.match(raw) is not None:
            current = None
            continue
        # A nested brace would make the field list ambiguous. None exist today;
        # if one appears the script should be taught about it rather than
        # quietly skipping the struct.
        depth += raw.count("{") - raw.count("}")
        if depth != 0:
            continue
        line = raw.split("/*")[0].strip()
        if not line or line.startswith("*") or "(" in line:
            continue
        field = FIELD.match(line)
        if field is not None:
            structs[current].append(field.group(1))
    return structs


def working_code():
    chunks = []
    for directory in IMPLEMENTATION_DIRS:
        for path in sorted((ROOT / directory).rglob("*.[ch]")):
            if path.name not in PLUMBING:
                chunks.append(path.read_text())
    return "\n".join(chunks)


def refused_options(text):
    """Option fields whose only role in the library is to be rejected."""
    return {
        match.group(1)
        for match in re.finditer(
            r"if\s*\(\s*options->(\w+)[^)]*\)\s*\{[^}]*RG_ERR_UNSUPPORTED_OPTION", text, re.S
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on any finding")
    args = parser.parse_args()

    structs = public_struct_fields()
    text = working_code()
    everything = text + "\n".join(
        (ROOT / directory / name).read_text()
        for directory in IMPLEMENTATION_DIRS
        for name in PLUMBING
        if (ROOT / directory / name).exists()
    )

    decoration = []
    for struct, fields in sorted(structs.items()):
        for field in fields:
            if f"{struct}.{field}" in EXPECTED_PLUMBING_ONLY:
                continue
            if re.search(r"(?:\.|->)\s*" + re.escape(field) + r"\b", text):
                continue
            decoration.append((struct, field))

    refused = refused_options(everything)
    undeclared = sorted(refused - set(EXPECTED_REFUSED_OPTIONS))
    stale = sorted(set(EXPECTED_REFUSED_OPTIONS) - refused)

    field_count = sum(len(f) for f in structs.values())
    print(f"{len(structs)} public structs, {field_count} fields")

    for struct, field in decoration:
        print(f"  named only by the plumbing, so nothing uses it: {struct}.{field}")
    for field in undeclared:
        print(f"  declared and refused, and not recorded as such: {field}")
    for field in stale:
        print(f"  recorded as declared-and-refused but no longer is: {field}")

    if refused:
        print(f"declared and refused, by record: {', '.join(sorted(refused))}")
    if not decoration and not undeclared and not stale:
        print("every public field is named by code that does the work")

    return 1 if args.check and (decoration or undeclared or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
