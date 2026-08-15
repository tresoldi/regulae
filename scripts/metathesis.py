#!/usr/bin/env python3
"""Generates the two metathesis fixtures under testdata/soundlaws/.

A regular transposition is one event, and the alignment search is monotone, so
its natural output is two correspondences running in opposite directions: /s/
answering /k/ and /k/ answering /s/. That is not a merger, it is not two
changes, and a reader who takes it at face value concludes something false
about both segments.

Two fixtures, because the two cases need different machinery. Adjacent
transposition fits inside a chunk. Long-distance transposition does not: the
search can only express it as one link covering everything between the two
segments that moved, which is wider than a chunk is allowed to be.

Both carry unrelated controls so the segment inventory has ordinary identity
correspondences to converge on.

Usage: scripts/metathesis.py
"""

import itertools
import pathlib

ONSETS = "ptkmn"
VOWELS = "aeiou"


def write(name, rows, note):
    out = ["cognate_id\tlect_id\tsegments"]
    for cid, proto, daughter in rows:
        out.append(f"{cid}\tproto\t{proto}")
        out.append(f"{cid}\tdaughter\t{daughter}")
    path = pathlib.Path("testdata/soundlaws") / f"{name}.tsv"
    path.write_text("\n".join(out) + "\n")
    moved = sum(1 for _, p, d in rows if p != d)
    print(f"{name:<30} {len(rows):>3} sets, {moved:>3} showing the {note}")


# Adjacent: proto /sk/ answers daughter /ks/, everywhere it occurs.
rows = []
for i, (c, v1, v2) in enumerate(itertools.product(ONSETS, VOWELS, VOWELS)):
    if len(rows) >= 20:
        break
    rows.append((f"w{i:02d}", f"{c} {v1} s k {v2}", f"{c} {v1} k s {v2}"))
for i, (c, v) in enumerate(itertools.product(ONSETS, VOWELS)):
    if i >= 10:
        break
    rows.append((f"c{i:02d}", f"{c} {v} l {v}", f"{c} {v} l {v}"))
write("metathesis_adjacent", rows, "transposition")

# Long-distance: /r/ and /l/ exchange places across three intervening
# segments, which is the Spanish milagro < Latin miraculo shape. Nothing else
# in the word changes, so any analysis that pairs them positionally has to
# claim that /r/ corresponds to /l/ and /l/ to /r/.
rows = []
for i, (c, v) in enumerate(itertools.product(ONSETS, VOWELS)):
    if len(rows) >= 20:
        break
    rows.append((f"w{i:02d}", f"{c} {v} r a k u l o", f"{c} {v} l a k u r o"))
for i, (c, v) in enumerate(itertools.product(ONSETS, VOWELS)):
    if i >= 10:
        break
    rows.append((f"c{i:02d}", f"{c} {v} m a n u", f"{c} {v} m a n u"))
write("metathesis_distant", rows, "transposition")
