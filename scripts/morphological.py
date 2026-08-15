#!/usr/bin/env python3
"""Generates testdata/soundlaws/morphological_rhotacism.tsv.

Latin rhotacism is the textbook case of a change that a phonological
environment alone gets wrong. Intervocalic /s/ became /r/ inside a morpheme --
*honos-is > honoris -- but not across a compound seam: de+sub, ni+si, tri+sulcus
keep their /s/ between the same two vowels. "Intervocalic" is necessary and not
sufficient, and what separates the two sets is the morpheme boundary.

So the corpus carries three kinds of /s/, and each of the three has to be
present or the fixture cannot fail:

  - intervocalic inside a morpheme  -> r      the change
  - intervocalic across a seam      -> s      the boundary contrast
  - not intervocalic                -> s      the phonological contrast

Without the second, "intervocalic" answers perfectly and the boundary is never
tested. Without the third, "morpheme-internal" answers perfectly and the
phonology is never tested.

Usage: scripts/morphological.py
"""

import pathlib

VOWELS = ["o", "e", "a", "u", "i"]
FRAMES = [(v1, v2) for v1 in VOWELS for v2 in VOWELS if v1 != v2][:16]
ONSETS = ["d", "n", "t", "p"]

rows = []


def add(cid, old, new, breaks=""):
    rows.append((cid, old, new, breaks))


# The two intervocalic sets are the *same word*. One is monomorphemic and
# rhotacises; the other is a prefix plus a stem and does not. Nothing
# phonological distinguishes them, which is the point: any environment the
# search states in features alone must be wrong on half of them.
for i, (v1, v2) in enumerate(FRAMES):
    onset = ONSETS[i % len(ONSETS)]
    old = f"{onset} {v1} s {v2} t e r"
    new = f"{onset} {v1} r {v2} t e r"
    add(f"internal{i:02d}", old, new)
    add(f"seam{i:02d}", old, old, "2")

# Not intervocalic, and no boundary: initial, final and preconsonantal /s/, so
# "morpheme-internal" cannot answer on its own either.
for i, (v1, v2) in enumerate(FRAMES[:8]):
    onset = ONSETS[i % len(ONSETS)]
    add(f"initial{i:02d}", f"s {v1} {onset} {v2}", f"s {v1} {onset} {v2}")
    add(f"final{i:02d}", f"{onset} {v1} {onset} {v2} s", f"{onset} {v1} {onset} {v2} s")
    add(f"cluster{i:02d}", f"{onset} {v1} s t {v2}", f"{onset} {v1} s t {v2}")

out = ["cognate_id\tlect_id\tsegments\tbreaks"]
for cid, old, new, breaks in rows:
    out.append(f"{cid}\told_latin\t{old}\t{breaks}")
    out.append(f"{cid}\tlatin\t{new}\t{breaks}")
path = pathlib.Path("testdata/soundlaws/morphological_rhotacism.tsv")
path.write_text("\n".join(out) + "\n")
changed = sum(1 for _, o, nw, _ in rows if o != nw)
print(f"{len(rows)} sets, {changed} showing the change")
