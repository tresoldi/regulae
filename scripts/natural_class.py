#!/usr/bin/env python3
"""Generates the natural-class pair under testdata/soundlaws/.

One change, and the question of what it costs to spread it over the segments it
applies to.

`natural_class.tsv` voices every voiceless continuant between vowels -- f~v,
θ~ð, x~ɣ, s~z -- and leaves all four alone after a consonant. One change, four
segments, eight examples and eight counterexamples each.

`natural_class_control.tsv` is the same change with the same total evidence on
*one* segment: f~v, thirty-two intervocalic and thirty-two after a consonant.
Same number of aligned positions, same environment, same contrast; the only
difference is whether the evidence sits in one cell or four.

The pair measures a known limitation nothing yet puts a number on: "one change
fragments into one correspondence per segment it applies to". The control finds the rule with room to spare. The spread version has to
find it four times over, from a quarter of the evidence each time, and whatever
it does is what fragmentation costs. `linguistic_research_sources.md` asks for
exactly this measurement -- evidence held constant while the same displacement
applies to 1 and to 4 source segments -- and warns in the same breath against
assuming the pooled answer is available: active classes are frequently not
featurally natural, so a tool that pools has to earn it against the fragmented
alternative rather than assume it.

This one *is* featurally natural, on purpose. A fixture measuring the cost of
fragmentation should not also be testing whether the class can be named.

The contrast set matters as much here as anywhere. Without post-consonantal
fricatives the environment "between vowels" is unfalsifiable -- every fricative
in the corpus would be intervocalic and "medial", "after a vowel" and "before a
vowel" would all be equally right. The fixture carries all four consonants in
both environments so the wrong answers are available.

Nonsense words, for the reason the graded ladder uses them: real lexical
material brings correlations, and a failure would then be ambiguous between the
environment and its company.

Usage: scripts/natural_class.py     # rewrites both fixtures
"""

import itertools
import pathlib

OUT = pathlib.Path("testdata/soundlaws")

# The class, and the displacement every member undergoes. Voicing a continuant
# is one feature change, which is what makes the four rows one event.
VOICING = [("f", "v"), ("θ", "ð"), ("x", "ɣ"), ("s", "z")]

VOWELS = ["a", "e", "i", "o", "u"]
# No fricative among them: a blocking consonant that was itself a member of the
# class would put the class in both environments and blur the contrast. The
# same inventory serves as word onsets, which are there only to keep the forms
# distinct -- a word-initial consonant is nowhere near the fricative.
BLOCKERS = ["m", "n", "l", "r", "k", "t", "p", "ŋ"]

PER_CELL = 8


def write(name, rows):
    out = ["cognate_id\tlect_id\tsegments"]
    for cid, proto, daughter in rows:
        out.append(f"{cid}\tproto\t{proto}")
        out.append(f"{cid}\tdaughter\t{daughter}")
    (OUT / f"{name}.tsv").write_text("\n".join(out) + "\n")
    changed = sum(1 for _, p, d in rows if p != d)
    print(f"{name:<30} {len(rows):>3} sets, {changed:>3} showing the change")


def frames(count, offset):
    """(onset, v1, v2, blocker) drawn on a stride, so the two fixtures see the
    same material in the same order and neither gets an easier lexicon. The
    pool is walked far enough apart that no cell repeats a word: eight
    identical forms are one observation eight times, not eight of them."""
    pool = list(itertools.product(BLOCKERS, VOWELS, VOWELS, BLOCKERS))
    return [pool[(offset + i * 37) % len(pool)] for i in range(count)]


def rows_for(pairs, tag="", base=0):
    """Eight intervocalic and eight post-consonantal words per member."""
    rows = []
    for index, (voiceless, voiced) in enumerate(pairs):
        for i, (onset, v1, v2, _) in enumerate(frames(PER_CELL, (base + index) * 311)):
            cid = f"{voiceless}v{tag}{i:02d}"
            rows.append((cid, f"{onset} {v1} {voiceless} {v2}",
                         f"{onset} {v1} {voiced} {v2}"))
        for i, (onset, v1, v2, blocker) in enumerate(frames(PER_CELL, (base + index) * 311 + 149)):
            cid = f"{voiceless}c{tag}{i:02d}"
            # Same fricative, same vowels, blocked by a preceding consonant.
            rows.append((cid, f"{onset} {v1} {blocker} {voiceless} {v2}",
                         f"{onset} {v1} {blocker} {voiceless} {v2}"))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("natural_class", rows_for(VOICING))
    # The whole class's evidence on one member, so the totals match and only
    # the spread differs.
    rows = []
    for index in range(len(VOICING)):
        rows.extend(rows_for([("f", "v")], tag=f"{index}_", base=index))
    write("natural_class_control", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
