#!/usr/bin/env python3
"""Generates the graded conditioning ladder under testdata/soundlaws/.

Seven corpora with the same shape and the same change -- proto /p/ answers to
daughter /f/ -- differing only in what conditions it. The point is to separate
"the tool cannot find this pattern" from "the tool cannot find patterns of this
*kind*". A single fixture that fails tells you something is wrong; a ladder
tells you where the ceiling is.

Every rung carries its own contrast: the same /p/ in the environments where the
change does not apply. Without that a predicate can be right by accident, which
is how an earlier rhotacism fixture came to be satisfied by "medial".

The forms are nonsense words, deliberately. A rung is testing one property of
the search, and real lexical material would bring correlations with it -- real
words that share an environment tend to share other things too, and then a
failure is ambiguous between the environment and its company.

Usage: scripts/graded.py        # rewrites the graded_*.tsv fixtures
"""

import itertools
import pathlib

ONSETS = ["k", "t", "m", "n", "l", "s"]
VOWELS = ["a", "e", "i", "o", "u"]
FRONT = {"e", "i"}
STRESS = "ˈ"


def write(name, rows):
    out = ["cognate_id\tlect_id\tsegments"]
    for cid, proto, daughter in rows:
        out.append(f"{cid}\tproto\t{proto}")
        out.append(f"{cid}\tdaughter\t{daughter}")
    path = pathlib.Path("testdata/soundlaws") / f"{name}.tsv"
    path.write_text("\n".join(out) + "\n")
    changed = sum(1 for _, p, d in rows if p.replace(STRESS, "") != d)
    print(f"{name:<28} {len(rows):>3} sets, {changed:>3} showing the change")


def frames():
    """(onset, v1, v2) triples, cycled so every rung sees the same material."""
    return [(o, v1, v2)
            for o, v1, v2 in itertools.product(ONSETS, VOWELS, VOWELS)
            if v1 != v2][:40]


def build(name, rule, shape=None):
    """rule(o, v1, v2) -> (applies, proto_word, daughter_word)."""
    rows = []
    for i, (o, v1, v2) in enumerate(frames()):
        cid = f"w{i:02d}"
        proto, daughter = rule(o, v1, v2)
        rows.append((cid, proto, daughter))
    write(name, rows)


# 0. No conditioning at all: /p/ answers to /f/ wherever it stands.
build("graded_0_unconditioned",
      lambda o, v1, v2: (f"{o} {v1} p {v2}", f"{o} {v1} f {v2}"))

# 1. One adjacent segment: only before a front vowel. The same /p/ before a
#    back vowel is the contrast.
build("graded_1_adjacent",
      lambda o, v1, v2: (f"{o} {v1} p {v2}",
                         f"{o} {v1} {'f' if v2 in FRONT else 'p'} {v2}"))

# 2. Position: only word-finally. Every word carries a medial /p/ too, so
#    "the word contains /p/" cannot stand in for "the /p/ is final".
build("graded_2_position",
      lambda o, v1, v2: (f"{o} {v1} p {v2} p", f"{o} {v1} p {v2} f"))

# 3. Stress: only when the preceding vowel carries the accent. This is
#    Verner's shape, with everything else held constant.
build("graded_3_stress",
      lambda o, v1, v2: (f"{o} {STRESS}{v1} p {v2}" if hash((o, v1, v2)) % 2 == 0
                         else f"{o} {v1} p {STRESS}{v2}",
                         f"{o} {v1} {'f' if hash((o, v1, v2)) % 2 == 0 else 'p'} {v2}"))

# 4. Two predicates at once: preceded by a nasal *and* followed by a front
#    vowel. All four combinations are present, so neither half predicts the
#    change on its own and a single-predicate answer is available and wrong.
#    An earlier draft asked for "intervocalic and before a front vowel" on a
#    corpus where every /p/ was intervocalic, which is one predicate wearing
#    two names.
def conjunction(o, v1, v2):
    index = VOWELS.index(v1) + VOWELS.index(v2)
    onset = "n" if index % 2 == 0 else "k"
    follower = v2 if v2 in FRONT else ("e" if index % 4 < 2 else "o")
    applies = onset == "n" and follower in FRONT
    proto = f"{onset} p {follower} {v1}"
    daughter = f"{onset} {'f' if applies else 'p'} {follower} {v1}"
    return proto, daughter
build("graded_4_conjunction", conjunction)


# 5. Distance two: conditioned by the segment two places back. The adjacent
#    segment is always the same vowel, so it carries no information, and the
#    trigger alternates independently of everything else in the word.
def distance(o, v1, v2):
    trigger = "n" if (VOWELS.index(v1) + VOWELS.index(v2)) % 2 == 0 else "l"
    proto = f"{o} {trigger} a p {v2}"
    daughter = f"{o} {trigger} a {'f' if trigger == 'n' else 'p'} {v2}"
    return proto, daughter
build("graded_5_distance_two", distance)


# 6. Existential: conditioned by a nasal somewhere later, at a distance that
#    varies from word to word. Nothing at a fixed offset predicts it, and the
#    trigger is independent of the neighbouring vowels -- an earlier draft tied
#    it to the preceding vowel's frontness, and the search duly found the
#    vowel, which was a fact about the fixture and not about the tool.
def existential(o, v1, v2):
    index = VOWELS.index(v1) + VOWELS.index(v2)
    nasal = index % 2 == 0
    trailer = "m" if nasal else "l"
    if index % 3 == 0:
        tail = f"{v2} {trailer} a"
    elif index % 3 == 1:
        tail = f"{v2} t a {trailer} e"
    else:
        tail = f"{v2} t a s e {trailer} o"
    proto = f"{o} {v1} p {tail}"
    daughter = f"{o} {v1} {'f' if nasal else 'p'} {tail}"
    return proto, daughter
build("graded_6_existential", existential)
