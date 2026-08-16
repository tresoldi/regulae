#!/usr/bin/env python3
"""Generates the graded conditioning ladder under testdata/soundlaws/.

Ten corpora with the same shape and the same change -- proto /p/ answers to
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
#
#    Which words take the accent used to be `hash((o, v1, v2)) % 2`, and
#    Python salts string hashes per process: the rung was a different corpus
#    every time this script ran, so the committed fixture could not be
#    reproduced from the generator that claims to produce it. Any index will
#    do here as long as it is one -- what the rung needs is that the accent
#    falls independently of the segments around the /p/, not that it falls
#    unpredictably.
def stress(o, v1, v2):
    accented = (ONSETS.index(o) + VOWELS.index(v2)) % 2 == 0
    proto = f"{o} {STRESS}{v1} p {v2}" if accented else f"{o} {v1} p {STRESS}{v2}"
    daughter = f"{o} {v1} {'f' if accented else 'p'} {v2}"
    return proto, daughter
build("graded_3_stress", stress)

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


# 7. A trigger set that is not a natural class: /p/ answers /f/ after any of
#    r, u, k or i, and stays /p/ after a, e, o, m, n, l, t or s. This is the
#    RUKI law's shape -- PIE *s retracts after exactly those four segments in
#    Indo-Iranian, Balto-Slavic, Armenian and Albanian -- and it is on the
#    ladder because the four share no articulatory feature. Two are high
#    vowels, one is a dorsal stop, one is a coronal liquid; every feature true
#    of all four is true of something in the contrast set as well.
#
#    A context here is a conjunction of feature constraints, and a conjunction
#    narrows. There is no way to write a disjunction, so the best available
#    answer is a decision list: one rule per trigger, all with the same
#    outcome. That is also what a comparativist writes on the board, so the
#    rung is not asking for something the field does differently -- it asks
#    whether the search finds the four rules or gives up after one.
#
#    The word shape is held constant across triggers. A trigger that was
#    sometimes a vowel and sometimes a consonant would change the length of
#    the word with it, and "the /p/ is fourth" would be a predicate the corpus
#    handed over for free.
TRIGGERS = ["r", "u", "k", "i"]
INERT = ["a", "e", "o", "m", "n", "l", "t", "s"]


def disjunction(o, v1, v2):
    index = ONSETS.index(o) * 5 + VOWELS.index(v2)
    pool = TRIGGERS if index % 3 == 0 else INERT
    pre = pool[index % len(pool)]
    applies = pre in TRIGGERS
    proto = f"{o} {v1} {pre} p {v2}"
    daughter = f"{o} {v1} {pre} {'f' if applies else 'p'} {v2}"
    return proto, daughter
build("graded_7_disjunction", disjunction)


# 8. Syllable weight: /p/ answers /f/ when the syllable before it is heavy,
#    and stays /p/ after a light one. Sievers' Law is this shape, and so is
#    every rule stated over moras rather than over segments -- Latin's penult
#    accent, Germanic's high-vowel deletion, the metrical half of Verner's
#    environment.
#
#    Weight is disjunctive at the segmental level, deliberately: a syllable is
#    heavy here either because its vowel is long or because it has a coda, and
#    both kinds are present in equal number. That is what weight *is*, and it
#    is why the rung is not a repeat of rung 7. Rung 7 asks whether a
#    disjunction can be written; this one asks whether the term that makes
#    writing one unnecessary is available at all.
#
#    Two confounds had to be taken out of it, and both were the fixture's fault
#    rather than the tool's.
#
#    The coda is drawn from five segments sharing no feature -- n, s, l, m, r
#    span nasal, fricative, lateral, rhotic, voiced and voiceless. It was `n`
#    every time until 2026-08-17, and then "the syllable is closed" and "the
#    preceding segment is a voiced sonorant" were the same partition: the rung
#    reported `prev-syl[voiced:+]`, which was true and said nothing about
#    weight. That is rung 4's recorded hole in another place.
#
#    `k` and `t` are not among them, and the reason is a fact about the
#    syllabifier rather than a preference: a stop before /p/ is analysed as the
#    onset of the next syllable rather than as a coda, so those words come out
#    with an *open* first syllable and the rung would be asserting something
#    the corpus does not contain. Measured, not assumed -- n, s, l, m and r
#    close the syllable and k and t do not.
#
#    And the onset carries an optional second consonant, in all three groups
#    equally. Without it a coda was the only thing that could make a word
#    longer, so the /p/ stood one place later in exactly the closed-syllable
#    words and `pre@2[vowel:+]` predicted closure perfectly -- a fact about
#    where the segment sits, not about the syllable it sits after.
CODAS = ["n", "s", "l", "m", "r"]


def weight(o, v1, v2):
    index = ONSETS.index(o) * 5 + VOWELS.index(v2)
    onset = f"{o} r" if index % 2 == 0 else o
    kind = index % 3
    if kind == 0:
        first, heavy = f"{onset} {v1}ː", True                 # heavy: long vowel
    elif kind == 1:
        coda = CODAS[index % len(CODAS)]
        first, heavy = f"{onset} {v1} {coda}", True                # heavy: closed
    else:
        first, heavy = f"{onset} {v1}", False                      # light
    proto = f"{first} p {v2}"
    daughter = f"{first} {'f' if heavy else 'p'} {v2}"
    return proto, daughter
build("graded_8_weight", weight)


# 9. The conditioning environment is gone from the daughter. Rung 1's change
#    exactly -- /p/ answers /f/ before a front vowel -- with the vowel that
#    conditioned it deleted in the daughter, so that in the daughter alone
#    nothing distinguishes the words that changed from the words that did not.
#
#    This is opacity, and it is the ordinary case rather than an exotic one.
#    Germanic i-umlaut fronted a vowel and then the *i* that fronted it fell,
#    which is why English has *foot/feet* with no /i/ anywhere in sight; the
#    same sequence gave Old Norse its umlaut, French its nasal vowels, and
#    Mandarin its tones. A change that destroys its own environment leaves the
#    daughter looking irregular, and the environment survives only in a
#    relative that did not run the second change.
#
#    Which is the argument for comparing more than two lects at a time, and it
#    is testable: the environment here is on the proto's side of the pair and
#    on no other, so a search that only looked at the segment being explained
#    would find nothing.
build("graded_9_lost_trigger",
      lambda o, v1, v2: (f"{o} {v1} p {v2}",
                         f"{o} {v1} {'f' if v2 in FRONT else 'p'}"))
