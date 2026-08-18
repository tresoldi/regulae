#!/usr/bin/env python3
"""Generates the restraint fixtures under testdata/restraint/.

Every other fixture in this repository asks whether regulae can find something.
These ask whether it can decline to. A tool a linguist leans on has to be wrong
in the safe direction: a missed correspondence costs an afternoon, an invented
one costs a paper, and the second failure is the one that does not announce
itself. Spurious conditioning reads exactly like discovery.

Six shapes, and none of them contains a conditioned sound law:

    chance      -- two lects with no historical connection at all
    contact     -- two unrelated lects, half of one borrowed from the other
    diffusion   -- a real change spread over the lexicon at random
    merger_gap  -- an unconditioned merger whose source has a lexical gap
    stratum     -- two correspondence sets in one pair, as borrowing leaves them
    sparse_*    -- one real conditioned change, at five corpus sizes
    arity_*     -- one real conditioned change, at four lect counts
    distant_*   -- one real conditioned change, trigger a syllable away
    noise_unconditioned -- a real relationship, noise, and nothing to condition

Four of them have a right answer of "no conditioned rule", and regulae gives it
for three. `merger_gap` is the one it fails, and it fails it at the strongest
score in the corpus with every safeguard agreeing; the README says why, and the
cause is a conflation in the search rather than a threshold. `contact` has a
right answer regulae cannot give -- the correspondences are real and their
origin is not in the data -- and is here so that what it does give is on
record. The ladder's right answer changes with the rung, which is the point:
it measures how much data the search needs before it can see a change, and
whether it stays quiet below that or starts guessing.

The forms are nonsense words drawn from one inventory, for the same reason the
graded ladder's are. Real lexical material brings correlations with it, and a
fixture about restraint cannot afford them: if the changed words happened to
share anything, a predicate that found it would be right about this corpus and
the fixture would be measuring the wrong thing.

Randomness is a 32-bit LCG written out here rather than `random`, so the
fixtures are a function of this file and nothing else.

Usage: scripts/restraint.py     # rewrites testdata/restraint/*.tsv
"""

import itertools
import pathlib

OUT = pathlib.Path("testdata/restraint")

ONSETS = ["p", "t", "k", "m", "n", "l", "s", "w"]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "n", "s", "l", "k"]


class Lcg:
    """Numerical Recipes' LCG. Small, exactly specified, and reproducible
    without depending on a language's random module staying put."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        # High bits only. The low bits of an LCG cycle with a tiny period, and
        # `state % 8` on this one is close to a counter -- which put the same
        # onset opposite the same onset in 43 of the 80 unrelated pairs the
        # first time this file generated `chance`, a correspondence built by
        # the generator and not by the languages.
        return self.state >> 16

    def pick(self, items):
        return items[self.next() % len(items)]

    def shuffled(self, items):
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.next() % (i + 1)
            out[i], out[j] = out[j], out[i]
        return out


def skew_of(frames, marks, slots):
    """Worst relative imbalance of any environment value, over `slots` -- each
    a function pulling one environment out of a frame."""
    tally = {}
    for frame, mark in zip(frames, marks):
        for index, slot in enumerate(slots):
            cell = tally.setdefault((index, slot(frame)), [0, 0])
            cell[0 if mark else 1] += 1
    return max(abs(a - b) / (a + b) for a, b in tally.values())


def balanced_marks(frames, slots, seeds=4000):
    """Half the frames marked, chosen so that no environment the search can
    name is much fuller of marked frames than of unmarked ones.

    A shuffle alone does not give this. Drawn at random over groups of two
    dozen, the worst group in a corpus this size lands a third of the way to
    one side often enough to matter, and a third is a slope a greedy search
    will happily walk up. So the draws are searched and the flattest kept: the
    assignment stays arbitrary -- which is the property the fixture needs --
    without being arbitrary in a direction.
    """
    half = len(frames) // 2
    pool = [True] * half + [False] * (len(frames) - half)
    best = None
    for seed in range(seeds):
        marks = Lcg(seed).shuffled(pool)
        skew = skew_of(frames, marks, slots)
        if best is None or skew < best[0]:
            best = (skew, marks)
    return best[1], best[0]


def write(name, rows, note):
    out = ["cognate_id\tlect_id\tsegments"]
    for cid, lect, form in rows:
        out.append(f"{cid}\t{lect}\t{form}")
    path = OUT / f"{name}.tsv"
    path.write_text("\n".join(out) + "\n")
    sets = len({cid for cid, _, _ in rows})
    print(f"{name:<16} {sets:>4} sets   {note}")


# ---------------------------------------------------------------- chance

def chance():
    """Two lects with no historical connection.

    Both wordlists are drawn from the same inventory and the same shapes, which
    is the hard version of the test rather than the easy one: languages of a
    region share phonotactics whether or not they share an ancestor, and a
    method that only rejects unrelatedness when the inventories differ has not
    rejected unrelatedness. Accidental segment matches are frequent here by
    construction, and a few concepts will look like evidence.

    The right answer is that nothing stands above the corpus's own shuffled
    baseline -- because shuffling this corpus does not remove anything.
    """
    rng = Lcg(20260816)
    rows = []
    for i in range(80):
        cid = f"c{i:03d}"
        for lect in ("alpha", "omega"):
            form = [rng.pick(ONSETS), rng.pick(VOWELS), rng.pick(ONSETS), rng.pick(VOWELS)]
            coda = rng.pick(CODAS)
            if coda:
                form.append(coda)
            rows.append((cid, lect, " ".join(form)))
    write("chance", rows, "unrelated lects, one inventory")


# ------------------------------------------------------------- diffusion

def diffusion():
    """One change, spread over the lexicon rather than over an environment.

    Proto /p/ answers daughter /f/ in half the words and stays /p/ in the other
    half, and which half a word falls in is not a phonological fact about it.
    The two groups are balanced across every environment the search can name:
    each onset, each preceding vowel and each following vowel occurs about
    equally often on both sides, so no predicate separates them better than a
    coin does.

    This is what lexical diffusion looks like from the outside, and it is also
    what an incomplete change, a dialect mixture and a half-finished analogy
    look like. The right answer is two unconditioned correspondences for one
    proto segment -- p ~ p and p ~ f -- and no environment at all. Reporting an
    environment here would be reporting the accident that half of something has
    to fall somewhere.
    """
    rows = []
    # No /p/ or /f/ among the onsets. With one there, every medial /p/ has a
    # vowel before it and every initial one does not, so `pre[vowel:+]` splits
    # the corpus perfectly and truthfully -- and the fixture then reports a
    # conditioned rule for a reason that has nothing to do with diffusion.
    onsets = [o for o in ONSETS if o not in ("p", "f")]
    frames = [(o, v1, v2) for o in onsets for v1 in VOWELS for v2 in VOWELS]
    slots = (lambda f: f[0], lambda f: f[1], lambda f: f[2])
    marks, skew = balanced_marks(frames, slots)
    for i, ((o, v1, v2), changes) in enumerate(zip(frames, marks)):
        cid = f"d{i:03d}"
        rows.append((cid, "proto", f"{o} {v1} p {v2}"))
        rows.append((cid, "daughter", f"{o} {v1} {'f' if changes else 'p'} {v2}"))
    write("diffusion", rows, f"half changed, worst environment skew {skew:.0%}")


# --------------------------------------------------------------- stratum

def stratum():
    """Two correspondence sets in one lect pair, as borrowing leaves them.

    The inherited layer runs the whole voiceless series through a spirantising
    shift -- p~f, t~s, k~x -- and the borrowed layer leaves all three alone.
    Which layer a word belongs to is a fact about its history and not about its
    shape, so the two layers occupy the same environments.

    Splitting a *set* of correspondences at once is what makes a stratum
    visible to a comparativist, and it is the reason this fixture is not just
    `diffusion` with more segments. English has *father* beside *paternal* and
    *three* beside *triple*, and the second member of each pair is not an
    exception to Grimm's Law but a word that was not in the language when
    Grimm's Law ran.

    The right answer is six unconditioned correspondences and no environment.
    Neither layer is an error; a report that named an environment for either
    would be.
    """
    rows = []
    shift = {"p": "f", "t": "s", "k": "x"}
    frames = [(c, o, v1, v2) for c in ("p", "t", "k") for o in ("m", "l")
              for v1 in VOWELS for v2 in VOWELS]
    # The stop itself is a slot too: a layer that took two thirds of the /k/
    # words would be a stratum with a phonological shape, which is not what a
    # borrowed layer is.
    slots = (lambda f: f[0], lambda f: f[1], lambda f: f[2], lambda f: f[3])
    marks, skew = balanced_marks(frames, slots)
    for i, ((c, o, v1, v2), inherited) in enumerate(zip(frames, marks)):
        cid = f"s{i:03d}"
        rows.append((cid, "proto", f"{o} {v1} {c} {v2}"))
        rows.append((cid, "daughter", f"{o} {v1} {shift[c] if inherited else c} {v2}"))
    write("stratum", rows, f"two layers, worst environment skew {skew:.0%}")


# ------------------------------------------------------------ sparse_NNN

def sparse():
    """One real conditioned change, at five corpus sizes.

    Proto /p/ answers daughter /f/ before a front vowel and stays /p/ before a
    back one: the same change as rung 1 of the graded ladder, which regulae
    finds comfortably on 40 sets. The question here is not whether the change
    is findable but how much of it has to be present before it is, and what the
    search does with the material below that line.

    Both readings matter to somebody with a real wordlist. A field linguist
    with thirty cognates wants to know whether a rule the tool reports is worth
    writing down, and a reviewer wants to know whether its silence on a small
    corpus is evidence of absence. Neither is answerable from a single
    fixture: a rung that finds nothing tells you the corpus was too small, and
    a rung that finds something tells you it was not, and only the ladder tells
    you where the line is.

    Every rung is a prefix of the same word list, so a rung differs from the
    one below it in size and in nothing else.
    """
    rng = Lcg(511)
    front = {"e", "i"}
    back = [v for v in VOWELS if v not in front]
    frames = []
    # The following vowel alternates front and back rather than being drawn,
    # so exactly half of every rung shows the change. Drawn, the rungs differed
    # in how much evidence they carried as well as in how many sets -- the
    # 8-set rung came out with one changed word in it -- and a ladder whose
    # rungs differ in two things at once measures neither.
    for i in range(128):
        onset = rng.pick(["k", "t", "m", "n", "l", "s"])
        v1 = rng.pick(VOWELS)
        v2 = rng.pick(sorted(front)) if i % 2 == 0 else rng.pick(back)
        frames.append((onset, v1, v2))
    for size in (8, 16, 32, 64, 128):
        rows = []
        changed = 0
        for i, (onset, v1, v2) in enumerate(frames[:size]):
            cid = f"n{i:03d}"
            applies = v2 in front
            changed += applies
            rows.append((cid, "proto", f"{onset} {v1} p {v2}"))
            rows.append((cid, "daughter", f"{onset} {v1} {'f' if applies else 'p'} {v2}"))
        write(f"sparse_{size:03d}", rows, f"{changed} showing the change")


# --------------------------------------------------------------- contact

def contact():
    """Two unrelated lects, half of one's vocabulary borrowed from the other.

    Thirty of the sixty concepts are loans, adapted through a regular
    substitution -- the donor's f, θ, x and z have no place in the borrower's
    inventory and come out as p, t, k and s. The other thirty are native on
    both sides and share nothing.

    This is the shape of an areal relationship, and it is the commonest way a
    long-range comparison goes wrong. The Balkans, mainland Southeast Asia,
    South Asia, the Pacific Northwest and Australia all contain pairs like
    this, and Japanese and Chinese are the textbook example: half the lexicon
    in systematic correspondence, and no common ancestor.

    There is no restraint available here, and that is the point. The
    correspondences are real, they are regular, and regulae reports them --
    correctly. What it cannot do is say where they came from, because nothing
    in the distribution of segments does: telling inheritance from borrowing
    needs the semantic fields involved, the direction of cultural flow and the
    dates, and it is the linguist's judgement.

    What the corpus does leave is a signature, and it is worth knowing how to
    read. The outlier ranking comes out bimodal -- the loans align well and the
    native vocabulary does not -- where an inherited relationship of the same
    strength is unimodal. A wordlist that splits in half like that is a
    wordlist to ask a different question about.
    """
    rng = Lcg(65029)
    adapt = {"f": "p", "\u03b8": "t", "x": "k", "z": "s"}
    donor_only = list(adapt)
    shared = ["m", "n", "l", "t", "k", "s"]
    rows = []
    for i in range(60):
        cid = f"x{i:03d}"
        borrowed = i % 2 == 0
        c1 = rng.pick(donor_only if borrowed else shared)
        c2 = rng.pick(donor_only + shared)
        v1, v2 = rng.pick(VOWELS), rng.pick(VOWELS)
        donor = f"{c1} {v1} {c2} {v2}"
        rows.append((cid, "donor", donor))
        if borrowed:
            rows.append((cid, "borrower",
                         f"{adapt.get(c1, c1)} {v1} {adapt.get(c2, c2)} {v2}"))
        else:
            n1, n2 = rng.pick(shared), rng.pick(shared)
            w1, w2 = rng.pick(VOWELS), rng.pick(VOWELS)
            rows.append((cid, "borrower", f"{n1} {w1} {n2} {w2}"))
    write("contact", rows, "half the wordlist borrowed, half unrelated")


# ---------------------------------------------------------------- merger_gap

# The one fixture here written by hand rather than sampled. The others need
# their forms to carry no correlation at all, which is what the LCG buys; this
# one needs the opposite -- a distribution that is skewed all the way, because
# a lexical gap is not a random draw. Sampling it would be sampling until the
# gap appeared, which is the same as writing it down.
MERGER_GAP = [
    # *o, and every one of them before a labial.
    ("root", "r o p u"), ("fall", "t o p a"), ("bark", "k o b i"),
    ("sleep", "s o m u"), ("flat", "n o p i"), ("leaf", "l o b e"),
    ("strike", "d o p a"), ("resin", "g o m u"), ("small", "t o b i"),
    ("thunder", "r o m a"), ("belly", "k o p u"), ("above", "s o b e"),
    ("name", "n o m i"), ("wide", "l o p a"), ("deep", "d o b u"),
    ("seed", "g o m a"),
    # *a, and never before a labial. That is what makes the gap a gap.
    ("tooth", "r a t i"), ("stone", "k a d e"), ("salt", "s a l u"),
    ("night", "n a t e"), ("lake", "l a k i"), ("day", "d a g u"),
    ("body", "t a n u"), ("sand", "g a r i"), ("seven", "s a t e"),
    ("shell", "r a k u"), ("river", "n a d i"), ("road", "k a l u"),
    ("father", "t a d e"), ("sky", "l a g i"), ("gift", "d a n u"),
    ("rise", "s a g i"),
    # Other vowels, so *k > x has room to be seen as the unconditioned change
    # it is and the corpus is unmistakably one language pair.
    ("bird", "p i t i"), ("fly", "m u k e"), ("star", "s i n i"),
    ("dog", "k u r i"), ("green", "b e l u"), ("hit", "t u k i"),
    ("thin", "m i n e"), ("fish", "p e s u"),
]

LABIALS = frozenset("pbm")


def merger_gap():
    """A merger whose source has a gap in the lexicon, and no conditioning.

    Proto *o and *a both give /a/; *k gives x. All three unconditioned. The
    only structure is that *o occurs solely before a labial, so among daughter
    /a/ before a labial the proto source is *o -- true, wanted by a
    reconstructor, and not a sound change.

    The gap is asserted rather than trusted, in both directions, because the
    fixture is worthless if a later edit puts one *a before a labial: the
    tempting wrong answer stops being perfectly supported and the corpus
    quietly starts passing.
    """
    rows = []
    for gloss, proto in MERGER_GAP:
        segments = proto.split()
        for i, segment in enumerate(segments):
            following = segments[i + 1] if i + 1 < len(segments) else None
            assert not (segment == "a" and following in LABIALS), \
                f"{gloss}: proto *a before a labial closes the gap"
            assert not (segment == "o" and following not in LABIALS), \
                f"{gloss}: proto *o away from a labial closes the gap"
        daughter = " ".join(
            {"o": "a", "k": "x"}.get(segment, segment) for segment in segments
        )
        rows.append((gloss, "proto", proto))
        rows.append((gloss, "daughter", daughter))
    write("merger_gap", rows, "unconditioned merger, gapped source distribution")



# ---------------------------------------------------------------- arity

def arity():
    """One conditioned change, at four lect counts, over the same words.

    Every rung is a projection of one five-lect corpus onto its first N lects,
    so a rung differs from the one below it in how many languages are in the
    sample and in nothing else -- the same discipline `sparse_*` applies to
    corpus size. The change is deterministic and exceptionless in every rung:
    lect `a` keeps /k/, and every other lect backs it to /q/ before a back
    vowel and keeps /k/ before a front one. That is Turkic velar-uvular
    allophony, which is the commonest shape a multi-lect conditioning search
    will be asked to find.

    Two nuisances are in the corpus because real wordlists have them, and
    neither touches the rule. A one-in-sixteen chance that a segment in one
    lect is written as some other segment -- a loan, a misjudged cognate, a
    transcription variant -- gives the correspondence table the tail of one-off
    reflexes every real table has. A one-in-four chance that a lect is missing
    from a set gives the ragged coverage every real wordlist has. Neither is
    correlated with the vowel that conditions the change.

    The right answer is the same at every rung, because the rungs hold the same
    change with more languages attesting it: the conditioned split on the
    following vowel. regulae gives it at two lects and loses it at three, and
    the reason is that the class-level criterion prices a split by the number of
    distinct SISTER TUPLES in the pivot -- which counts the coverage pattern and
    the one-off reflexes alongside the correspondence, and so grows with the
    sample rather than with the structure.
    """
    rng = Lcg(4409)
    lects = ["a", "b", "c", "d", "e"]
    back = ["a", "o", "u"]
    front = ["e", "i", "y"]
    others = ["t", "n", "s", "l", "m", "r", "p", "b"]
    # Everything random is drawn once, for the five-lect corpus, so that a
    # rung really is a column subset of the rung above it.
    sets = []
    for i in range(160):
        is_back = i % 2 == 0
        harmony = back if is_back else front
        v1, v2 = rng.pick(harmony), rng.pick(harmony)
        medial = rng.pick(others)
        forms = {}
        for lect in lects:
            velar = "k" if lect == "a" or not is_back else "q"
            forms[lect] = [velar, v1, medial, v2]
        for lect in lects:
            for slot in range(4):
                if rng.next() % 16 == 0:
                    forms[lect][slot] = rng.pick(others + back + front)
        present = {lect: lect == "a" or rng.next() % 4 != 0 for lect in lects}
        sets.append((f"h{i:03d}", forms, present))

    for width in (2, 3, 4, 5):
        keep = lects[:width]
        rows = []
        changed = 0
        for cid, forms, present in sets:
            here = [lect for lect in keep if present[lect]]
            if len(here) < 2:
                continue
            changed += any(forms[lect][0] == "q" for lect in here)
            for lect in here:
                rows.append((cid, lect, " ".join(forms[lect])))
        write(f"arity_{width}", rows, f"{width} lects, {changed} showing the change")


# ---------------------------------------------------------------- distant

def distant():
    """The same ladder as `sparse_*`, for a trigger one syllable away.

    `sparse_*` measures the evidence floor for a change conditioned by the
    immediate neighbour and finds eight examples and eight counterexamples.
    Umlaut, vowel harmony, Verner's Law and dissimilation at distance are not
    that shape: their trigger sits in another syllable, and the search prices
    and gates those candidates separately. This ladder asks the same question of
    them, so the two numbers can be compared.

    Proto /u/ answers daughter /y/ when the next syllable holds /i/, and stays
    /u/ when it holds /a/. Deterministic, exceptionless, and stated on a slot
    the candidate vocabulary carries. Every rung is a prefix of the same word
    list.
    """
    rng = Lcg(6607)
    frames = []
    for i in range(8):
        frames.append((rng.pick(ONSETS), rng.pick(["t", "s", "n", "m", "l", "r"])))
    for size in (3, 4, 5, 8):
        rows = []
        for i, (onset, medial) in enumerate(frames[:size]):
            rows.append((f"i{i:02d}", "proto", f"{onset} u {medial} i"))
            rows.append((f"i{i:02d}", "daughter", f"{onset} y {medial} i"))
            rows.append((f"a{i:02d}", "proto", f"{onset} u {medial} a"))
            rows.append((f"a{i:02d}", "daughter", f"{onset} u {medial} a"))
        write(f"distant_{size:03d}", rows, f"{size} showing the change, {size} not")


# A spirantising map with no conditioning anywhere: every proto segment has
# exactly one reflex.
NOISE_MAP = {
    "p": "f", "t": "θ", "k": "x", "b": "p", "d": "t", "g": "k",
    "m": "m", "n": "n", "l": "l", "r": "r", "s": "s", "w": "w",
}
# Segments that appear only as transcription noise -- a loan, a misjudged
# cognate, a slip -- never as a regular reflex.
NOISE_SEGMENTS = ["h", "j", "ʃ", "ɣ", "q", "ts", "dz", "β"]


def noise_unconditioned():
    """A strong, wholly unconditioned relationship with a little noise.

    Every proto segment has one reflex, so there is no conditioned sound law
    anywhere. Then one segment in twelve is replaced by a token that is not a
    regular reflex at all -- the loans, misjudged cognates and transcription
    slips a real wordlist carries. The corpus aligns a hundred standard
    deviations better than its own shuffles, so it is unmistakably one language
    pair; the right answer is still zero conditioned rules.

    What the search does instead is commit a handful, each resting on the three
    or four words where a noise token happened to line up with a neighbour. This
    is the shape `docs/m6_evaluation.md`'s finding is about: on data with a real
    relationship but no conditioning, the pairing shuffle -- which destroys the
    correspondences and so measures a low bar -- certified every one of these as
    STANDS, and did it more the larger the corpus got. The per-pivot
    context-permuted null, which keeps the correspondences and shuffles only the
    environment, and the eight-example floor beneath it, decline all of them.

    So the assertion is not that nothing is committed -- something is -- but that
    nothing STANDS. That is a different restraint from `chance`, where nothing is
    committed at all, and it is the one the standing verdict exists to provide.
    """
    rng = Lcg(90112)
    consonants = [c for c in NOISE_MAP if c not in VOWELS]
    rows = []
    for i in range(280):
        proto = []
        for _ in range(2 + rng.next() % 2):
            proto += [rng.pick(consonants), rng.pick(VOWELS)]
        daughter = [NOISE_MAP.get(s, s) for s in proto]
        for k in range(len(daughter)):
            # One segment in twelve becomes a non-reflex token.
            if rng.next() % 100 < 12:
                daughter[k] = rng.pick(NOISE_SEGMENTS)
        rows.append((f"s{i:03d}", "proto", " ".join(proto)))
        rows.append((f"s{i:03d}", "daughter", " ".join(daughter)))
    write("noise_unconditioned", rows, "one map, one-in-twelve noise, no environment")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chance()
    contact()
    diffusion()
    merger_gap()
    stratum()
    sparse()
    arity()
    distant()
    noise_unconditioned()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
