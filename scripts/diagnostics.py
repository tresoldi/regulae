#!/usr/bin/env python3
"""Generates the data-quality fixtures under testdata/diagnostics/.

`testdata/soundlaws/` asks whether regulae can find a change. `testdata/restraint/`
asks whether it can decline to. These ask a third question, and it is the one a
working comparativist spends most of the week on: **is my corpus wrong, and
where?**

Nobody's wordlist is clean. Cognate judgements are made by people and some are
mistaken; two sources transcribing the same language do not agree with each
other; compounds are cognate in one half and not the other. None of that is a
failure of the method, and all of it changes what the method reports. What a
tool owes its user is not immunity to bad data -- there is no such thing -- but
a way of finding out.

Three shapes:

    contaminated  -- a regular correspondence set with non-cognate pairs in it
    drift         -- one language, two transcription conventions
    partial       -- compounds cognate in one element and not the other

Regenerate with `scripts/diagnostics.py`. The clean material is nonsense words
carrying a systematic correspondence set, for the same reason the graded
ladder's is: a fixture about noticing damage cannot afford words that bring
their own.
"""

import pathlib

OUT = pathlib.Path("testdata/diagnostics")

# A correspondence set a comparativist would call regular: seven consonants
# each with one reflex, vowels unchanged. Nothing here is conditioned, so
# anything the search reports as conditioned came from the damage.
SHIFT = {"p": "f", "t": "θ", "k": "x", "s": "h", "l": "r", "m": "m", "n": "n"}
ONSETS = list(SHIFT)
VOWELS = ["a", "e", "i", "o", "u"]


class Lcg:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state >> 16

    def pick(self, items):
        return items[self.next() % len(items)]


def write(name, rows, note):
    out = ["cognate_id\tlect_id\tsegments"]
    out += [f"{cid}\t{lect}\t{form}" for cid, lect, form in rows]
    (OUT / f"{name}.tsv").write_text("\n".join(out) + "\n")
    print(f"{name:<16} {len({r[0] for r in rows}):>4} sets   {note}")


def clean_frames(rng, count):
    """(id, proto, daughter) triples under the regular correspondence set."""
    frames = []
    for i in range(count):
        c1, c2 = rng.pick(ONSETS), rng.pick(ONSETS)
        v1, v2 = rng.pick(VOWELS), rng.pick(VOWELS)
        proto = f"{c1} {v1} {c2} {v2}"
        daughter = f"{SHIFT[c1]} {v1} {SHIFT[c2]} {v2}"
        frames.append((f"w{i:03d}", proto, daughter))
    return frames


# ----------------------------------------------------------- contaminated

def contaminated():
    """A regular correspondence set with five non-cognate pairs mixed in.

    The five are not corrupted versions of the right answer; they are two
    unrelated words filed under one identifier, which is what a mistaken
    cognate judgement actually is. They carry no confidence column, because a
    linguist who knew which ones were wrong would have removed them.

    The claim being tested is the one that makes `regulae outliers` a tool
    rather than a statistic: that ranking cognate sets by how badly they align
    under the trained model puts the bad ones at the top. If it does, the
    command is a way to re-read forty judgements in the order most likely to
    repay it; if it does not, it is a column of numbers.
    """
    rng = Lcg(3307)
    rows = []
    for cid, proto, daughter in clean_frames(rng, 40):
        rows.append((cid, "proto", proto))
        rows.append((cid, "daughter", daughter))
    for i in range(5):
        cid = f"bad{i}"
        c1, c2 = rng.pick(ONSETS), rng.pick(ONSETS)
        v1, v2 = rng.pick(VOWELS), rng.pick(VOWELS)
        rows.append((cid, "proto", f"{c1} {v1} {c2} {v2}"))
        # Drawn afresh. The daughter form is built from the daughter's own
        # inventory, so the bad sets are not findable by inventory alone -- but
        # nothing in it is a reflex of anything in the proto form.
        #
        # The first draft reused the proto's own consonants here, shifted and
        # displaced by one. That is a scrambled right answer rather than a
        # wrong judgement, and the ranking duly put two of the five below the
        # median: they contained the correct reflexes and the aligner found
        # them. The fixture was measuring its own generator.
        d1, d2 = rng.pick(ONSETS), rng.pick(ONSETS)
        w1, w2 = rng.pick(VOWELS), rng.pick(VOWELS)
        rows.append((cid, "daughter", f"{SHIFT[d1]} {w1} {SHIFT[d2]} {w2}"))
    write("contaminated", rows, "40 regular sets, 5 not cognate at all")


# ------------------------------------------------------------------ drift

def drift():
    """One language, two transcription conventions.

    The `narrow` lect writes what the `broad` lect writes as single graphemes
    in the way a different source would: the affricate as a stop plus a
    fricative, aspiration as a following /h/, a long vowel as two vowels. Every
    form is the same word, segmented differently. There is no sound change here
    at all.

    This is the commonest way a comparative dataset goes wrong, and the least
    visible. Both transcriptions are valid IPA, every grapheme resolves, and
    `regulae check` reports nothing, because nothing is unreadable -- the two
    sources simply disagree about where a segment ends. What the disagreement
    produces is a family of one-to-many correspondences that are indexed,
    counted and published exactly like real ones.

    The fixture asserts what actually comes out, so that a later normalisation
    step shows up as a test that has to be rewritten.
    """
    rng = Lcg(881)
    wide = {"tʃ": "t ʃ", "tʰ": "t h", "kʰ": "k h", "aː": "a a", "iː": "i i"}
    stock = ["tʃ", "tʰ", "kʰ", "m", "n", "l", "s", "p"]
    nuclei = ["aː", "iː", "a", "e", "o"]
    rows = []
    for i in range(40):
        c1, c2 = rng.pick(stock), rng.pick(stock)
        v1, v2 = rng.pick(nuclei), rng.pick(nuclei)
        parts = [c1, v1, c2, v2]
        rows.append((f"d{i:03d}", "broad", " ".join(parts)))
        rows.append((f"d{i:03d}", "narrow", " ".join(wide.get(p, p) for p in parts)))
    write("drift", rows, "the same forty words, segmented two ways")


# ---------------------------------------------------------------- partial

def partial():
    """Compounds cognate in one element and not the other.

    Half the sets are compounds whose first element is cognate and whose second
    is a different word entirely -- the shape of *hand-ring* against
    *hand-band*, and of a great many entries in a wordlist gathered by concept
    rather than by etymon. The other half are simplex and wholly cognate.

    A cognate set that is half right is not a cognate set that is wrong, and
    dropping it loses the half that is good. What the corpus needs from the
    tool is that the regular correspondences still come out of the cognate
    halves, and that the sets carrying a non-cognate element are visible as
    such rather than averaged in.
    """
    rng = Lcg(1723)
    rows = []
    for i in range(20):
        cid = f"s{i:03d}"
        c1, c2 = rng.pick(ONSETS), rng.pick(ONSETS)
        v1, v2 = rng.pick(VOWELS), rng.pick(VOWELS)
        rows.append((cid, "proto", f"{c1} {v1} {c2} {v2}"))
        rows.append((cid, "daughter", f"{SHIFT[c1]} {v1} {SHIFT[c2]} {v2}"))
    for i in range(20):
        cid = f"c{i:03d}"
        head1, head2 = rng.pick(ONSETS), rng.pick(ONSETS)
        hv1, hv2 = rng.pick(VOWELS), rng.pick(VOWELS)
        head = f"{head1} {hv1} {head2} {hv2}"
        shifted_head = f"{SHIFT[head1]} {hv1} {SHIFT[head2]} {hv2}"
        t1, t2 = rng.pick(ONSETS), rng.pick(ONSETS)
        tv1, tv2 = rng.pick(VOWELS), rng.pick(VOWELS)
        d1, d2 = rng.pick(ONSETS), rng.pick(ONSETS)
        dv1, dv2 = rng.pick(VOWELS), rng.pick(VOWELS)
        rows.append((cid, "proto", f"{head} {t1} {tv1} {t2} {tv2}"))
        rows.append((cid, "daughter",
                     f"{shifted_head} {SHIFT[d1]} {dv1} {SHIFT[d2]} {dv2}"))
    write("partial", rows, "20 simplex sets, 20 compounds cognate in the first half")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contaminated()
    drift()
    partial()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
