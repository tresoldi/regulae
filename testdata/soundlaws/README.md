# Sound-law fixtures

Small corpora, one per sound change, each built so that exactly one answer is
right and the wrong answers are available. `tests/c/test_sound_laws.c` asserts
what regulae must find in each.

These are **curated illustrations, not field data**. The forms are the standard
textbook examples for changes the field settled long ago, written in IPA and
regularised: enough real lexical material for the correspondence to be the one a
comparative linguist would draw, and no more. Anything requiring a judgement
call was left out rather than invented. For messy real data see `experiments/`,
and for the field's shared reference corpora see `scripts/lexibank.py`.

The point of a fixture here is not that regulae finds *a* pattern. It is that
the change is not in doubt, so a failure is unambiguous: the method has stopped
being able to recover a relationship every handbook agrees on.

## What each one tests, and why it is built the way it is

### `grimm.tsv` — Grimm's Law (PIE → Proto-Germanic)

Three shifts running at once: voiceless stops spirantise (p→f, t→θ, k→x),
voiced stops devoice (d→t, g→k), voiced aspirates lose aspiration (bʰ→b, dʰ→d,
gʰ→g). None of them is conditioned.

The test asserts all of them, and asserts they come out as *unconditioned*
classes. Finding one correspondence proves little; finding a chain shift whole
is the claim. A tool that split any of them on environment would be inventing
conditioning the change does not have, which is the more insidious failure —
spurious conditioning reads as a discovery.

### `rhotacism.tsv` — Latin rhotacism

/s/ becomes /r/ between vowels (*honōsis* → *honōris*) and stays /s/ everywhere
else.

The corpus deliberately carries /s/ in three environments that do **not**
rhotacise: word-initial (*sāl*, *sedeō*), word-final (*lupus*, *ōs*), and
preconsonantal but word-medial (*castra*, *vesper*, *magister*). That last group
is what makes the fixture a test rather than a demonstration. Without it,
"medial" predicts the change perfectly, BIC prefers it as the simpler predicate,
and the environment actually doing the work is never examined — an earlier
version of this file had exactly that hole, and regulae answered "medial",
correctly for the data it was given.

A fixture where only the intended answer is available cannot fail, and so
cannot test anything.

### `rhotacism_reordered.tsv` — the same corpus, rows in the other order

Byte-identical content, with each cognate set's two rows swapped so the corpus
mentions `old_latin` before `latin`. Row order is not data and must not change
the model.

It used to. Lects were held in the order the corpus first mentioned them, while
reconciliation, class discovery and the outlier ranking all walk pairs in
ascending lect-id order — so a corpus that did not list its lects alphabetically
had those stages align every pair in the opposite direction from the one its
model was trained in. A model of P(b|a) read as P(a|b) misses nearly every
lookup and falls back to the untrained prior, and the classes came out of an
alignment that had learned nothing.

### `lenition.tsv` — Western Romance intervocalic voicing

Latin voiceless stops voice between vowels (*vīta* → *vida*, *amīcu* →
*amigo*). The contrast set is the same stops after a consonant, where they stay
voiceless (*campu* → *campo*, *altu* → *alto*).

### `grassmann.tsv` — Grassmann's Law, and two limits it documents

Of two aspirates in a word, the first loses its aspiration. The conditioning
segment is never adjacent and never at a fixed distance, so only an existential
predicate can express it.

This fixture currently **fails to recover the environment**, and is kept for
that reason. The test asserts only what is true today — that the
correspondence is found — with a comment saying what would have to change for
it to assert more. Two limits stand in the way, both written up in
`docs/c_conversion_roadmap.md`: conditioning is discovered from the
alphabetically first lect of a pair, and a change is only visible from the side
that has the split; and existential predicates are never tried as a refinement
of a positional one, so "word-initial *and* an aspirate somewhere after" cannot
be stated.

A fixture that documents a limit is worth more than one that avoids it. This is
the corpus to run after changing anything in discovery.

## Adding one

A fixture earns its place by being able to fail. Before adding one, write down
the environments where the change does *not* apply and put them in the corpus;
if there are none, the fixture is a demonstration and belongs in the
documentation instead.
