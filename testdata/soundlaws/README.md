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

### `grassmann.tsv` — Grassmann's Law

Of two aspirates in a word, the first loses its aspiration. The hardest shape
here, and the one that took three separate capabilities to state:

    greek t ~ pie tʰ   /  @initial ∧ somewhere-following[aspirated:+]

An existential predicate, because the conditioning aspirate is neither adjacent
nor at a fixed distance. The ability to conjoin one onto a positional split,
because word-initial alone separates the aspirates that *can* dissimilate from
those that cannot and then has nothing more to say. And aspiration in the
conditioning vocabulary, because otherwise there is no term for the thing doing
the work.

Any two of the three give a *wrong* answer rather than no answer, which is why
the corpus carries words whose only later stop is unaspirated — `tʰekʰo` beside
`tʰeko`. Without them, "a stop somewhere after" predicts the change perfectly
and aspiration is never tested. With them, a search that cannot see aspiration
commits both outcomes under one environment: nine words that keep the aspirate
and six that lose it, all filed under `@initial ∧ somewhere-fol[stop:+]`.

This fixture documented a limit for a day and now documents a capability. The
paragraph above is kept because the shape of the failure is the useful part.

### `place_assimilation.tsv` — nasal place assimilation

The nasal takes the place of the consonant after it: `m` before a labial, `ŋ`
before a dorsal, unchanged before a coronal. The commonest conditioned change
there is, and until place entered the conditioning vocabulary it produced no
conditioned class at all — nothing in the candidate tables named a place, so
"before a labial" had no term to be stated in.

All three environments are present, so no two-way predicate can stand in for
the three-way one. Two come out named and the third as the elsewhere case,
which is how a sound law is conventionally written.

The same rule turns up in the real Latin/Italian data:
`italian ŋ ~ latin n / next-syl[dorsal:+]`, which is *lingua* and *anca*.

### `place_dissimilation.tsv` — labial dissimilation

An initial labial goes coronal when another labial appears later in the word,
at no fixed distance. Place in an existential environment, which is what place
conditioning looks like when it is not adjacent, and the reason the major
classes are searchable at long range rather than only next door.

## The graded ladder — `graded_*.tsv`

Seven corpora with the same shape and the same change, proto /p/ answering to
daughter /f/, differing only in what conditions it. Regenerate with
`scripts/graded.py`.

A single fixture that fails tells you something is wrong. A ladder tells you
where the ceiling is:

| rung | conditioning | found |
| --- | --- | --- |
| 0 | none | correctly leaves the change unconditioned |
| 1 | adjacent segment | yes, `fol[front:+]` |
| 2 | word position | yes, though by a correlate rather than the position predicate |
| 3 | stress on the preceding vowel | yes, `pre-stress[stress:primary]` |
| 4 | two predicates at once | yes, `pre[voiced:+] ∧ fol[front:+]` |
| 5 | a segment two places back | yes, `prev-syl[nasal:+]` |
| 6 | a segment somewhere later | yes, `somewhere-fol[nasal:+]` |

Rungs 5 and 6 found nothing at all until 2026-08-15. Of the thirteen features
regulae computes for every segment, four — `nasal`, `stop`, `fricative`,
`sonorant` — were attached to every context and never searched as candidates,
so no manner could ever condition anything. Nasal assimilation is among the
commonest conditioned changes there is, and it was inexpressible. Completing
the candidate tables changed nothing on the real corpora, which is the expected
result: the new predicates only win where manner genuinely conditions.

Rung 4 was the standing limit until 2026-08-15. The multi-lect stage committed
a single-predicate context and moved on, so a change conditioned by two came out
as one predicate with contradictory outcomes under it. It refines now, and the
broader environment stays as the fallback: `f ~ p` under both predicates, `p ~ p`
under the first alone. That is a decision list, not a contradiction — the
narrower rule wins where it applies.

What remains is that the ladder tests one predicate *kind* per rung. A rung
combining stress with distance, or three predicates at once, would say more
about the ceiling than rung 4 now does.

Rung 4 also does not test what its generator says it tests. The onsets are `n`
against `k`, which differ in nasality, sonorancy **and** voicing at once, so all
three predicates separate the corpus identically and no search can prefer one.
The rung therefore tests that *some* onset predicate is conjoined with the
follower, not that the onset predicate is nasality. Contrasting `n` with `l` —
both voiced sonorants — would isolate it. Recorded 2026-08-15, when the
conditioning vocabulary became corpus-derived and the reported name moved from
`voiced` to `sonorant` with nothing about the search having changed.

The forms are nonsense words on purpose. A rung tests one property of the
search, and real lexical material brings correlations with it — real words that
share an environment tend to share other things too, and then a failure is
ambiguous between the environment and its company. An earlier draft of rung 6
tied the trigger to the preceding vowel's frontness, and the search duly found
the vowel, which was a fact about the fixture rather than about the tool.

### `rounding_harmony.tsv` — a change conditioned by lip rounding

Proto `p` answers daughter `f` before a front **rounded** vowel (`y`, `ø`) and
stays `p` before a front unrounded one (`i`, `e`). Twenty-four sets each side,
perfectly regular, and the two environments differ in rounding alone: both are
front, both are vowels, and the preceding vowel is drawn from the same pair
either way, so nothing but rounding separates them.

This is the fixture that would have caught the old conditioning vocabulary.
Until 2026-08-15 the searchable feature list was 27 hand-picked names with no
`rounded` among them, and this corpus produced **zero** conditioned classes —
not a weak rule or a wrong environment, but silence, on a change as regular as
any in the directory. Rounding harmony organises the Turkic and Uralic vowel
systems and `experiments/turkish_azerbaijani/` has been in this repository the
whole time.

It is here as a standing check that the vocabulary is derived from the corpus
rather than from whichever languages the author had in mind.

### `morphological_rhotacism.tsv` — a change the phonology gets wrong

Latin rhotacism again, but built to fail the phonological answer. Intervocalic
/s/ became /r/ inside a morpheme — *honos-is > honoris — and did not across a
compound seam, where it stands between the same two vowels. "Intervocalic" is
necessary and not sufficient, and what separates the two sets is the morpheme
boundary.

The two intervocalic sets are the **same word**. One is monomorphemic and
rhotacises; the other carries a boundary after the prefix and does not. Nothing
phonological distinguishes them, so any environment stated in features alone is
wrong on half of them. Word-initial, word-final and preconsonantal /s/ are
present too, so "morpheme-internal" cannot answer on its own either.

This is the fixture for `rg_context_spec.morphological`, which until
2026-08-15 was copied, compared, sorted on and printed by regulae and assigned
by nothing at all. The boundaries come from a `breaks` column, which the long
TSV loader learned to read for this — the wide loader has carried
`<lect>_breaks` all along.

### `metathesis_adjacent.tsv` and `metathesis_distant.tsv` — one event, not two

Proto /sk/ answers daughter /ks/ in the first; /r/ and /l/ exchange places
across three intervening segments in the second, which is the Spanish *milagro*
< Latin *miraculo* shape.

The alignment search is monotone, so its natural output for a transposition is
two correspondences running in opposite directions: /s/ answering /k/ **and**
/k/ answering /s/. That is not a merger, it is not two changes, and a reader
who takes it at face value concludes something false about both segments. It is
what regulae reported until 2026-08-15.

Both fixtures assert the opposite: each transposed segment corresponds to
itself, because nothing about the segments changed — the order did.

The two are separate because they need different machinery. Adjacent
transposition fits inside a chunk. Long-distance transposition does not: the
search can only express it as one link covering everything between the two
segments that moved, which is wider than a chunk is allowed to be.

## Adding one

A fixture earns its place by being able to fail. Before adding one, write down
the environments where the change does *not* apply and put them in the corpus;
if there are none, the fixture is a demonstration and belongs in the
documentation instead.
