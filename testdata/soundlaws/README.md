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

### `opaque_umlaut.tsv` — the environment is in a different language

Germanic i-umlaut across three lects: a Gothic-shaped one that kept the final
vowel and never fronted, an Old-High-German-shaped one that fronted and kept
it, and an Old-English-shaped one that fronted and lost it. In the third alone
`g e s t` and `g a s t` are a minimal pair with nothing to separate them; the
*i* that explains them is in the other two.

This is the standard classroom argument for why Gothic matters to the history
of English, and it turns out to be a testable claim about a method rather than
only a story. The environment comes out as `next-syl[close:+]`, read off the
lects that kept the trigger, for the lect that did not.

The fixture also records a limit, and it is the more useful half. **Umlaut is
one change and it surfaces here as four correspondences, one per vowel
quality**: `a ~ e` with fifteen examples, and `uː ~ yː`, `u ~ y`, `oː ~ øː`
with two to four each. Only the first crosses the evidence floor that
`testdata/restraint/sparse_*` measures, so only the first gets its environment;
the other three are published as unconditioned splits, with the same trigger
standing next to them in the same words.

A change that applies to a natural class of segments is divided by the size of
that class before the search ever sees it, and each fragment has to carry its
own evidence. Palatalisation, lenition, nasalisation and every chain shift have
this shape, which makes it one of the more consequential limits in the tool: a
change can be overwhelmingly attested as a change and invisible as any one of
its correspondences.

The first draft of this fixture had ten stems and found no environment at all.
Adding ten more `a`-stems — and nothing else — was enough. That is what pins
the cause on class size rather than on the opacity.

### `great_vowel_shift.tsv` — a chain, not a merger

Middle English /eː/ raised to /iː/ while /iː/ was diphthongising out of the
way, and /oː/ raised to /uː/ while /uː/ did the same. Every step lands where
the next one has just left.

Which is what makes it a test. A method that keeps no separate account of the
two sources reports a merger — ME /eː/ and /iː/ both answering Modern English
/iː/ — and that is false about both, and would say that *feet* and *five* had
the same vowel in 1400. The fixture asserts each link separately and asserts
that no class puts two Middle English sources together.

The diphthongs are written as two segments, so the raised vowel answers the
*nucleus* of `a ɪ` rather than the diphthong whole. That is a decision the
corpus makes and not the tool: a correspondence is stated at the granularity
the transcription was written at, and a dataset that wrote `aɪ` as one grapheme
would get a different and equally correct answer. It is worth knowing before
comparing two sources that made that choice differently.

### `compensatory_lengthening.tsv` — a change caused by a deletion

The Ingvaeonic nasal spirant law: Proto-Germanic lost a nasal before a
fricative and lengthened the vowel in front of it, so *gans* answers Old
English *gōs* and *tanþ* answers *tōþ*. The same nasal before a stop is
untouched — *hand* stays *hand* — and a nasal with nothing after it is
untouched too, so neither "before a consonant" nor "before anything" will do.

Two events with one cause, and the segment that explains the vowel is the one
that is no longer there. The environment has to name both the nasal and what
followed it, which is a conjunction with a distance term in it:

    oe oː ~ pgmc a   /  fol[nasal:+] ∧ fol@2[fricative:+]

Greek, Latin, Old Irish, Hindi and Middle Korean all have a version of this,
and it is the mechanism behind a large share of the world's long vowels,
nasal vowels and tone systems. It is here because the alignment is the part
that could go wrong: the nasal has to be absorbed into a chunk with the vowel
rather than aligned against the fricative that follows it.

### `final_devoicing.tsv` — a neutralisation, and what a baseline is for

German *Rad/Räder*, *Tag/Tage*, *Kind/Kinder* against *Wort/Worte*,
*Blut/Blutes*, *Licht/Lichter*. Two "lects" that are two slots of one
paradigm, which makes this internal reconstruction rather than comparison —
the same machinery, and the method a linguist reaches for when there is only
one language to work with.

Word-final /t/ has two sources and nothing in the citation form says which, so
the right answer is two correspondences and no environment: `t ~ t` and
`t ~ d`, `k ~ k` and `k ~ g`, `p ~ p` and `p ~ b`. regulae commits four
conditioned rules, every one of them a correlate of the alternation rather
than a cause of it — the alternating /d/-words happen to have sonorants before
them more often than the others do — and **all four fall below the level the
same search reaches on the shuffled corpus**, while the corpus itself sits
twenty standard deviations below its own baseline.

Read without the baseline it reports four environments for a change that has
none. Read with it, it reports a neutralisation. That is the argument for
`permutation_count`, made on data nobody disputes, and it is why the
`tests/c/test_restraint.c` assertions for this fixture live with the restraint
suite rather than here.

The file was in this directory for a week before any test read it, and it
wrote length with an ASCII colon. Both are fixed; the second is the sort of
thing `regulae check` reports and nobody runs.

## The graded ladder — `graded_*.tsv`

Ten corpora with the same shape and the same change, proto /p/ answering to
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
| 7 | any one of four unrelated segments | yes, as a decision list: `pre[close:+]`, `pre[stop:+]`, `pre[trill:+]` |
| 8 | the weight of the preceding syllable | **partly** — one of the two things that make a syllable heavy |
| 9 | a segment the daughter has since lost | yes, `fol[front:+]`, read off the proto |

Rung 8 is the ceiling, and it is asserted as it stands so that raising it shows
up as a test that has to be rewritten rather than as one that starts passing by
accident.

### Rung 7 — a trigger set that is not a natural class

/p/ answers /f/ after any of `r`, `u`, `k` or `i`, and stays /p/ after `a`,
`e`, `o`, `m`, `n`, `l`, `t` or `s`.

This is the RUKI law's shape. PIE *s retracts after exactly those four segments
in Indo-Iranian, Balto-Slavic, Armenian and Albanian, and the four share no
articulatory feature: two are high vowels, one is a dorsal stop, one is a
coronal liquid, and every feature true of all four is true of something in the
contrast set as well. The field has argued about whether they form an *acoustic*
class for a century; nobody claims they form a featural one.

A context is a conjunction of feature constraints and a conjunction narrows, so
a disjunction cannot be written as one environment. What a comparativist writes
on the board instead is a decision list — one rule per trigger, all with the same
outcome — and that is what comes out, because discovery is greedy and each rule
is committed against what the earlier ones left:

    f ~ p / pre[close:+]    8   ({i, u}, which do share a feature)
    f ~ p / pre[stop:+]     4   ({k})
    f ~ p / pre[trill:+]    4   ({r})

**This rung documented a ceiling for a day, and the ceiling was in the report
rather than in the search.** Until 2026-08-17 the multi-lect table merged rows
on the correspondence tuple alone, so three of these four rules were discarded
before publication and the survivor was whichever environment carried the most
constraints — here a two-term conjunction describing none of them well. The
rules were in the pairwise tables the whole time, which is how the collapse
went unnoticed, and the same merge was throwing away 32 of the 58 splits
committed on `testdata/corpora/real_romance_4lect.tsv`.

A fourth row is committed on a broader correlate overlapping the first, and the
shuffled baseline marks it within-noise. That is the right place for the
judgement: absorbing it during the merge would absorb the genuine
`pre[close:+]` underneath it too, because a superset swallows what it contains.

### Rung 8 — syllable weight

/p/ answers /f/ after a heavy syllable and stays /p/ after a light one, with
heavy-by-long-vowel and heavy-by-coda present in equal number.

Sievers' Law has this shape, and so does every rule stated over moras rather
than segments: Latin's penultimate accent, Germanic's high-vowel deletion, the
metrical half of Verner's environment, most of what a metrist means by a rule.

The disjunction is deliberate, and it is why this is not rung 7 again. There
the missing thing was the ability to write a disjunction; here it is the term
that would make writing one unnecessary.

    f ~ p / prev-syl[syllable_weight:heavy]    28, elsewhere 0

Three syllable properties are available and none of them is a segment feature:
`syllable_shape` (open or closed), `syllable_nucleus` (long or short) and
`syllable_weight` (heavy or light). The first two are facts. The third is a
**verdict** — heavy means a long nucleus or a coda, which is the majority
convention and the one Latin, Greek, Arabic and Sanskrit metrics use, and it is
not universal. It is offered anyway because a disjunction over segments is
exactly what a context cannot hold, and because the verdict is the term the
field states these rules in; the two facts sit beside it so a language whose
tradition draws the line elsewhere can still be described.

Until 2026-08-17 the rung reported `pre[long:+]` and left the closed syllables
to a second rule. That was correct — the two rules covered all twenty-eight —
and it was not what anybody wants to read. Two things had to change: the
predicates had to reach the pass that sees the ungrouped corpus (they were
offered only in the long-range pass, which does not decompose promoted chunks
and so never saw the corpus whole), and the multi-lect stage had to be given
them at all.

Two confounds had to come out of the fixture first, and both were the
fixture's fault rather than the tool's. The coda was `n` every time, so
"closed" and "preceded by a voiced sonorant" were the same partition. And
without an optional second consonant in the onset, a coda was the only thing
that could lengthen a word, so `pre@2[vowel:+]` predicted closure exactly — a
fact about where the segment sits rather than about the syllable before it.
`k` and `t` are not among the codas either, and that is measured rather than
chosen: a stop before /p/ is syllabified as the onset of the next syllable, so
those words have an *open* first syllable and the rung would be asserting
something the corpus does not contain.

### Rung 9 — an environment the daughter no longer has

Rung 1 exactly, with the conditioning vowel deleted in the daughter. Within the
daughter nothing distinguishes the words that changed from the words that did
not.

This is opacity, and it is the ordinary case rather than an exotic one.
Germanic i-umlaut fronted a vowel and then the *i* that fronted it fell, which
is why English has *foot/feet* with no /i/ anywhere in sight; the same sequence
gave Old Norse its umlaut, French its nasal vowels and Mandarin its tones. A
change that destroys its own environment leaves the daughter looking irregular,
and the environment survives only in the proto, or in a relative that did not
run the second change.

It is found, fully covered, from the proto's side of the pair — which is what
`context_is_target` is for, and which is the argument for comparing more than
two lects at a time stated as a property of the search. `opaque_umlaut.tsv`
below is the same claim on curated material.

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

## Reproducibility

Every generated fixture has to be a function of its generator and nothing else,
or the committed file cannot be checked against the script that claims to
produce it.

`graded_3_stress.tsv` was not, until 2026-08-16. Which words took the accent
was `hash((onset, v1, v2)) % 2`, and Python salts string hashes per process, so
running `scripts/graded.py` rewrote the fixture with a different accent
assignment every time. Nothing failed — the rung kept passing, because the rung
is about stress conditioning and any assignment gives it — which is why it went
unnoticed. The index is arithmetic now.

## The other half of the suite

`testdata/restraint/` holds the corpora that contain **no** conditioned sound
law, and asserts that regulae reports none: two lects with no history between
them, a change spread over the lexicon rather than over an environment, a
borrowed stratum, and a ladder measuring how much evidence the search needs
before it can see a rule at all. A tool that finds every law in this directory
and also finds laws in that one has not been shown to work.

## Adding one

A fixture earns its place by being able to fail. Before adding one, write down
the environments where the change does *not* apply and put them in the corpus;
if there are none, the fixture is a demonstration and belongs in the
documentation instead.
