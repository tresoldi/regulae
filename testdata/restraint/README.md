# Restraint fixtures

Corpora that contain no conditioned sound law, and three ladders that say how
much evidence it takes before regulae can see one that is there — in cognate
sets, in languages, and in how far from the segment its trigger sits.
`tests/c/test_restraint.c` asserts what regulae must **not** find in each.

Restraint has two halves, and they need different fixtures. Most here ask
whether the search stays quiet — commits no conditioned rule at all.
`noise_unconditioned` asks the other half: when the search *does* commit
something on data with a real relationship and no conditioning, whether the
standing verdict declines it. The pairing shuffle did not; the per-pivot
context-permuted null and the evidence floor do.

One fixture, `merger_gap.tsv`, regulae currently fails, and its section says so
in full. A directory of restraint fixtures that all pass is a directory that
has stopped looking.

Everything else in `testdata/` asks whether a discovery can be made. These ask
whether it can be declined, which is the harder half of being useful. A method
that misses a correspondence costs its user an afternoon. A method that invents
an environment costs them a claim they will have to withdraw — and that failure
does not announce itself, because a spurious conditioned rule is formatted
exactly like a real one, carries a count and a confidence interval, and reads as
a finding.

Regenerate with `scripts/restraint.py`.

## `chance.tsv` — two lects with no history between them

Eighty concepts, both wordlists drawn independently from one inventory and one
set of word shapes. There is no relationship to find, and accidental segment
matches are frequent by construction.

Drawing both from the same inventory is the hard version rather than the easy
one. Languages of a region share phonotactics whether or not they share an
ancestor — that is what an areal feature *is* — and a method that only rejects
unrelatedness when the inventories differ has not rejected unrelatedness.

Trained with 12 shuffles after correcting the multinomial parameter count, it
reports:

```
cost/segment      -0.333      shuffled  -0.331 ± 0.022     z = -0.1
unconditioned     79          shuffled  80.0
conditioned       0           shuffled   5.8
rules above noise 0 of 0
```

Two things to read off it, and they point in opposite directions.

**The corpus-level comparison works.** `z = -0.12` says this wordlist aligns no
better than its own shuffles, against `-33` on the fixture next door that has a
real change in it. That is evidence about structure in these supplied pairings,
not a genealogical verdict.

**Seventy-nine accidental unconditioned correspondences are still published.**
The class count is therefore still not evidence by itself. The corrected
`(K−1) ln(n)` charge publishes no conditioned environment on top of them.

**The shuffled searches still find environments.** They train with the search
charge disabled so their maximum margin can calibrate the real run. Their mean
of 5.8 conditioned classes is the remaining demonstration that an adaptive,
unpriced search finds structure in noise.

## `diffusion.tsv` — a real change, spread over the lexicon

Proto /p/ answers daughter /f/ in half the words and stays /p/ in the other
half, and which half a word falls in is not a phonological fact about it. The
two groups are balanced across every environment the search can name: no onset,
preceding vowel or following vowel is more than 12% fuller of one than of the
other.

This is what lexical diffusion looks like from outside, and it is also what an
unfinished change, a dialect mixture and a half-completed analogy look like.
The right answer is two correspondences for one proto segment — p ~ p and
p ~ f — and no environment at all.

```
cost/segment      -1.509                                   z = -33.3
unconditioned     14          shuffled  73.1
conditioned       0           shuffled  63.8
```

Fourteen classes against a shuffled seventy-three, and not one environment.
Restraint here is not doubt about the data: the corpus is thirty-three standard
deviations from chance. It is the search declining to explain a split that has
no phonological explanation.

An earlier draft of this fixture had /p/ among the onsets as well as in the
medial slot, and regulae reported `f ~ p / pre[vowel:+]` — perfectly true,
because every medial /p/ has a vowel before it and every initial one does not,
and perfectly irrelevant to diffusion. The fixture was measuring the confound.

## `stratum.tsv` — two correspondence sets in one pair

An inherited layer that ran a spirantising shift over the whole voiceless
series — p~f, t~s, k~x — and a borrowed layer that left all three alone. Which
layer a word belongs to is a fact about its history and not about its shape, so
the two layers occupy the same environments.

A whole *set* of correspondences splitting at once is what makes a stratum
visible to a comparativist, and it is why this is not `diffusion` with more
segments. English has *father* beside *paternal* and *three* beside *triple*,
and the second member of each pair is not an exception to Grimm's Law but a
word that was not in the language when Grimm's Law ran. Japanese, Persian,
Tagalog, Swahili and Vietnamese all have layers like this, and telling one from
a conditioned split is ordinary daily work.

```
cost/segment      -1.621                                   z = -34.2
unconditioned     13          shuffled  47.0
conditioned       0           shuffled  82.7
```

Six correspondences where a comparativist would draw six, and no environment.
Neither layer is an error; naming an environment for either would be.

What regulae does not do is say *which* layer is inherited. Nothing in the
distribution can: that judgement needs the donor language, the semantic fields
involved and the dates, and it is the linguist's.

## `contact.tsv` — real correspondences, no common ancestor

Two unrelated lects, half of one's vocabulary borrowed from the other. Thirty
of the sixty concepts are loans adapted through a regular substitution — the
donor's `f`, `θ`, `x` and `z` have no place in the borrower's inventory and
come out as `p`, `t`, `k` and `s` — and the other thirty are native on both
sides and share nothing.

This is an areal relationship, and it is the commonest way a long-range
comparison goes wrong. Japanese and Chinese are the textbook case: half the
lexicon in systematic correspondence and no common ancestor. The Balkans,
mainland Southeast Asia, South Asia, the Pacific Northwest and much of
Australia contain pairs like it.

**There is no restraint available here, which is why the fixture is worth
having.** The correspondences are real, they are regular, and regulae reports
them — correctly. The corpus sits fourteen standard deviations below its own
shuffles, and that is a true statement about the data and not a statement
about descent. Nothing in the distribution of segments distinguishes
inheritance from borrowing; that judgement needs the semantic fields involved,
the direction of cultural flow and the dates, and it belongs to the linguist.
A tool that claimed otherwise would be claiming to have solved the problem the
field has been arguing about since Trubetzkoy.

What the corpus does leave is a signature:

```
worst-aligning 30 sets:  all 30 native
best-aligning  30 sets:  all 30 loans
```

No interleaving at all, and the model reports it without being asked to rank
anything:

```
  the sets fall in two groups, 50% of them in the worse-aligning
  one, 6.3 standard deviations apart.
```

Read the share with the separation. A mistaken judgement or two leaves a short
tail — five sets out of forty-five in `diagnostics/contaminated`, which comes
out at 7.2 standard deviations with **11%** on the worse side. Half the corpus
ranking apart is a different object.

And the statistic is not a borrowing test, which this directory can demonstrate
rather than assert: `stratum` scores 23.0 and `diffusion` 31.5, both at 50%,
and in neither is anything borrowed at all — half the words underwent a change
and half did not, so half align one way and half the other. What a high
separation says is that the corpus is not one thing. Which of the four reasons
it is — a borrowed layer, a block of bad judgements, two sources, or an
unfinished change — nothing in the distribution of segments can say.

## `merger_gap.tsv` — a gap in the proto lexicon, read as an environment

Forty sets. Proto \*o and \*a both give daughter /a/, and \*k gives x. All three
are unconditioned; there is no conditioned sound law anywhere in the corpus.

The accident is that \*o occurs only before a labial. Not because anything
conditioned it — because the sixteen words that happen to carry \*o happen to be
those words. Real lexicons are full of gaps like this, and a proto segment with
a skewed distribution is the normal case rather than a contrived one.

So among daughter /a/ before a labial, the proto source is \*o. That is true,
and a reconstructor wants it. It is not a sound change, and regulae publishes
it as one:

```
count=16 vs 21 as #0   daughter:a ~ proto:o   [daughter fol[labial:+]]
   dBIC=−40.4   margin=7.13   STANDS   predictive=confirmed gain=+0.705
```

**Every safeguard agrees with it, and none of them is malfunctioning.** It
clears its pivot's context-permuted null, because it is not noise — permuting
the environment cannot dissolve the fact that /o/ occurs only before labials,
which is a real distributional regularity even though it is ancestry and not a
change. It carries sixteen observations, well above the evidence floor. Held-out
prediction confirms it, because a retrodiction generalises exactly as well as a
law does. `environment_alternatives` reads 0, because no rival feature carves
the split differently. The rule is real. It is answering a different question
from the one the table's format implies, and no null keyed to the distribution
can catch that — only direction typing can.

The cause is in `model_context.c`: `target_side` means both *the environment
was read from the target form* and *the target grapheme is the pivot*, so a
target-side row states P(source | target) — which reflex descends from what —
while a source-side row states P(target | source), a conditioned change. Both
land in one table in one format, and nothing on the row says which.

Note that historical direction cannot be the discriminator. The pairwise source
lect is whichever sorts first, so `grimm` and `verner` both train
daughter-to-ancestor; regulae does not know which lect is ancestral and must
not start guessing. The discriminator is which conditional the environment
informs. Here P(daughter | proto \*o) is degenerate before any environment is
applied — \*o gives /a/ and nothing else — so there is no uncertainty for an
environment to reduce, and a row conditioning that direction cannot be a change
whatever it scores.

`test_a_lexical_gap_in_the_source_is_published_as_conditioning` asserts the
count that is published rather than the count that is right, and says so. M8's
direction typing is what should change it.

## `noise_unconditioned` — a real relationship, and nothing to condition

Every fixture above either commits no conditioned rule (`chance`, `diffusion`,
`stratum`) or commits one that is wrong for a reason no distribution can catch
(`merger_gap`). This one commits several and asks a different question: not
whether the search stays quiet, but whether the *standing verdict* declines
what the search does say.

280 cognate sets, one spirantising map with no conditioning anywhere — every
proto segment has exactly one reflex — and then one segment in twelve replaced
by a token that is not a regular reflex at all: the loans, misjudged cognates
and transcription slips a real wordlist carries. The corpus aligns about a
hundred standard deviations better than its own shuffles, so it is unmistakably
one language pair. The right answer is still zero conditioned rules.

What the search commits instead is a handful of rules, each resting on the three
or four words where a noise token lined up with a neighbour — and one on seven,
above the floor's neighbourhood. This is the shape `docs/m6_evaluation.md`'s
finding is about: on data with a real relationship and no conditioning, the
**pairing shuffle certified every one of these as `STANDS`, and did it more the
larger the corpus grew**, because the shuffle destroys the correspondences and
so measures a bar that does not rise with the evidence.

```
cost/segment      -1.7 (illustrative)                      z = -110
conditioned       6           of which above their null    0
```

Two things make the verdict decline them. Each rule is cut against its own
**pivot's context-permuted null** — the same bucket with its environments
shuffled against their outcomes, correspondences left intact — so the bar a rule
must clear is what the adaptive search reaches on that pivot with nothing to
find. And beneath that sits an **eight-example floor**: below it a clean real
change and a thin accident both leave the permutation rarely reaching their
margin, the first because its signal is destroyed and the second because there
is too little to resample, so the count is what separates them. The count-seven
rule clears the floor and is declined by the permutation; the rest are declined
by the floor.

This is the half of restraint the standing verdict exists for, and `chance`
cannot test it because `chance` commits nothing to decline.
`test_an_unconditioned_relationship_commits_but_nothing_stands` asserts the
split: the search commits, and not one of them stands.

## `sparse_008` … `sparse_128` — the evidence floor

The same conditioned change in each — /p/ answers /f/ before a front vowel, the
graded ladder's rung 1 — at 8, 16, 32, 64 and 128 cognate sets. Each rung is a
prefix of the one above, and half of every rung shows the change, so a rung
differs from its neighbour in size and in nothing else.

This is the number a field linguist with thirty cognates actually wants, and it
is not answerable from any single fixture: a corpus that yields nothing has
either too little data or no pattern in it, and only the ladder says which.

| sets | rules found | the intended rule | above its pivot's null |
| ---: | ---: | --- | ---: |
| 8 | 0 | — | 0 of 0 |
| 16 | 2 | found, as decision **#1** | 1 of 2 |
| 32 | 2 | found, as decision #0 | 2 of 2 |
| 64 | 2 | found, as decision #0 | 2 of 2 |
| 128 | 2 | found, as decision #0 | 2 of 2 |

Measured with 12 shuffles, each rule cut against its own pivot's
context-permuted null and the eight-example floor.

**Below the floor the search stays quiet rather than guessing.** Eight sets,
four of them showing a perfectly regular change, and nothing is committed. That
is the half of this that matters most: silence on a small corpus is not
evidence of absence, and it is also not the tool inventing something to say.

**Sixteen sets is enough to find the intended association, and now to tell it
from its correlate.** The intended `daughter:f ~ proto:p` before a front vowel
clears its pivot's null; the retention side `daughter:p ~ proto:p`, committed
in the same decision, does not — it is a covariate, not a change, and the
per-pivot null declines it where the old pairing shuffle passed both. What the
standing verdict still cannot do is choose one historical interpretation among
several stable correlates; it separates selected structure from noise, not a
cause from a co-occurrence.

Eight examples of a change and eight counterexamples, then. Below that, run
the baseline and expect to be told nothing; above it, expect the search to be
right about which rule is which but not yet about which came first.

## `arity_2` … `arity_5` — one change, at four lect counts

Four projections of one five-lect corpus onto its first N lects. Lect `a` keeps
/k/; every other lect backs it to /q/ before a back vowel and keeps /k/ before a
front one, which is Turkic velar-uvular allophony and the commonest shape a
multi-lect conditioning search is asked to find. The change is deterministic and
exceptionless in every rung, and a rung differs from the one below it in how
many languages attest it and in nothing else — the discipline `sparse_*` applies
to corpus size, applied to arity.

Two nuisances are in the corpus because real wordlists have them, and neither
touches the rule or correlates with the vowel that conditions it. A one-in-
sixteen chance that a segment in one lect is written as some other segment — a
loan, a misjudged cognate, a transcription variant — gives the correspondence
table the tail of one-off reflexes every real table has. A one-in-four chance
that a lect is missing from a set gives the ragged coverage every real wordlist
has.

The right answer is the same at every rung, and the evidence for it only grows.
What comes out instead:

| rung | sets | classes | conditioned | rows stating /k ~ q/ | largest such row |
| --- | ---: | ---: | ---: | ---: | ---: |
| `arity_2` | 126 | 55 | 3 | 2 | 54 |
| `arity_3` | 146 | 142 | 7 | 3 | 7 |
| `arity_4` | 157 | 235 | 7 | 2 | 6 |
| `arity_5` | 158 | 293 | **0** | 0 | — |

**Adding a language subtracts evidence.** The same rule, attested by one more
lect each time, is stated across thinner and thinner rows until the search stops
committing anything.

The cause is in `multilect_classes.c`, and it is one mechanism with two
symptoms. A class-level split is priced by the number of distinct *sister
tuples* in the pivot bucket, and a sister tuple is the full list of
(lect, grapheme) pairs — so it counts which lects a cognate set happened to
cover, and any one-off reflex in any one sister, alongside the correspondence
itself. Corrected BIC charges `K−1` parameters for a split. `K` therefore grows
with the sample rather than with the structure, while a binary environment can
only ever buy about one bit per observation, so the charge outruns anything the
evidence can pay. The same tuple identity is what publishes one correspondence
across several rows.

This is not the fragmentation hypothesis the roadmap tested and refuted. That
one aliased each partial tuple to its *unique widest compatible parent*, which
cannot reach the tail: on a four-lect Turkic pivot with nineteen sister keys,
two have a unique widest parent, ten are compatible with several and seven with
none, so the alias moves `K` from 19 to 17. The keys that dominate the charge
are exactly the ones no parent can absorb.

`test_adding_a_lect_costs_the_conditioned_rule` asserts the counts that are
published rather than the counts that are right, and says which is which.

## `distant_003` … `distant_008` — the floor for a trigger one syllable away

`sparse_*` measures the evidence floor for a change conditioned by the immediate
neighbour. Umlaut, vowel harmony, Verner's Law and dissimilation at a distance
are not that shape, and the search prices and gates those candidates
separately. This ladder asks them the same question, so the two numbers can be
read side by side. Proto /u/ answers daughter /y/ when the next syllable holds
/i/ and stays /u/ when it holds /a/.

| pairs a side | rules found | the intended rule | dBIC |
| ---: | ---: | --- | ---: |
| 3 | 0 | — | — |
| 4 | 0 | — | — |
| 5 | 1 | found | −7.2 |
| 8 | 1 | found | −14.4 |

**The step is a gate, not a floor the evidence climbs.**
`long_range_min_split_observations` is 5 against 3 for an immediate neighbour,
and it is applied per side. At four a side the best split in the bucket has a
likelihood gain of exactly zero — no long-range predicate survives the gate to
be scored at all. At five a side the same predicate commits at −7.2, which is a
comfortable margin rather than a marginal pass. Adding one minimal pair to
`soundlaws/opaque_umlaut.tsv` is enough to make its `oː ~ øː` environment
appear, for the same reason.

Worth knowing next to the eight-and-eight the guide quotes, which is measured on
`sparse_*` and so on an immediate neighbour. A change over a natural class is
divided by the size of that class before the search sees it, so a distance-
conditioned change over four vowel qualities needs forty examples before any one
of its correspondences can be conditioned at all.

## The fixture in `soundlaws/` that marks the surface/historical boundary

`final_devoicing.tsv` — German *Rad/Räder*, *Tag/Tage*, *Kind/Kinder* against
*Wort/Worte*, *Blut/Blutes*, *Licht/Lichter*. Word-final /t/ has two sources
and nothing in the citation form says which, so the right answer is two
historical correspondences with no causal environment. regulae commits four
conditioned surface associations, all correlates of the alternation rather than
causes of it, and after the `K−1` correction all four exceed the shuffled
search.

This no longer serves as a restraint success. It serves as the sharper limit:
search standing can distinguish an association from adaptive-search noise, but
not a surface correlate from a historical cause. The neutralisation analysis
still requires internal reconstruction.

## Adding one

A restraint fixture earns its place by having a wrong answer that is *tempting*.
Before adding one, write down the environment a careless search would commit
and check that the corpus makes it available; if no plausible spurious rule
exists, the fixture cannot fail and belongs in the documentation instead.

The mirror of the rule in `soundlaws/README.md`, and the same rule.
