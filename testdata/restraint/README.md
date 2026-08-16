# Restraint fixtures

Corpora that contain no conditioned sound law, and one ladder that says how
much evidence it takes before regulae can see one that is there.
`tests/c/test_restraint.c` asserts what regulae must **not** find in each.

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

## `sparse_008` … `sparse_128` — the evidence floor

The same conditioned change in each — /p/ answers /f/ before a front vowel, the
graded ladder's rung 1 — at 8, 16, 32, 64 and 128 cognate sets. Each rung is a
prefix of the one above, and half of every rung shows the change, so a rung
differs from its neighbour in size and in nothing else.

This is the number a field linguist with thirty cognates actually wants, and it
is not answerable from any single fixture: a corpus that yields nothing has
either too little data or no pattern in it, and only the ladder says which.

| sets | rules found | the intended rule | its margin | above noise |
| ---: | ---: | --- | ---: | ---: |
| 8 | 0 | — | — | 0 of 0 |
| 16 | 2 | found, as decision **#1** | 1.09 | 2 of 2 |
| 32 | 2 | found, as decision #0 | 1.12 | 2 of 2 |
| 64 | 2 | found, as decision #0 | 2.99 | 2 of 2 |
| 128 | 2 | found, as decision #0 | 7.38 | 2 of 2 |

Measured after correcting the multinomial parameter count, with 12 shuffles.

**Below the floor the search stays quiet rather than guessing.** Eight sets,
four of them showing a perfectly regular change, and nothing is committed. That
is the half of this that matters most: silence on a small corpus is not
evidence of absence, and it is also not the tool inventing something to say.

**Sixteen sets is enough to find the intended association, but not to identify
one unique description.** A correlated environment (`fol[open:+]`, margin
0.65) enters ahead of `fol[front:+]`, and both exceed this fixture's shuffled
search. Search standing separates selected structure from the null; it does not
choose a historical interpretation among stable correlates.

Eight examples of a change and eight counterexamples, then. Below that, run
the baseline and expect to be told nothing; above it, expect the search to be
right about which rule is which but not yet about which came first.

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
