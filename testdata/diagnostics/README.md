# Data-quality fixtures

Corpora that are wrong in the ways real corpora are wrong, and what regulae
says about them. `tests/c/test_diagnostics.c` asserts each one.

`soundlaws/` asks whether a change can be found. `restraint/` asks whether a
non-change can be declined. Both assume the wordlist is right, and no wordlist
is. Cognate judgements are made by people and some are mistaken; two sources
transcribing one language do not agree with each other; a compound is cognate
in one element and not the other. None of that is a failure of the method, all
of it changes what the method reports, and noticing it is most of what a
comparativist's week is spent on.

What a tool owes its user is not immunity to bad data — there is no such
thing — but a way of finding out. These fixtures measure how much of that
regulae provides, **including where the answer is none**.

Regenerate with `scripts/diagnostics.py`.

## `contaminated.tsv` — five judgements that are wrong

Forty sets under a regular correspondence set (p~f, t~θ, k~x, s~h, l~r, m~m,
n~n) with five non-cognate pairs mixed in. The five are two unrelated words
filed under one identifier, which is what a mistaken judgement actually is, and
they carry no confidence column: a linguist who knew which five were wrong
would have taken them out.

Ranked by how badly each set aligns under the trained model — what
`regulae outliers --model` prints:

```
bad4    3.84  ┐
bad3    3.82  │
bad2    2.05  ├ the five
bad0    1.67  │
bad1    1.46  ┘
w037   -0.06  ← the worst of the forty good ones
w022   -0.06
```

All five first, and the gap between the fifth and the sixth is wider than the
whole spread of the forty. So the ranking is not just ordered but *separated*,
and a reader who stops at the gap stops in the right place. That is what makes
`regulae outliers` a way to re-read a wordlist rather than a column of numbers.

**What this does not test.** The five bad daughters share nothing with their
protos. A wrong judgement between two words that happen to resemble each other
is not findable this way, and is not findable by any distributional method
whatever — which is the reason cognacy is a judgement and not a measurement.
An earlier draft built the bad daughters by displacing the proto's own reflexes
by one segment, and two of the five ranked below the median; correctly, since a
scrambled right answer is mostly a right answer. The fixture was measuring its
own generator.

## `partial.tsv` — cognate in one half

Twenty compounds whose first element is cognate and whose second is a different
word entirely, against twenty simplex sets cognate throughout. This is ordinary
in a wordlist gathered by concept rather than by etymon: both languages name
the thing with a compound and they do not use the same second element.

A set that is half right is not a set that is wrong, and dropping it loses the
half that is good. Both halves of what the corpus needs hold: the regular
correspondences still come out of the cognate elements, and **all twenty
compounds rank in the worse half** of the outlier list, so the sets to look at
again are the ones at the top.

The field's answer to this is partial cognate annotation — cognacy assigned per
morpheme rather than per word — and regulae has no way to express it. The
ranking is what it has instead, and it is enough to find them, not enough to
use the good half and ignore the bad one.

## `drift.tsv` — one language, two conventions

The `narrow` lect writes what `broad` writes as single graphemes the way a
different source would: the affricate as a stop plus a fricative, aspiration as
a following /h/, a long vowel as two vowels. Every form is the same word. There
is no sound change in this corpus at all.

What comes out:

```
broad:tʃ ~ narrow:ʃ    14
broad:tʰ ~ narrow:t    12
broad:iː ~ narrow:i    14
broad:kʰ ~ narrow:k     9
broad:aː ~ narrow:a     8
```

A reader takes that for three well-attested sound changes — deaffrication, loss
of aspiration, loss of vowel length — and writes them up. None of them
happened.

**Nothing in the model catches it, and nothing in the model can.**

* The fit is excellent: `z = -26.7` against the corpus's own shuffles.
* The shuffled baseline does not catch it and **cannot**. A baseline separates
  a pattern from chance, and this pattern is not chance — it is perfectly
  systematic, which is what a sound law is. Both conditioned rules the drift
  invents stand above the noise floor with room to spare.
* `cost_split_separation` does not catch it either: the corpus is one
  population, because every set is the same word twice.

So the question had to be asked somewhere else, and about the *writing* rather
than about the sounds. `regulae check` runs it:

```
drift	5
DRIFT	broad	narrow	iː	i i	14/14
DRIFT	broad	narrow	tʃ	t ʃ	13/13
DRIFT	broad	narrow	tʰ	t h	12/12
DRIFT	broad	narrow	kʰ	k h	9/9
DRIFT	broad	narrow	aː	a a	7/7
```

Take a grapheme one lect uses and the other never does, work out what the other
lect would have to write instead — segment it, unpick a modifier letter, undo a
length mark — and go and see whether it writes it. That last step is what keeps
this from being a resemblance heuristic: two different languages, one of which
lost its affricates, have exactly the inventory asymmetry that drift has, and
what they do not have is the *same cognate sets* showing the pieces in the same
order.

Which is why the row ends in a ratio rather than a verdict. Here every one is
1.0, because every word is the same word. On Kessler's Latin/French wordlist
the single row it produces — Latin `kʷ` against French `k w` — sits at 1 of 5,
which is *qu* → /k/ and not a convention. No corpus in this repository other
than this fixture reports a row at all, and that includes `restraint/chance`,
whose two lects have no history between them, and `restraint/contact`, where
the borrower's inventory is a subset of the donor's by construction — the two
shapes a similarity heuristic would fire on.

It reports and never refuses. A corpus can honestly hold one language with
affricates and one without; only the person who assembled it can tell that from
two sources disagreeing.

It is worth being plain about how common this is. Lexibank exists in part
because it was not solvable dataset by dataset: the same language in two
datasets is transcribed by two editors following two traditions, and CLTS is
the field's answer. A user who assembles a corpus from a published dataset and
their own fieldwork has done exactly what this fixture does.

## Adding one

A diagnostics fixture earns its place by breaking the corpus in a way somebody
actually breaks corpora, and by having a definite answer about whether regulae
notices. "Notices" and "does not notice" are both results; a fixture whose
answer is "it depends" is measuring something else.
