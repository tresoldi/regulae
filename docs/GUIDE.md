# regulae guide

This guide is the source for the in-page walkthrough; `scripts/guide.py`
generates `web/guide-content.js` from it. Each `##` heading is a step, and a
step may name a corpus the page loads into the editor.

## What regulae does, and what it does not

Give regulae a set of cognates across two or more lects and it returns the
systematic sound correspondences between them: which segment in one lect
answers to which segment in another, how often, in which cognates, and under
which conditions.

That is the whole of it. regulae does not decide which words are cognate, does
not reconstruct proto-forms, does not build family trees, does not date
anything, and does not claim a direction of change. Those are separate
questions belonging to a later stage of historical inference, and a tool that
answered them silently while doing this job would make its output impossible to
audit.

The notation follows from that. A correspondence is written
`latin:p ~ spanish:f`, never `p → f`. The tilde says these two segments answer
to each other in this data. The arrow would say one became the other, which is
a historical claim regulae has no basis for and does not make. If you want to
read direction into what you see, that is a judgement you are adding, and it
should be visible as yours.

What regulae gives you is the synchronic structure. What that structure means
historically is your problem, and it should be.

## Your first run

Cognate data is written one row per meaning, one column per lect, with whole
words. The first column names the cognate set; the rest are lects, named by
their column headers.

<!-- example: experiments/finnish_estonian/cognates.tsv -->

```
gloss   finnish   estonian   finnish_breaks   estonian_breaks
hand    kæsi      kæsi       -                -
eye     silmæ     silm       -                -
water   vesi      vesi       -                -
fish    kɑlɑ      kɑlɑ       -                -
```

That is the head of the corpus this step loads. The `_breaks` columns are
optional and carry morpheme boundaries; `-` means none, and you can leave them
out entirely.

Words are split into segments through merkmal, so multi-character graphemes
stay whole: `pʰ`, `t͡ʃ`, `kʷ` and a vowel carrying a combining diacritic are
each one segment, where splitting on characters would break every one of them.
You do not need to segment anything yourself.

The tie bar is what marks an affricate as a single segment. `t͡ʃ` is one
segment; untied `tʃ` is two, because that is what the transcription says. This
matters in both directions: writing the tie bar where you mean one segment is
the way to be understood, and leaving it off where you mean a sequence is
equally respected.

Press Run. What comes back is a list of correspondence classes, each one a
segment from every participating lect, with how much evidence supports it and
which cognates that evidence came from.

## Reading a correspondence class

A class is one row per lect, plus a count and the cognates behind it.

<!-- example: experiments/latin_spanish/cognates.tsv -->

The count is not a number of cognates. It is the summed evidence weight across
every aligned position where the correspondence held, so a cognate contributing
the same correspondence twice contributes twice, and a cognate marked with low
confidence contributes proportionally less. Selecting a class highlights every
position realising it in the alignments below, which is the fastest way to see
whether a correspondence rests on genuinely independent evidence or on the same
root appearing five times.

Classes are ordered by count, so what you see first is what the data supports
best. A class resting on one or two cognates is not necessarily wrong, but it
is a different kind of object from one resting on forty, and the ordering keeps
that distinction in front of you rather than in a footnote.

## Conditioned environments

Some correspondences hold everywhere. Others hold only in particular
environments, and regulae looks for those rather than making you spot them.

<!-- example: experiments/georgian_svan/cognates.tsv -->

The search is a greedy split: for each segment with more than one outcome, it
tries conditioning the choice on the surrounding phonology and keeps a split
only when it pays for itself under the configured categorical criterion. The
default is corrected BIC; exact multinomial NML and a symmetric-Dirichlet
marginal likelihood are experimental alternatives. That gate matters. It
is easy to explain away every exception by adding enough conditions, and the
criterion is what stops the tool doing that: a split has to buy more in
explained variation than it costs in added complexity, or it is discarded.

A conditioned class shows its environment beside the lect it constrains, read
as a filter on where that correspondence applies:

- `following: vowel:+` — holds when a vowel follows
- `preceding: voiceless:+` — holds after a voiceless segment
- `position: initial` — holds word-initially
- `same_syllable: close:+` — holds when a close segment shares the syllable
- `preceding@2: front:+` — holds when a front segment sits two positions back

The last two reach beyond the immediate neighbours, which is what catches
umlaut, vowel harmony, and other conditioning that operates at a distance.

An unconditioned class for the same segment usually remains alongside the
conditioned ones. That is not a contradiction: the conditioned entry is the
more specific statement, and the unconditioned one is what applies everywhere
else.

### Reading a rule against its complement

A conditioning claim is a comparison. "Latin `r` answers old Latin `s` between
vowels, count 14" is not a claim until you know what `s` does when it is *not*
between vowels. So every conditioned class reports that too:

```
count=14 elsewhere=0  cov=0.74 dBIC=-11.5   latin:r ~ old_latin:s  [between vowels]
count=6  elsewhere=26 cov=0.13 dBIC=-28.2   latin:s ~ old_latin:s  [before a vowel]
```

The first is the clean surface association expected from the rhotacism
challenge: fourteen times in the environment, never outside it. It is still
not, by itself, a historical event claim. The second does not support its stated
conditioning: the same correspondence occurs twenty-six times outside. Both
were committed by the same split gate, and before `elsewhere` was printed they
differed only in a count you had nothing to weigh against.

`dBIC` is the default scorer's value: how much better the model got, charged
for the `K−1` outcome parameters and the distinct candidate partitions searched.
More negative is a stronger split. Alternative runs label this `dNML` or
`dDir`; machine output publishes `score_kind` and `delta_score`. None is a
p-value. The shuffled baseline separately measures what the adaptive search
reaches after cross-lect pairings are broken.

## More than two lects

With three or more lects, regulae aligns each pair, then reconciles those
pairwise alignments into classes binding every lect at once.

<!-- example: testdata/corpora/real_romance_4lect.tsv -->

Reconciliation works over positions: if the Latin position 2 aligns with the
Spanish position 2, and that Spanish position aligns with the Italian position
3, all three are the same correspondence. Where the pairwise alignments
disagree about a position, the group is dropped rather than forced, so a class
you see is one every pair agreed on.

Cognate sets do not need every lect. A set covering three of four lects
contributes to the pairs it can, and the classes it supports simply bind fewer
lects. The lect count is shown per class for that reason.

This example is in the long format rather than the wide one, with a row per
lect and pre-segmented forms. Both formats reach the same place.

## Is there anything here at all?

Ask this before reading a single correspondence, because the model will produce
some either way.

A trained model reports `cost/segment`: how well, on average, the corpus aligns
under the model it produced. On its own the number means little; its scale
depends on the corpus. So compare it against the same corpus with the answer
taken out. `--permutations 20` retrains twenty times on a version of your data
where every wordlist is intact and every cognate set is the same size, but
which form pairs with which has been shuffled. There are no correspondences
left in that. Whatever the model finds is what the method finds in nothing:

```
cost/segment:        -1.8164 over 31 sets
shuffled baseline:   -0.5791 +/- 0.0837 over 10 shuffles, z = -14.8
  the same shuffles give 58.1 unconditioned and 11.9 conditioned classes
```

Two things to read here. The corpus aligns about fifteen standard deviations
better than its own noise: there is a relationship, and it is not close. And
the shuffled data produced **more** classes than the real data: 58 against 20,
12 against 3.

That second line is the one to take to heart. **The number of classes is not
evidence of relatedness.** An under-charged search over a large inventory of
possible environments finds more of them in noise than in signal. The current
categorical charge makes the unrelated restraint corpus publish no conditioned
classes, while its unpriced shuffled searches still find several. The fit
statistic measures whether the supplied pairings contain structure beyond that
comparison, and it costs one training run per shuffle.

This is a statement about the corpus, not about any one rule. It answers
"is there a relationship here", which has to be answered first.

## Does *this* rule stand?

Asking whether a corpus has a relationship and whether one rule's environment is
real are different questions, and they need different nulls. The pairing shuffle
above answers the first: it breaks the correspondences, so what its search
reaches is what the method finds in a corpus with no cross-lect structure at
all. That is the wrong bar for a single rule, whose correspondences are not in
doubt; the question is only whether its *environment* is.

So each conditioned class is judged against a second null, printed on the same
run: its own pivot's **context-permuted** search. The bucket that produced the
rule is re-searched many times with its environments shuffled against their
outcomes and the correspondences left exactly as they are. A rule stands when
its margin beats that shuffled environment on 95% of draws (a permutation
p-value), so a lone overfit draw cannot sink a real rule, and a well-attested
environment the shuffled search reaches just as often cannot pass.

```
context-permuted null: each conditioned class is judged against its own
  pivot's environment shuffled against its outcome, correspondences held intact
verdict:             3 of 5 conditioned rules stand above their pivot's null
                     3 of 5 per-pair conditioned correspondences above the
                     pairing-shuffle level, counted per pair
```

and every rule then carries its own verdict:

```
  count=14 elsewhere=0  #0 STANDS cov=0.74 dBIC=-7.3 margin=1.26 ...
```

Beneath the p-value sits an **eight-example floor**, the same one the
`sparse_016`…`sparse_128` ladder measures: a class on fewer than eight
observations is `within-noise` whatever its margin. Below the floor a clean real
change and a thin accident both leave the permutation rarely reaching their
margin: the first because its signal is destroyed, the second because there is
too little to resample. No distribution separates them, and the honest
verdict is that the corpus is too small to say. That is why a fragmented law
often reads `within-noise` on every one of its pieces even when the change is
not in doubt: each fragment is below the floor. It is the predictive check
below, not the standing verdict, that tells a real thin rule from an accident.

`STANDS` means the margin cleared the rule's own pivot null and the floor;
`within-noise` means it did not. **Without `--permutations` there is no verdict
at all**: every rule reads `unmeasured`, which is not a pass. It says the
comparison was never made. Small corpora produce a lot of `within-noise`, and
the answer is usually more data rather than a different reading.

Two verdicts are printed because regulae works at two levels, and they are
counted differently, and, for now, against different nulls. The first counts
the multi-lect rules, the classes reconciled across every lect, each against
its pivot's context-permuted null. The second counts the conditioned
correspondences each *pair* of lects carries in its own model, still against the
pairing shuffle, and those are counted per pair, so a rule visible in every pair
of a four-lect corpus is six there and one above. They are not added together
for that reason.

**Read both.** On the Grassmann fixture the two multi-lect classes
(`greek:t ~ pie:tʰ` and `greek:k ~ pie:kʰ`) rest on six and five observations,
below the floor, so the multi-lect view is silent; the per-pair view, judged
against the shuffle, still shows the aspirate-dissimilation correspondences. A
corpus's strongest surface association can be in either view, and a small
fixture can carry it per-pair while its reconciled fragments fall below the
floor.

## Does it predict an unseen reflex?

The shuffle asks whether a rule stands above adaptive search noise in this
corpus. Prediction asks a different question: whether an association selected
from some histories helps on histories the search did not see.

Run it explicitly because it retrains once per fold and per lect orientation:

```sh
regulae train corpus.tsv --human --predictive-folds 5
```

Rows sharing a cognate id, `etymon_group` or `source_group` are kept together.
Within each training partition regulae rebuilds the feature vocabulary,
alignments and complete greedy decision list; the held-out partition only sees
that frozen model. The report compares conditioned prediction with identity,
inventory frequency, feature distance and the unconditioned correspondence
table, and publishes log loss, top-k coverage, calibration and abstention. With
three or more lects it also predicts each lect by pooling the others.

The predictive verdict is separate from `STANDS` and the split score:

- `confirmed` means conditioning lowered group-held-out log loss.
- `not_confirmed` means it did not. The rule may still describe the supplied
  corpus accurately.
- `descriptive_only` means there were too few independent histories, or the
  exact association was not rediscovered often enough, for confirmation.
- `unmeasured` means no predictive run was requested.

This is surface cloze prediction: the other lect's form and the surrounding
surface context are observed while the reflex is predicted. It is not a
proto-form reconstruction score and not a claim that regulae discovered a
sound law.

## How much data do I need?

About **eight examples of a change and eight counterexamples**, for a trigger
in the immediate neighbour.

A trigger a syllable away needs five examples a side, not three, and it is a gate
rather than a floor the evidence climbs: the `distant_003`…`distant_008` ladder
commits nothing at three or four a side and recovers the rule at five, and
lowering the long-range minimum to three lets a three-a-side split commit, so the
extra evidence the search charge asks for does not by itself account for the
step. The eight-and-eight figure does not transfer to a distance-conditioned
change; that ladder pins the one that does.

`testdata/restraint/sparse_008` … `sparse_128` is one conditioned change (/p/ answering /f/
before a front vowel) at 8, 16, 32, 64 and 128 cognate sets, each corpus a
prefix of the next so that the only thing that varies is size. At 8 sets
nothing at all is committed. At 16 the two sides of the intended contrast are
the only committed classes, and the intended change (/p/ answering /f/ before a
front vowel) clears its pivot's null while the retention side committed beside
it does not. From 32 upward both the change and its recovery are stable.

Two readings, and the second is the one people miss.

**Silence on a small corpus is not evidence of absence.** Below the floor the
search stays quiet rather than guessing, which is what you want it to do, and
it means a run that reports nothing on thirty cognates has told you about your
corpus and not about your languages.

**Discovery at the floor still needs a baseline.** Rules are listed in the
order they were decided and that order carries meaning: a later rule refines
what an earlier one left. At 16 sets the corrected model is already stable on
this clean fixture, but a real corpus may contain stronger correlated
predicates, and `--permutations` is what measures the search they create.

One more thing changes the arithmetic, and it catches people out. **A change
that applies to a whole class of segments is divided by the size of that class
before the search sees it.** Germanic i-umlaut is one change, and it arrives as
four correspondences (`a ~ e`, `uː ~ yː`, `u ~ y`, `oː ~ øː`), each of which
has to carry its own evidence. In `testdata/soundlaws/opaque_umlaut.tsv` only
the first crosses the floor, so only the first gets its environment, and the
other three are published as bare splits with the trigger standing right next
to them in the same words. Palatalisation, lenition, nasalisation and every
chain shift have this shape. If a change you know is there comes out
unconditioned, count how many segments it applies to before concluding
anything.

## Two correspondences, with or without a surface environment

Sometimes one proto segment answers two daughter segments and the search
commits no environment. That is an answer, not a failure, and there are at
least four different things it can mean:

- **A borrowed layer.** English *father* beside *paternal*: the second is not
  an exception to Grimm's Law, it is a word that was not in the language when
  Grimm's Law ran. The tell is that a whole *set* of correspondences splits at
  once, not just one.
- **A neutralisation.** German word-final /t/ has two sources and nothing in
  the citation form says which; the alternation in the inflected stem is the
  only evidence. The fixture contains stable lexical correlates that regulae
  reports as environments, but those associations are not causes. This is what
  internal reconstruction is for.
- **Lexical diffusion, or a change still in progress.** No environment because
  there is none: which words changed is a fact about the words.
- **An environment regulae cannot state.** Syllable weight is the clearest
  case; see `docs/capabilities.md` for the standing list of these.

Distinguishing them is your job, and the distributions will not do it. A
selected environment explains a surface distribution; it does not decide
which historical interpretation produced that distribution.

## What the answer cannot tell you

Three limits worth knowing before the output is quoted anywhere.

**Contact against inheritance.** A wordlist half borrowed from an unrelated
neighbour produces real, regular, well-supported correspondences over the
borrowed half and sits far below its own shuffled baseline. Japanese and
Chinese are the textbook case. Nothing in the distribution of segments
separates a loan stratum from an inherited one, so nothing in this output does
either.

There is a signature, and the report names it when it is strong:

```
  the sets fall in two groups, 50% of them in the worse-aligning
  one, 6.3 standard deviations apart.
```

Read the percentage with the separation. A high separation with a *small*
share (a tenth of the corpus) is a tail of sets that do not belong, which is
what a handful of bad cognate judgements looks like. A high separation at about
half is a corpus that is two populations. Which of the two it is, and why, the
distributions cannot say: a borrowed layer, a block of bad judgements, two
sources, and half a lexicon that underwent a change the other half did not all
produce it. It is a reason to ask a different question, not an answer.

**Transcription drift.** Two sources for one language that disagree about
where a segment ends (`tʃ` against `t ʃ`, `tʰ` against `t h`, `aː` against
`a a`) produce a family of clean correspondences that read as deaffrication,
loss of aspiration and loss of length. Nothing in the *model* catches it: every
grapheme is valid, the fit is excellent, and the shuffled baseline cannot help,
because a systematic difference is exactly what a sound law is.

`regulae check` asks the other question, about the writing rather than the
sounds, and prints a `DRIFT` line per suspect grapheme with the evidence
beside it:

```
DRIFT   broad   narrow   tʃ   t ʃ   13/13
```

Thirteen cognate sets where `broad` writes `tʃ`, and in all thirteen `narrow`
writes `t ʃ`. A ratio near 1 is a transcription difference; a low one is a
sound change — Latin `kʷ` against French `k w` comes out at 1 of 5, which is
*qu* → /k/. It reports and never refuses: a corpus can honestly hold one
language with affricates and one without, and only you can tell that from two
sources disagreeing.

**Which lect is the innovator.** A correspondence is symmetric. Deciding which
side changed is reconstruction, and regulae does not do reconstruction.

## Confidence and outliers

Rows are not always independent lexical histories. Eight paradigm cells can
repeat one etymon, and a transcription assembled from two publications can
carry a source-wide convention. Optional `etymon_group` and `source_group`
columns name those dependencies. regulae never guesses them from a shared id
prefix or similar forms.

Every class already carries a Wilson interval on the rate it applies at. With
bootstrap intervals enabled (through the API, or the page's *Bootstrap the
intervals* box), `AUTO` resamples etymon groups when supplied and otherwise
treats cognate sets as independent. The fit report prints that unit and its
effective count. Source-publication resampling
is explicit because one publication often contains the entire corpus; treating
it as one automatic draw would erase rather than quantify the evidence.

Not every cognate set deserves equal weight. A confidence column carries that:

<!-- example: experiments/contaminated_cognates_synthetic/cognates.tsv -->

Confidence is an evidence weight in `[0, 1]`. A set at `0.5` contributes half
as much to every count it touches; a set at `0.0` contributes nothing while
staying in the corpus, so it remains visible and auditable rather than being
quietly deleted. Where a corpus gives different confidences for different rows
of the same set, the lowest wins: a set is only as good as its weakest member.

Separately, regulae ranks cognate sets by how badly they align under the model
it just learned, as a z-score against the corpus mean. A set scoring far above
the rest is one whose forms do not behave like the rest of the data — a
borrowing, a misjudged cognate, a transcription problem, or a real irregularity
worth looking at.

This is a diagnostic, not a filter. Nothing is removed and no set is judged.
The ranking tells you where to look; whether what you find there is an error or
the most interesting thing in your data is not something a program can decide.

## What regulae cannot read yet

Some of the corpora in the example list will not load, and the honest reason is
worth stating plainly, because it is not what it looks like.

<!-- example: experiments/morph_boundary_synthetic/cognates.tsv -->

Run this one and regulae tells you `"+"` is CLDF/CLTS markup rather than a
transcribed sound. It is not refusing to handle morpheme boundaries — they are
implemented, and the arcaverborum format carries them today. The wide format
simply has no convention yet for a `+` written inside a word, so the marker
reaches the feature system as if it were a sound. The gap is in the input path.

The other blocked corpus is the same shape: `stress_conditioned_synthetic`
fails on `-`, a syllable separator, not on the stress mark, which is read fine.
Stress conditioning itself is implemented.

Tone used to be the example here, and it is worth saying what changed, because
it is the same lesson. Every tonal corpus in the example list was unreadable —
not because tone was missing, but because they wrote it as an ASCII `1`, which
is not tone notation and is not something a reader can safely guess at. Written
the way the field writes it, tone goes straight in: `ma³³` gives `m` and an `a`
carrying `³³`, whether the tone is bound to its nucleus or spelled as a token
of its own. A `<lect>_tone` column is the other way in, for corpora that record
tone categories rather than pitch: one value per segment, `-` for a segment you
are not annotating. Nothing in the engine changed; the corpora were rewritten.

[`docs/capabilities.md`](capabilities.md) lists all of this: what is
implemented, what the loaders can express, and which grapheme blocks each
corpus that will not load. It is generated by running the corpora, so it cannot
drift from what the tool actually does.

The distinction matters. A corpus that will not load says something about the
input path, not about the engine.

## Input formats

Four readers are available, and all of them arrive at the same model.

**Wide** is the default and the one used above: a cognate per row, a lect per
column, whole unsegmented words. A `<lect>_breaks` column supplies morpheme
boundary positions for that lect, and a `confidence` column weights the set.
A `<lect>_tone` column supplies tone per segment, and is recognised so it is
not mistaken for a lect. `<lect>_stress` does the same for stress.

Stress can also be written in the word, with the IPA marks: `ˈpater` puts the
accent on the first syllable, `paˈter` on the second. The mark stands before a
syllable and the accent is realised on its nucleus, so that is where it lands —
which is what makes "the vowel before this consonant was accented" a question
the model can answer. That question is Verner's Law.

**Long TSV** wants `cognate_id`, `lect_id` and `segments` columns, with
segments already space-separated, plus an optional `confidence`. Use it when
you have made the segmentation decisions yourself and want them respected.

```
cognate_id  lect_id   segments    confidence
father      latin     p a t e r   1.0
father      spanish   p a d r e   1.0
```

**GLED** reads that project's export, grouping by `COGSET` and taking segments
from the `IPA` column.

**Arcaverborum** reads merged Lexibank data, grouping by the first `Cognacy`
identifier and taking morpheme boundaries from `+` tokens in `Segments`.

Rows that leave a cell empty, or hold `-`, simply do not contribute that lect.

## Beyond the browser

This page runs the same C library as everything else, compiled to WebAssembly,
and the test suite asserts the two produce byte-identical output.

The command line takes the same corpora:

```
regulae train corpus.tsv --format wide      # machine-readable summary
regulae train corpus.tsv --human            # a readable report
regulae train corpus.tsv --json             # model, alignments and outliers
regulae outliers corpus.tsv --top-k 10
regulae align corpus.tsv --model
```

The JSON is a debug snapshot, labelled as such in the payload. It is
deliberately not an interchange format: the framework this belongs to requires
anything making historical claims to carry ensembles and uncertainty rather
than one best answer, and a single set of correspondences is not that.

To use the library directly, `include/regulae.h` is the contract. Training runs
through `rg_train_model`, results are read through borrowed accessors, and
every call that can fail returns a status. The
[README](https://github.com/tresoldi/regulae) has a worked example.
