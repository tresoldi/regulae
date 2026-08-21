# regulae guide

This guide is the source for the in-page walkthrough; `scripts/guide.py`
generates `web/guide-content.js` from it. Each `##` heading is a step, and a
step may name a corpus the page loads into the editor.

## Try it

<!-- example: experiments/finnish_estonian/cognates.tsv -->

Load this example and press Run. The corpus is a set of Finnish and Estonian
cognates — one row per meaning, one column per lect, with whole unsegmented
words. What comes back is a table of correspondence classes: which segment in
one lect answers to which segment in the other, how often, and in which
cognate sets.

That is what regulae does. It takes cognate data and returns the systematic
sound correspondences in it. Everything else on this page — the environments,
the statistics, the outlier ranking — is structure discovered inside that
table.

The words are split into segments through merkmal, so multi-character
graphemes stay whole: `pʰ`, `t͡ʃ`, `kʷ` and a vowel carrying a combining
diacritic are each one segment. You do not need to segment anything yourself.

## Reading the table

Click a row in the class table. Two things happen: the alignments below
filter to the cognate sets that realise that class, and the columns belonging
to it light up. That is the fastest way to see whether a correspondence
rests on genuinely independent evidence or on the same root appearing several
times.

The count column is not a number of cognates. It is the summed evidence
weight across every aligned position where the correspondence held, so a
cognate contributing the same correspondence twice contributes twice. The bar
beside it is a 95% confidence interval on the rate — how often, where the
class applies, it is the correspondence taken.

Click an alignment column to go the other way: the class it belongs to
lights up in the table above.

## A richer corpus

<!-- example: experiments/latin_spanish/cognates.tsv -->

Load the Latin/Spanish example and run it. There are more classes here, and
the table is sorted by count, so what you see first is what the data supports
best. Scroll down — classes resting on one or two cognates are still listed,
but they are a different kind of object from those resting on forty, and the
ordering keeps that distinction visible.

The filter at the top of the table narrows the view: type `p` to see every
class involving /p/, or `~ f` to find everything answering /f/. The chips
below it switch between conditioned and unconditioned classes, or show only
the ones where something changed.

## Conditioned correspondences

<!-- example: experiments/georgian_svan/cognates.tsv -->

Some correspondences hold everywhere. Others hold only in particular
phonological environments. Load this Georgian/Svan corpus, run it, and look
for rows that carry an environment beside the correspondence — something
like `following: vowel:+` or `position: initial`.

The search is a greedy split: for each segment with more than one outcome,
regulae tries conditioning the choice on the surrounding phonology and keeps
a split only when it pays for itself under the model's complexity criterion.
The default is corrected BIC. A split has to buy more in explained variation
than it costs in added complexity, or it is discarded. That gate matters —
it is easy to explain away every exception by adding enough conditions, and
the criterion is what stops the tool doing that.

The environment notation reads as a filter on where the correspondence
applies:

- `following: vowel:+` — holds when a vowel follows
- `preceding: voiceless:+` — holds after a voiceless segment
- `position: initial` — holds word-initially
- `same_syllable: close:+` — holds when a close segment shares the syllable
- `preceding@2: front:+` — holds when a front segment sits two positions back

The last two reach beyond the immediate neighbours, which is what catches
umlaut, vowel harmony, and other conditioning that operates at a distance.

### A rule and its complement

A conditioning claim is a comparison. "Latin `r` answers old Latin `s`
between vowels, count 14" is not a claim until you know what `s` does when
it is *not* between vowels. Every conditioned class shows its complement
inline — what the same segment does elsewhere. A row that reads "else
latin:s ~ old_latin:s" beside a conditioned `latin:r ~ old_latin:s` is
saying: the change happened in this environment, the retention happened
outside it.

Click the caret beside a conditioned class to open its evidence drawer.
The score there (`dBIC`) says how much better the model got from the split,
charged for the parameters and the partitions searched. More negative is a
stronger split. It is not a p-value.

### Rule notation

Under each environment is the same environment written a second time, in rule
notation:

```
latin r ~ old_latin s · latin — previous_syllable: syllable_shape:open …
r  ~  s  /  latin: σ⁻[open] _ ;  old_latin: V _ V
```

The first line is what the model states and is what to quote. The second is
the reading. `_` is this position, `/` is "where", `V` is a vowel, `#` a word
boundary, `+` a morpheme boundary, `…` any number of segments, `·` exactly
one, and `σ⁻` `σ⁼` `σ⁺` the previous, same and next syllable.

One thing in it is a claim rather than a convention. The standard notation is
`A > B / X _ Y` — directional, with the environment sitting, unstated, in the
ancestor's word. regulae has no ancestor, and its environment belongs to a
**named lect**: only a lect that still shows the conditioning contrast has an
environment to state, and which lect that is says what each lect preserved. So
the lect is named before its frame, and the label is dropped only when every
lect states the same one. Above, Latin conditions on an open preceding
syllable and old Latin on being between vowels, and those are two different
findings about two different words.

Some of what regulae can condition on has no notation anywhere in the
literature — the syllable-scoped predicates, the counted distances, the
morpheme index. Where none existed, regulae defines one and publishes the key:
see [docs/NOTATION.md](https://github.com/tresoldi/regulae/blob/master/docs/NOTATION.md).

## Events

The events pane groups classes that look like one change. Rows join along
whichever axis holds them together: one environment over several outcomes (a
change across a class of segments), one outcome over several environments (a
change stated as a decision list), or, where there is no environment, a shared
feature displacement — Grimm's `p ~ f`, `t ~ θ` and `k ~ x` agree on
stop→fricative and disagree on place, and that shared step is what groups them.
Click an event row: the member classes light up in the table, and the
alignments filter to the cognate sets behind them.

Each event states the environment **every** member of it conditions on, in its
own column. Only the shared part: the members were searched separately, so one
routinely carries a conjunct the others did not need, and on Verner's Law the
event states `following_stress[primary]` and drops the "before a vowel" that
one member also picked up. Where the members differ in environment — that is
what a decision-list grouping *is* — the column says so instead of going blank,
because a blank there would read as unconditioned. Where the corpus cannot tell the environment
from a rival, the row names the rivals rather than only counting them — "after
a sonorant, or equally: following [front:+]" is something you can go and check
against the wordlist, where a count is only a warning. A rival written with
"not" holds exactly where the stated environment does not, which is the same
split seen from the other side.

Two kinds, and they mean different things. "or equally" is a confound: that
predicate carves exactly the same words, and no more data collected this way
would separate the two. "also fits" is a near tie: it splits the words
*differently* and would still have been committed on its own, so the corpus did
prefer the stated environment — the number beside it is how strongly, against
the margin of the one that won. Latin rhotacism commits a syllable-shape
environment at 3.50 and reports `following[vowel:+]` at 2.14, which is the
intervocalic reading, in the running and beaten.

Only rivals that every member of an event shares are shown, so a grouping can
be pinned down where none of its rules is: each of lenition's three rules is
confusable with something on its own five words, and no rival survives all
three.

A change that lands on a single segment still appears, alone, with a member
count of one — vowel harmony, umlaut and rhotacism are each one class in these
corpora, and a pane that showed only groupings was blank on exactly the corpora
whose change is easiest to name. Only a conditioned class earns a row that way:
it was committed by a search, against a contrast, and carries the margin that
says how well it paid. An unconditioned correspondence standing alone is an
aggregate no search decided, and admitting those would reprint the
correspondence table.

So an empty pane now means the corpus conditioned nothing and nothing grouped —
and where the change was a reordering, look in the chunks pane instead, because
a swap is not a segment change. Retentions never appear: a row where nothing
changed is evidence about a split, and the member row's contrast link is where
to read it.

This is a proposal, not an assertion. Whether the grouped classes really
reflect a single historical event is a question about history, and regulae
does not answer questions about history. It groups what the surface
distribution groups, and leaves the interpretation to you.

## Gaps and outliers

Below the class table, two panels.

The **gaps** panel lists segments that answer to nothing: a segment present
in one lect and absent in the other, with how often that happens. Click a
gap row to see the alignments where it occurs.

The **residue** panel ranks every cognate set by how badly it aligns under
the model. A set scoring far above the rest is one whose forms do not
behave like the others — a borrowing, a misjudged cognate, a transcription
problem, or a real irregularity. Click a residue row to see the alignment.
This is a diagnostic, not a filter: nothing is removed and no set is judged.

## Is there a relationship here?

Ask this before reading any correspondence, because the model will produce
classes either way — even on unrelated data.

Check the **Shuffled baseline** box in the options before running. This
retrains the model on versions of your data where the cognate pairings have
been randomised. Whatever structure it finds in that is what the method finds
in nothing. If your corpus aligns about as well as the shuffled version,
the correspondences are noise.

**The number of classes is not evidence of relatedness.** The shuffled data
often produces *more* classes than the real data. An adaptive search over a
large inventory of possible environments finds more of them in noise than
in signal. The fit statistic measures whether your supplied pairings contain
structure beyond that comparison.

## Does a rule stand?

Whether a corpus has a relationship and whether one rule's environment is
real are different questions. The shuffled baseline answers the first.
For the second, each conditioned class is judged against its own pivot's
**context-permuted** search: the environments are shuffled against their
outcomes while the correspondences are held intact. A rule stands when its
margin beats that shuffled environment on 95% of draws.

Every conditioned row carries a verdict:

- `STANDS` — the margin cleared the rule's own null and the eight-example floor
- `within-noise` — it did not
- `unmeasured` — no permutation run was requested

Without the shuffled baseline, every rule reads `unmeasured`, which is not
a pass. It says the comparison was never made. Small corpora produce a lot
of `within-noise`, and the answer is usually more data rather than a
different reading.

The eight-example floor is there because below it a clean real change and a
thin accident look the same to any resampling test. A fragmented law often
reads `within-noise` on every one of its pieces even when the change is not
in doubt: each fragment is below the floor, and the honest verdict is that
the corpus is too small to say.

## Does it predict?

The shuffle asks whether a rule stands above adaptive search noise in this
corpus. Prediction asks whether an association selected from some histories
helps on histories the search did not see.

Check the **Cross-validation** box. Rows sharing a cognate id are kept
together. Within each training partition regulae rebuilds everything from
scratch; the held-out partition only sees the frozen model. The report
compares conditioned prediction with identity, inventory frequency, feature
distance and the unconditioned correspondence table.

The predictive verdict is separate from standing:

- `confirmed` — conditioning lowered group-held-out log loss
- `not_confirmed` — it did not; the rule may still describe the corpus accurately
- `descriptive_only` — too few independent histories for confirmation
- `unmeasured` — no predictive run was requested

This is surface cloze prediction: the other lect's form and the surrounding
context are observed while the reflex is predicted. It is not a proto-form
reconstruction score.

## How much data

About **eight examples of a change and eight counterexamples**, for a
trigger in the immediate neighbour.

A trigger a syllable away needs five examples a side, not three. The figure
does not transfer to a distance-conditioned change.

**Silence on a small corpus is not evidence of absence.** Below the floor
the search stays quiet rather than guessing, which is what you want it to
do. A run that reports nothing on thirty cognates has told you about your
corpus, not about your languages.

One thing catches people out. **A change that applies to a whole class of
segments is divided by the size of that class before the search sees it.**
Germanic i-umlaut is one change, but it arrives as four correspondences
(`a ~ e`, `uː ~ yː`, `u ~ y`, `oː ~ øː`), each of which has to carry
its own evidence. Palatalisation, lenition, nasalisation and every chain
shift have this shape. If a change you know is there comes out
unconditioned, count how many segments it applies to before concluding
anything.

## Two outcomes, no environment

Sometimes one proto segment answers two daughter segments and the search
commits no environment. That is an answer, not a failure, and there are at
least four different things it can mean:

- **A borrowed layer.** English *father* beside *paternal*: the second is
  not an exception to Grimm's Law, it is a word that was not in the
  language when Grimm's Law ran.
- **A neutralisation.** German word-final /t/ has two sources and nothing
  in the citation form says which.
- **Lexical diffusion.** No environment because there is none: which words
  changed is a fact about the words.
- **An environment regulae cannot state.** Syllable weight is the clearest
  case.

Distinguishing them is your job. A selected environment explains a surface
distribution; it does not decide which historical interpretation produced
that distribution.

## What the answer cannot tell you

Three limits worth knowing before the output is quoted anywhere.

**Contact against inheritance.** A wordlist half borrowed from an unrelated
neighbour produces real, regular, well-supported correspondences over the
borrowed half. Japanese and Chinese are the textbook case. Nothing in the
distribution of segments separates a loan stratum from an inherited one, so
nothing in this output does either.

There is a signature, and the report names it when it is strong: the
residue section shows whether the cognate sets fall into two groups, with
what share sits in the worse-aligning one and how far apart the two groups
are. A high separation at about half is a corpus that is two populations.
Which of the two it is, and why, the distributions cannot say.

**Transcription drift.** Two sources for one language that disagree about
where a segment ends (`tʃ` against `t ʃ`, `tʰ` against `t h`) produce
clean correspondences that read as deaffrication or loss of aspiration.
Nothing in the model catches it — a systematic difference is exactly what
a sound law is. The drift banner at the top of the results reports suspect
graphemes when it finds them.

**Which lect is the innovator.** A correspondence is symmetric. Deciding
which side changed is reconstruction, and regulae does not do
reconstruction.

## Confidence and outliers

<!-- example: experiments/contaminated_cognates_synthetic/cognates.tsv -->

Not every cognate set deserves equal weight. A `confidence` column in the
input carries that: a value in `[0, 1]` that weights how much a set
contributes to every count it touches. A set at `0.5` contributes half; a
set at `0.0` contributes nothing while staying visible and auditable.

Optional `etymon_group` and `source_group` columns name dependencies between
rows — eight paradigm cells repeating one etymon, or a transcription
assembled from two publications. regulae never guesses them.

With bootstrap intervals enabled, the resampling respects those groups.
The fit report prints the resampling unit and its effective count.

## More than two lects

<!-- example: testdata/corpora/real_romance_4lect.tsv -->

With three or more lects, regulae aligns each pair, then reconciles those
pairwise alignments into classes binding every lect at once. Where the
pairwise alignments disagree about a position, the class is dropped rather
than forced. Use the lect-pair selector above the alignments to narrow the
view to one pair at a time.

Cognate sets do not need every lect. A set covering three of four lects
contributes to the pairs it can.

This example is in the long format rather than the wide one, with a row per
lect and pre-segmented forms. Both formats reach the same place.

## Morpheme boundaries and tone

<!-- example: experiments/morph_boundary_synthetic/cognates.tsv -->

The wide format reads `+` inside a word as a morpheme boundary and `-` as a
syllable break: `pat+a` segments to four sounds with a boundary after the
third. A `<lect>_breaks` column, if present, takes precedence over inline
marks.

Tone is written the way the field writes it: `ma³³` gives `m` and an `a`
carrying `³³`, whether the tone is bound to its nucleus or spelled as a
separate token. Plain ASCII digits `0`–`5` are accepted too — they are
normalised to superscript Chao form during segmentation. A `<lect>_tone`
column is the other way in, for corpora that record tone categories rather
than pitch.

Stress works the same way: `ˈpater` puts the accent on the first syllable,
and the accent is realised on the nucleus, which is what makes "the vowel
before this consonant was accented" a question the model can answer.

## Input formats

Four readers are available, and all of them arrive at the same model.

**Wide** is the default: a cognate per row, a lect per column, whole
unsegmented words. A `<lect>_breaks` column supplies morpheme boundaries,
a `<lect>_tone` column supplies tone per segment, and a `confidence` column
weights the set.

**Long TSV** wants `cognate_id`, `lect_id` and `segments` columns, with
segments already space-separated. Use it when you have made the segmentation
decisions yourself and want them respected.

```
cognate_id  lect_id   segments    confidence
father      latin     p a t e r   1.0
father      spanish   p a d r e   1.0
```

**GLED** reads that project's export, grouping by `COGSET` and taking
segments from the `IPA` column.

**Arcaverborum** reads merged Lexibank data, grouping by the first
`Cognacy` identifier.

## What this is and is not

regulae finds the systematic sound correspondences in a set of cognates.
It writes `latin:p ~ spanish:f`, never `p → f`. The tilde says these two
segments answer to each other in this data. The arrow would say one became
the other, which is a historical claim the tool has no basis for and does
not make. If you want to read direction into what you see, that is a
judgement you are adding, and it should be visible as yours.

It does not decide which words are cognate, does not reconstruct
proto-forms, does not build family trees, and does not date anything. Those
are separate questions belonging to a later stage of historical inference.
What regulae gives you is the synchronic structure. What that structure
means historically is your problem, and it should be.

## Beyond the browser

This page runs the same C library as everything else, compiled to
WebAssembly. Nothing is sent to a server.

The command line takes the same corpora:

```
regulae train corpus.tsv --format wide
regulae train corpus.tsv --human
regulae train corpus.tsv --json
```

The JSON is a debug snapshot, labelled as such in the payload. It is
deliberately not an interchange format: the framework this belongs to
requires anything making historical claims to carry ensembles and
uncertainty rather than one best answer, and a single set of
correspondences is not that.

To use the library directly, `include/regulae.h` is the contract.
The [README](https://github.com/tresoldi/regulae) has a worked example.
