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
only when it pays for itself under a BIC criterion. That last part matters. It
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

## Confidence and outliers

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
