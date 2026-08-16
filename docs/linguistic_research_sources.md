# Linguistic research memo: what regulae must prove

Snapshot reviewed: `5de30f8` (2026-08-16). This is a source memo for a
linguistic evaluation, not an implementation plan. It draws on the repository's
documentation, commit history and four live runs of the C CLI, and on the
primary or authoritative sources linked throughout. The product claim was
subsequently clarified: regulae explains relationships among supplied surface
forms; it does not discover sound laws. The evaluation below should be read at
that descriptive level. Historical-event recovery is a downstream research
question, not a core acceptance criterion. The resulting implementation plan
is in `docs/surface_relationship_roadmap.md`.

The live measurements in this memo describe that reviewed snapshot, before the
categorical `K−1` parameter-count correction. M1's implementation record and
post-correction acceptance panel are in the roadmap.

## Executive judgment

regulae has the right epistemic centre for a potentially field-changing tool.
It treats recurring sound correspondences and their conditioning as things to
discover, quantify and audit, while refusing to infer cognacy, ancestry,
proto-forms, chronology, borrowing or phylogeny. That boundary agrees with the
comparative method: recurrent correspondences are indispensable evidence, but
historical reconstruction also needs directionality, morphological analysis,
relative chronology and judgments about inheritance and contact ([Rankin
2003](https://doi.org/10.1002/9780470756393.ch1); [Campbell
2020](https://doi.org/10.1515/9781474463133-013)). Automatic proto-form systems
make the extra phylogenetic and directional assumptions explicit, as they
should ([Bouchard-Côté et al. 2013](https://doi.org/10.1073/pnas.1204678110);
[Jäger and List 2018](https://doi.org/10.1163/22105832-00801002)).

The project is already unusually serious about negative evidence. Its
complement-attested conditioning, BIC gate, shuffled baselines, set-level
resampling, restraint corpora, transcription-drift diagnostic, outlier ranking
and explicit `unmeasured` verdict are better foundations than a showcase that
only recovers famous laws. Its present weakness is not a lack of interesting
output. It is that there is not yet enough independent, expert-annotated
evidence to know the descriptive fidelity, predictive generalization,
calibration and analyst usefulness of that output.

The transformative claim should therefore be framed narrowly and strongly:

> regulae can make the evidential middle of comparative work—alignment,
> recurrent correspondences, conditioning contrasts, residuals and
> uncertainty—reproducible and reviewable at a scale a linguist could not
> inspect unaided.

It should not be framed as “automatic sound-law discovery,” because that is not
the package's claim. A detected surface correspondence is not necessarily one
historical event, one direction of change, or one population of inherited
words; those are possible downstream interpretations.

## Repository audit: the claim is ahead of the validation

The C core is coherent enough to support serious scientific evaluation. The
scientific evidence is not yet coherent enough to establish how faithfully and
generally it explains a supplied surface relationship. The distinction matters:
this is a validation verdict, not a code-quality verdict.

The commit history contains 117 commits from 2026-04-16 through the reviewed
snapshot. The C port began on 2026-07-30, and most of the linguistic restraint,
search calibration and challenge fixtures landed on 2026-08-14--16. The same
development process designed the algorithm, constructed the fixtures, chose the
expected analyses and wrote the assertions. That work found real defects and is
excellent exploratory science, but it is not independent validation. There is
no tagged evaluation release, blind expert surface-analysis set, held-out
prediction benchmark or outside replication in the repository.

The real-data experiments do not fill that role. Most checked-in corpora are
30--100 manually selected pairs, often described only as “standard literature,”
Wiktionary, or a delegated-agent compilation. Several mix normalized or
simplified ancestral forms with modern broad transcription; Latin--French
deliberately keeps non-strict cognates, and Latin--Spanish documents a pair that
is not a whole-word cognate. Item-level sources, edition identifiers, raw forms,
normalization decisions, licences and expert adjudications are generally absent.
Several narrative findings no longer reproduce. These are productive pilot
corpora, not publishable gold data.

Three examples show why the distinction is substantive rather than
bibliographic. The Proto-Polynesian--Hawaiian findings say the corpus produces
no conditioned splits; the current C build produces three. The Old
English--Modern English corpus was selected to include enough examples of every
Great Vowel Shift outcome, so recovering those outcomes measures fixture
construction as much as generalization. The Arabic--Hebrew corpus describes its
Hebrew as modern Israeli pronunciation while writing pharyngeals and an
“emphatic” /tsˤ/ specifically to preserve historical feature visibility. That
is a hybrid analytical transcription, not an observed modern system, and it
can make a feature correspondence easier by putting the answer in the input.
The Proto-Polynesian corpus likewise represents length through doubled vowels
while the loader treats those as two segments. These choices can be useful in a
semi-synthetic probe, but they must be labelled, sourced per item and varied
factorially before they can validate typological behavior.

The Lexibank family survey is valuable coverage testing, but it reports no gold
rule precision or recall and was recorded without shuffled baselines. Selecting
the most-attested lects can densely sample one subgroup; reporting pair counts
then magnifies that sampling decision. Its `kesslersignificance` row is labelled
“control,” but the conversion retains expert cognate sets and is not an
unrelated-wordlist negative control.

The C assertions also favour recall over exactness. Most sound-law tests ask
whether at least one expected correspondence or feature occurs. They rarely
assert that unsupported rules are absent, that the complete environment is
right, or that one historical event has been recovered as one event. The
clearest mismatch is Grimm's Law: the fixture documentation says all shifts are
asserted to be unconditioned, but `test_grimm` only checks that eight expected
segment pairings exist. The current model also publishes three conditioned
classes on that corpus. Ten shuffles reject all three, but the default run does
not execute shuffles and labels them merely `unmeasured`.

Some fixtures also test a narrower synthetic proxy while naming a broader
historical law. The Grassmann corpus requires the affected aspirate to be
word-initial, although word-initiality is not part of Grassmann's Law; it is a
correlate introduced by the selected word shapes. The morphological rhotacism
fixture usefully distinguishes an internal string from a compound seam, but
Latin rhotacism famously applies across stem--suffix boundaries, so describing
the law as “morpheme-internal” gives the morphological predicate the wrong
historical interpretation. Fixture names and expected environments need expert
review just as much as the engine output does.

Two documentation contradictions affect scientific interpretation. In
`docs/correspondence_discovery.md`, “Why BIC” says BIC supplies the
multiple-comparison adjustment, while a later section correctly explains that
BIC prices added parameters but not the adaptive search over roughly 55 or 135
candidates. Those cannot both be true. In `docs/capabilities.md`, bootstrap is
listed as unported and ignored, while `bootstrap_class_intervals`, the API,
tests and commit `df68421` show that whole-cognate-set bootstrap is live. These
are not cosmetic discrepancies: they change what a reader believes a rule's
evidence and interval mean.

### Measurements from the reviewed C build

The following are exit-status-checked runs, not readings of stored findings:

| probe | observed result | linguistic reading |
| --- | --- | --- |
| Grimm, ten shuffles | 3 conditioned, 0 standing | expected unconditioned correspondences recovered; false conditioning is still emitted |
| Rhotacism, ten shuffles | 3 conditioned, 3 standing | the true rule stands, but so do `r ~ r` after a stop and `s ~ s` before a vowel |
| Lenition, ten shuffles | 4 conditioned, 1 standing | one of three intended stop-voicing rows clears the corpus null; one unrelated vowel row is emitted |
| Grassmann, ten shuffles | 3 conditioned, 0 multi-lect standing; 1/2 pairwise standing | the known law can be found but not certified at the headline level |
| Verner, ten shuffles | 5 conditioned, 0 standing at either level | a textbook law is below the search's own null on this fixture |
| Final devoicing, ten shuffles | 4 conditioned, 0 standing | good abstention after calibration; misleading without it |

These results show why “standing” must be read as “survives this adaptive-search
null,” not “is a sound law.” On rhotacism, a corpus-specific correlate survives
the null because shuffling destroys that correlate along with the law. A null
can reject chance search artefacts; it cannot distinguish historical cause from
a stable accidental association in the observed lexicon.

The intended rhotacism row itself is over-specified. Alongside the old-Latin
intervocalic context it includes a Latin-side open-previous-syllable condition.
That predicate is true of the selected observations, but the corpus does not
show that it belongs in the historical environment. A rule representation that
prints every committed correlate as one conjunction risks turning a valid
discovery into a stronger and less defensible historical claim.

### New challenge corpora and what they expose

The review added `testdata/linguistic/`, deliberately outside the pass/fail C
suite until the desired scientific behavior is decided.

An exact 32-set change has rule margin 5.47 in every taxon-sampling variant. In
the two-lect corpus it stands against a null margin of 3.17. Adding one unchanged
conservative sister raises the shared null to 6.49, so the unchanged rule becomes
`within noise`; adding another innovative sister raises the null to 7.72. The
rule evidence did not change. The number of pairwise opportunities did. This is
a concrete case where a verdict depends on genealogical sampling.

`repeated_etymon.tsv` repeats two stems over sixteen apparent cognate sets. The
engine cannot know that these are paradigm cells rather than sixteen lexical
histories and reports the eight-example conditioned rule as above noise
(`z=-7.85`). Whole-set bootstrap does not solve this if cognate-set identifiers
are already pseudo-replicated. An etymon, paradigm or source-group identifier is
needed at the data or evaluation layer.

The paired confound corpora make identifiability visible. When the only trigger
is /i/ against /a/, closeness, frontness and segment identity partition the data
together; regulae prints `following[close:+]`. When height and rounding are
balanced across eight vowels, it prints `following[front:+]`. Only the second
corpus identifies frontness. The first supports an equivalence class of
descriptions, not the winning label as a unique historical cause.

A real negative control was derived reproducibly from Kessler's CC-BY CLDF data
without being vendored: 200 Hawaiian and Navajo forms paired by meaning while
the expert cognate assignments were intentionally ignored. These are bad
cognate inputs, but that is exactly the stress test for the guide's stronger
claim that corpus fit answers whether a relationship exists. The result was:

| shuffles | cost/segment z | multi-lect standing/measured | pairwise standing/measured |
| ---: | ---: | ---: | ---: |
| 10 | -5.47 | 10/107 | 13/416 |
| 20 | -4.08 | 10/107 | 13/416 |
| 50 | -4.58 | 10/107 | 13/416 |

For comparison, meaning-matched English--German and Latin--French scored
`-36.60` and `-20.56` with ten shuffles. The fit contains strong relationship
signal, but the unrelated real control overlaps the `z < -5` criterion used in
positive project tests at one routine shuffle count and produces standing
rules. It should be presented as a graded model-fit statistic, not a categorical
relatedness verdict, until real null distributions across inventories, areas,
word lengths and semantic lists establish operating thresholds.

## What the repository already demonstrates

The history since `b8ecf5c` is linguistically substantive, not merely a port:
fixtures were rebuilt so wrong analyses were available; conditioning became a
comparison with an attested complement; searches were made bidirectional;
place, stress, morphology, syllable weight, distant triggers and disjunctive
triggers became expressible; transposition became one event; uncertainty moved
to cognate-set resampling; shuffled data became a rule-level baseline; and
restraint and diagnostic corpora were added. Those decisions should be
preserved as scientific commitments.

Four current CLI runs are especially informative:

- `opaque_umlaut.tsv` recovers the strong `a ~ e` environment, but one
  historical umlaut process is fragmented by affected segment. The other
  frontings lack enough evidence independently. This is the most important
  known recall limitation.
- `tone_chinese_like_clean/cognates.tsv`, loaded as wide data, returns no
  cross-dimensional rule. Merely changing the lect identifiers so that the
  historically conditioning lect sorts first makes the two constructed rules
  appear. A finding cannot depend on a metadata name.
- `metathesis_distant.tsv` recovers identity correspondences internally but
  publishes no reordering event; its fit diagnostic nevertheless describes
  two populations, 67% in the worse-aligning group. The historical event and
  the corpus-mixture diagnostic are being conflated by a limitation of the
  published representation.
- `restraint/stratum.tsv` correctly publishes no conditioned rule while
  reporting a sharply separated 50/50 mixture. That is exactly the right
  abstention: statistics can show two systems without deciding whether the
  cause is borrowing, chronology, source mixture or diffusion.

Scientific auditability also requires documentation consistency.
`docs/capabilities.md` currently says bootstrap uncertainty was never ported,
while the current API documentation, roadmap, history (`df68421`) and project
instructions say whole-cognate-set bootstrap is implemented. A reader cannot
evaluate an uncertainty claim while the capability ledger disagrees with the
implementation record.

## 1. Correspondence discovery is not historical direction

`latin:p ~ spanish:f` is the correct output shape. Direction requires an
ancestral state, a topology, dates or an independently motivated scenario.
Phonetic and typological evidence can help choose among directions, but that is
additional evidence; laboratory work on `/p/ > /f/ > /h/` illustrates the kind
of external evidence involved ([Foulkes
1997](https://doi.org/10.1177/002383099704000303)). `decision_index` is a
greedy explanatory order, not historical chronology.

Required evaluation:

- Rename lects, reverse pair orientation, permute lect columns and permute
  rows. Canonical correspondence content must remain invariant.
- Include endpoint data compatible with more than one history, and feeding,
  bleeding and counterfeeding chains whose chronology is underdetermined. The
  expected result is an undirected description or an explicit ambiguity.
- When an attested ancestor is supplied for evaluation, score agreement with a
  directional gold analysis externally; do not let the benchmark quietly turn
  the core model into a proto-language model.

Cross-dimensional discovery is the urgent failure here. It must be evaluated
in both orientations before any tonogenesis result is presented to linguists.

## 2. A row is a surface unit, not a historical event

Historical linguists usually state one change over a class: voiceless stops
spirantize, vowels front before *i*, or dorsals palatalize. regulae currently
tests many of these as separate segment-pair rows, dividing the evidence by
the size of the affected class. List's multilingual correspondence work makes
the same distinction between alignment columns, recurrent patterns and the
regular core used for prediction ([List
2019](https://doi.org/10.1162/coli_a_00344)). Hruschka et al. model regular
sound change as a concerted event across lexical characters ([Hruschka et al.
2015](https://doi.org/10.1016/j.cub.2014.10.064)).

Pooling cannot simply mean “same feature displacement.” Natural classes are
representation-dependent and sometimes not featurally natural at all. In a
survey of nearly 600 languages, no tested universal feature theory expressed
more than 71% of attested active classes ([Mielke
2008](https://doi.org/10.1093/oso/9780199207916.001.0001)). The existing RUKI
decision-list fixture is therefore not an edge case but a warning against
equating “compact in merkmal” with “historically real.”

This fragmentation is not a failure of the core claim: separate surface
correspondences can be correct and useful. It is nevertheless important to
measure because it affects compression, evidence strength, readability and the
downstream tool's ability to group rows into a historical hypothesis.

Required evaluation:

- Hold total evidence constant while the same surface displacement applies to
  1, 2, 4 and 8 source segments. Measure row-level coverage, predictive gain
  and how much evidence fragmentation costs. Event-level recall belongs to a
  downstream law-induction evaluation.
- Test natural classes, disjunctive but attested-style classes, and deliberately
  arbitrary lexical classes.
- Run the same corpora under every supported feature system. Record semantic
  equivalence, not exact printed feature names, and expose cases where the
  finding itself changes.
- Add correlated distractors—word position, segment identity, syllable shape
  and morphology—that select almost the same observations. An honest result
  may be an equivalence class of analyses rather than one “true” label.

## 3. Conditioning must survive confounds and unseen data

Testing an environment against its complement is essential, and the present
design gets this right. The remaining risk is adaptive search: the winning
predicate is selected from many correlated possibilities and then measured on
the same data. BIC prices parameters, but it is not a general false-discovery
guarantee for an adaptive greedy search. Conventional intervals after model
selection do not have their usual coverage ([Berk et al.
2013](https://doi.org/10.1214/12-AOS1077)); the repository's `post_selection`
flag is therefore important, but it is a warning rather than a correction.

Every conditioned claim should face two forms of confirmation:

1. corpus-specific null calibration by rerunning the entire search on shuffled
   data; and
2. discovery on one set of cognates followed by effect estimation and
   prediction on held-out cognates.

The second matters because a rule can be well above shuffled noise and still
be an overfit description of this lexicon. Vary sample size, word length,
inventory size, candidate-feature count, number of lects and imbalance
independently in null corpora. Report the chance that a null corpus publishes
at least one standing rule and the false-discovery proportion among all rules.
If a project-wide list of “discoveries” is presented, false-discovery control
is the relevant statistical concept ([Benjamini and Hochberg
1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)).

## 4. Borrowing, diffusion and lexical strata are observational competitors

Contact is not a nuisance confined to cultural vocabulary. The World Loanword
Database shows large, structured differences in borrowability by meaning and
word class ([Tadmor, Haspelmath and Taylor
2010](https://doi.org/10.1075/dia.27.2.04tad)); network analyses recover
horizontal lexical transmission that a tree alone misses ([Nelson-Sathi et al.
2011](https://doi.org/10.1098/rspb.2010.1917)). Regularly adapted loans can
produce perfectly systematic correspondences. Conversely, a regular change
may spread lexically and remain incomplete ([Wang
1969](https://doi.org/10.2307/411748)), with word frequency affecting diffusion
([Bybee 2002](https://doi.org/10.1017/S0954394502143018)).

The existing `contact`, `diffusion` and `stratum` fixtures correctly show that
the same surface split admits different histories. They are too clean to
measure the difficult distinctions.

The next datasets should include:

- WOLD languages with item-level loan annotations, scored not as a demand that
  regulae “detect loans,” but as a test of what known loan layers do to rules,
  residuals and mixture diagnostics;
- adapted and unadapted loans, several donor periods, and contact between close
  relatives;
- two internally regular lexical strata with different complete
  correspondence systems;
- diffusion ordered by lexical frequency, compared with a random half-lexicon
  split; and
- contact, bad cognacy and transcription-source mixtures matched to have the
  same mixture fraction, so that causal labels cannot be smuggled into a fit
  statistic.

The 2025 IE-CoR data model is useful here because it represents a loan event at
the same structural level as cognate history rather than as a boolean flag on
one word ([Anderson et al.
2025](https://doi.org/10.1038/s41597-025-05445-3)).

## 5. Cognacy uncertainty must propagate, and partial cognacy must be usable

Expert cognate judgments are indispensable but not infallible. Automatic
comparison reaches high but imperfect agreement with expert benchmarks and is
best understood as part of an iterative expert workflow ([List, Greenhill and
Gray 2017](https://doi.org/10.1371/journal.pone.0170046)). Whole-set confidence
weights are useful, but they cannot express “the root is cognate and the suffix
is not.” Partial cognacy is a morpheme-level relation ([List, Lopez and
Bapteste 2016](https://doi.org/10.18653/v1/P16-2097)), and IE-CoR likewise
notes that cognacy belongs to morphemes rather than automatically to whole
words ([Anderson et al.
2025](https://doi.org/10.1038/s41597-025-05445-3)).

The existing partial-compound diagnostic finds suspect sets but then discards
the usable half. A transformative expert tool should eventually make partial
cognacy a first-class input. Before that representation exists, the linguistic
benchmark can still be built:

- annotate full, partial and non-cognate sets at morpheme level;
- include shared affixes with unrelated roots and inherited roots with replaced
  affixes;
- obtain two independent expert codings and an adjudication for a real subset;
- rerun the analysis across plausible cognacy weights and report rule
  stability; and
- use leave-one-cognate-set-out influence, because an outlier ranked by a model
  trained on itself is not an independent diagnosis.

## 6. Alignment uncertainty and reordering need their own evidence

Treating transposition as one event is linguistically correct. Metathesis can
be adjacent, non-local or featural and is shaped by perceptual and sequence
conditions rather than reducible to two substitutions ([Blevins and Garrett
2004](https://doi.org/10.1017/CBO9780511486401.005); [Hume
2004](https://doi.org/10.1353/lan.2004.0083)). The Benchmark Database of
Phonetic Alignments was explicitly designed to include metathesis and diverse
sequence problems ([List and Prokić
2014](https://aclanthology.org/L14-1269/)).

Current evaluation uses the winning alignment. The missing quantity is how
close plausible alternatives were and whether downstream rules survive them.
Tests should cover repeated segments, insertion/deletion around the swap,
metathesis plus substitution, and several equal-cost alignments. Distant
reordering needs a publishable event independent of chunk promotion; otherwise
the alignment can be right while the linguistic output denies that an event
occurred.

Transcription granularity must also be factorial: affricates, diphthongs,
prenasalized stops and contours should be represented once as one grapheme and
once as a sequence. The expected differences must be explicit. CLTS exists
because transcription systems and source practices are highly idiosyncratic
([Anderson et al.
2018](https://doi.org/10.2478/yplm-2018-0002)); normalization is not a
theory-neutral clerical step.

## 7. Morphological association is not automatically morphological causation

Caller-supplied boundaries are the right division of responsibility: regulae
must not infer morphology. But a change associated with a boundary, morpheme
index or grammatical category may reflect analogy, paradigm levelling,
lexical strata or phonological history. Whether morphology can directly
condition sound change has itself been a theoretical controversy ([Pierce
2016](https://doi.org/10.2218/pihph.1.2016.1702)). The output should therefore
remain “correspondence in a morphological environment,” not a causal label.

Required challenge data include prefix, suffix, compound and word boundaries
with the same segmental context; analogical extension; paradigm levelling;
sandhi across word but not morpheme boundaries; root-and-pattern morphology;
reduplication; and multiple paradigm cells weighted so that one lexeme does
not masquerade as many independent historical examples.

## 8. Tone, stress and weight cannot always be segment properties

The present `tone`, `stress` and `length` dimensions cover useful cases, and
the loader's recognition of Chao tones is a real advance. They do not cover the
full object. Tonogenesis commonly reinterprets consonant-induced pitch effects
as tonal contrasts ([Hombert, Ohala and Ewan
1979](https://doi.org/10.2307/412518)), often through interactions of source
tone and onset/coda laryngeal state rather than one predictor. Tone can spread,
dock, float, downstep and function as a morpheme independently of the segment
on which it surfaces; that empirical independence motivated autosegmental
phonology ([Goldsmith 1976](https://dspace.mit.edu/handle/1721.1/16388)).
Prosodic typology is property-based and does not reduce cleanly to a linear
“stress–pitch accent–tone” scale ([Hyman
2009](https://doi.org/10.1016/j.langsci.2008.12.007)). Syllable weight is also
language-specific: different systems count vowel length, codas and sometimes
onsets differently ([Gordon
2002](https://doi.org/10.1353/lan.2002.0020)).

The tonal challenge set should cover source tone × onset interaction, retained
and lost triggers, register and phonation, contour split and merger, tone
spreading, floating morphological tone, sandhi, mora- versus syllable-bearing
tone, and stress/weight conditioned by an earlier system no longer visible in
the daughter. The family survey's five rule-producing tonal datasets are
valuable smoke tests, not a typology: only six of 43 surveyed sources transcribe
tone at all, and the current directionality defect can turn a perfect rule into
zero.

## 9. Multi-lect evidence is genealogically and areally dependent

Six lect pairs among four closely related varieties are not six independent
historical events. Typological inference that ignores common descent
overstates evidence; this is the classic comparative-data problem ([Felsenstein
1985](https://doi.org/10.1086/284325)). Linguistic analyses likewise find
lineage-specific pathways and use phylogenetic models to control dependence
([Dunn et al. 2011](https://doi.org/10.1038/nature09923); [Jäger and Wahle
2021](https://doi.org/10.3389/fpsyg.2021.682132)).

regulae need not build a tree. It does need to prevent consumers from reading
pair counts as independent confirmations. Evaluation should duplicate one
daughter into near-identical sisters, compare one inherited innovation in ten
daughters with several independent innovations, and report both corpus/lect
counts and independent-family counts. Generalization splits must hold out
whole families, macroareas and source datasets, not random rows.

## 10. Typological plausibility is a check, not a verdict

Lexibank provides standardized, reusable wordlists and expert cognacy for a
large cross-family collection ([List et al.
2022](https://doi.org/10.1038/s41597-022-01432-0)); PHOIBLE and CLTS provide
inventory and transcription infrastructure ([PHOIBLE
2.0](https://phoible.org/), [CLTS 2.3](https://clts.clld.org/)). UNIDIA was
designed specifically to derive diachronic tendencies, but its initial 3,750
changes over roughly 190 languages were concentrated in Bantu, Sinitic and
Daic ([Ben Hamed and Flavier
2009](https://doi.org/10.1075/cilt.308.21ham)). These resources should be used
with provenance and genealogical/areal stratification, not converted into a
universal prior that rules unattested-looking findings in or out.

An “unrelated” null must also be harder than random pseudowords. Similar
phonotactics, areal convergence and sound-symbolic associations can create
structured similarities; sound–meaning biases have been demonstrated across
thousands of languages ([Blasi et al.
2016](https://doi.org/10.1073/pnas.1605782113)). Build null panels from
unrelated geographic neighbours with similar inventories, not only from
independently generated forms.

## A benchmark capable of supporting the surface-model claim

The benchmark should have four layers, each answering a different question.

| layer | purpose | indispensable content |
| --- | --- | --- |
| Minimal synthetic fixtures | Can one known contrast be found or declined? | One intended surface analysis, tempting wrong analyses, exact generated truth |
| Factorial simulations | Where are the power and false-positive boundaries? | Sample size, inventory, class size, confounds, diffusion, loans, noise, missing data, alignment ambiguity, lect count |
| Semi-synthetic corpora | Does the method work on realistic phonotactics with known truth? | Injected changes in real word shapes/inventories, retained untouched controls |
| Expert real-data analysis sets | Is the output descriptively useful and defensible? | Alternative alignments, accepted surface correspondences, environments, exceptions, cognacy, partial cognacy, loans/strata, citations, confidence |

Lexibench is a useful model for a versioned expert benchmark, though it targets
cognacy rather than correspondence modeling ([Häuser and List
2025](https://doi.org/10.15475/calcip.2025.1.2)). The List–Prokić alignment
database can seed alignment gold data. WOLD can supply contact annotations.
IE-CoR can supply carefully documented cognacy, morphology and loan events.
Lexibank can supply broad family coverage. None alone is the gold standard
regulae needs. A gold inventory of historical laws would instead evaluate a
downstream package.

### Dataset design rules

- Split by etymon/cognate set, family and source publication. Never let
  alternate reflexes or duplicated datasets of the same etymology cross train
  and confirmation partitions.
- Annotate competing scholarly analyses. Exact string equality against one
  feature label is not a fair rule metric.
- Retain raw source transcription beside normalized segments and record every
  transformation.
- Balance phenomena and negative cases, not just families. “More Romance” does
  not test tone, templatic morphology or register.
- Keep source licenses and versioned provenance. A recorded run without the
  data revision is not reproducible.
- Blind the adjudicators to regulae's output when establishing the first gold
  set, then separately measure whether the tool helps them find omissions or
  errors.

### Metrics

Report at least:

- alignment link accuracy plus ambiguity/margin;
- unconditioned correspondence precision and recall;
- conditioned-correspondence precision, recall and false-conditioning rate;
- coverage and predictive gain across related affected segments;
- environment equivalence under feature synonyms and logically equivalent
  decision lists;
- held-out predictive log loss or surprisal improvement;
- calibration of intervals, `standing` and abstention;
- family-wise chance of any false standing rule and false discovery
  proportion;
- stability under lect renaming, orientation, row order, feature system,
  transcription granularity and plausible cognacy weights;
- results stratified by phenomenon, corpus size, family, macroarea, time depth,
  contact intensity and transcription source; and
- human outcomes: time to a defensible analysis, errors found, accepted versus
  rejected suggestions, and inter-analyst agreement with and without the tool.

Prediction is critical. List's correspondence-pattern study evaluated the
ability to predict missing reflexes precisely because no exhaustive
expert-annotated rule gold standard existed ([List
2019](https://doi.org/10.1162/coli_a_00344)). regulae should create a small
expert surface-analysis set and use held-out prediction as the scalable check
that a selected environment generalizes. Historical event grouping can be
scored separately when the downstream tool exists.

## Priority order for the linguistic programme

1. Correct the categorical split parameter count while holding all other gates
   fixed, then review every moved model deliberately.
2. Treat lect-name/orientation invariance as a release-blocking scientific
   invariant for every discovery stage.
3. Calibrate the complete adaptive search on multiple nulls and held-out
   cognate sets; distinguish descriptive fit from selection and prediction.
4. Define etymon/source grouping so resampling and holdout units match lexical
   histories rather than rows.
5. Commission a small blind, expert-adjudicated surface-analysis set before
   expanding the family survey further.
6. Measure evidence fragmentation across affected segment classes without
   requiring the core to infer one historical event.
7. Design a morpheme-level partial-cognacy representation and benchmark.
8. Build the autosegmental/prosodic challenge set and state explicitly what the
   current segment-attached representation cannot express.
9. Add real contact, strata, diffusion, analogy and source-mixture corpora with
   external annotations; keep causal interpretation outside regulae.
10. Keep the central boundary intact: the tool explains structured surface
    correspondence evidence; downstream historical inference explains it.

If these tests succeed, regulae would not replace the comparative method. It
would change its practice by making a large and previously tacit part of the
method inspectable, repeatable and falsifiable. That is a more credible—and
more consequential—claim.
