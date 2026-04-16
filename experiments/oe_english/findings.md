# Old English → Modern English Experiment: Findings

## Motivation

Third real-data experiment. After Latin→Spanish (Romance,
context-conditioned gradient changes) and PPn→Hawaiian (clean
exceptionless mergers in a small inventory), this pair was chosen
to test a completely different type of phenomenon:

* **Chain shifts** rather than mergers. The Great Vowel Shift
  rearranged the long vowels in height/frontness space without
  (mostly) merging them — each long vowel moved but stayed
  distinct.
* **Consonant cluster simplifications** at word edges: initial
  `kn-` → `n-`, `hl-` → `l-`, loss of `x/h` in `-ht-` clusters.
* **Fricative voicing** (intervocalic) — a classic context-dependent
  split that the learned model should fail on predictably.
* **Palatalization** — OE `/k/` before front vowels → ModE `/tʃ/`,
  another context-conditioned split.

The change profile is genuinely different from both previous
experiments. And it's Germanic, not Romance or Polynesian — a third
language family branch.

## Setup

* **Corpus**: 100 Old English → Modern English cognate pairs, hand
  curated. Basic vocabulary plus enough GVS instances to test every
  long-vowel change.
* **Reconstructions**: OE in simplified phonetic form (length marks
  stripped at parse time — see caveat below), Modern English in
  broad phonemic IPA with diphthongs as single graphemes (`aɪ`,
  `oʊ`, `aʊ`).
* **Caveat**: the parser drops OE length marks (`ː`), so `staːn`
  parses as the same 4 segments as `stan` would. This means OE
  short `a` and long `ā` are indistinguishable to the framework for
  this experiment. Since ModE reflects them differently (short `a` →
  `æ`, long `ā` → `oʊ`), the framework recovers the shift but at
  the cost of treating the OE input as "ambiguous length." A follow-
  up experiment could add proper length-aware segments.
* **Training**: default learned-model hyperparameters, no tuning.

## Headline numbers

| metric | value |
|---|---|
| corpus pairs | 100 |
| prior-only total cost | 108.38 |
| learned-model total cost | −791.10 |
| Reduction | 899.48 |
| distinct segment correspondences | 44 |
| distinct feature displacements | 27 |
| promoted chunks | 24 |

Compare across all three experiments:

| metric | Lat→Spa | PPn→Haw | OE→ModE |
|---|---:|---:|---:|
| corpus pairs | 97 | 96 | 100 |
| segment correspondences | 49 | 20 | 44 |
| displacement vectors | 29 | 9 | 27 |
| promoted chunks | 33 | 3 | 24 |
| "competing correspondences" above threshold | 8 | 0 | 10 |

The OE→ModE profile sits between the other two: more complex than
Hawaiian (which had nothing to chunk), less chunk-heavy than
Spanish (which had cluster changes plus diphthongizations plus
palatalization). The context-dependent splits count is the highest
of the three — Germanic has several simultaneous conditioned
changes (fricative voicing, palatalization, x-loss) which all surface
as competing correspondences.

## What the learned model recovered

### The Great Vowel Shift — all six long-vowel changes

```
Top segment correspondences showing GVS:
  e → i: 13    ← ē → iː  (see, he, me, tree, three, feet, green)
  a → oʊ: 11   ← ā → oʊ  (stone, bone, home, oak, road, goat...)
  i → aɪ: 11   ← ī → aɪ  (wife, mine, wine, time, ride, bite...)
  u → aʊ: 11   ← ū → aʊ  (house, mouse, out, brown, cow, how, now...)
  o → u: 4     ← ō → uː  (tooth, moon, food, goose)
  o → ʊ: 4     ← ō → ʊ   (foot, book, good, took — post-GVS shortening)
  i → ɪ: 8     ← short i → ɪ (king, ring, sing, swim...)
```

All six GVS changes are in the top 15 by count, recovered cleanly
from the data. This is a chain shift, so several different vowels
all moved to different targets — the framework had to learn each
correspondence separately (no obvious merger pattern). It did.

The split between `o → u` (long ō, 4 counts) and `o → ʊ` (shortened
ō, 4 counts) is interesting: these are the same OE input with two
different outcomes in Modern English, so the learned model sees them as competing
correspondences. In reality they're a split conditioned on consonantal
environment. More context-discovery motivation.

### Chain-shift signatures in the displacement layer

```
Feature displacements:
  identity: 181
  [close-mid: P→A, close: A→P]: 15               ← e→i + o→u (mid → high)
  [close: P→A, near-close: A→P,
   near-front: A→P, open: A→P]: 11               ← i → aɪ (height & front)
  [front: P→A, open: P→A, unrounded: P→A,
   back: A→P, close-mid: A→P, ...]: 11           ← ā → oʊ (front/central →back)
  [back: P→A, close: P→A, front: A→P,
   near-back: A→P, near-close: A→P, ...]: 10     ← ū → aʊ (high back → low back)
  [close: P→A, front: P→A,
   near-close: A→P, near-front: A→P]: 7          ← i → ɪ (a minor shift)
  [voiceless: P→A, voiced: A→P]: 6               ← fricative voicing
  [back: P→A, close-mid: P→A,
   near-back: A→P, near-close: A→P]: 5           ← ō → ʊ
  [alveolar: P→A, post-alveolar: A→P]: 4         ← palatalization
```

This is **the most complex displacement profile of any experiment
so far**. The Great Vowel Shift, which in traditional descriptions
is often called "a single rule" but is actually a bundle of
coordinated chain shifts, surfaces here as multiple large feature-
level displacements, each representing one part of the chain.

Notably, the displacements are *not* a single unified "all vowels
move" pattern. They're separate per-vowel movements: one for ē → iː,
another for ā → oʊ, etc. This is how the historical linguistics
literature actually describes the GVS — it's a chain of moves, not
a single uniform shift. The framework's distinction between these
is linguistically accurate.

### Chunks: consonant cluster simplifications

```
Top promoted chunks:
  (eta, it)          various endings
  (nga, ŋ)           ng simplification
  (isk, ɪʃ)          OE -isk → -ɪʃ (fish, dish)
  (ing, ɪŋ)          OE -ing → -ɪŋ
  (iht, aɪt)         OE -iht → -aɪt (right, night, light — x-loss + GVS)
  (eof, ɛv)          OE eof → ɛv (seven, heaven — eo→ɛ + f→v voicing)
  (reo, ri)          OE reo → ri (three, tree — eo→i)
  (nih, naɪ)         night-type
  (ofo, və)          over-type (f→v voicing)
  (tan, t)           various
  (sk, ʃ)            -sk → -ʃ cluster
  (ng, ŋ)            -ng → -ŋ
  (kn, n)            initial kn- → n-
  (hl, l)            initial hl- → l-
  (ht, t)            -ht → -t (x/h loss)
```

**These are all real linguistic phenomena.** The framework correctly
promoted:

* Initial cluster simplification: `kn-`, `hl-` → `n-`, `l-`
* Medial cluster simplification: `-ht-` → `-t-` (loss of /x/)
* Velar palatalization chunks: `-sk-` → `-ʃ`
* Nasal cluster changes: `-ng-` → `-ŋ`
* Large multi-segment chunks that bundle several changes at once:
  `-iht → -aɪt` (night-type, x-loss plus GVS), `-eof → -ɛv` (seven-
  type, vowel shift plus f-voicing)

The big multi-segment chunks are interesting: they capture the
combination of multiple co-occurring sound changes. A simple rule
view would decompose these into separate rules, but from the
framework's standpoint, they recurred together often enough to
justify promotion as single entries. This is actually closer to
the traditional idea of "sound law in a specific environment" than
to abstract rules.

## What the framework got wrong (or ambiguously)

### Chunk-preferring alignments can obscure the vowel shift

Example: `stone: stan → stoʊn`

```
s ~ s       [0.000]
ta ~ t      [0.750]
ε ~ oʊ      [0.500]
n ~ n       [0.000]
```

The linguistically transparent alignment would be `s~s, t~t, a~oʊ,
n~n` with the GVS change visible as `a~oʊ`. Instead the framework
chose to absorb the medial vowel into a `(ta, t)` chunk and insert
the `oʊ` afterwards. The *alternative alignment* has higher cost
under the trained model because `(ta, t)` is a promoted chunk with
a very negative cost (−5.738), which strongly attracts the search.

Root cause: chunk promotion creates gravity wells that can grab
related alignments even when a transparent segment-by-segment
alignment exists. This is the same cost-optimal-vs-transparent
tradeoff noted in the Latin-Spanish findings. The cost math is
correct; the interpretability is suboptimal.

**Candidate improvement**: a "prefer segment-level" tiebreaker that
penalizes chunks when an equally-cheap segment alignment exists.
Low priority but would improve readability.

### Competing correspondences: the expected context-discovery targets

```
's': s×22, ʃ×4  [total 26]         — sk-type palatalization
'n': n×22, ŋ×4  [total 26]         — ng → ŋ
'k': k×9, tʃ×4  [total 13]         — k palatalization before front V
'f': f×9, v×3  [total 12]          — fricative voicing
'θ': θ×8, ð×2  [total 10]          — fricative voicing
```

Every one of these is a classic context-conditioned change:

* `s → s/ʃ` — OE `-sk-` → ModE `-ʃ-` when clustered, else `s` remains
* `n → n/ŋ` — OE `-ng-` → ModE `-ŋ-` when clustered
* `k → k/tʃ` — palatalization before front vowels (cheese, church)
* `f → f/v` — intervocalic voicing (over, seven, heaven)
* `θ → θ/ð` — intervocalic voicing (mother, father, brother)

These are **exactly the phenomena context discovery is
supposed to handle**. The learned model handles them as competing
unconditioned correspondences — not wrong, but not resolved. The
"winning" correspondence is always the more frequent / unconditioned
version, and the conditioned minority shows up as competing mass.

### The `kn-` loss chunk: real but too big

The framework correctly promoted `(kn, n): −6.019` for initial
`kn-` clusters. This is real. But it also promoted some redundant
variants like `(nih, naɪ)` and `(iht, aɪt)` that overlap: the
`night`-type alignment gets both `(nih, naɪ)` and `(iht, aɪt)` as
candidate chunks, and they're both promoted.

This is milder than the Latin-Spanish over-promotion issue because
the overlapping chunks are genuinely different structures (one
starts at `n`, the other ends at `t`), but it's a minor form of
chunk redundancy that could benefit from post-promotion cleanup.

## Cross-experiment comparison

With three experiments in hand we can see the shape of the learned model clearly:

### Per-corpus profile

| property | Lat→Spa | PPn→Haw | OE→ModE |
|---|:-:|:-:|:-:|
| mergers present | Yes | Yes (exceptionless) | No (chain shift) |
| context-dependent splits | Many | None | Several |
| cluster/chunk changes | Many | None | Several |
| vowel rearrangements | Few | None | Many |
| fricative voicing | No | N/A | Yes |

### Learned-model per-profile behavior

* **Clean mergers** (all three have some): recovered as direct
  segment correspondences with dominant counts. Displacement layer
  picks up the feature-level signature. No issues.
* **Context-dependent splits** (Spanish, Germanic): visible as
  competing correspondences with both targets having substantial
  mass. The framework doesn't resolve them but reports them
  diagnostically. Identical failure mode across experiments.
* **Chain shifts** (English only): each vowel move learned
  separately. Displacement layer shows multiple coordinated
  displacements. This is both correct (each move IS a separate
  change) and surprising (I expected more unification).
* **Cluster changes / chunks** (Spanish, Germanic): promoted when
  they beat the compositional fit under BIC. Polynesian had none
  and the framework correctly promoted almost nothing.
* **Cost optimal vs transparent**: in all three experiments, the
  framework sometimes prefers a mathematically cheaper alignment
  that's less historically transparent. Not a bug but a consistent
  limitation.

### Context-discovery priorities confirmed

The three experiments together anchor context-discovery priorities in concrete
real-world data:

**Context conditioning: critical.** Every experiment with
context-dependent splits shows them as unresolved competing
correspondences. Spanish `k~k/θ`, English `k~k/tʃ`, English `f~f/v`,
English `θ~θ/ð` — five separate instances of the same class of
phenomenon. Context conditioning would resolve all five.

**Rule abstraction: partially resolved, still useful.**
The sub-alignment compositional cost fix handled the worst cases,
but chunk redundancy persists. OE→ModE promoted `(nih, naɪ)` and
`(iht, aɪt)` as overlapping chunks that describe partially the same
change. A principled rule-merging layer would help.

**Non-cognate robustness: uncommitted but minor.** None of
the three experiments had badly non-cognate pairs beyond the
Latin/Spanish `kaput/kabeθa` case. This is mostly a data-quality
concern, not a framework priority.

**Cost/transparency tradeoff.** The framework consistently
prefers chunk-heavy interpretations where segment-by-segment would
be more linguistically transparent. In three experiments this came
up. Worth considering a tiebreaker that prefers segment-level
decompositions when their cost is equal or near-equal.

## What this third experiment changes about the plan

Not much. The major findings from Latin→Spanish are confirmed:

* The learned model works on real data.
* The learned model recovers unconditioned sound changes cleanly across multiple
  language families.
* The learned model fails on context-dependent splits in a diagnostically useful
  way (competing correspondences report).
* Context-discovery priorities (context conditioning, rule abstraction) are the
  right ones.

The new finding is about **chain shifts vs mergers**. This is a
deeper phonological distinction than I'd expected the learned model to handle:
chain shifts keep categories distinct as they move, while mergers
collapse them. The framework handled both cleanly at the segment
level AND surfaced the difference in the displacement layer. This
was not an explicit design goal; it fell out of the Bayesian
hierarchical structure naturally.

That's a point in favor of the design. The layered model
(segment + displacement + chunk) captures more than the sum of its
intended parts.

## Reproducing this experiment

```sh
cd /home/tiagot/nas-dev/new_chl/regulae
source .venv/bin/activate
python experiments/oe_english/run_experiment.py
```

The experiment script is deterministic; same corpus → same model.
