# Latin → Spanish Experiment: Findings

## Setup

* **Corpus**: 97 Latin → Spanish cognate pairs, curated manually, in
  `cognates.tsv`. Basic vocabulary (Swadesh-style): body parts,
  natural world, numerals, basic verbs.
* **Reconstructions**: simplified Latin (no macrons, reduced case
  inflection, e.g. `kaput` not `caput` and `pater` not `patrem`) and
  phonetic Spanish (IPA-adjacent: `tʃ` for ⟨ch⟩, `ʎ` for ⟨ll⟩, `ɲ` for
  ⟨ñ⟩, `θ` for Castilian ⟨z⟩/⟨c⟩-before-front-V, `x` for ⟨j⟩).
* **Training**: default learned-model hyperparameters (τ=1.0, β=5.0, w_seg=0.7,
  w_disp=0.3, max_chunk_size=3). Single invocation of `train_model`.
* **Runtime**: < 5 seconds end to end on 97 pairs.

## Headline numbers

| metric | value |
|---|---|
| corpus pairs | 97 |
| prior-only total alignment cost | 143.37 |
| learned-model total alignment cost | −889.25 |
| Reduction | 1032.62 |
| distinct segment correspondences | 49 |
| distinct feature displacements | 29 |
| promoted chunks | 45 |

The huge learned-model cost drop reflects the layered model heavily discounting
observed correspondences via both the Dirichlet posterior and the
negative log-Z offset. Absolute magnitude is not meaningful; the
direction (learned < prior-only) is.

## What the framework recovered well

### Regular unconditioned correspondences

Top segment correspondences after training:

```
  a → a: 38    r → r: 38    e → e: 33    u → o: 32
  o → o: 15    l → l: 14    n → n: 14    i → i: 11
  k → k: 11    m → m: 11    s → s: 10    d → d: 9
  i → e: 9     v → b: 9     p → p: 8     b → b: 7
```

This is exactly what the comparative method would find:

* **Final `u → o` (32 counts)** — Latin 2nd-declension `-us/-u` → Spanish `-o`.
  The single most decisive diagnostic of Latin-to-Romance vowel change
  after losing final `-s`, and the framework recovered it cleanly.
* **`v → b` (9 counts)** — Spanish betacism, merging Latin `v` [w/β] and `b`.
  Dominated by intervocalic cases.
* **`i → e` (9 counts)** — Latin short `ĭ` lowering to Spanish `e`, a
  major vowel-quality shift in Western Romance.
* Initial stops (`p`, `t`, `k`, `b`, `d`, `m`, `n`, `l`, `r`, `s`) all
  preserved with solid counts.

### Feature-level regularities

The displacement distribution captures three big patterns:

```
identity: 239
[close: P→A, close-mid: A→P]: 46
[fricative: P→A, labio-dental: P→A, bilabial: A→P, stop: A→P]: 9
[voiceless: P→A, voiced: A→P]: 9
[back: P→A, rounded: P→A, front: A→P, unrounded: A→P]: 8
[alveolar: P→A, palatal: A→P]: 7
[front: P→A, unrounded: P→A, back: A→P, rounded: A→P]: 4
[nasal: P→A, trill: A→P]: 4
```

* **`close → close-mid` (46 counts)** — this is the aggregate signature
  of the vowel lowering `ĭ → e` and `ŭ → o`, captured at the *feature*
  level. 46 = 32 (u→o) + 9 (i→e) + 5 (others). The framework
  automatically generalized across natural classes.
* **`fricative → stop at a labial place` (9 counts)** — this is Spanish
  betacism, `v → b`, seen as a feature-level change.
* **`voiceless → voiced` (9 counts)** — this is intervocalic lenition
  generalized across `p → b`, `t → d`, `k → g`. The learned model couldn't learn the
  context-dependent rule, but it correctly bundled the instances into
  a single feature-level pattern.

### Chunk promotion

The learned model promoted 45 chunks. Many are linguistically meaningful:

```
(kte, tʃe)       Latin "-cte-" → Spanish "-ch-"  (noctem → noche)
(okt, otʃ)       Latin -oct- → Spanish -och-    (octo → ocho)
(kt, tʃ)         the general cluster change
(au, o)          Latin au → Spanish o            (causam → cosa)
(ll, ʎ)          Latin geminate l → palatal     (kaballu → kabaʎo)
(ce, θ)          Latin ce- → Spanish θe-        (centum → ciento)
(ua, wa)         (aqua → agua)
(kka, ka)        geminate simplification
(att, at)        geminate simplification
(de, dje)        stressed e → je diphthongization (dentem → djente)
(pe, pje)        same                            (petram → pjedra)
(ve, bje)        same + betacism                 (veteran → bjexo)
(o, we)          stressed o → we                 (novus → nueβo)
```

These are real sound changes. The framework discovered them from raw
form pairs with no prior knowledge.

## What the framework got wrong (or weakly)

### Context-dependent correspondences look like noise

The `k` source segment ends up with competing targets:

```
k → k: 11, θ: 5 (+ others)
```

This is **the classic Spanish palatalization case**. Before back vowels
(a, o, u), Latin `k` → Spanish `k`. Before front vowels (e, i), Latin `k`
→ Spanish `θ`. The learned model has no way to distinguish these and treats both as
competing unconditioned correspondences. This is exactly the phenomenon
context discovery is supposed to handle.

Same pattern for intervocalic lenition:

```
t → t: 10, d: 3 (+ others)
```

Latin `t` is preserved word-initially and in clusters, but becomes `d`
intervocalically. The learned model sees both and doesn't know the conditioning.

### Chunk over-promotion

The framework promoted 45 chunks. Some are redundant or should be
captured as rules rather than entries:

```
(en, jen), (el, jel), (ve, bje), (pe, pje), (de, dje)
```

These are all **instances of the same rule**: stressed short `e` →
`je` after various consonants. The learned model has no way to abstract the rule,
so it memorizes each specific chunk. A context-conditioned rule
would be far more economical: *one* chunk `(e, je)` conditioned on
following stressed syllable, rather than five specific combinations.

Similarly:

```
(ere, er), (ere, ir), (are, ar), (ire, ir)
```

These are Latin infinitive endings → Spanish infinitive endings.
Three separate entries for what could be one feature-conditioned
rule.

**Implication for context discovery**: the anomaly-detection engine should not only
surface context-dependent splits from *within* a correspondence class,
but also detect chunks that could be merged into a single
context-conditioned entry.

### Suboptimal segment-level interpretations

Some alignments chose chunk-interpretation when a segment-by-segment
one would be more linguistically transparent. Example:

```
[father] pater → padre
  p ~ p       [0.000]
  ate ~ ad    [1.125]
  r ~ r       [0.000]
  ε ~ e       [0.500]
```

Total cost 1.625. The alternative `p~p a~a t~d e~ε r~r ε~e` would
cost 1.375 due to two gap penalties, so the chunked version wins
mathematically. Historically, the segment-level interpretation is
more accurate: Latin `pater` syncopated to `patr` then the final `r`
got a supporting `-e`. The framework's cheaper choice groups these
events into an opaque `ate → ad` chunk.

This is a **cost/truth mismatch**: what the model prefers is not what
the linguist would choose. Not a bug per se, but a reminder that the
framework finds correspondences that optimize its cost function, not
"historically correct" decompositions.

### Bad input handling

One alignment is genuinely wrong:

```
[head] kaput → kabeθa
  k ~ k       t ~ a  (bogus!)
```

This is because `kaput` and `kabeθa` are **not strict cognates** at the
full-word level. Spanish `kabeθa` comes from Vulgar Latin `*capitia`
(a feminine diminutive form), not from the nominative `caput` I gave
it. The framework tried to align forms with different etymologies and
produced garbage in the middle. This is a **data quality** issue, not
a framework bug — but it highlights that the framework has no way to
detect "these shouldn't really be aligned."

**Implication**: cognate vetting is an upstream responsibility. The
framework will align any two forms you give it, even when they
shouldn't be aligned.

### Other mild issues

* `(re, r)`, `(er, r)`, `(er, re)`, `(er, ir)` all got promoted. These
  are word-final syllable fragments; most are not linguistically
  coherent units. BIC promoted them because their specific
  source-target co-occurrence is frequent enough to beat the
  compositional score.
* `(m, mb)` and `(m, mbr)` got promoted — these are epenthetic-b
  insertions in words like `nomen → nombre` and `omen → hombre`. This
  is correct, but the segmentation `m → mbr` is awkward.
* The `(u, xo)` chunk at cost −1.211 reflects `-ulu → -xo` specifically
  (from `oculu → oxo`). A one-off compound from the `-kl-` palatalization
  that couldn't be decomposed because the framework doesn't know about
  the consonantal change inside it.

## What the experiment confirmed

1. **The learned model does the job it was designed for.** Unconditioned segment
   correspondences, feature-level generalization, and context-free
   chunk promotion all work on real data.

2. **Grimm's-law-style recovery scales.** On 97 real cognates, the
   framework recovered multiple phonological shifts simultaneously
   and bundled them at the feature level.

3. **The log-Z offset makes the prior-only/learned-model interface clean.** At
   initialization the model behaves exactly like prior-only scoring; the learning
   pushes observed correspondences below the merkmal floor smoothly.

## What the experiment says about context discovery scope

Three concrete context-discovery requirements emerged from real-data failure modes:

### Immediate-neighbor context conditioning

The `k → k/θ` and `t → t/d` splits are the dominant "competing
correspondence" failures. Context discovery must be able to split
these by the following vowel (for `k`) and by surrounding context
(for `t`). Without this, the framework can't learn the single biggest
class of Romance sound changes.

**Priority: high.** Affects more pairs than any other failure.

### Chunk merging / rule abstraction

Five separate chunk entries for "stressed `e` → `je`" is wasteful
and opaque. Context discovery should be able to:

* Detect that multiple promoted chunks share a common "core" plus
  varying context
* Merge them into a single context-conditioned correspondence
* Keep the memory footprint compact

This is a different kind of context conditioning — not "split a
single correspondence by context" but "merge multiple chunks that
differ only in their context." Both belong in context discovery's scope.

**Priority: medium.** Affects elegance and interpretability more
than accuracy.

### Robustness to non-cognate pairs

The `kaput / kabeθa` case shows that the framework doesn't detect
when form pairs aren't really cognate. A downstream "relatedness"
layer can use alignment quality as input to a cognate-vs-not
decision, but the alignment layer itself can at least:

* Report per-pair confidence (some signal of alignment quality)
* Flag pairs where most links are high-cost even after training
* Not promote chunks that appear in such low-confidence pairs

**Priority: low.** This is a cross-layer concern; the
immediate alignment priority should be context conditioning.

## What the experiment says about the framework more broadly

* **Real data works with almost no tuning.** Default hyperparameters
  produced linguistically meaningful output on the first run. No
  temperature sweep, no concentration adjustment, no custom gap
  costs. That's encouraging.

* **Chunk promotion is more aggressive than expected.** 45 chunks
  from 97 pairs — roughly half a chunk per pair. Some are clear
  wins, some are bookkeeping artifacts. A tighter BIC threshold
  might help, but the real fix is rule abstraction so that
  related chunks don't accumulate as separate entries.

* **The displacement layer earns its keep.** The fact that 46
  separate segment-pair observations compressed into one
  feature-level displacement (`close → close-mid`) is exactly what
  the Bayesian hierarchical setup is supposed to do. This is a
  result we might have missed with a segment-only model.

* **The framework is not a substitute for cognate vetting.** It's
  a tool for *analyzing* cognate sets, not filtering them. Garbage
  in, garbage out.

## Follow-up: chunk over-promotion fix (2026-04-13)

After the initial experiment revealed that the learned model was over-promoting
rule-like chunks (five separate entries for the "stressed e → je"
diphthongization rule), I traced the root cause to a broken
**compositional cost calculation for asymmetric chunks**.

The old `_compositional_chunk_cost_raw` paired segments left-to-right,
which is wrong for asymmetric chunks. For `(en, jen)` it computed the
cost as `e→j + n→e + gap(n)` — two wild substitutions that inflate
the cost massively, making BIC over-eager to promote the chunk.

The fix: compute the compositional cost via a **sub-alignment** —
run a mini `align_forms` with `max_chunk_size=1` and an empty chunk
table (no recursion), then sum the raw cost over the chosen links.
This gives the true best-pairing cost for any chunk, symmetric or
asymmetric.

### Results after the fix

| metric | before | after |
|---|---|---|
| promoted chunks | 45 | 33 |
| rule-like over-promotions | 10 | 0 |
| real phonological chunks | ~30 | ~30 |

The rule-like chunks that disappeared: `(en, jen)`, `(el, jel)`,
`(ve, bje)`, `(pe, pje)`, `(de, dje)`, `(kka, ka)`, `(u, xo)`,
`(r, ir)`, `(e, re)`, `(er, r)`.

A new chunk appeared: `(e, je): cost=-1.462` — **the general
diphthongization rule**. The fix consolidated the five position-
specific variants into one generic entry, partially resolving
the rule-abstraction concern as a side effect.

All real chunks survived:
`(kt, tʃ)`, `(kte, tʃe)`, `(okt, otʃ)`, `(ll, ʎ)`, `(au, o)`,
`(ce, θ)`, `(ua, wa)`.

The full test suite (174 tests) continues to pass.

### What this teaches us

The compositional cost calculation was a hidden assumption of the
BIC comparison: "what would it cost to NOT promote this chunk?"
With a broken baseline (left-to-right pairing), the comparison was
unfair and promoted chunks that didn't deserve it. A **principled
baseline via sub-alignment** produces the right answer without any
tuning knobs.

This is a valuable design lesson: when a BIC-like comparison gives
suspicious results, check the baseline first. Our "baseline" was
the compositional cost under the assumption of no chunk; if that
assumption is computed wrongly, the whole comparison is wrong.

## Reproducing this experiment

```sh
cd /home/tiagot/nas-dev/new_chl/regulae
source .venv/bin/activate
python experiments/latin_spanish/run_experiment.py
```

The experiment script is deterministic; the same corpus produces the
same model every time.
