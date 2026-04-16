# Alignment Details: Concrete Design

## Link structure

Each alignment link carries:

```
Link {
  source_chunk:   one or more segments from Lect A (can be empty)
  target_chunk:   one or more segments from Lect B (can be empty)
  context: {
    position:      word-initial | medial | final | ...
    preceding:     feature specification (merkmal) or nil
    following:     feature specification (merkmal) or nil
    morphological: stem | prefix | suffix | boundary | any
  }
  feature_displacement: {
    dimension_1: value_A -> value_B,
    dimension_2: value_A -> value_B,
    ...
  }
  confidence:     float [0, 1]
  correspondence_class: latent class ID (for multi-lect reconciliation)
}
```

## Feature displacement vectors

Each link includes the merkmal feature decomposition of the correspondence.

Key property: SYSTEMATIC correspondences share the same displacement vector
across different places/manners. This makes the correspondence system
*generative* — once {continuant: - -> +} is established across voiceless
stops, it predicts correspondences for new segment pairs.

Example — Grimm-type correspondences all share {continuant: - -> +}:
  p ~ f  displacement: {continuant: - -> +}  (same)
  t ~ θ  displacement: {continuant: - -> +}  (same)
  k ~ h  displacement: {continuant: - -> +}  (same)

The regularity is at the feature level, not the segment level.

## Conditioning context

Full conditioning tuple: (position, preceding, following, morphological).
Specifications are in merkmal features, not specific segments.

**Most context is immediate** (one segment to the left/right). But the
framework must support:

(a) **Long-range context.** Some correspondences depend on segments beyond
    the immediate neighbors (vowel harmony triggers, distance assimilation,
    syllable-level effects).

(b) **Combining context.** Sometimes the conditioning is not a single
    segment but a *combination* of segments acting together. The condition
    fires only when all members of the combination are present
    (e.g., "before [+front] vowel AND in stressed syllable").

Context specifications must support both single-segment and
multi-segment-combination conditions.

Context granularity discovery: start coarse, split when finer conditioning
yields better correspondence patterns — *always driven by evidence, never
stipulated*. E.g., start with "k ~ tS / before front vowel" and split to
"k ~ tS / before front non-high" vs "k ~ s / before front high" only when
the data support it.

Example — Latin /k/ in French with fine conditioning:
  k ~ ʃ  / _{+front, -high}    (caput > chef, cantare > chanter)
  k ~ s  / _{+front, +high}    (centum > cent, caelum > ciel)
  k ~ k  / _{-front}           (cor > coeur, clamo > clame)

## Chunk correspondence learning

Start compositional (scoring from merkmal). Promote recurring chunk
correspondences to first-class phrase-table entries.

No fixed maximum chunk size — the model discovers granularity.
Constraint: chunks should be supported by multiple independent form pairs.

Example — Latin /kt/ ~ Italian /tt/:
  Initially scored as two links: k~t + t~t (compositional)
  After seeing noctem~notte, octo~otto, factum~fatto:
  Promoted to single chunk link: [kt] ~ [tt] with own probability

## Two-level architecture for complex phenomena

Phenomena spanning multiple segments or feature dimensions are captured at
the SYSTEM level, not forced into word-level alignment:

### System-level correspondence patterns:
- Vowel harmony: feature spreading in A <-> independent specs in B
- Tonogenesis: laryngeal contrast <-> tonal contrast
- Regular feature displacement: {feature: val_A -> val_B} across a class
- Mergers: two categories in A <-> one category in B

### Word-level alignment:
- Uses system-level patterns as priors/constraints
- Handles word-specific residual detail
- Feeds back anomalies to system level

The two levels iterate (EM-style).

### Tonogenesis example:
System level discovers:
  Lect A [+voice] on onset stops <-> Lect B [low tone] on vowels
  Lect A [-voice] on onset stops <-> Lect B [high tone] on vowels
  
Word level then uses this:
  Lect A "ba" ~ Lect B "pa[low]"  -- system pattern applied
  
Displacement is CROSS-DIMENSIONAL:
  consonant.voice -> vowel.tone  (coupled displacement)

### Harmony example:
System level discovers:
  Lect A has [+/-front] harmony across root+suffix domain
  Lect B has independent vowel specifications per syllable
  
Word level treats all harmonizing vowels in A as one specification,
not as independent segments.

## Multi-lect correspondence classes

With N lects, each position in aligned forms is assigned to a latent
correspondence class. Classes are N-tuples (one segment/chunk per lect).

**No upper bound on N.** Each additional lect strengthens or refines
class assignments.

**Robustness to missing/noisy data is mandatory:**
- A lect may be missing the cognate entirely (replacement, lexical loss)
- A form may be attested in only some of the lects
- Some lects have far less data than others
- Transcription noise varies across sources
- Cognates may be uncertain (the form pair itself may be wrong)

The model must operate gracefully when any of these hold. A correspondence
class is valid even if only a subset of lects contribute evidence to it.

Number of classes is inferred (Dirichlet process or similar).
Each additional lect provides evidence for splitting or merging classes.

Example — Polynesian merger disambiguation:
  Class C1: Hawaiian k ~ Tongan t ~ Samoan t ~ Maori t ~ Rapanui t
  Class C2: Hawaiian k ~ Tongan k ~ Samoan ʔ ~ Maori k ~ Rapanui k

Workflow:
1. Pairwise alignments first (the primitive operation)
2. Reconcile into multi-lect correspondence classes
3. Multi-lect classes feed back into pairwise alignments (iterative)
4. Each additional lect strengthens or refines class assignments

## Staged discovery

Complex patterns (tonogenesis, harmony, context-dependent splits) require
simpler patterns to be established first. The training pipeline is
therefore staged: each stage reduces the residual structure that the
next stage has to explain. By the time anomaly detection runs for
cross-dimensional rule discovery, most systematic segmental
correspondences and feature-level regularities have already been
captured — anomaly detection then focuses on what's left, which is
exactly the kind of phenomenon where residual-MI tests are most
useful.

The full staging and its rationale live in `docs/training_pipeline.md`.
The discovery mechanisms for each kind of correspondence
(segment-level, context-conditioned, tonal, cross-dimensional,
long-range) live in `docs/correspondence_discovery.md`.

## Open design questions

- **Expert annotation.** How does expert annotation enter the
  alignment — hard constraints on specific links, prior weights on
  correspondence types, soft penalties? No urgency until there is a
  real expert-annotated corpus to drive the decision.
- **Data asymmetry across lects.** Multi-lect reconciliation currently
  trusts every pairwise model equally. With very different amounts of
  data across lects, some pairwise models will be much noisier; the
  reconciliation layer may need to down-weight noisy pairs explicitly
  rather than relying on the union-find majority.
- **Expert-assisted cognate curation.** `find_cognate_outliers` ranks
  outliers but doesn't close the loop: there's no convenient workflow
  for "drop these N outliers, retrain, compare". The diagnostics
  layer could grow an explicit curation loop.
