# regulae rule notation

Every environment regulae publishes is printed twice: once in the slot syntax
that is the contract, and once in the notation defined here.

```
  count=14 sets=14 over 1 class  margin=3.50  latin:{r}=[trill] ~ old_latin:{s}=[sibilant]
      latin: prev-syl[syllable_shape:open]  old_latin: pre[vowel:+] fol[vowel:+]
    r  ~  s  /  latin: σ⁻[open] _ ;  old_latin: V _ V
```

The first form is exact and is what a consumer should parse. The second is what
a comparativist reads. Anything ambiguous in the notation can be resolved by
looking one line up, which is why the notation is allowed to be brief.

## Why a new notation was needed

The core is agreed across the literature and is adopted unchanged: `_` for the
target position, `/` for "where", `#` for a word boundary, `∅` for zero, `{}`
for a set of segments, `[±F]` for a feature matrix, `σ` for a syllable, `…` for
"any number of segments further along".

Nothing past that core is agreed. The syllable boundary is `$` in the
handbooks, `%` in the *Index Diachronica*, `.` in Lexurgy. Cover symbols past
`V` and `C` collide between traditions. And six of regulae's eighteen
environment slots — the syllable-scoped predicates, the counted distances, the
syllable role and position, the morpheme index — have no notation anywhere.

There is also one thing regulae states that the standard notation cannot.
`A > B / X _ Y` is directional and puts the environment, implicitly, in the
ancestor's word. regulae has no ancestor: it publishes surface correspondences,
and its environment belongs to a **named lect** — the *computational
orientation*, "which of a pair's two lects a search states its environment
over". Which lect that is says what each lect preserved, so it is part of the
finding and not a formatting detail.

Hence `A ~ B / lect: X _ Y`, and hence this document.

## Frame

| | |
|---|---|
| `~` | correspondence. Never `>` or `→`: no direction is claimed |
| `/` | "where" |
| `;` | separates one lect's frame from another's |
| `_` | the target position — a *different* grapheme in each lect |
| `…` | any number of segments, including none |
| `·` | exactly one unspecified segment |
| `∅` | zero |
| `{a,b}` | a set. Braces are dropped for a single grapheme |

## Neighbours

| | |
|---|---|
| `X _` | X immediately precedes |
| `_ X` | X immediately follows |
| `X · _` | X two before; `X · · _` three before, one dot per skipped position |
| `_ · X` | X two after |
| `X … _` | X somewhere before, at any distance |
| `_ … X` | X somewhere after |

The skipped positions are spelled out rather than subscripted because a reader
checking the claim against a wordlist counts segments.

A counted-distance slot is written as **one skeleton**, not one claim per
constraint: every position from the farthest constraint in to the target
appears, carrying its constraint if it has one and a `·` if it does not. Two
before and three before together are `[+open] [+coronal] · _` — three
positions, the nearest unconstrained because offset 1 is the immediate
neighbour and belongs to a different slot.

A "somewhere" slot with more than one constraint is a conjunction of
independent existentials — a segment before carrying one, a segment before
carrying the other, possibly the same segment and possibly not — so it is
written `[+f] & [+g] … _`. `[+f] … [+g] … _` would claim an order and
`[+f,+g] … _` would claim one segment carries both; the model states neither.

## Syllable

| | |
|---|---|
| `σ⁻[X] _` | the previous syllable carries X |
| `_ σ⁺[X]` | the next syllable carries X |
| `_ σ⁼[X]` | the same syllable carries X, elsewhere in it |

Which syllable of the word the target sits in, and its role in that syllable,
are properties of the target and go inside the target bracket (below):
`σ₁` first, `σ₋₁` last, `σ₋₂` penultimate, `σ₋₃` antepenultimate;
`⟨onset⟩`, `⟨nucleus⟩`, `⟨coda⟩`, `⟨ambisyllabic⟩`.

There is **no syllable-boundary symbol**. regulae's syllable predicates say
"a syllable carries feature X", not "a boundary falls here", and borrowing `$`
or `%` would describe something the model does not claim.

## Edges

One system serves the word and the morpheme alike. The symbol adjacent to the
target means the target is *at* that edge; the symbol with an ellipsis between
it and the target means the target is somewhere strictly inside; the symbol on
both sides, adjacent, means the target is the whole span.

| | |
|---|---|
| `# _` | word-initial |
| `_ #` | word-final |
| `# … _ … #` | word-internal |
| `+ _` | morpheme-initial |
| `_ +` | morpheme-final |
| `+ _ +` | the target is a whole morpheme |
| `+ … _ … +` | morpheme-internal |

## The target itself

Everything said about the target segment, rather than about its neighbours,
goes in one bracket around the underscore:

```
[_ +long]            the target is long
[_ ⟨coda⟩]           the target is in a coda
[_ σ₋₂]              the target is in the penultimate syllable
[_ ´]                the target bears primary stress
[_ ⟨morpheme 2⟩]     the target is in the third morpheme
[_ +long, ⟨coda⟩]    both
```

The brackets go *around* the underscore rather than beside it so that nothing
about the target is one space away from being a claim about the following
segment. `_ [+long]` and `_[+long]` would otherwise be different claims that
look alike.

`⟨morpheme n⟩` counts from the start of the word and is zero-based. The
counter saturates: `⟨morpheme 7+⟩` means the seventh or any later one.

## Stress

`´` primary, `ˋ` secondary, on whichever position bears it:

| | |
|---|---|
| `[´] _` | the preceding segment bears primary stress |
| `_ [´]` | the following segment bears primary stress |
| `[_ ´]` | the target bears primary stress |

Any other stress value is written out as `stress=value`.

## Feature matrices

| | |
|---|---|
| `[+f]` | the feature is present |
| `[−f]` | the feature is absent. U+2212, not a hyphen — feature names contain hyphens (`close-mid`) |
| `[+f,+g]` | both, on one segment |
| `[dim=value]` | a dimension-valued constraint, e.g. `[tone=2]` |

Inside a σ bracket the `syllable_` prefix is dropped, since the three syllable
dimensions have disjoint vocabularies: `σ⁻[open]` is `syllable_shape:open`,
`σ⁺[short]` is `syllable_nucleus:short`, `σ⁻[heavy]` is
`syllable_weight:heavy`.

## Cover symbols

Each abbreviates exactly **one** positive merkmal feature. They are used only
where a bare symbol can stand alone — the adjacent, counted-distance and
"somewhere" positions. Inside a σ bracket or a target bracket, constraints are
always written as matrices, so that one abbreviation never sits beside the
expansion of the same kind of thing.

| | | | | | | |
|---|---|---|---|---|---|---|
| `V` | vowel | `S` | stop | `J` | approximant | |
| `C` | consonant | `A` | affricate | `Z` | continuant | |
| `N` | nasal | `O` | obstruent | `P` | labial | |
| `F` | fricative | `R` | sonorant | `K` | velar | |
| `E` | front | `B` | back | | | |

Never for a negated feature: `voiced:-` is `[−voiced]` and never a symbol.

The set is the *Index Diachronica*'s, restricted to the symbols that survive
contact with merkmal's feature vocabulary. Dropped, with reasons:

- `L` liquid, `W` semivowel, `Q` uvular, `M` diphthong — merkmal has no single
  feature for any of them.
- `U` syllable — would collide with `σ`.
- `H` laryngeal — merkmal has separate `glottal` and `guttural`.
- `D` voiced plosive, `T` voiceless plosive — conjunctions. Writing
  `[+voiced,+stop]` as `D` would make one symbol say what the matrix notation
  already says, and say it only sometimes.

Three of the survivors diverge from *Index Diachronica*, and a reader who
misses the divergence misreads the output:

- **`E` and `B` are not vowel-restricted.** They abbreviate merkmal's `front`
  and `back`, which apply to consonants too. *Index Diachronica*'s mean front
  and back *vowel*.
- **`S` is `stop`.** Much handbook use has `S` for a sibilant. Here `sibilant`
  renders `[+sibilant]`.
- **`F` is `fricative`.** Some traditions use `F` for a front vowel; that is
  `E` here.

These are feature *names*, so they hold for the `descriptive` vocabulary
regulae defaults to. Under another feature system the names do not match, no
symbol fires, and every constraint renders as a matrix — correct, and less
pretty.

## Attribution

The lect label is dropped only when **every** side states a frame and all of
them are identical. Every other case keeps its labels, including the common one
where only a single lect conditions.

```
r ~ s / old_latin: V _ V                         one side conditions
r ~ s / latin: σ⁻[open] _ ;  old_latin: V _ V    both do, differently
eː ~ iː / _ [+close]                             both do, identically
p ~ p ~ f / ancestor: _ [+close] ;
            conservative: _ [+close]             two of three do
```

The last case is why the rule is demanding. `ancestor` and `conservative` both
condition on a following close vowel and `innovator` does not; an unlabelled
`p ~ p ~ f / _ [+close]` would say the frame holds in the innovator too, which
is the opposite of what was found.

## Cross-dimensional rules

One lect carries a suprasegmental value where another lect's environment holds.
Not a correspondence, so not written with `~`:

```
daughter [tone=⁵⁵]  /  proto: [−voiced] _
daughter [tone=⁵⁵]  /  [−voiced] _
```

The second is a lect-internal rule: the environment is read in the same lect
that carries the value, and that lect is already named, so an unlabelled frame
can only be its own.

## Where it appears

- the human model report, indented under the row it restates
- `--json`, as a `notation` string on conditioned classes, proposed events and
  cross-dimensional rules. Present only where some side states an environment
- the web reports, under the exact environment

**Not** on alignment links, though they carry an `rg_context_spec` like
everything else here. A link's context is not a rule: it is the exhaustive
description of one segment's surroundings — every feature of both neighbours,
the union over everything anywhere before and after, every syllable, both edges
— which is the space rules are *searched over*. A committed rule keeps one or
two constraints out of it. Rendered as a rule it comes out as a hundred-conjunct
frame that no reader would write and no search committed, and it would read as
a finding because everything else in this notation is one.

## What was surveyed

The claim that nothing past the core is agreed rests on these, and each
contributed something here:

- **Chomsky & Halle's `A → B / X _ Y`**, as it reaches the field through the
  handbooks and rule-writing guides — the target slot, the slash, `#`, `∅`,
  braces, feature matrices, `σ`. Everything regulae adopts unchanged.
- **[*Index Diachronica*](https://chridd.nfshost.com/diachronica/)**, whose key
  is the most widely circulated cover-symbol set there is: `{}` for a set of
  segments, `(…X)` for "any number remaining", `!` for "except", `%` for a
  syllable boundary, and the V C N F S A O R L J W Z / P K Q E B M U H D T
  series this restricts.
- **[Rosenfelder's SCA²](https://www.zompist.com/scahelp.html)**, where `…`
  means "anywhere later in the word" — the shape `_ … X` takes here.
- **[Lexurgy](https://www.meamoria.com/lexurgy/html/sc.html)** and
  **[Brassica](https://bradrn.com/brassica/)**, the two sound-change appliers
  with worked-out feature and suprasegmental syntax, which use `.` and `$` for
  boundaries — a third and fourth answer to the same question.

The disagreements are not incidental. A syllable boundary is `$`, `%` or `.`
depending on which of those you read; `F` is a fricative in one tradition and a
front vowel in another; `S` is a plosive in *Index Diachronica* and a sibilant
in common handbook use. And none of them writes a correspondence: they all
write a change, from something to something, in one language.

## Worked examples

```
{b,z}      ~  {f,s}       /  pgmc: _ [´]                   Verner
{k,t}      ~  {kʰ,tʰ}     /  pie: _ … [+aspirated]         Grassmann
{v,z,ð,ɣ}  ~  {f,s,x,θ}   /  proto: V _                    intervocalic voicing
{k,p,t}    ~  {b,d,ɣ}     /  latin: V _                    lenition
r          ~  s           /  latin: σ⁻[open] _ ;
                             old_latin: V _ V              rhotacism
k          ~  tʃ          /  A: R _                        palatalisation
oː         ~  a           /  pgmc: _ · F
f          ~  p           /  proto: _ … N
s          ~  s           /  old_latin: + _ σ⁺[closed]
```
