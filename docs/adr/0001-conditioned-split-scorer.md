# ADR 0001: keep corrected BIC as the conditioned-split default

Status: accepted on 2026-08-16.

## Context

Pairwise conditioned correspondences, multi-lect conditioned classes and
cross-dimensional associations all compare one pooled categorical distribution
with two distributions induced by an environment. Earlier, two copies of the
operation used a corrected BIC-shaped score while cross-dimensional discovery
implemented the likelihood comparison separately. The multi-lect immediate
pass also added `2/(n-1)` to `ln(n)`, without a derivation for a BIC penalty.

The three discovery stages were placed behind one categorical-scoring seam, and
corrected BIC, exact multinomial normalized maximum likelihood (NML)
and a symmetric-Dirichlet marginal likelihood were compared. NML follows the exact
multinomial normalization and recurrence described by
[Kontkanen et al. (2003)](https://proceedings.mlr.press/r4/kontkanen03a.html)
and [Kontkanen and Myllymäki (2007)](https://doi.org/10.1016/j.ipl.2007.04.003).

The protocol and development/held-back split are recorded in
`testdata/evaluation/m3/protocol.json`. Protocol version 3 records two
amendments. The first repaired an unreachable fractional-mass requirement and
made the un-derived small-sample term ineligible. The second added the
repository's pre-existing `chance`, `diffusion`, and `stratum` restraint
fixtures as a mandatory development gate after the previously selected
configuration failed `chance.tsv`. No candidate formula or original synthetic
metric was changed; the amendment prevents a generated benchmark from
overruling an older, linguistically stronger negative control.

## Decision

Keep the corrected BIC-shaped score as the production default:

```text
delta = 2 (split_nll - pooled_nll)
      + (K - 1) ln(n)
      + gamma 2 ln(distinct_partitions)
```

with a zero score threshold and `gamma = 1`, the full charge for choosing one
partition from the observed model space. Remove the
multi-lect `2/(n-1)` addition from the default. It remains an explicit CLI and
C-option compatibility experiment, not part of the selected model.

`distinct_partitions` counts the different unordered two-way divisions
candidates make on the observations being searched. Two names for the same
division, including predicates that exchange its yes and no sides, are one
search opportunity. Correlated but non-identical divisions remain separate.
This makes the score invariant to duplicate predicate encodings.

The pooled observed outcomes define the alphabet for every criterion, and the
same alphabet is used on both sides even when a value is absent from one side.
Here `n` remains weighted aligned-position exposure within the source/class
bucket. It is not presented as a count of independent lexical histories;
grouped resampling and held-out evaluation address that separate dependence
question.

NML and the Dirichlet marginal likelihood remain selectable experimental
scorers. Published evidence names its scorer and exposes `delta_score`.
`delta_bic` remains a compatibility alias and must not be interpreted as BIC
when another scorer is named.

## Evidence

Under the recorded development gates:

- corrected BIC with the full model-space charge had null FWER `0.000`,
  negative-cell abstention `1.000`, intended-association power `1.000` for
  `n >= 16`, mean synthetic held-out log-loss gain `0.4390`, exact-order
  stability `0.700`, runtime ratio `1.00`, and abstention on all three
  established no-environment fixtures;
- full-model-code NML failed the required-input gate, because exact NML has no
  defined integer-sample normalizer for fractional confidence mass, failed
  established restraint, and selected an association in `4/60` generated null
  corpora (`FWER = 0.067`);
- symmetric-Dirichlet total prior mass `0.5` met the generated null, power and
  runtime gates but failed established restraint; masses `1`, `2`, and `5`
  additionally had null FWER `0.067`;
- half-strength search charges for NML and Dirichlet also had null FWER `0.067`;
- the legacy corrected-BIC configuration with the un-derived multi-lect
  addition passed established restraint but remained ineligible by protocol.

Corrected BIC with the full model-space charge was the only eligible
configuration. On
the held-back partition it obtained factorial positive power `0.824`, negative
abstention `1.000`, null FWER `0.000`, power `1.000` at `n >= 16`, and mean
synthetic predictive log-loss gain `0.4576`.

These results do not establish BIC as universally optimal. Several criteria
tie on the clean synthetic panels, which are easy once a search charge is
applied, and twenty null replicates per kind give a wide uncertainty interval
around zero. The established restraint fixtures provide the discrimination
the generated panel did not. The decision is to retain the only compatible
default supported by all current evidence, not to treat the approximation as
a linguistic truth.

## Consequences

- The selected default removes the un-derived multi-lect addition, counts
  distinct rather than raw candidate partitions, changes `gamma` from `0.5`
  to `1`, and removes the redundant negative score buffers.
- Exact NML returns `RG_ERR_UNSUPPORTED_OPTION` when a scored split contains
  fractional mass. It never rounds confidence weights into fictitious tokens.
- The Dirichlet scorer uses a separate symmetric total concentration. It does
  not reuse the feature-distance prior or concentration from alignment EM.
- Permutation nulls remain validation gates rather than being folded into the
  within-sample score. The pairing shuffle still supports the production
  standing verdict.
- The narrow fixed-position predictive panel must be replaced with nested,
  group-held-out prediction over learned alignments and arbitrary environments.
