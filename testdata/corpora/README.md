# Test corpora

Small corpora used by the C tests, the CLI smoke tests and `scripts/bench.sh`.
Between them they cover three lects with partial pair coverage,
correspondences supported by a subset of lects, conditioned and unconditioned
classes for the same segment, class reconciliation across unequal alignments
and gaps, zero-confidence and reduced-confidence sets, morpheme boundaries
through the arcaverborum loader, long-range syllable conditioning, tone, and
five sets derived from the experiment data.

`tonogenesis_cldf.tsv` is the odd one out and is here for a reason worth
stating. It writes tone the way every CLDF wordlist for a tonal language does —
`b a ⁵⁵`, the Chao token separate, after the syllable it belongs to — and its
voiced onsets condition a low tone in the daughter. Read on 2026-08-15 it
produced nothing: the tone token stayed a segment, no segment carried a tone,
and cross-dimensional discovery had nothing to work from, while the *same*
forms through the wide loader produced a tonal correspondence. It now comes out
as `pre[voiced:+] -> tone=¹¹` at confidence 1.00. The corpus exists so that the
CLDF shape is exercised end to end and not only at the loader.

The other three suites are elsewhere and ask different questions:
`../soundlaws/` whether a settled change can be found, `../restraint/` whether
a change that is not there can be declined, `../diagnostics/` whether a corpus
that is wrong can be identified as wrong. Each has its own README.

These were the parity corpora: until merkmal 1.0 they were run through both
the C implementation and the frozen Go reference and diffed. That comparison
ended with the reference — see the parity section of
`docs/c_conversion_roadmap.md` — and the corpora outlived it because the
coverage they were chosen for is still the coverage the C tests need.
