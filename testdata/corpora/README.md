# Test corpora

Small corpora used by the C tests, the CLI smoke tests and `scripts/bench.sh`.
Between them they cover three lects with partial pair coverage,
correspondences supported by a subset of lects, conditioned and unconditioned
classes for the same segment, class reconciliation across unequal alignments
and gaps, zero-confidence and reduced-confidence sets, morpheme boundaries
through the arcaverborum loader, long-range syllable conditioning, tone, and
five sets derived from the experiment data.

These were the parity corpora: until merkmal 1.0 they were run through both
the C implementation and the frozen Go reference and diffed. That comparison
ended with the reference — see the parity section of
`docs/c_conversion_roadmap.md` — and the corpora outlived it because the
coverage they were chosen for is still the coverage the C tests need.
