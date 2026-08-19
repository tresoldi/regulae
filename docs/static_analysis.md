# Static analysis

`clang-tidy`, configured by `.clang-tidy`, run by `scripts/tidy.sh`. The
baseline is **zero findings** across `src/`, `cmd/`, `tests/c/` and `fuzz/`, and
`scripts/check.sh --full` and CI both fail on any output.

```sh
./scripts/tidy.sh              # about two minutes
```

Two minutes is why it is not on the default `check.sh` path.

## What is enabled, and what is not

`clang-analyzer-*` is the reason this exists. It is path-sensitive, so it finds
what a compiler warning by construction cannot — a leak on the third branch of
an error path, a pointer that is null on one route through a function. Planting
a use-after-free and an uninitialised return in a scratch file confirmed it was
actually running before any of the results below were believed.

`bugprone-*` and `misc-*` are on. Most of `readability-*` is a style opinion
this project does not share, so three checks are opted into individually rather
than the family being enabled and then argued with.

Three checkers are disabled, each with its reason in `.clang-tidy`:
`bugprone-multi-level-implicit-pointer-conversion` (78 findings, every one an
ordinary `free`, `memcpy`, `realloc` or `qsort` on an array of pointers),
`clang-analyzer-security.ArrayBound` (13, all on heap arrays whose extent is
held in a count the analyzer does not relate to the allocation), and
`clang-analyzer-optin.portability.UnixAPI` (2, both guarded zero-size
allocations). `bugprone-easily-swappable-parameters` and
`bugprone-narrowing-conversions` are off as well — the first fires on every
`(source, target)` pair in the library, and the second duplicates `-Wconversion`,
which is already an error under `check.sh`.

## The first baseline: 27 findings

Four were real and are fixed.

**An uninitialised diagnosis read by the CLI.** `report_load_failure` reads
`diagnosis->message[0]`, and every loader clears the struct on entry — but
`load_corpus` returns `RG_ERR_UNSUPPORTED_OPTION` for an unrecognised format
without reaching a loader, so the CLI read uninitialised stack. Introduced in
the same session, and reachable only by passing `--format` a value
the CLI does not know.

**An `ftell` that can return -1.** `tests/c/test_loaders.c` did
`malloc((size_t)size + 1)`, and `(size_t)-1 + 1` is 0: a zero-byte allocation
followed by `text[size]` writing before the buffer.

**Two `qsort` calls that could pass NULL.** `qsort`'s parameters are
non-null-qualified, so a NULL with a count of zero is undefined even though
every implementation in existence ignores it. Guarded.

Two more were real and were found in the same pass, in `loaders.c`:
`parse_record` appends fields as it goes and can fail after some are built,
leaving the partly-filled row to its caller; neither caller released it. That is
a leak on the out-of-memory path, of the same shape as one the fuzzer found
earlier — and one no fuzzer would have found, because fuzzing does not produce
allocation failures. `loader_row_clear` now exists and both callers use it.

## The six suppressions

Six findings are suppressed, each with the reason on the line above the first
annotation. They span twelve `NOLINTNEXTLINE` lines, because a finding reported
at several points in one function needs one per point; the reason is written
once, at the first.

Two of the three in library code are one root cause, worth stating plainly
because it is a cost of a decision taken deliberately elsewhere:

**`rg_free_owned_internal` is opaque to the analyzer.** It removes the `const`
from library-owned storage by copying the pointer value with `memcpy`, which is
what keeps the conversion defined and `-Wcast-qual` clean. The analyzer does not
follow the allocation's identity across that copy, so every free that goes
through it reads as a leak. Writing the copy as a union instead was tried and
changes nothing. The alternative is ninety-odd unexplained casts, which is the
thing the helper exists to prevent, so it is annotated rather than unwound:
`src/model_chunks.c` in the chunk promoter, and `src/search_features.c` where
the feature matrix is released on an error path.

The third, in `src/vocabulary.c`, is a correlation the analyzer does not carry:
the entry array is null only when the count is zero, and the loop it guards then
does not run. The two are set together by the append helper and it does not
follow that far.

The fourth, in `src/predictive.c`, is another lost correlation: a zeroed array
of probability buffers is allocated by an indexed loop and released by an
indexed loop over the same bounds. The analyzer treats a later index as though
it could replace an earlier allocation, then no longer associates that
allocation with the pointer passed to the clear helper. The full success and
partial-failure paths both release every buffer that can have been allocated.

Two more live in the tests, for deliberate behaviour: an `assert` with a side
effect, in the file whose whole purpose is to detect a build where `NDEBUG`
deleted the assertions, and a cast of an out-of-range value to an enum, in the
test that the library refuses one.

**A suppression is not the first resort.** A `clang-analyzer-core.NullDereference`
on the feature cache's entry lookup was fixed on 2026-08-16 by changing the
function to return the entry rather than write it through an out-param, so
"no entry" and "the lookup failed" became one condition the analyzer can see,
instead of a postcondition that held only by inspection. Annotating it would
have hidden a contract that genuinely was not enforced.

## Two measurement errors worth remembering

Both produced a *clean* result from a broken measurement, which is the
direction that matters.

`Checks:` in `.clang-tidy` is a YAML block scalar, so a `#` inside it is part of
the check list rather than a comment. Putting the reasoning there corrupted the
list silently and turned a real baseline into 101 unrelated findings.

Filtering the output with `grep -oE '\[[a-z0-9.-]+\]$'` drops every
`clang-analyzer` check, because their names are CamelCase —
`clang-analyzer-security.ArrayBound`. That reported zero findings when there
were 27. Count the `warning:` lines, not the tags.
