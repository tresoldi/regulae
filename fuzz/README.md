# Fuzz harnesses

One libFuzzer target per parse entry point. These are the only places regulae
reads input nobody wrote for it, and they already take a buffer rather than a
path — the WebAssembly build, which links without a filesystem, forced that
shape, so no refactoring was needed to fuzz them.

```
fuzz_parse_tsv            rg_corpus_parse_tsv
fuzz_parse_wide_tsv       rg_corpus_parse_wide_tsv   (also merkmal's segmenter)
fuzz_parse_gled           rg_corpus_parse_gled
fuzz_parse_arcaverborum   rg_corpus_parse_arcaverborum
```

## Running them

```sh
cmake -S . -B build/fuzz -DCMAKE_C_COMPILER=clang \
    -DREGULAE_BUILD_FUZZERS=ON -DREGULAE_ENABLE_SANITIZER=address \
    -DREGULAE_BUILD_TESTS=OFF -DREGULAE_BUILD_CLI=OFF
cmake --build build/fuzz -j"$(nproc)"

mkdir -p build/fuzz/corpus
./build/fuzz/fuzz_parse_tsv build/fuzz/corpus testdata/corpora testdata/soundlaws
```

The first path is the corpus libFuzzer **writes** to; the rest are read-only
seeds. Point the first one at `testdata/` and it will start adding files to the
repository.

`scripts/check.sh --full` builds all four and runs each for fifteen seconds.
`REGULAE_FUZZ_SECONDS` raises that. Fifteen seconds is a regression check, not
a proof.

## What the first run found

Within the first runs, three defects in the loaders, all fixed and now regression
tests in `tests/c/test_loaders.c` rather than something only a fuzzer reaches:

- **A read past the end of a truncated stress mark.** `lift_stress_mark`
  established that `grapheme[0]` was `0xCB` and then read `grapheme[2]` to see
  whether anything followed the mark. A lone `0xCB` is one byte and a NUL, so
  index 2 is past the allocation. A file cut at a byte boundary produces this.
- **A leak on a refused row.** A stress column with fewer values than the row
  has segments is correctly refused, and that one branch of eight in the row
  loop had been written without the two `free` calls the other seven have.
- **A leak on a stressed tone-only token.** Tone lifting attaches a standalone
  Chao digit to the previous segment and removes its token. A malformed token
  carrying both a stress mark and only tone also owned the lifted stress value;
  removing it abandoned that allocation. Such a token is now retained for
  diagnosis rather than guessed away.

After the fixes, 3.2 million executions across the four targets found nothing
further.

## Notes

`fuzz_parse_wide_tsv` runs roughly thirty times slower than the others because
it segments through merkmal. merkmal itself is **not** instrumented — the
coverage flags are a directory property set after its `add_subdirectory`, so
inputs reach its segmenter but the fuzzer is blind inside it, and ASAN does not
watch its allocations either. Instrumenting a dependency is a separate
decision; this is recorded so the coverage is not mistaken for more than it is.
