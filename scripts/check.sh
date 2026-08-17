#!/usr/bin/env bash
#
# Everything that has to hold before a commit, in one place, failing loudly.
#
#   scripts/check.sh            native build, tests, generated artifacts
#   scripts/check.sh --full     the above plus AddressSanitizer and WebAssembly
#
# This exists because two separate sessions' worth of work were verified
# against artifacts that had not been rebuilt. Once a probe binary was not
# relinked; once `cmake --build` failed at the configure step and the previous
# library stayed in place, while the check being used -- piping the build into
# `grep -c error` -- reported nothing wrong, because CMake writes "Error". A
# reported "100% tests passed" was not true.
#
# So: no greps for the word "error". Every step is checked by its exit status,
# `set -e` stops at the first failure, and the build's own output is shown when
# it fails. The generated artifacts are regenerated and the tree is then
# required to be clean, which is the only way to be sure they were current
# rather than merely present.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

full=0
if [ "${1:-}" = "--full" ]; then
    full=1
fi

step() {
    printf '\n=== %s\n' "$1"
}

fail() {
    printf '\nFAILED: %s\n' "$1" >&2
    exit 1
}

step "native build"
# -Werror here rather than in the build's defaults. A warning is a reason to
# stop before committing; it is not a reason to deny a downstream consumer on
# some other compiler a build at all. The flags themselves are a directory
# property in CMakeLists.txt, so every target carries them -- the CLI, the
# tests and the WebAssembly shim all compiled with none until 2026-08-15.
cmake -S . -B build/c -DREGULAE_WERROR=ON >/dev/null || fail "cmake configure"
# Not piped: a configure failure has to be visible, and piping it into a
# filter is exactly how it was missed before.
cmake --build build/c -j"$(nproc)" || fail "cmake build"

# Each artifact must be newer than every source it is built from. A build
# system that silently declines to rebuild is the failure this whole script is
# for.
#
# The CLI is checked against its own binary rather than against the archive.
# cmd/ was in the archive's list, which is wrong in a way that only shows on a
# commit that touches the CLI and nothing else: main.c is not compiled into
# libregulae.a, so it stays newer than the archive forever and the step fails
# with a message naming the wrong artifact.
step "build is current"
newest_source="$(find src include third_party -newer build/c/libregulae.a -type f 2>/dev/null | head -1 || true)"
if [ -n "$newest_source" ]; then
    fail "libregulae.a is older than $newest_source"
fi
newest_cli_source="$(find cmd -newer build/c/regulae -type f 2>/dev/null | head -1 || true)"
if [ -n "$newest_cli_source" ]; then
    fail "the regulae binary is older than $newest_cli_source"
fi

# The WebAssembly artifacts are committed and the native suite checks them
# against the native build, so they have to be rebuilt *before* the tests run,
# not after. Getting this order wrong is what the first run of this script
# found.
if [ "$full" = "1" ]; then
    step "WebAssembly build"
    if [ -z "${EMSDK:-}" ]; then
        fail "--full needs the Emscripten SDK: source ~/emsdk/emsdk_env.sh"
    fi
    ./web/build-wasm.sh >/dev/null || fail "web/build-wasm.sh"
fi

step "generated artifacts"
python3 scripts/capabilities.py >/dev/null || fail "scripts/capabilities.py"
python3 scripts/corpora.py >/dev/null || fail "scripts/corpora.py"
python3 scripts/guide.py >/dev/null || fail "scripts/guide.py"

# The fixture generators, for the same reason and with the same consequence:
# the staleness gate below reads `git status`, so a generator that no longer
# produces its committed fixture fails here rather than at the next person to
# run it. `graded_3_stress.tsv` could not be reproduced from its generator for
# months -- Python salts string hashes per process and the rung keyed on one --
# and nothing noticed, because the rung kept passing.
python3 scripts/graded.py >/dev/null || fail "scripts/graded.py"
python3 scripts/restraint.py >/dev/null || fail "scripts/restraint.py"
python3 scripts/diagnostics.py >/dev/null || fail "scripts/diagnostics.py"
python3 scripts/linguistic_probes.py >/dev/null || fail "scripts/linguistic_probes.py"
python3 scripts/evaluate_m2.py >/dev/null || fail "scripts/evaluate_m2.py"
python3 scripts/evaluate_m3.py --verify >/dev/null || fail "scripts/evaluate_m3.py"
python3 scripts/evaluate_m5.py >/dev/null || fail "scripts/evaluate_m5.py"

# A second compiler, because every assumption GCC happens to be lenient about
# is otherwise untested -- and until 2026-08-15 the string "clang" appeared
# nowhere in this repository. The build is the cheap part and catches almost
# all of the divergence: two seconds against the twenty the suite costs. So the
# default path builds under clang and --full also runs the suite there, which
# is where a difference that only shows at runtime would surface. CI runs both,
# unconditionally.
if command -v clang >/dev/null 2>&1; then
    step "second compiler"
    cmake -S . -B build/c-clang -DCMAKE_C_COMPILER=clang -DREGULAE_WERROR=ON >/dev/null \
        || fail "clang configure"
    cmake --build build/c-clang -j"$(nproc)" || fail "clang build"
    if [ "$full" = "1" ]; then
        ctest --test-dir build/c-clang --output-on-failure || fail "clang ctest"
    fi
else
    printf '\n=== second compiler: clang not installed, skipped (CI runs it)\n'
fi

step "native tests"
ctest --test-dir build/c --output-on-failure || fail "ctest"

# The loaders are the only part of regulae that reads input nobody wrote for
# it. A short run is not a proof; it is enough to catch a regression in the
# parsing paths, and the two defects the first run found -- a read past the end
# of a truncated stress mark, and a leak on a refused row -- are regression
# tests in tests/c/test_loaders.c now rather than something only a fuzzer
# reaches. REGULAE_FUZZ_SECONDS raises the per-target time for a real session.
if [ "$full" = "1" ] && command -v clang >/dev/null 2>&1; then
    step "fuzz harnesses"
    cmake -S . -B build/c-fuzz -DCMAKE_C_COMPILER=clang \
        -DREGULAE_BUILD_FUZZERS=ON -DREGULAE_ENABLE_SANITIZER=address \
        -DREGULAE_BUILD_TESTS=OFF -DREGULAE_BUILD_CLI=OFF -DREGULAE_WERROR=ON \
        >/dev/null || fail "fuzz configure"
    cmake --build build/c-fuzz -j"$(nproc)" >/dev/null || fail "fuzz build"
    mkdir -p build/c-fuzz/corpus
    for target in parse_tsv parse_wide_tsv parse_gled parse_arcaverborum; do
        # The first path is the corpus libFuzzer *writes* to, so it has to be
        # under build/; the rest are read-only seeds.
        "./build/c-fuzz/fuzz_$target" \
            -max_total_time="${REGULAE_FUZZ_SECONDS:-15}" \
            -artifact_prefix=build/c-fuzz/ \
            build/c-fuzz/corpus testdata/corpora testdata/soundlaws \
            || fail "fuzz_$target found something; the input is under build/c-fuzz/"
    done
fi

# Two minutes, so --full only. The baseline is zero and the script says what to
# do when it is not; docs/static_analysis.md has the reasoning behind every
# disabled checker and every annotation.
if [ "$full" = "1" ] && command -v clang-tidy >/dev/null 2>&1; then
    step "static analysis"
    ./scripts/tidy.sh build/c-tidy || fail "clang-tidy"
fi

if [ "$full" = "1" ]; then
    step "sanitizer build and tests"
    cmake -S . -B build/c-asan -DREGULAE_ENABLE_SANITIZER=address -DREGULAE_WERROR=ON >/dev/null || fail "asan configure"
    cmake --build build/c-asan -j"$(nproc)" || fail "asan build"
    ctest --test-dir build/c-asan --output-on-failure || fail "asan ctest"

    # The build adds -fno-sanitize-recover=all for this one, so a finding is a
    # failing exit status rather than a line of output nobody reads. It was
    # skipped here for weeks on the belief that this machine could not link
    # libubsan; it can, and could all along.
    cmake -S . -B build/c-ubsan -DREGULAE_ENABLE_SANITIZER=undefined -DREGULAE_WERROR=ON >/dev/null || fail "ubsan configure"
    cmake --build build/c-ubsan -j"$(nproc)" || fail "ubsan build"
    ctest --test-dir build/c-ubsan --output-on-failure || fail "ubsan ctest"
fi

# Regenerating must not have changed anything: if it did, what was committed
# was stale. Untracked files are the author's business; modifications are not.
#
# regulae.js and regulae.wasm are deliberately not on this list. Emscripten's
# output is not byte-reproducible -- it embeds build metadata -- so rebuilding
# always changes them and the check would fail every time in --full mode,
# which is worse than not having it. BUILD_INFO is the proxy: it records the
# hashes of the sources the artifacts were built from, it is deterministic, and
# the `wasm_current` test fails when it falls behind. The artifacts themselves
# are checked by `wasm_smoke`, which trains real corpora in both builds and
# requires byte-identical JSON.
# The suite asserts that particular rules are found; it does not assert that
# the published model is the one it was before, and those are very different
# claims. A refactor that quietly changes what every corpus learns passes
# ctest. This trains all of them and compares hashes, which takes about four
# seconds and is the only thing here that can tell a refactor from a change.
# The baseline is byte-identical under GCC and clang, so a difference is the
# engine, not the compiler.
step "learned models"
python3 scripts/model_hashes.py || fail "the learned models moved; see above"

step "generated artifacts were already current"
# Tracked files only. The fixture directories joined this list when their
# generators did, and they hold files a person writes by hand as well as files
# a script writes -- a new hand-authored fixture is untracked until it is
# committed, and reporting it as a stale generated artifact is a wrong answer
# to a question nobody asked. What the gate is for is a *committed* file that
# its generator no longer produces.
dirty="$(git status --porcelain -- docs/capabilities.md web/corpora.js web/guide-content.js web/BUILD_INFO testdata/soundlaws testdata/restraint testdata/diagnostics docs/m5_evaluation.md docs/m5_evaluation_probes.tsv testdata/evaluation/m5 | grep -v '^??' || true)"
if [ -n "$dirty" ]; then
    printf '%s\n' "$dirty" >&2
    fail "generated artifacts were stale; they have been regenerated, review and commit them"
fi

printf '\nAll checks passed.\n'
