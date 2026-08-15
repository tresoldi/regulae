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
cmake -S . -B build/c >/dev/null || fail "cmake configure"
# Not piped: a configure failure has to be visible, and piping it into a
# filter is exactly how it was missed before.
cmake --build build/c -j"$(nproc)" || fail "cmake build"

# The archive must be newer than every source it is built from. A build system
# that silently declines to rebuild is the failure this whole script is for.
step "build is current"
newest_source="$(find src include third_party cmd -newer build/c/libregulae.a -type f 2>/dev/null | head -1 || true)"
if [ -n "$newest_source" ]; then
    fail "libregulae.a is older than $newest_source"
fi

step "native tests"
ctest --test-dir build/c --output-on-failure || fail "ctest"

step "generated artifacts"
python3 scripts/capabilities.py >/dev/null || fail "scripts/capabilities.py"
python3 scripts/corpora.py >/dev/null || fail "scripts/corpora.py"
python3 scripts/guide.py >/dev/null || fail "scripts/guide.py"

if [ "$full" = "1" ]; then
    step "WebAssembly build"
    if [ -z "${EMSDK:-}" ]; then
        fail "--full needs the Emscripten SDK: source ~/emsdk/emsdk_env.sh"
    fi
    ./web/build-wasm.sh >/dev/null || fail "web/build-wasm.sh"

    step "sanitizer build and tests"
    cmake -S . -B build/c-asan -DREGULAE_ENABLE_SANITIZER=address >/dev/null || fail "asan configure"
    cmake --build build/c-asan -j"$(nproc)" || fail "asan build"
    ctest --test-dir build/c-asan --output-on-failure || fail "asan ctest"

    step "native tests again, against the rebuilt artifacts"
    ctest --test-dir build/c --output-on-failure || fail "ctest after regeneration"
fi

# Regenerating must not have changed anything: if it did, what was committed
# was stale. Untracked files are the author's business; modifications are not.
step "generated artifacts were already current"
dirty="$(git status --porcelain -- docs/capabilities.md web/corpora.js web/guide-content.js web/regulae.js web/regulae.wasm web/BUILD_INFO)"
if [ -n "$dirty" ]; then
    printf '%s\n' "$dirty" >&2
    fail "generated artifacts were stale; they have been regenerated, review and commit them"
fi

printf '\nAll checks passed.\n'
