#!/usr/bin/env bash
#
# Static analysis over first-party C. One definition, used by both gates:
# scripts/check.sh --full runs it locally, .github/workflows/ci.yml runs it on
# every push.
#
# The baseline is zero. It was 27 when first measured, and the difference is
# recorded in docs/static_analysis.md -- four real defects fixed, three checkers
# disabled with reasons in .clang-tidy, five sites annotated where the analyzer
# cannot follow a free through a pointer copy. A new finding means new code, so
# this exits non-zero on any output at all.
#
# Not in the default check.sh path: a full run is about two minutes against the
# suite's twenty seconds.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "tidy: clang-tidy is not installed" >&2
    exit 1
fi

build_dir="${1:-build/c-tidy}"

# The analyzer needs to know how each file is compiled, so every directory it
# reads has to be in the compilation database -- including the fuzz harnesses,
# which only exist when REGULAE_BUILD_FUZZERS is on. Without that they are not
# skipped with a complaint, they are analysed against a default command that
# cannot even find regulae.h, and the run reports clean having read nothing.
#
# Tests are in scope too. They are first-party code, and the ftell that could
# return -1 and index before a buffer was found in one of them.
if ! command -v clang >/dev/null 2>&1; then
    echo "tidy: clang is not installed; it is needed to configure the fuzz targets" >&2
    exit 1
fi
# A cache configured with another compiler cannot be reconfigured in place --
# CMake refuses rather than silently using the old one, which is the right
# behaviour and an unhelpful failure here. Wipe and retry once.
if ! cmake -S . -B "$build_dir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER=clang -DREGULAE_BUILD_FUZZERS=ON >/dev/null 2>&1; then
    rm -rf "$build_dir"
    cmake -S . -B "$build_dir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER=clang -DREGULAE_BUILD_FUZZERS=ON >/dev/null
fi

files="$(find src cmd tests/c fuzz -name '*.c' | sort)"
missing=0
for f in $files; do
    if ! grep -q "\"$PWD/$f\"" "$build_dir/compile_commands.json"; then
        echo "tidy: $f is not in the compilation database" >&2
        missing=1
    fi
done
[ "$missing" = "0" ] || exit 1

output="$(printf '%s\n' "$files" | xargs clang-tidy -p "$build_dir" --quiet 2>/dev/null || true)"

if [ -n "$output" ]; then
    printf '%s\n' "$output"
    echo >&2
    echo "tidy: the baseline is zero and this run was not." >&2
    echo "tidy: fix it, or suppress it narrowly with the reason, as the five" >&2
    echo "tidy: existing NOLINT annotations do. See docs/static_analysis.md." >&2
    exit 1
fi

echo "tidy: clean"
