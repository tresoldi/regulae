#!/usr/bin/env bash
# Times whole-corpus training over testdata/parity, C against the Go reference.
#
# Both sides load the same file through their own loader and train with default
# options, so this measures the thing a user actually waits for rather than a
# microbenchmark. Corpora are timed in increasing size; the 4-lect Romance set
# dominates and is the one to watch.
#
# The Go side needs ../merkmal/go restored (see docs/c_conversion_handoff.md);
# without it the script reports C timings alone.
#
# Usage: scripts/bench.sh [runs]        # default 3 runs, best time reported
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

runs="${1:-3}"
cli="${REGULAE_CLI:-$root/build/c/regulae}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ ! -x "$cli" ]; then
    echo "bench: C CLI not found at $cli (build it first)" >&2
    exit 2
fi

have_go=0
if [ -d ../merkmal/go ] && go build -o "$work/goref" ./tools/goref 2>/dev/null; then
    have_go=1
else
    echo "bench: Go reference unavailable, reporting C only" >&2
fi

# Best of N wall-clock seconds; best rather than mean, to suppress scheduler noise.
best_time() {
    local best=""
    local i t
    for i in $(seq 1 "$runs"); do
        t=$( { /usr/bin/time -f "%e" "$@" >/dev/null; } 2>&1 | tail -1 )
        case "$t" in
            *[!0-9.]*) return 1 ;;
        esac
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then
            best="$t"
        fi
    done
    printf '%s' "$best"
}

printf '%-26s %8s %8s %8s\n' corpus c go ratio
printf '%-26s %8s %8s %8s\n' -------------------------- -------- -------- --------

for corpus in testdata/parity/*.tsv testdata/parity/*.csv; do
    [ -e "$corpus" ] || continue
    name="$(basename "$corpus")"
    fmt=tsv
    case "$corpus" in *.csv) fmt=arcaverborum ;; esac

    c_time="$(best_time "$cli" train --format "$fmt" "$corpus")" || c_time="err"
    go_time="-"
    ratio="-"
    if [ "$have_go" -eq 1 ] && [ "$fmt" = tsv ]; then
        go_time="$(best_time "$work/goref" summary "$corpus")" || go_time="err"
        if [ "$c_time" != err ] && [ "$go_time" != err ]; then
            ratio="$(awk "BEGIN{printf \"%.2fx\", $c_time/$go_time}")"
        fi
    fi
    printf '%-26s %8s %8s %8s\n' "$name" "$c_time" "$go_time" "$ratio"
done

echo
echo "ratio below 1.00x means C is faster than the Go reference"
