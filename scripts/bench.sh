#!/usr/bin/env bash
# Times whole-corpus training over testdata/corpora.
#
# The CLI loads each file through its own loader and trains with default
# options, so this measures the thing a user actually waits for rather than a
# microbenchmark. Corpora are timed in increasing size; the 4-lect Romance set
# dominates and is the one to watch.
#
# Usage: scripts/bench.sh [runs]        # default 3 runs, best time reported
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

runs="${1:-3}"
cli="${REGULAE_CLI:-$root/build/c/regulae}"

if [ ! -x "$cli" ]; then
    echo "bench: C CLI not found at $cli (build it first)" >&2
    exit 2
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

printf '%-26s %8s\n' corpus seconds
printf '%-26s %8s\n' -------------------------- --------

for corpus in testdata/corpora/*.tsv testdata/corpora/*.csv; do
    [ -e "$corpus" ] || continue
    name="$(basename "$corpus")"
    fmt=tsv
    case "$corpus" in *.csv) fmt=arcaverborum ;; esac

    c_time="$(best_time "$cli" train --format "$fmt" "$corpus")" || c_time="err"
    printf '%-26s %8s\n' "$name" "$c_time"
done
