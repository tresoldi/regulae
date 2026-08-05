#!/usr/bin/env bash
# Diffs the C implementation against the frozen Go reference on every corpus in
# testdata/parity. Both sides load the same TSV through their own loader and
# print the same compact summary, so a run compares loading, training,
# reconciliation, discovery and cross-dimensional lifting in one pass.
#
# The Go reference needs ../merkmal/go, which current merkmal no longer ships;
# restore it first (see docs/c_reference_freeze.md):
#
#   git -C ../merkmal archive d59b987^ go | tar -x -C ../merkmal
#
# Known intentional deviation: Go's published chunk table drops the promoted
# observation counts and their Wilson intervals (chunks.go builds `final`
# without copying ObservationCounts/Uncertainty), so CHUNK rows are compared on
# cost only. C retains the counts.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

cli="${REGULAE_CLI:-$root/build/c/regulae}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ ! -x "$cli" ]; then
    echo "parity: C CLI not found at $cli (build it first)" >&2
    exit 2
fi
if [ ! -d ../merkmal/go ]; then
    echo "parity: ../merkmal/go missing; restore the Go reference first" >&2
    exit 2
fi

go build -o "$work/goref" ./tools/goref || exit 2

known="testdata/parity/KNOWN_DEVIATIONS"
failures=0
expected=0
checked=0

is_known() {
    [ -f "$known" ] || return 1
    grep -v '^#' "$known" | grep -qF "$(printf '%s\t%s\t' "$1" "$2")"
}
for corpus in testdata/parity/*.tsv testdata/parity/*.csv; do
    name="$(basename "$corpus")"
    fmt=tsv
    case "$corpus" in *.csv) fmt=arcaverborum ;; esac

    "$work/goref" summary "$corpus" > "$work/go.summary" 2>"$work/go.err" || {
        echo "FAIL $name (go summary): $(head -1 "$work/go.err")"; failures=$((failures + 1)); continue; }
    "$cli" train --format "$fmt" "$corpus" > "$work/c.summary" 2>"$work/c.err" || {
        echo "FAIL $name (c summary): $(head -1 "$work/c.err")"; failures=$((failures + 1)); continue; }

    "$work/goref" pairwise "$corpus" 2>/dev/null | sed 's/\(^CHUNK\t.*\t[-0-9.]*\)\t[-0-9.]*$/\1/' | sort > "$work/go.pairwise"
    "$cli" train --format "$fmt" --pairwise "$corpus" 2>/dev/null | sed 's/\(^CHUNK\t.*\t[-0-9.]*\)\t[-0-9.]*$/\1/' | sort > "$work/c.pairwise"

    "$work/goref" outliers "$corpus" > "$work/go.outliers" 2>/dev/null
    "$cli" outliers --format "$fmt" "$corpus" > "$work/c.outliers" 2>/dev/null

    ok=1
    for part in summary pairwise outliers; do
        if ! diff -u "$work/go.$part" "$work/c.$part" > "$work/diff.$part"; then
            if is_known "$name" "$part"; then
                echo "tie  $name ($part) - known float tie, see $known"
                expected=$((expected + 1))
            else
                echo "FAIL $name ($part)"
                head -20 "$work/diff.$part"
                ok=0
            fi
        fi
    done
    checked=$((checked + 1))
    if [ "$ok" -eq 1 ]; then
        echo "ok   $name ($(grep -c . "$work/c.summary") summary rows, $(grep -c '^COND' "$work/c.summary") conditioned)"
    else
        failures=$((failures + 1))
    fi
done

echo
echo "parity: $((checked - failures))/$checked corpora match ($expected known float ties tolerated)"
[ "$failures" -eq 0 ]
