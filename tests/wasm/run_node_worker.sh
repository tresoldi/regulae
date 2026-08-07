#!/usr/bin/env bash
# Runs the worker protocol test under Node.
#
# The worker cannot be a real Web Worker here, but its message protocol is what
# the page depends on, and that is what this exercises. Skips when Node or the
# built artifacts are missing.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

if ! command -v node >/dev/null 2>&1; then
    echo "worker test: node not found, skipping" >&2
    exit 0
fi
if [ ! -f "$repo/web/regulae.js" ] || [ ! -f "$repo/web/regulae.wasm" ]; then
    echo "worker test: web/regulae.{js,wasm} not built, skipping" >&2
    exit 0
fi

exec node "$here/worker.mjs"
