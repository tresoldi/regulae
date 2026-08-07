#!/usr/bin/env bash
# Runs the WebAssembly smoke test under Node.
#
# Needs a built web/regulae.js (see web/build-wasm.sh) and a native CLI to
# compare against, since the central assertion is that the two agree byte for
# byte. Skips rather than fails when either is missing, so a checkout without
# the Emscripten toolchain still passes its test suite.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cli="${REGULAE_CLI:-$repo/build/c/regulae}"

if ! command -v node >/dev/null 2>&1; then
    echo "wasm smoke: node not found, skipping" >&2
    exit 0
fi
if [ ! -f "$repo/web/regulae.js" ] || [ ! -f "$repo/web/regulae.wasm" ]; then
    echo "wasm smoke: web/regulae.{js,wasm} not built, skipping" >&2
    exit 0
fi
if [ ! -x "$cli" ]; then
    echo "wasm smoke: native CLI not found at $cli, skipping" >&2
    exit 0
fi

REGULAE_CLI="$cli" exec node "$here/smoke.mjs"
