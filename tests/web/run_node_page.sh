#!/usr/bin/env bash
# Runs the page test under Node.
#
# There is no browser here; the shim covers the DOM app.js touches. Skips when
# Node, the generated page data or the CLI are missing.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cli="${REGULAE_CLI:-$repo/build/c/regulae}"

if ! command -v node >/dev/null 2>&1; then
    echo "page test: node not found, skipping" >&2
    exit 0
fi
for file in web/app.js web/corpora.js web/guide-content.js; do
    if [ ! -f "$repo/$file" ]; then
        echo "page test: $file missing, skipping" >&2
        exit 0
    fi
done
if [ ! -x "$cli" ]; then
    echo "page test: native CLI not found at $cli, skipping" >&2
    exit 0
fi

REGULAE_CLI="$cli" exec node "$here/page.mjs"
