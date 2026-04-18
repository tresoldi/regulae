#!/usr/bin/env bash
# Run mypy --strict over the package. Exits non-zero on any error.
#
# Usage:
#     scripts/typecheck.sh          # check src/regulae/
#     scripts/typecheck.sh tests    # check an additional path

set -euo pipefail

cd "$(dirname "$0")/.."

exec python -m mypy --no-incremental src/regulae "$@"
