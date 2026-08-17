"""A thin command line over the native core: read a corpus, print the model JSON.

The C project ships its own ``regulae`` binary with a fuller CLI; this exists so
`pip install regulae` gives a runnable command without one, and so the Python
package has a `__main__`.
"""

from __future__ import annotations

import argparse
import json
import sys

from regulae import RegulaeError, __version__, _native


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regulae", description=__doc__)
    parser.add_argument("corpus", help="path to a corpus, or - for stdin")
    parser.add_argument(
        "--format",
        default="wide",
        choices=("wide", "tsv", "gled", "arcaverborum"),
        help="corpus loader (default: wide)",
    )
    parser.add_argument(
        "--options",
        default=None,
        help="training options as a JSON object",
    )
    parser.add_argument("--version", action="version", version=f"regulae {__version__}")
    args = parser.parse_args(argv)

    if args.corpus == "-":
        text = sys.stdin.read()
    else:
        with open(args.corpus, encoding="utf-8") as handle:
            text = handle.read()
    try:
        payload = _native.train(text, args.format, args.options)
    except (RegulaeError, ValueError) as exc:
        print(f"regulae: {exc}", file=sys.stderr)
        return 1
    # Round-trip so the output is compact and deterministic regardless of how
    # the core spaced it.
    json.dump(json.loads(payload), sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
