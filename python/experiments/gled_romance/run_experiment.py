"""GLED Romance multi-lect experiment.

Loads a 7-way Romance subset from the GLED database and runs the
multi-lect training pipeline. Produces a human-readable report of:

- per-pair trained learned models
- reconciled unconditioned multi-lect correspondence classes
- conditioned multi-lect classes from context discovery

Run with:

    python experiments/gled_romance/run_experiment.py

This script is a WALK-THROUGH, not a test. It runs against a real GLED
TSV checkout expected at ``/tmp/gled_clone/releases/20221127/gled.tsv``
— adjust ``GLED_TSV`` below if your checkout lives elsewhere.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from regulae import MultiLectModel, load_gled, train_model

GLED_TSV = Path("/tmp/gled_clone/releases/20221127/gled.tsv")

ROMANCE_DOCULECTS = {
    "LATIN",
    "SPANISH",
    "PORTUGUESE_2",
    "FRENCH_2",
    "ITALIAN_2",
    "CATALAN_3",
    "ROMANIAN_2",
}


def main() -> None:
    if not GLED_TSV.exists():
        raise SystemExit(
            f"GLED data not found at {GLED_TSV}. "
            "Clone https://github.com/tresoldi/gled and update the path."
        )

    print(f"Loading GLED Romance subset from {GLED_TSV}...")
    corpus = load_gled(
        GLED_TSV,
        family="Indo-European",
        doculects=ROMANCE_DOCULECTS,
    )

    lects_present: Counter[str] = Counter()
    size_hist: Counter[int] = Counter()
    for cs in corpus:
        for lect in cs.forms:
            lects_present[lect] += 1
        size_hist[len(cs.forms)] += 1

    print(f"Loaded {len(corpus)} Romance cognate sets.")
    print(f"  per-lect coverage: {dict(lects_present)}")
    print(f"  cognate set sizes: {dict(sorted(size_hist.items()))}")
    print()
    print("Training multi-lect model...")
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)

    print(f"  lects: {model.lect_ids}")
    print(f"  pairwise models: {len(model.pairwise_models)}")
    print(f"  unconditioned classes: {len(model.unconditioned_classes)}")
    print(f"  conditioned classes:   {len(model.conditioned_classes)}")
    print()

    print("--- Top 30 unconditioned correspondence classes ---")
    for k in model.unconditioned_classes[:30]:
        n_lects = len(k.segments)
        tupled = " ".join(
            f"{lect}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  [{n_lects}-way] count={k.count:5.1f}  {tupled}")
    print()

    print("--- All conditioned classes ---")
    if not model.conditioned_classes:
        print("  (none — context discovery found no splits that beat BIC on this corpus)")
    for k in model.conditioned_classes:
        segs = " ".join(
            f"{lect}:{g}" for lect, g in sorted(k.segments.items())
        )
        ctxs = []
        if k.contexts is not None:
            for lect, ctx in sorted(k.contexts.items()):
                if ctx is None or ctx.constraint_count() == 0:
                    continue
                parts: list[str] = []
                if ctx.position is not None:
                    parts.append(f"pos={ctx.position}")
                if ctx.preceding:
                    parts.append(
                        "prec=[" + ",".join(f"{c.feature}" for c in ctx.preceding) + "]"
                    )
                if ctx.following:
                    parts.append(
                        "foll=[" + ",".join(f"{c.feature}" for c in ctx.following) + "]"
                    )
                if parts:
                    ctxs.append(f"{lect}: {' '.join(parts)}")
        ctx_str = " | ".join(ctxs) if ctxs else "(no constraints)"
        print(f"  count={k.count:5.1f}  {segs}   [{ctx_str}]")


if __name__ == "__main__":
    main()
