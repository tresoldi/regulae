"""GLED Polynesian multi-lect experiment.

The motivating multi-lect reconciliation test: Hawaiian merger disambiguation. Hawaiian
``k`` reflects both Proto-Polynesian ``*t`` and ``*k``. A pairwise
Hawaiian–Samoan model can only see that "Hawaiian k corresponds to
both Samoan t and Samoan ʔ". The multi-lect pipeline should produce
two DISTINCT correspondence classes with Hawaiian k at the same
position, resolved by the other lects' reflexes.

Run with:

    python experiments/gled_polynesian/run_experiment.py

Adjust ``GLED_TSV`` if your checkout lives elsewhere.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from regulae import MultiLectModel, load_gled, train_model

GLED_TSV = Path("/tmp/gled_clone/releases/20221127/gled.tsv")

POLY_DOCULECTS = {
    "HAWAIIAN_2",
    "MAORI",
    "SAMOAN",
    "TAHITIAN",
    "RAROTONGAN",
    "RAPA_NUI",
}


def main() -> None:
    if not GLED_TSV.exists():
        raise SystemExit(
            f"GLED data not found at {GLED_TSV}. "
            "Clone https://github.com/tresoldi/gled and update the path."
        )

    print(f"Loading GLED Polynesian subset from {GLED_TSV}...")
    corpus = load_gled(
        GLED_TSV,
        family="Austronesian",
        doculects=POLY_DOCULECTS,
    )

    lects_present: Counter[str] = Counter()
    size_hist: Counter[int] = Counter()
    for cs in corpus:
        for lect in cs.forms:
            lects_present[lect] += 1
        size_hist[len(cs.forms)] += 1

    print(f"Loaded {len(corpus)} Polynesian cognate sets.")
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

    print("--- Unconditioned classes involving HAWAIIAN_2:k ---")
    haw_k = [
        k
        for k in model.unconditioned_classes
        if k.segments.get("HAWAIIAN_2") == "k"
    ]
    if not haw_k:
        print("  (none — reconciliation did not emit any class with Haw:k)")
    for k in sorted(haw_k, key=lambda x: -x.count):
        tupled = " ".join(
            f"{lect}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  count={k.count:5.1f}  {tupled}")
    print()

    print("--- Unconditioned classes where Samoan has t (proto *t reflex) ---")
    sam_t = [
        k
        for k in model.unconditioned_classes
        if k.segments.get("SAMOAN") == "t"
    ]
    for k in sorted(sam_t, key=lambda x: -x.count):
        tupled = " ".join(
            f"{lect}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  count={k.count:5.1f}  {tupled}")
    print()

    print("--- Top 25 unconditioned classes overall ---")
    for k in model.unconditioned_classes[:25]:
        tupled = " ".join(
            f"{lect}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  [{len(k.segments)}-way] count={k.count:5.1f}  {tupled}")
    print()

    print("--- Conditioned classes (top 15 by count) ---")
    for k in sorted(model.conditioned_classes, key=lambda x: -x.count)[:15]:
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
                    parts.append("prec=[" + ",".join(c.feature for c in ctx.preceding) + "]")
                if ctx.following:
                    parts.append("foll=[" + ",".join(c.feature for c in ctx.following) + "]")
                if parts:
                    ctxs.append(f"{lect}: {' '.join(parts)}")
        ctx_str = " | ".join(ctxs) if ctxs else "(no constraints)"
        print(f"  count={k.count:5.1f}  {segs}   [{ctx_str}]")


if __name__ == "__main__":
    main()
