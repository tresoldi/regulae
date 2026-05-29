"""arcaverborum Polynesian multi-lect experiment.

The follow-up to ``experiments/gled_polynesian``: this uses a larger
Polynesian dataset from arcaverborum (Walworth's Proto-Polynesian
wordlist) that CRUCIALLY includes **Tongan**, which preserves the
Proto-Polynesian ``*k`` distinction. With Tongan as the anchor, the
multi-lect context discovery should be able to disambiguate Samoan ``ʔ``
(from either ``*ʔ`` or ``*k``) and Hawaiian ``ʔ``.

Run with:

    python experiments/arcaverborum_polynesian/run_experiment.py

Data:

    Zenodo archive 10.5281/zenodo.17294927, CoreCog collection. The
    loader expects an unpacked forms.csv at
    /tmp/arcaverborum_corecog/arcaverborum-A-corecog-20251008/forms.csv.
    Adjust ``CSV_PATH`` below if yours lives elsewhere.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from regulae import MultiLectModel, load_arcaverborum, train_model

CSV_PATH = Path(
    "/tmp/arcaverborum_corecog/arcaverborum-A-corecog-20251008/forms.csv"
)

POLY_LECTS = {
    "walworthpolynesian_Hawaiian",
    "walworthpolynesian_Samoan",
    "walworthpolynesian_Tongan",
    "walworthpolynesian_Maori",
    "walworthpolynesian_Tahitian",
    "walworthpolynesian_Niuean",
    "walworthpolynesian_Tuvalu",
}

# Shorter display labels for the report.
SHORT = {
    "walworthpolynesian_Hawaiian": "Haw",
    "walworthpolynesian_Samoan": "Sam",
    "walworthpolynesian_Tongan": "Ton",
    "walworthpolynesian_Maori": "Mao",
    "walworthpolynesian_Tahitian": "Tah",
    "walworthpolynesian_Niuean": "Niu",
    "walworthpolynesian_Tuvalu": "Tuv",
}


def short(lect: str) -> str:
    return SHORT.get(lect, lect)


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"arcaverborum data not found at {CSV_PATH}. "
            "Download from https://doi.org/10.5281/zenodo.17294927 "
            "(CoreCog collection) and unpack."
        )

    print(f"Loading arcaverborum walworthpolynesian from {CSV_PATH}...")
    corpus = load_arcaverborum(
        CSV_PATH,
        dataset="walworthpolynesian",
        language_ids=POLY_LECTS,
    )

    lects_present: Counter[str] = Counter()
    size_hist: Counter[int] = Counter()
    for cs in corpus:
        for lect in cs.forms:
            lects_present[lect] += 1
        size_hist[len(cs.forms)] += 1

    print(f"Loaded {len(corpus)} Polynesian cognate sets.")
    print(f"  per-lect coverage: {{{', '.join(f'{short(k)}: {v}' for k,v in sorted(lects_present.items()))}}}")
    print(f"  cognate set sizes: {dict(sorted(size_hist.items()))}")
    print()
    print("Training multi-lect model...")
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)

    print(f"  lects: {tuple(short(l) for l in model.lect_ids)}")
    print(f"  pairwise models: {len(model.pairwise_models)}")
    print(f"  unconditioned classes: {len(model.unconditioned_classes)}")
    print(f"  conditioned classes:   {len(model.conditioned_classes)}")
    print()

    print("--- Top 25 unconditioned classes overall ---")
    for k in model.unconditioned_classes[:25]:
        tupled = " ".join(
            f"{short(lect)}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  [{len(k.segments)}-way] count={k.count:5.1f}  {tupled}")
    print()

    print("--- Classes with Tongan:k (proto *k anchor) ---")
    ton_k = [
        k
        for k in model.unconditioned_classes
        if k.segments.get("walworthpolynesian_Tongan") == "k"
    ]
    for k in sorted(ton_k, key=lambda x: -x.count)[:15]:
        tupled = " ".join(
            f"{short(lect)}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  count={k.count:5.1f}  {tupled}")
    print()

    print("--- Classes with Samoan:ʔ (glottal — mixed *ʔ and *k) ---")
    sam_glot = [
        k
        for k in model.unconditioned_classes
        if k.segments.get("walworthpolynesian_Samoan") == "ʔ"
    ]
    for k in sorted(sam_glot, key=lambda x: -x.count)[:15]:
        tupled = " ".join(
            f"{short(lect)}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  count={k.count:5.1f}  {tupled}")
    print()

    print("--- Classes with Hawaiian:ʔ (glottal — mixed *ʔ and *k) ---")
    haw_glot = [
        k
        for k in model.unconditioned_classes
        if k.segments.get("walworthpolynesian_Hawaiian") == "ʔ"
    ]
    for k in sorted(haw_glot, key=lambda x: -x.count)[:15]:
        tupled = " ".join(
            f"{short(lect)}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  count={k.count:5.1f}  {tupled}")
    print()

    # Key diagnostic: is there a class where Sam:ʔ corresponds to Ton:k?
    # That's the merger disambiguation — proto *k → Haw ʔ / Sam ʔ / Ton k
    # vs proto *ʔ → Haw ʔ / Sam ʔ / Ton Ø (or whatever).
    merger_disambig = [
        k
        for k in model.unconditioned_classes
        if k.segments.get("walworthpolynesian_Samoan") == "ʔ"
        and k.segments.get("walworthpolynesian_Tongan") == "k"
    ]
    print(f"--- DIAGNOSTIC: Sam:ʔ + Ton:k classes (merger-disambig signal): {len(merger_disambig)} ---")
    for k in sorted(merger_disambig, key=lambda x: -x.count):
        tupled = " ".join(
            f"{short(lect)}:{g}" for lect, g in sorted(k.segments.items())
        )
        print(f"  count={k.count:5.1f}  {tupled}")
    print()

    print("--- Top 20 conditioned classes ---")
    for k in sorted(model.conditioned_classes, key=lambda x: -x.count)[:20]:
        segs = " ".join(
            f"{short(lect)}:{g}" for lect, g in sorted(k.segments.items())
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
                    ctxs.append(f"{short(lect)}: {' '.join(parts)}")
        ctx_str = " | ".join(ctxs) if ctxs else "(no constraints)"
        print(f"  count={k.count:5.1f}  {segs}   [{ctx_str}]")


if __name__ == "__main__":
    main()
