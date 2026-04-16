"""Latin → Spanish cognate alignment experiment.

Trains the learned model on a ~95 Latin-Spanish cognate pair corpus
(loaded from cognates.tsv) and inspects what the framework learned.

Run with:

    python -m experiments.latin_spanish.run_experiment

or directly:

    cd experiments/latin_spanish && python run_experiment.py

This is an EXPERIMENT, not a test. It produces a human-readable report,
not an automated pass/fail. The goal is to see what the learned model
captures on real data, find where it struggles, and generate concrete
targets for context discovery.
"""

from __future__ import annotations

from pathlib import Path

from regulae import (
    align_forms,
    alignment_cost,
    cognate_sets_from_pairs,
    format_alignment,
    format_model,
    train_model,
)
from regulae.types import Form, Segment


def parse_segments(ipa: str) -> tuple[Segment, ...]:
    """Parse an IPA string into Segments.

    Handles multi-character graphemes (tʃ, nj, dʒ, etc.) by a small
    explicit list rather than full Unicode normalization. For this
    experiment we only care about the affricates we actually use.
    """
    multi = ["tʃ", "dʒ", "nj", "ts"]
    segments: list[Segment] = []
    i = 0
    while i < len(ipa):
        matched = False
        for cluster in multi:
            if ipa[i : i + len(cluster)] == cluster:
                segments.append(Segment(cluster))
                i += len(cluster)
                matched = True
                break
        if not matched:
            segments.append(Segment(ipa[i]))
            i += 1
    return tuple(segments)


def load_corpus(tsv_path: Path) -> list[tuple[str, Form, Form]]:
    """Load the cognates TSV: returns list of (gloss, latin_form, spanish_form)."""
    corpus: list[tuple[str, Form, Form]] = []
    with tsv_path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "latin", "spanish"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, latin, spanish = parts
            src = Form(lect_id="latin", segments=parse_segments(latin))
            tgt = Form(lect_id="spanish", segments=parse_segments(spanish))
            corpus.append((gloss, src, tgt))
    return corpus


def report_training_stats(
    label: str, corpus: list[tuple[Form, Form]], trained_model
) -> None:
    """Print headline numbers comparing prior-only and learned-model total corpus cost."""
    m2_cost = sum(alignment_cost(align_forms(s, t)) for s, t in corpus)
    m3_cost = sum(
        alignment_cost(align_forms(s, t, model=trained_model), model=trained_model)
        for s, t in corpus
    )
    print(f"{label}:")
    print(f"  Prior-only total cost: {m2_cost:.2f}")
    print(f"  Learned-model total cost: {m3_cost:.2f}")
    print(f"  Reduction:     {m2_cost - m3_cost:.2f}")


def show_interesting_alignments(
    labeled_corpus: list[tuple[str, Form, Form]], trained_model
) -> None:
    """Print a handful of alignments that illustrate specific phenomena."""
    to_show = [
        "father",  # initial p, intervocalic t→d
        "night",  # cluster ct → tʃ
        "eight",  # cluster ct → tʃ
        "milk",  # cluster ct → tʃ
        "head",  # k before a, medial p→b
        "eye",  # kl → x (cluster palatalization)
        "tooth",  # e → je diphthongization
        "new",  # novum with o → we
        "sky",  # k before ae → θ
        "five",  # kinkwe → θinko (two k positions)
        "wolf",  # simple initial + intervocalic
        "fire",  # o → we
    ]
    print()
    print("=" * 70)
    print("Selected alignments (trained model)")
    print("=" * 70)
    by_gloss = {g: (s, t) for g, s, t in labeled_corpus}
    for gloss in to_show:
        if gloss not in by_gloss:
            continue
        src, tgt = by_gloss[gloss]
        alignment = align_forms(src, tgt, model=trained_model)
        print()
        print(f"[{gloss}]")
        print(format_alignment(alignment, show_costs=True, show_displacement=False))


def find_context_dependent_problems(
    labeled_corpus: list[tuple[str, Form, Form]], trained_model
) -> None:
    """Find segment correspondences where the same source segment maps
    to multiple distinct targets with substantial mass each.

    These are the cases the learned model can't resolve: they would be
    split into context-dependent correspondences by context discovery.

    The segment table keys are ``ConditionedCorrespondence`` objects.
    Entries with non-empty contexts are already split.
    """
    # Gather (src, tgt, count) by source (unconditioned entries only,
    # since split entries are already handled).
    by_source: dict[str, dict[str, float]] = {}
    for key, n in trained_model.segment_table.counts.items():
        if key.context.constraint_count() != 0:
            continue  # split — already resolved
        by_source.setdefault(key.src, {})[key.tgt] = n

    print()
    print("=" * 70)
    print("Source segments with competing target correspondences")
    print("(top 5 sources with multiple high-mass targets — these are the")
    print(" cases where context-conditioning would help)")
    print("=" * 70)
    competing: list[tuple[str, list[tuple[str, float]], float]] = []
    for src, targets in by_source.items():
        if len(targets) < 2:
            continue
        top_two = sorted(targets.items(), key=lambda kv: -kv[1])[:2]
        if top_two[1][1] < 2.0:  # second choice must have nontrivial mass
            continue
        total = sum(targets.values())
        competing.append((src, top_two, total))
    competing.sort(key=lambda x: -x[2])
    for src, top_two, total in competing[:8]:
        parts = [f"{t}×{int(n)}" for t, n in top_two]
        others_total = total - sum(n for _, n in top_two)
        tail = f" (+{int(others_total)} others)" if others_total > 0 else ""
        print(f"  {src!r}: {', '.join(parts)}{tail}  [total {int(total)}]")


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus: list[tuple[Form, Form]] = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Latin-Spanish cognate pairs.")
    print()

    print("Training learned model...")
    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel for the one pair. The surrounding code still uses
    # the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(corpus, ("latin", "spanish"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"latin", "spanish"})]
    print()

    print("=" * 70)
    print("Trained model summary")
    print("=" * 70)
    print(format_model(trained, top_segments=20, top_displacements=8))

    print()
    print("=" * 70)
    print("Headline numbers")
    print("=" * 70)
    report_training_stats("Full corpus", corpus, trained)

    show_interesting_alignments(labeled, trained)
    find_context_dependent_problems(labeled, trained)


if __name__ == "__main__":
    main()
