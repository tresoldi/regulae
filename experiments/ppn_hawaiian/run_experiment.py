"""Proto-Polynesian → Hawaiian cognate alignment experiment.

Second real-data experiment. The motivation is cross-family
validation: the Latin→Spanish experiment worked well, but it's one
family, one direction, one specific phonological profile. Polynesian
is structurally very different — small phoneme inventories,
near-exceptionless mergers, different segmental profile, different
lect pair entirely.

Hawaiian underwent dramatic phonological reduction from Proto-
Polynesian: *t → k (most contexts), *k → ʔ, *f → h, *ŋ → n, and
*s → h. If the framework recovers these mergers and shows the
expected "competing correspondence" signature on cases where the
merger isn't absolute, we have a second independent data point.

Usage:

    python run_experiment.py
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
    """Parse an IPA string into Segments, handling multi-char graphemes."""
    multi = ["tʃ", "dʒ", "ts"]
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
    """Load the cognates TSV."""
    corpus: list[tuple[str, Form, Form]] = []
    with tsv_path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "ppn", "hawaiian"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, ppn, haw = parts
            src = Form(lect_id="ppn", segments=parse_segments(ppn))
            tgt = Form(lect_id="hawaiian", segments=parse_segments(haw))
            corpus.append((gloss, src, tgt))
    return corpus


def report_training_stats(
    label: str, corpus: list[tuple[Form, Form]], trained_model
) -> None:
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
    """Selected alignments illustrating specific phenomena."""
    to_show = [
        "one",  # *t → k merger
        "three",  # *t → k + *r → l
        "four",  # *f → h
        "seven",  # *f → h + *t → k
        "person",  # complex: *t → k, *ŋ → n
        "mouth",  # *ŋ → n
        "skin",  # *k → ʔ
        "chief",  # *r → l + *k → ʔ
        "bird",  # conservation
        "ten",  # conservation + *f → h
        "go",  # f→h in isolation
        "canoe",  # k→ʔ
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


def find_competing_correspondences(
    labeled_corpus: list[tuple[str, Form, Form]], trained_model
) -> None:
    """Show the top sources with multiple high-mass targets — these
    are the targets for context conditioning."""
    by_source: dict[str, dict[str, float]] = {}
    for key, n in trained_model.segment_table.counts.items():
        if key.context.constraint_count() != 0:
            continue  # context-conditioned entry, already split
        by_source.setdefault(key.src, {})[key.tgt] = n

    print()
    print("=" * 70)
    print("Source segments with competing target correspondences")
    print("=" * 70)
    competing: list[tuple[str, list[tuple[str, float]], float]] = []
    for src, targets in by_source.items():
        if len(targets) < 2:
            continue
        top_two = sorted(targets.items(), key=lambda kv: -kv[1])[:2]
        if top_two[1][1] < 2.0:
            continue
        total = sum(targets.values())
        competing.append((src, top_two, total))
    competing.sort(key=lambda x: -x[2])
    for src, top_two, total in competing[:10]:
        parts = [f"{t}×{int(n)}" for t, n in top_two]
        others_total = total - sum(n for _, n in top_two)
        tail = f" (+{int(others_total)} others)" if others_total > 0 else ""
        print(f"  {src!r}: {', '.join(parts)}{tail}  [total {int(total)}]")


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus: list[tuple[Form, Form]] = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Proto-Polynesian → Hawaiian cognate pairs.")
    print()

    print("Training learned model...")
    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(corpus, ("ppn", "hawaiian"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"ppn", "hawaiian"})]
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
    find_competing_correspondences(labeled, trained)


if __name__ == "__main__":
    main()
