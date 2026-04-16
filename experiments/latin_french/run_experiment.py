"""Latin → French cognate alignment experiment.

Fourth real-data experiment. French shows more extreme reductive
changes than Spanish — heavy nasalization of vowel+nasal sequences,
loss of most word-final consonants, aggressive palatalization of
velars before front vowels, loss of unstressed syllables, and
systematic vowel shifts. All from the same Vulgar Latin starting
point as Latin→Spanish, so the intra-family comparison is direct.

This experiment tests whether context discovery recovers a *different*
split profile for French than for Spanish, reflecting the different
historical developments, without any hyperparameter retuning.

Data caveat: the corpus uses IPA-adjacent notation for modern
French. Standard Parisian French pronunciation; some glosses with
borrowing-based replacements (e.g., ``brother → frɛʁ`` is cognate,
but ``wood → bwa`` is from Gallo-Romance *boscum*, not lignum — I
kept a few such replacements because this is about training
behavior not strict cognate purity).
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
    # Multi-character graphemes we might encounter in Modern French IPA.
    multi = ["tʃ", "dʒ", "nj", "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"]
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
    corpus: list[tuple[str, Form, Form]] = []
    with tsv_path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["gloss", "latin", "french"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, latin, french = parts
            src = Form(lect_id="latin", segments=parse_segments(latin))
            tgt = Form(lect_id="french", segments=parse_segments(french))
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
    to_show = [
        "father",    # p preserved initially
        "night",     # kt cluster → nasal+vowel
        "eight",     # kt cluster + GVS-like vowel
        "hundred",   # see if 'kentu' shows up
        "head",      # k before a → palatal ʃ
        "tongue",    # nasalization
        "name",      # nasalization preservation
        "five",      # kinkwe → sɛ̃k, palatalization + nasalization
        "wolf",      # clean reduction
        "king",      # rege → ʁwa
        "water",     # aqua → o, extreme reduction
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
    by_source: dict[str, dict[str, float]] = {}
    for key, n in trained_model.segment_table.counts.items():
        if key.context.constraint_count() != 0:
            continue
        by_source.setdefault(key.src, {})[key.tgt] = n

    print()
    print("=" * 70)
    print("Unconditioned source segments with competing targets")
    print("(context discovery should have split most real cases; remaining competitions")
    print(" are either spurious or below BIC threshold)")
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
    if not competing:
        print("  (none above threshold)")
    for src, top_two, total in competing[:10]:
        parts = [f"{t}×{int(n)}" for t, n in top_two]
        others_total = total - sum(n for _, n in top_two)
        tail = f" (+{int(others_total)} others)" if others_total > 0 else ""
        print(f"  {src!r}: {', '.join(parts)}{tail}  [total {int(total)}]")


def show_context_splits(trained_model) -> None:
    """Print all conditioned segment correspondences from context discovery."""
    conditioned = [
        (k, v)
        for k, v in trained_model.segment_table.counts.items()
        if k.context.constraint_count() > 0
    ]
    print()
    print("=" * 70)
    print(f"Context-conditioned splits ({len(conditioned)} total)")
    print("=" * 70)
    by_src: dict[str, list] = {}
    for k, v in conditioned:
        by_src.setdefault(k.src, []).append((k, v))
    for src in sorted(by_src):
        entries = sorted(by_src[src], key=lambda x: -x[1])
        for k, v in entries:
            ctx = k.context
            parts = []
            if ctx.position:
                parts.append(f"pos={ctx.position}")
            if ctx.preceding:
                parts.append("prec=[" + ",".join(c.feature for c in ctx.preceding) + "]")
            if ctx.following:
                parts.append("foll=[" + ",".join(c.feature for c in ctx.following) + "]")
            print(f"  {k.src} → {k.tgt}: {int(v)}  {' '.join(parts)}")


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus: list[tuple[Form, Form]] = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Latin → French cognate pairs.")
    print()

    print("Training learned model...")
    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(corpus, ("latin", "french"))
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[frozenset({"latin", "french"})]
    print()

    print("=" * 70)
    print("Trained model summary")
    print("=" * 70)
    print(format_model(trained, top_segments=25, top_displacements=8))

    print()
    print("=" * 70)
    print("Headline numbers")
    print("=" * 70)
    report_training_stats("Full corpus", corpus, trained)

    show_interesting_alignments(labeled, trained)
    find_context_dependent_problems(labeled, trained)
    show_context_splits(trained)


if __name__ == "__main__":
    main()
