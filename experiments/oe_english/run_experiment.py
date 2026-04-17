"""Old English → Modern English cognate alignment experiment.

Third real-data experiment. After Latin→Spanish (Romance,
context-conditioned shifts) and Proto-Polynesian→Hawaiian (clean
mergers), this one tests a completely different type of change:

* **Great Vowel Shift**: long vowels rearranged in height/front-back
  space rather than merging. Ī → aɪ, ē → iː, ā → oʊ, etc. This is a
  chain shift, not a merger, so the framework should show very
  different displacement activity than the previous experiments.
* **Consonant losses**: initial /kn/ → /n/, /gn/ → /n/, loss of
  /x/ (spelled h) in light/night/right etc.
* **Fricative voicing**: intervocalic /θ/ → /ð/, /f/ → /v/,
  /s/ → /z/. A context-conditioned change — should show up as a
  competing-correspondence in the learned model's diagnostic report.
* **Palatalization**: OE /k/ before front vowels → Modern /tʃ/
  (church, cheese, chicken). Another context-conditioned split.

This is Germanic rather than Romance or Polynesian, a third language
family branch, and the change profile is largely distinct from both
previous experiments.

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
    """Parse an IPA string into Segments, handling multi-char graphemes.

    Merkmal handles long vowels (aː, eː, iː, oː, uː) and diphthongs
    (aɪ, oʊ, aʊ, ɔɪ, eɪ) as single graphemes in the descriptive
    system, so we match those first before falling back to single
    characters.

    Length marks are preserved: OE /aː/ is distinct from OE /a/, and
    the length distinction is what drives the Great Vowel Shift
    conditioning. Earlier versions of this experiment stripped ː to
    work around an import-time misconception — ``merkmal`` does in
    fact cover long vowels natively.
    """
    multi = (
        "aː", "æː", "ɑː", "eː", "iː", "oː", "ɔː", "uː", "yː",
        "aɪ", "oʊ", "aʊ", "ɔɪ", "eɪ",
        "tʃ", "dʒ",
    )
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
        assert header == ["gloss", "old_english", "modern_english"]
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gloss, oe, me = parts
            src = Form(lect_id="old_english", segments=parse_segments(oe))
            tgt = Form(lect_id="modern_english", segments=parse_segments(me))
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
        "stone",  # GVS: ā → oʊ
        "tooth",  # GVS: ō → uː
        "house",  # GVS: ū → aʊ
        "wife",  # GVS: ī → aɪ
        "see",  # GVS: ē → iː
        "knee",  # initial kn- → n-
        "knight",  # kn- + x loss
        "night",  # x loss
        "light",  # x loss + GVS
        "mother",  # intervocalic θ → ð
        "cheese",  # palatalization k → tʃ
        "church",  # palatalization
        "blood",  # shortening ō → ʌ
        "foot",  # shortening ō → ʊ
        "fish",  # sk → ʃ
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
    """Show the top sources with multiple high-mass targets."""
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
    if not competing:
        print("  (none above threshold — no context-dependent splits)")
    for src, top_two, total in competing[:10]:
        parts = [f"{t}×{int(n)}" for t, n in top_two]
        others_total = total - sum(n for _, n in top_two)
        tail = f" (+{int(others_total)} others)" if others_total > 0 else ""
        print(f"  {src!r}: {', '.join(parts)}{tail}  [total {int(total)}]")


def main() -> None:
    tsv_path = Path(__file__).parent / "cognates.tsv"
    labeled = load_corpus(tsv_path)
    corpus: list[tuple[Form, Form]] = [(s, t) for _, s, t in labeled]
    print(f"Loaded {len(corpus)} Old English → Modern English cognate pairs.")
    print()

    print("Training learned model...")
    # Wrap the pair list in CognateSets and extract the inner
    # LearnedModel. The rest of the script uses the LearnedModel API.
    cognate_corpus = cognate_sets_from_pairs(
        corpus, ("old_english", "modern_english")
    )
    multi_model = train_model(cognate_corpus)
    trained = multi_model.pairwise_models[
        frozenset({"old_english", "modern_english"})
    ]
    print()

    print("=" * 70)
    print("Trained model summary")
    print("=" * 70)
    print(format_model(trained, top_segments=25, top_displacements=10))

    print()
    print("=" * 70)
    print("Headline numbers")
    print("=" * 70)
    report_training_stats("Full corpus", corpus, trained)

    show_interesting_alignments(labeled, trained)
    find_competing_correspondences(labeled, trained)


if __name__ == "__main__":
    main()
