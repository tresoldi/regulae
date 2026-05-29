"""Tests for  inspection helpers (format_multi_lect_model,
describe_multi_lect_class)."""

from regulae import (
    CognateSet,
    Context,
    Form,
    LearnedModel,
    MultiLectCorrespondenceClass,
    MultiLectModel,
    Segment,
    cognate_sets_from_pairs,
    describe_multi_lect_class,
    format_multi_lect_model,
    train_model,
)
from regulae.types import FeatureConstraint


def _form(lect: str, word: str) -> Form:
    return Form(lect, tuple(Segment(c) for c in word))


# ----- format_multi_lect_model -------------------------------------------


def test_format_empty_model() -> None:
    """COMMITMENT: an empty model renders without crashing and shows
    explicit 'none' markers for both class tables."""
    out = format_multi_lect_model(MultiLectModel.empty())
    assert "MultiLectModel" in out
    assert "lects (0)" in out
    assert "(none)" in out


def test_format_n2_model_shows_lect_ids_and_classes() -> None:
    pairs = [(_form("A", "pa"), _form("B", "fa")) for _ in range(4)]
    model = train_model(cognate_sets_from_pairs(pairs, ("A", "B")))
    assert isinstance(model, MultiLectModel)
    out = format_multi_lect_model(model)
    assert "A" in out
    assert "B" in out
    assert "pairwise models:" in out
    assert "unconditioned cls:" in out


def test_format_truncates_to_top_n() -> None:
    """COMMITMENT: top_unconditioned caps the number of unconditioned
    classes shown and mentions how many more exist."""
    classes = tuple(
        MultiLectCorrespondenceClass(
            class_id=i,
            segments={"A": chr(ord("a") + i), "B": chr(ord("a") + i)},
            count=float(10 - i),
        )
        for i in range(5)
    )
    model = MultiLectModel(
        pairwise_models={},
        unconditioned_classes=classes,
        conditioned_classes=(),
        cognate_corpus=(),
        lect_ids=("A", "B"),
    )
    out = format_multi_lect_model(model, top_unconditioned=2)
    assert "3 more" in out


def test_format_renders_conditioned_classes_with_contexts() -> None:
    front = Context(following=(FeatureConstraint("front", "+"),))
    cond = MultiLectCorrespondenceClass(
        class_id=0,
        segments={"A": "k", "B": "s"},
        contexts={"A": front, "B": Context()},
        count=8.0,
    )
    model = MultiLectModel(
        pairwise_models={},
        unconditioned_classes=(),
        conditioned_classes=(cond,),
        cognate_corpus=(),
        lect_ids=("A", "B"),
    )
    out = format_multi_lect_model(model)
    assert "A:k" in out
    assert "B:s" in out
    # Context appears with the pivot lect label and feature name.
    assert "A=" in out
    assert "front" in out


def test_format_groups_identical_contexts_across_lects() -> None:
    """COMMITMENT: when multiple lects in a conditioned class share
    exactly the same context (common after multi-lect context discovery pivot-lect dedup),
    the formatter collapses them into one ``{lect1,lect2,...}=ctx``
    entry instead of listing each lect separately."""
    front = Context(following=(FeatureConstraint("front", "+"),))
    cond = MultiLectCorrespondenceClass(
        class_id=0,
        segments={"A": "ʔ", "B": "k", "C": "k", "D": "ʔ"},
        # A, C, D share the front-vowel context; B has empty.
        contexts={
            "A": front,
            "B": Context(),
            "C": front,
            "D": front,
        },
        count=6.0,
    )
    model = MultiLectModel(
        pairwise_models={},
        unconditioned_classes=(),
        conditioned_classes=(cond,),
        cognate_corpus=(),
        lect_ids=("A", "B", "C", "D"),
    )
    out = format_multi_lect_model(model)
    # The 3 lects sharing the context should be collapsed.
    assert "{A,C,D}=" in out
    # The empty-context lect should not appear in the display.
    assert "B=" not in out


def test_format_shows_per_class_confidence() -> None:
    """COMMITMENT: format_multi_lect_model prints the per-class
    coverage confidence alongside the count for conditioned
    classes."""
    cond = MultiLectCorrespondenceClass(
        class_id=0,
        segments={"A": "k", "B": "s"},
        contexts={"A": Context(following=(FeatureConstraint("front", "+"),))},
        count=8.0,
        confidence=0.73,
    )
    model = MultiLectModel(
        pairwise_models={},
        unconditioned_classes=(),
        conditioned_classes=(cond,),
        cognate_corpus=(),
        lect_ids=("A", "B"),
    )
    out = format_multi_lect_model(model)
    assert "cov=0.73" in out


# ----- describe_multi_lect_class -----------------------------------------


def test_describe_class_empty_model_shows_no_results() -> None:
    out = describe_multi_lect_class(MultiLectModel.empty(), "A", "k")
    assert "A:k" in out
    assert "(none)" in out


def test_describe_class_shows_matching_unconditioned_and_conditioned() -> None:
    uncond = MultiLectCorrespondenceClass(
        class_id=0,
        segments={"A": "k", "B": "k"},
        count=5.0,
    )
    front = Context(following=(FeatureConstraint("front", "+"),))
    cond = MultiLectCorrespondenceClass(
        class_id=1,
        segments={"A": "k", "B": "s"},
        contexts={"A": front, "B": Context()},
        count=3.0,
    )
    other = MultiLectCorrespondenceClass(
        class_id=2,
        segments={"A": "p", "B": "p"},
        count=7.0,
    )
    model = MultiLectModel(
        pairwise_models={},
        unconditioned_classes=(uncond, other),
        conditioned_classes=(cond,),
        cognate_corpus=(),
        lect_ids=("A", "B"),
    )
    out = describe_multi_lect_class(model, "A", "k")
    # Matches for A:k
    assert "A:k B:k" in out
    assert "A:k B:s" in out
    assert "Unconditioned entries (1)" in out
    assert "Conditioned entries (1)" in out
    # Other grapheme not present.
    assert "A:p" not in out


def test_describe_class_no_matches_shows_zero_counts() -> None:
    uncond = MultiLectCorrespondenceClass(
        class_id=0,
        segments={"A": "p", "B": "p"},
        count=5.0,
    )
    model = MultiLectModel(
        pairwise_models={},
        unconditioned_classes=(uncond,),
        conditioned_classes=(),
        cognate_corpus=(),
        lect_ids=("A", "B"),
    )
    out = describe_multi_lect_class(model, "A", "k")
    assert "Unconditioned entries (0)" in out
    assert "Conditioned entries (0)" in out


def test_format_integrates_with_trained_palatalization_model() -> None:
    """SMOKE: format_multi_lect_model runs cleanly on a real trained
    MultiLectModel that has both unconditioned and conditioned
    classes."""
    pairs = [
        ("kita", "sita"),
        ("kite", "site"),
        ("keta", "seta"),
        ("ketu", "setu"),
        ("kata", "kata"),
        ("kota", "kota"),
        ("kupa", "kupa"),
        ("kuma", "kuma"),
    ]
    corpus = [
        CognateSet(
            cognate_id=f"c{i}",
            forms={"A": _form("A", s), "B": _form("B", t)},
        )
        for i, (s, t) in enumerate(pairs)
    ]
    model = train_model(corpus)
    assert isinstance(model, MultiLectModel)
    out = format_multi_lect_model(model)
    assert "MultiLectModel" in out
    assert "A" in out
    assert "B" in out
